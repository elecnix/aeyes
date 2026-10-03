//! Regression tests for the daemon runtime registry (`daemon.pid` / `daemon.addr`).
//!
//! The registry lives in a world-shared path (`runtime_dir()`, i.e.
//! `/tmp/aeyes`) that every process on the machine reads and writes. These tests
//! pin the invariant that only the process actually serving the socket publishes
//! into it — see elecnix/aeyes#40.
//!
//! Two rules keep this file honest:
//!
//! * Every test that touches `AEYES_DAEMON` carries `#[serial]`. The variable is
//!   process-global while cargo runs this binary's tests on parallel threads, so
//!   an unserialised `set_var`/`remove_var` races the daemon code under test and
//!   races its own sibling tests.
//! * No test asserts absence off a fixed sleep. A test that sleeps 600ms and
//!   then checks "nothing was published" also passes when the daemon never came
//!   up at all, which is exactly the state where the assertion means nothing.
//!   Each test first proves its daemon reached the listening state, then asserts.

use aeyes::{
    run_daemon, runtime_dir, CameraBackend, CameraDescriptor, CameraOpenOptions, OpenCamera,
};
use anyhow::Result;
use serial_test::serial;
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

/// Grab an ephemeral port that is free right now, so the daemon under test can
/// bind it without colliding with the other tests in this suite.
fn free_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr")
}

/// How long a daemon under test gets to bind its port before the fixture is
/// declared broken. Generous, because CI machines are slow; still bounded, so a
/// daemon that never starts fails the test instead of hanging it.
const STARTUP_BUDGET: Duration = Duration::from_secs(10);

/// Remove `AEYES_DAEMON` for the duration of a test and restore it afterwards.
///
/// Restoring matters even under `#[serial]`: these tests share a process with
/// each other, and a leaked "1" would silently turn the next test's daemon into
/// a registry-publishing one.
fn without_daemon_env() -> Option<String> {
    let previous = std::env::var("AEYES_DAEMON").ok();
    std::env::remove_var("AEYES_DAEMON");
    previous
}

struct FakeBackend;

impl CameraBackend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn list_cameras(&self) -> Result<Vec<CameraDescriptor>> {
        Ok(vec![CameraDescriptor {
            id: "cam-a".into(),
            name: "Fake Cam".into(),
            backend: "fake".into(),
        }])
    }

    fn open(&self, _id: &str, _options: &CameraOpenOptions) -> Result<Box<dyn OpenCamera>> {
        Ok(Box::new(FakeCamera))
    }
}

struct FakeCamera;

impl OpenCamera for FakeCamera {
    fn set_auto_features(&mut self) -> Result<()> {
        Ok(())
    }

    fn capture_jpeg(&mut self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

fn read_registry() -> (Option<String>, Option<String>) {
    let pid = std::fs::read_to_string(runtime_dir().join("daemon.pid")).ok();
    let addr = std::fs::read_to_string(runtime_dir().join("daemon.addr")).ok();
    (pid, addr)
}

/// Poll `bind` until the daemon under test actually serves HTTP.
///
/// Returns whether it ever did; the caller must treat `false` as a broken
/// fixture, not as a licence to skip its assertions. The probe is a real
/// request rather than a bare `TcpStream::connect` because the connect succeeds
/// against any listener with spare backlog — including a squatter holding the
/// port, which would make "the daemon came up" true for a daemon that did not.
async fn wait_until_serving(bind: SocketAddr) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .expect("build client");
    let deadline = tokio::time::Instant::now() + STARTUP_BUDGET;
    loop {
        let serving = client
            .get(format!("http://{bind}/health"))
            .send()
            .await
            .is_ok_and(|resp| resp.status().is_success());
        if serving {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// An in-process `run_daemon` call — the shape the unit test suite uses — must
/// not write the machine-global registry.
///
/// Before this was fixed, every one of the seven tests that spawn `run_daemon`
/// unconditionally overwrote `/tmp/aeyes/daemon.{pid,addr}`. Running `cargo test`
/// on a workstation that had a real daemon running therefore silently repointed
/// every client at a throwaway test port and replaced the live daemon's pid with
/// a dead one — after which `aeyes stop` would signal that dead (and possibly
/// recycled) pid while the actual daemon kept running.
///
/// The registry is read *after* the daemon is proven to be listening, which is
/// the point at which `publish_runtime_registry` would already have run. Reading
/// it on a timer instead would pass just as happily for a daemon that never came
/// up.
#[tokio::test]
#[serial]
async fn in_process_daemon_does_not_publish_to_shared_registry() {
    // This test asserts the *absence* of publication, so it must not itself be
    // running in a process that looks like a spawned daemon.
    let previous = without_daemon_env();

    let before = read_registry();
    let bind = free_port();

    let handle = tokio::spawn(run_daemon(
        bind,
        "cam-a".into(),
        CameraOpenOptions::default(),
        Box::new(FakeBackend),
    ));

    let serving = wait_until_serving(bind).await;
    // Sampled after the daemon is serving: past the publish step, so the
    // comparison below covers the whole window in which it could have written.
    let after = read_registry();

    handle.abort();
    let _ = handle.await;
    if let Some(value) = previous {
        std::env::set_var("AEYES_DAEMON", value);
    }

    assert!(
        serving,
        "daemon under test never served {bind} within {STARTUP_BUDGET:?}, so the registry assertion would be vacuous"
    );
    assert_eq!(
        after, before,
        "in-process run_daemon wrote to the shared runtime registry"
    );
}

/// A successful in-process daemon does bind its port — proving the test above
/// exercised a daemon that actually came up rather than one that failed early
/// and therefore never reached the publish step.
///
/// This is the sibling reachability check the test above leans on, kept on its
/// own so a regression in the fixture is diagnosed here rather than showing up
/// as a confusing registry failure.
#[tokio::test]
#[serial]
async fn in_process_daemon_actually_binds_its_port() {
    let previous = without_daemon_env();

    let bind = free_port();
    let handle = tokio::spawn(run_daemon(
        bind,
        "cam-a".into(),
        CameraOpenOptions::default(),
        Box::new(FakeBackend),
    ));

    let serving = wait_until_serving(bind).await;

    handle.abort();
    let _ = handle.await;
    if let Some(value) = previous {
        std::env::set_var("AEYES_DAEMON", value);
    }

    assert!(
        serving,
        "daemon under test never served {bind} within {STARTUP_BUDGET:?}, so the registry assertions are vacuous"
    );
}
