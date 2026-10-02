//! Regression tests for the daemon runtime registry (`daemon.pid` / `daemon.addr`).
//!
//! The registry lives in a world-shared path (`runtime_dir()`, i.e.
//! `/tmp/aeyes`) that every process on the machine reads and writes. These tests
//! pin the invariant that only the process actually serving the socket publishes
//! into it — see elecnix/aeyes#40.

use aeyes::{
    run_daemon, runtime_dir, CameraBackend, CameraDescriptor, CameraOpenOptions, OpenCamera,
};
use anyhow::Result;
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

/// Grab an ephemeral port that is free right now, so the daemon under test can
/// bind it without colliding with the other tests in this suite.
fn free_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr")
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

/// An in-process `run_daemon` call — the shape the unit test suite uses — must
/// not write the machine-global registry.
///
/// Before this was fixed, every one of the seven tests that spawn `run_daemon`
/// unconditionally overwrote `/tmp/aeyes/daemon.{pid,addr}`. Running `cargo test`
/// on a workstation that had a real daemon running therefore silently repointed
/// every client at a throwaway test port and replaced the live daemon's pid with
/// a dead one — after which `aeyes stop` would signal that dead (and possibly
/// recycled) pid while the actual daemon kept running.
#[tokio::test]
async fn in_process_daemon_does_not_publish_to_shared_registry() {
    // This test asserts the *absence* of publication, so it must not itself be
    // running in a process that looks like a spawned daemon.
    std::env::remove_var("AEYES_DAEMON");

    let before = read_registry();
    let bind = free_port();

    let handle = tokio::spawn(run_daemon(
        bind,
        "cam-a".into(),
        CameraOpenOptions::default(),
        Box::new(FakeBackend),
    ));

    // Give the daemon long enough to reach the point where it would publish if
    // it were going to.
    tokio::time::sleep(Duration::from_millis(600)).await;
    handle.abort();
    let _ = handle.await;

    assert_eq!(
        read_registry(),
        before,
        "in-process run_daemon wrote to the shared runtime registry"
    );
}

/// A successful in-process daemon does bind its port — proving the test above
/// exercised a daemon that actually came up rather than one that failed early
/// and therefore never reached the publish step.
#[tokio::test]
async fn in_process_daemon_actually_binds_its_port() {
    std::env::remove_var("AEYES_DAEMON");

    let bind = free_port();
    let handle = tokio::spawn(run_daemon(
        bind,
        "cam-a".into(),
        CameraOpenOptions::default(),
        Box::new(FakeBackend),
    ));

    tokio::time::sleep(Duration::from_millis(600)).await;

    let reachable = tokio::net::TcpStream::connect(bind).await.is_ok();
    handle.abort();
    let _ = handle.await;

    assert!(
        reachable,
        "daemon under test never bound {bind}, so the registry assertions are vacuous"
    );
}
