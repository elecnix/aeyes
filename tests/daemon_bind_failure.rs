//! The runtime registry must only ever describe a socket that is really listening.
//!
//! `run_daemon` used to write `daemon.pid` / `daemon.addr` *before* calling
//! `TcpListener::bind`. When the bind failed — which is the normal outcome when
//! another daemon already owns the port — the process exited via `?` and left
//! both files behind describing a daemon that never existed. `stop_daemon` then
//! read that dead pid and signalled it. See elecnix/aeyes#40.
//!
//! This lives in its own integration-test binary because it mutates
//! `AEYES_DAEMON`, and environment mutation must not race other tests.

use aeyes::{
    run_daemon, runtime_dir, CameraBackend, CameraDescriptor, CameraOpenOptions, OpenCamera,
};
use anyhow::Result;
use std::net::TcpListener;

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

/// A daemon that looks like a real one (`AEYES_DAEMON` set) but loses the race
/// for the port must leave the shared registry untouched.
///
/// Pre-fix this test fails: the files are written before the bind, so the failed
/// bind leaves a pid/addr pair behind naming a process that has already exited.
#[tokio::test]
async fn failed_bind_leaves_no_stale_registry_files() {
    std::env::set_var("AEYES_DAEMON", "1");

    let before = read_registry();

    // Hold the port for the whole test so the daemon's bind is guaranteed to fail
    // with EADDRINUSE rather than succeeding.
    let squatter = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let bind = squatter.local_addr().expect("local addr");

    let result = run_daemon(
        bind,
        "cam-a".into(),
        CameraOpenOptions::default(),
        Box::new(FakeBackend),
    )
    .await;

    assert!(
        result.is_err(),
        "daemon unexpectedly bound a port that was already held"
    );

    assert_eq!(
        read_registry(),
        before,
        "a daemon whose bind failed published stale pid/addr files"
    );
}
