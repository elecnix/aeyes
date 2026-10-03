//! `aeyes stop` and `aeyes status` must be honest about what they actually did.
//!
//! Two defects are pinned here, both from the review of #42:
//!
//! * `stop` printed "Daemon stopped." even on the branch where it *refused* to
//!   signal the pid, because the pid belonged to a live process that was not this
//!   executable. The daemon kept serving and the user was told otherwise.
//! * A daemon that never published the runtime registry (anything not spawned by
//!   `aeyes start`) was reported as "Daemon not running." by `status` and was
//!   invisible to `stop`, even though its port was listening.
//!
//! Both tests drive the real binary in a subprocess so they never mutate the
//! developer's environment. `TMPDIR` redirects the runtime registry into a
//! private directory and `AEYES_BIND` redirects the fallback probe onto a port
//! chosen by the test, so a real daemon on the machine is left alone.

use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::process::Command;

/// An ephemeral port that is free right now, so nothing the test does can collide
/// with a daemon actually running on this machine.
fn free_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr")
}

/// Run the real `aeyes` binary with the registry and the fallback probe pointed
/// at test-private locations.
fn aeyes(dir: &Path, probe: SocketAddr, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aeyes"))
        .args(args)
        .env("TMPDIR", dir)
        .env("AEYES_BIND", probe.to_string())
        .output()
        .expect("run aeyes")
}

#[cfg(target_os = "linux")]
fn pid_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// `stop` must not signal a pid that belongs to something other than this
/// aeyes executable — and when it declines, it must say so instead of printing
/// "Daemon stopped.".
///
/// The bystander is a `sleep`, whose `/proc/<pid>/exe` resolves to `/bin/sleep`,
/// i.e. exactly the case that a substring test on the path would have waved
/// through. Only Linux can tell the two apart; on other unix targets the check
/// degrades to liveness by design (see `classify_daemon_pid`), so the assertion
/// is Linux-only.
#[cfg(target_os = "linux")]
#[test]
fn stop_refuses_to_signal_a_live_process_that_is_not_aeyes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut bystander = Command::new("sleep")
        .arg("60")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn bystander");
    let pid = bystander.id();

    let registry = dir.path().join("aeyes");
    std::fs::create_dir_all(&registry).expect("create registry dir");
    std::fs::write(registry.join("daemon.pid"), pid.to_string()).expect("write pid file");

    // No daemon.addr file: the registry claims a pid but names no address, which
    // is the state left behind by a crash, a failed bind or a lost start-up race.
    let out = aeyes(dir.path(), free_port(), &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    let survived = pid_is_alive(pid);
    let _ = bystander.kill();
    let _ = bystander.wait();

    assert!(
        survived,
        "aeyes stop signalled PID {pid}, which is a bystander, not an aeyes daemon (stdout: {stdout})"
    );
    assert!(
        stdout.contains("Did NOT stop"),
        "stop did not report the refusal; stdout: {stdout}"
    );
    assert!(
        !stdout.contains("Daemon stopped."),
        "stop reported success for an action it declined to take; stdout: {stdout}"
    );
    assert!(
        !registry.join("daemon.pid").exists(),
        "the stale registry was not cleared after the refusal"
    );
}

/// A daemon that never published the runtime registry is still a running
/// daemon. `status` must not answer "Daemon not running." for a port that is
/// listening.
#[test]
fn status_reports_a_listening_daemon_that_is_not_registered() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("aeyes")).expect("create registry dir");

    // Hold the probe port for the whole test: from `status`'s point of view this
    // is a daemon serving requests but absent from the registry.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let probe = listener.local_addr().expect("local addr");

    let out = aeyes(dir.path(), probe, &["status"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains(&probe.to_string()),
        "status did not report the unregistered daemon at {probe}; stdout: {stdout}"
    );
    assert!(
        !stdout.contains("Daemon not running."),
        "status called a listening daemon 'not running'; stdout: {stdout}"
    );
}
