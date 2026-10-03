//! `aeyes stop` and `aeyes status` must be honest about what they actually did.
//!
//! Three defects are pinned here, all from the review of #42:
//!
//! * `stop` printed "Daemon stopped." even on the branch where it *refused* to
//!   signal the pid, because the pid belonged to a live process that was not this
//!   executable. The daemon kept serving and the user was told otherwise.
//! * On that same refusal it deleted `daemon.pid`/`daemon.addr` — throwing away
//!   the only handle on a daemon it had just declined to signal.
//! * The fallback probe decided a daemon was there with a bare `TcpStream::connect`,
//!   so `stop` sent `GET /shutdown` to *whatever* held the bind port and then
//!   reported success. A daemon that never published the registry was invisible
//!   for the same reason: `status` called a listening port "not running".
//!
//! All tests drive the real binary in a subprocess so they never mutate the
//! developer's environment. `TMPDIR` redirects the runtime registry into a
//! private directory and `AEYES_BIND` redirects the fallback probe onto a port
//! chosen by the test, so a real daemon on the machine is left alone.

use std::collections::VecDeque;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

/// An ephemeral port in the kernel's ephemeral range, for tests that need a
/// port *nobody is listening on*.
///
/// The listener is dropped before the port is handed out, so this is a
/// momentary claim, not a reservation — another process could in principle take
/// the port in the gap. That is the standard trade-off for this technique and it
/// is acceptable here because the assertions do not depend on the port staying
/// empty: a bystander that grabbed it would answer the identity probe as
/// "not aeyes", which is exactly what these tests want to see. What this does
/// guarantee is that the port is not the fixed 43210 a real daemon uses.
#[cfg(target_os = "linux")]
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

/// A stand-in HTTP server that records every request path it is asked for.
///
/// Two personalities matter to `stop` and `status`, and neither can be faked
/// with a bare `TcpListener`:
///
/// * `aeyes()` answers `/health` with `ok` and `/` with this daemon's OpenAPI
///   document — i.e. it identifies as aeyes, so `stop` may shut it down.
/// * `foreign()` answers `/health` with `ok` and 404s everything else — it looks
///   plausible to a health check but is not aeyes, so `stop` must leave it
///   alone.
struct RecordingPeer {
    addr: SocketAddr,
    requests: Arc<Mutex<VecDeque<String>>>,
}

impl RecordingPeer {
    fn start(responds_as_aeyes: bool) -> RecordingPeer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(VecDeque::new()));
        let log = Arc::clone(&requests);

        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let log = Arc::clone(&log);
                std::thread::spawn(move || {
                    // Read until the end of the request headers. A single `read` can
                    // return a partial line, and stopping at the first `\r\n` would both
                    // mis-parse a truncated request and leave the peer's write half
                    // unfinished, which the client sees as a broken connection.
                    let mut raw: Vec<u8> = Vec::new();
                    let mut chunk = [0u8; 1024];
                    loop {
                        match std::io::Read::read(&mut stream, &mut chunk) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                raw.extend_from_slice(&chunk[..n]);
                                if raw.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    let request = String::from_utf8_lossy(&raw).to_string();
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let body: Option<Vec<u8>> = match path.as_str() {
                        "/health" => Some(b"ok".to_vec()),
                        "/" if responds_as_aeyes => Some(
                            serde_json::to_vec(&serde_json::json!({
                                "openapi": "3.0.3",
                                "info": { "title": "aeyes", "version": "0.1.0" }
                            }))
                            .expect("serialize openapi"),
                        ),
                        _ => None,
                    };
                    log.lock().expect("log lock").push_back(path.clone());

                    let response = match body {
                        Some(body) => {
                            let mut r = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            )
                            .into_bytes();
                            r.extend_from_slice(&body);
                            r
                        }
                        None => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    };
                    let _ = std::io::Write::write_all(&mut stream, &response);
                    let _ = std::io::Write::flush(&mut stream);
                });
            }
        });

        RecordingPeer { addr, requests }
    }

    /// Wait briefly for `path` to show up, so the assertion is not racing the
    /// server thread.
    fn await_request(&self, path: &str) -> bool {
        for _ in 0..100 {
            if self.paths().iter().any(|p| p == path) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        false
    }

    fn paths(&self) -> Vec<String> {
        self.requests
            .lock()
            .expect("log lock")
            .iter()
            .cloned()
            .collect()
    }
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
}

/// A refusal must not throw away the registry on its way out.
///
/// The pid file is the only handle this process has on a daemon it just declined
/// to signal. Deleting it leaves that daemon running with nothing left to stop
/// it: the next `stop` finds no registry at all and falls through to probing the
/// bind address, which is the very path that cannot attribute a pid.
#[cfg(target_os = "linux")]
#[test]
fn refusal_to_signal_keeps_the_runtime_registry() {
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
    let pid_file = registry.join("daemon.pid");
    std::fs::write(&pid_file, pid.to_string()).expect("write pid file");

    let out = aeyes(dir.path(), free_port(), &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    let _ = bystander.kill();
    let _ = bystander.wait();

    assert!(
        stdout.contains("Did NOT stop"),
        "stop did not report the refusal; stdout: {stdout}"
    );
    assert_eq!(
        std::fs::read_to_string(&pid_file).ok(),
        Some(pid.to_string()),
        "stop deleted the runtime registry on the refusal path, leaving a live daemon with no handle on it; stdout: {stdout}"
    );
}

/// A live pid whose executable cannot be read must not be treated as a pid that
/// is not there.
///
/// `fs::read_link("/proc/<pid>/exe")` fails both when the pid is gone and when
/// the process is not ours to inspect — another user owns it, `/proc` is
/// mounted `hidepid`. Collapsing the two into `Absent` meant `stop` deleted the
/// registry while a live process held the port. A zombie reproduces the exact
/// shape (`/proc/<pid>` present, `exe` unreadable) and is the only version of it
/// an unprivileged test binary can arrange.
///
/// Only Linux: `/proc` is the whole mechanism.
#[cfg(target_os = "linux")]
#[test]
fn stop_keeps_the_registry_when_the_pid_is_live_but_uninspectable() {
    let dir = tempfile::tempdir().expect("tempdir");

    // A child that exits but is never waited on stays in the process table.
    let mut child = Command::new("true")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn");
    let pid = child.id();

    // Wait for the uninspectable state to be real rather than a race we happened
    // to win. The sleep matters as much as the poll: the child may still be
    // running on the first attempt, and an unslept loop burns all its attempts
    // before the state it is waiting for exists.
    let mut uninspectable = false;
    for _ in 0..500 {
        if Path::new(&format!("/proc/{pid}")).exists()
            && std::fs::read_link(format!("/proc/{pid}/exe")).is_err()
        {
            uninspectable = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        uninspectable,
        "PID {pid} never reached a state with a /proc entry but no readable exe"
    );

    let registry = dir.path().join("aeyes");
    std::fs::create_dir_all(&registry).expect("create registry dir");
    let pid_file = registry.join("daemon.pid");
    std::fs::write(&pid_file, pid.to_string()).expect("write pid file");

    let out = aeyes(dir.path(), free_port(), &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    let _ = child.wait();

    assert!(
        stdout.contains("Did NOT stop"),
        "stop signalled a pid it could not confirm was its own; stdout: {stdout}"
    );
    assert_eq!(
        std::fs::read_to_string(&pid_file).ok(),
        Some(pid.to_string()),
        "stop deleted the runtime registry for a live-but-uninspectable pid; stdout: {stdout}"
    );
}

/// The same identity check has to cover the *registered* address, not only the
/// fallback.
///
/// `daemon.addr` was written by a daemon that has since died; the OS is free to
/// hand that port to anything. If `stop` trusted the file and used a bare
/// connect, it would deliver `GET /shutdown` to that stranger — the same harm
/// as the fallback case, reached without the probe.
///
/// Pre-fix this fails: the registered branch used `daemon_responding`, which is
/// a bare connect.
#[test]
fn stop_does_not_shut_down_a_foreign_service_at_the_registered_address() {
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = dir.path().join("aeyes");
    std::fs::create_dir_all(&registry).expect("create registry dir");

    let peer = RecordingPeer::start(false);
    std::fs::write(registry.join("daemon.addr"), peer.addr.to_string()).expect("write addr file");

    // Probe a port nobody is on, so the registered address is the only candidate.
    let quiet = RecordingPeer::start(false);
    let out = aeyes(dir.path(), quiet.addr, &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        !peer.await_request("/shutdown"),
        "aeyes stop sent /shutdown to a foreign service at the registered address {} (requests: {:?}); stdout: {stdout}",
        peer.addr,
        peer.paths()
    );
}

/// The fallback probe must establish that the peer is *aeyes* before `stop`
/// points a `GET /shutdown` at it.
///
/// Pre-fix this fails: the probe was a bare `TcpStream::connect`, so any TCP
/// listener on the bind port was enough. Here the peer speaks HTTP and even
/// answers `/health` with `ok` — it is simply not an aeyes daemon, and it must
/// not be shut down by `aeyes stop`.
#[test]
fn stop_does_not_shut_down_a_foreign_service_on_the_bind_port() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("aeyes")).expect("create registry dir");

    let peer = RecordingPeer::start(false);
    let out = aeyes(dir.path(), peer.addr, &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        !peer.await_request("/shutdown"),
        "aeyes stop sent /shutdown to a foreign service on {} (requests: {:?}); stdout: {stdout}",
        peer.addr,
        peer.paths()
    );
    assert!(
        !stdout.contains("Daemon stopped at"),
        "stop reported shutting down a daemon it never verified; stdout: {stdout}"
    );
}

/// The counterpart: a peer that really does answer as aeyes on the bind port is
/// a daemon started outside `aeyes start`, and `stop` must still reach it.
#[test]
fn stop_reaches_an_unregistered_daemon_that_identifies_as_aeyes() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("aeyes")).expect("create registry dir");

    let peer = RecordingPeer::start(true);
    let out = aeyes(dir.path(), peer.addr, &["stop"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        peer.await_request("/shutdown"),
        "stop never reached the unregistered aeyes daemon on {} (requests: {:?}); stdout: {stdout}",
        peer.addr,
        peer.paths()
    );
    assert!(
        stdout.contains("Daemon stopped at"),
        "stop did not report the daemon it stopped; stdout: {stdout}"
    );
}

/// A daemon that never published the runtime registry is still a running
/// daemon. `status` must not answer "Daemon not running." for a port that is
/// listening.
#[test]
fn status_reports_a_listening_daemon_that_is_not_registered() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("aeyes")).expect("create registry dir");

    let peer = RecordingPeer::start(true);
    let out = aeyes(dir.path(), peer.addr, &["status"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains(&peer.addr.to_string()),
        "status did not report the unregistered daemon at {}; stdout: {stdout}",
        peer.addr
    );
    assert!(
        !stdout.contains("Daemon not running."),
        "status called a listening daemon 'not running'; stdout: {stdout}"
    );
}

/// ...and the converse: an unrelated service on the bind port is not an aeyes
/// daemon, so `status` must not report it as one.
#[test]
fn status_does_not_report_a_foreign_service_as_a_daemon() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("aeyes")).expect("create registry dir");

    let peer = RecordingPeer::start(false);
    let out = aeyes(dir.path(), peer.addr, &["status"]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("Daemon not running."),
        "status reported a foreign service on {} as an aeyes daemon; stdout: {stdout}",
        peer.addr
    );
}
