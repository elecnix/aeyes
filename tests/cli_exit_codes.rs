//! The `aeyes motion` exit-code contract, exercised through the shipped binary.
//!
//! ```text
//! 0 = motion passed the filter
//! 1 = no motion within the budget        (pinned in src/lib.rs, needs a daemon)
//! >= 2 = the command failed
//! ```
//!
//! Exit 2 is the case that can be reached from a test without a camera, and it
//! is the one that matters most: it is the difference between "nothing moved"
//! and "the daemon is dead", which is exactly what a script could not tell
//! before. `TMPDIR`/`TMP`/`TEMP` are redirected into a temporary sandbox so the
//! daemon address and pid files this binary looks for cannot collide with a
//! real daemon's.

use std::process::{Command, Stdio};

#[test]
fn motion_exits_2_when_the_camera_cannot_be_resolved() {
    let sandbox = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_aeyes"))
        // A camera that cannot exist on any machine: resolving it fails before
        // any camera is opened, so this is deterministic.
        .args(["motion", "--camera", "aeyes-no-such-camera"])
        .env("TMPDIR", sandbox.path())
        .env("TMP", sandbox.path())
        .env("TEMP", sandbox.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("failed to run the aeyes binary");

    assert_eq!(
        status.code(),
        Some(2),
        "an unreachable/broken invocation must exit >= 2, not 1"
    );
}

#[test]
fn motion_rejects_an_unparseable_budget() {
    let sandbox = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_aeyes"))
        .args(["motion", "--timeout", "soon"])
        .env("TMPDIR", sandbox.path())
        .env("TMP", sandbox.path())
        .env("TEMP", sandbox.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("failed to run the aeyes binary");

    // clap rejects the value; the point is that it is a failure (>= 2) rather
    // than "no motion" (1), which a caller would read as a measurement.
    assert!(status.code().unwrap_or(0) >= 2);
}

#[test]
fn motion_flags_that_shape_the_detector_are_gone_from_the_client() {
    let sandbox = tempfile::tempdir().unwrap();
    for flag in ["--threshold", "--no-sobel", "--no-lbp"] {
        let status = Command::new(env!("CARGO_BIN_EXE_aeyes"))
            .args(["motion", flag, "1"])
            .env("TMPDIR", sandbox.path())
            .env("TMP", sandbox.path())
            .env("TEMP", sandbox.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("failed to run the aeyes binary");

        assert!(
            status.code().unwrap_or(0) >= 2,
            "{flag} must no longer be accepted on the client; it is a daemon setting now"
        );
    }
}

#[test]
fn start_flags_that_shape_the_detector_are_accepted_by_the_daemon() {
    let sandbox = tempfile::tempdir().unwrap();
    // `--help` for the start subcommand must mention the daemon-side motion
    // settings, so the documented surface is the shipped surface.
    let output = Command::new(env!("CARGO_BIN_EXE_aeyes"))
        .args(["start", "--help"])
        .env("TMPDIR", sandbox.path())
        .env("TMP", sandbox.path())
        .env("TEMP", sandbox.path())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("failed to run the aeyes binary");

    assert_eq!(output.status.code(), Some(0));
    let help = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--motion-hz",
        "--motion-width",
        "--motion-threshold",
        "--motion-no-sobel",
        "--motion-no-lbp",
    ] {
        assert!(help.contains(flag), "aeyes start must document {flag}");
    }
}
