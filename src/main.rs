use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use image::RgbImage;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::{info, warn, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "aeyes", about = "AI Eyes - Fast webcam captures via daemon")]
struct Cli {
    #[arg(long)]
    daemon: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the daemon
    Start,
    /// Capture image from daemon
    #[command(arg_required_else_help = false)]
    Capture {
        /// Output path for the captured frame
        #[arg(short, long, default_value_os_t = default_capture_output())]
        output: PathBuf,
    },
    /// Stop the daemon
    Stop,
    /// Check daemon status
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let cli = Cli::parse();

    if cli.daemon {
        run_daemon().await?;
    } else {
        match cli.command {
            Some(Commands::Start {}) => start_daemon().await?,
            Some(Commands::Capture { output }) => capture(&output).await?,
            Some(Commands::Stop {}) => stop_daemon().await?,
            Some(Commands::Status {}) => status().await?,
            None => {
                Cli::command().print_help()?;
                println!();
            }
        }
    }
    Ok(())
}

const DAEMON_ADDR: &str = "127.0.0.1:47321";

fn runtime_dir() -> PathBuf {
    env::temp_dir().join("aeyes")
}

fn pid_path() -> PathBuf {
    runtime_dir().join("aeyes.pid")
}

fn default_capture_output() -> PathBuf {
    runtime_dir().join("aeyes-last.jpg")
}

fn ensure_runtime_dir() -> Result<PathBuf> {
    let dir = runtime_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create runtime dir at {}", dir.display()))?;
    Ok(dir)
}

fn parse_capture_command(cmd: &str) -> Option<PathBuf> {
    cmd.strip_prefix("capture ")
        .and_then(|s| s.strip_suffix('\n'))
        .map(|s| PathBuf::from(s.trim()))
}

async fn connect_daemon() -> Result<TcpStream> {
    TcpStream::connect(DAEMON_ADDR)
        .await
        .with_context(|| format!("failed to connect to daemon at {DAEMON_ADDR}"))
}

#[cfg(windows)]
async fn terminate_process(pid: u32) -> Result<()> {
    let status = tokio::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()
        .await
        .with_context(|| format!("failed to invoke taskkill for PID {pid}"))?;

    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("taskkill failed for PID {pid} with status {status}")
    }
}

#[cfg(not(windows))]
async fn terminate_process(pid: u32) -> Result<()> {
    let status = tokio::process::Command::new("kill")
        .arg(pid.to_string())
        .status()
        .await
        .with_context(|| format!("failed to invoke kill for PID {pid}"))?;

    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("kill failed for PID {pid} with status {status}")
    }
}

async fn is_running() -> bool {
    TcpStream::connect(DAEMON_ADDR).await.is_ok()
}

async fn start_daemon() -> Result<()> {
    if is_running().await {
        println!("Daemon is already running.");
        return Ok(());
    }

    ensure_runtime_dir()?;

    let exe = env::current_exe()?;
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.arg("--daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let child = cmd.spawn()?;
    let pid = child.id().context("spawned daemon process did not report a PID")?;
    std::fs::write(pid_path(), pid.to_string().as_bytes())?;

    info!("Daemon started with PID {}", pid);
    println!("Daemon started (PID: {})", pid);

    sleep(Duration::from_millis(500)).await;
    Ok(())
}

async fn stop_daemon() -> Result<()> {
    if let Ok(pid_str) = std::fs::read_to_string(pid_path()) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            if let Err(error) = terminate_process(pid).await {
                warn!("Failed to stop daemon process {}: {:?}", pid, error);
            }
        }
    }
    let _ = std::fs::remove_file(pid_path());
    println!("Daemon stopped.");
    Ok(())
}

async fn status() -> Result<()> {
    if is_running().await {
        println!("Daemon is running.");
    } else {
        println!("Daemon is not running.");
    }
    Ok(())
}

async fn capture(output: &Path) -> Result<()> {
    if !is_running().await {
        anyhow::bail!("Daemon is not running. Run `aeyes start` first.");
    }

    ensure_runtime_dir()?;

    let mut stream = connect_daemon().await?;
    let cmd = format!("capture {}\n", output.to_string_lossy());
    stream.write_all(cmd.as_bytes()).await?;

    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await?;
    let response = String::from_utf8_lossy(&buf[..n]).trim().to_string();

    match response.as_str() {
        "OK" => {
            println!("Captured to {:?}.", output);
            Ok(())
        }
        other => anyhow::bail!("Daemon error: {}", other),
    }
}

async fn run_daemon() -> Result<()> {
    ensure_runtime_dir()?;
    let _ = std::fs::remove_file(pid_path());

    let listener = TcpListener::bind(DAEMON_ADDR).await?;
    let pid = std::process::id();
    std::fs::write(pid_path(), pid.to_string().as_bytes())?;

    info!("Daemon running, listening on {}", DAEMON_ADDR);

    let latest_image: Arc<RwLock<Option<Arc<RgbImage>>>> = Arc::new(RwLock::new(None));

    // Camera task
    let camera_latest = Arc::clone(&latest_image);
    tokio::spawn(async move {
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = match Camera::new(CameraIndex::Index(0), requested) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to create camera: {:?}", e);
                return;
            }
        };
        if let Err(e) = camera.open_stream() {
            warn!("Failed to open stream: {:?}", e);
            return;
        }
        info!("Camera opened");
        loop {
            match camera.frame() {
                Ok(frame) => {
                    match frame.decode_image::<RgbFormat>() {
                        Ok(img) => {
                            let arc_img = Arc::new(img);
                            let mut latest = camera_latest.write().await;
                            *latest = Some(arc_img);
                        }
                        Err(e) => warn!("Decode error: {:?}", e),
                    }
                }
                Err(e) => warn!("Frame error: {:?}", e),
            }
            sleep(Duration::from_millis(33)).await;
        }
    });

    // Server loop
    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(p) => p,
            Err(e) => {
                warn!("Accept error: {:?}", e);
                sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let latest_clone = Arc::clone(&latest_image);
        tokio::spawn(async move {
            let mut buf = vec![0u8; 4096];
            let n = match stream.read(&mut buf).await {
                Ok(n) => n,
                Err(_) => return,
            };
            let cmd = String::from_utf8_lossy(&buf[..n]);
            if let Some(path) = parse_capture_command(&cmd) {
                let guard = latest_clone.read().await;
                if let Some(arc_img) = guard.as_ref().cloned() {
                    drop(guard);
                    if let Err(e) = arc_img.save(&path) {
                        let _ = stream.write_all(format!("ERROR: {}", e).as_bytes()).await;
                    } else {
                        let _ = stream.write_all(b"OK").await;
                    }
                } else {
                    let _ = stream.write_all(b"NOFRAME").await;
                }
            }
            let _ = stream.shutdown().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{default_capture_output, parse_capture_command, pid_path, runtime_dir};

    #[test]
    fn runtime_paths_are_nested_under_temp_dir() {
        let temp_dir = std::env::temp_dir();

        assert!(runtime_dir().starts_with(&temp_dir));
        assert!(pid_path().starts_with(runtime_dir()));
        assert!(default_capture_output().starts_with(runtime_dir()));
    }

    #[test]
    fn parse_capture_command_extracts_output_path() {
        let parsed = parse_capture_command("capture C:/temp/frame.jpg\n");

        assert_eq!(parsed, Some(std::path::PathBuf::from("C:/temp/frame.jpg")));
    }

    #[test]
    fn parse_capture_command_rejects_unknown_commands() {
        assert!(parse_capture_command("status\n").is_none());
    }
}
