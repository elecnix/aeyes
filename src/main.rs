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
use tokio::net::{UnixListener, UnixStream};
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
        /// Output path, default /tmp/aeyes-last.jpg
        #[arg(short, long, default_value = "/tmp/aeyes-last.jpg")]
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

const SOCKET_PATH: &str = "/tmp/aeyes.sock";
const PID_PATH: &str = "/tmp/aeyes.pid";

async fn is_running() -> bool {
    UnixStream::connect(SOCKET_PATH).await.is_ok()
}

async fn start_daemon() -> Result<()> {
    if is_running().await {
        println!("Daemon is already running.");
        return Ok(());
    }

    let exe = env::current_exe()?;
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.arg("--daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);

    let mut child = cmd.spawn()?;
    let pid = child.id();
    std::fs::write(PID_PATH, pid.to_string().as_bytes())?;

    info!("Daemon started with PID {}", pid);
    println!("Daemon started (PID: {})", pid);

    sleep(Duration::from_millis(500)).await;
    Ok(())
}

async fn stop_daemon() -> Result<()> {
    if let Ok(pid_str) = std::fs::read_to_string(PID_PATH) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            let _ = tokio::process::Command::new("kill")
                .arg(pid.to_string())
                .status()
                .await;
        }
    }
    let _ = std::fs::remove_file(SOCKET_PATH);
    let _ = std::fs::remove_file(PID_PATH);
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

    let mut stream = UnixStream::connect(SOCKET_PATH).await?;
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
    let _ = std::fs::remove_file(SOCKET_PATH);
    let _ = std::fs::remove_file(PID_PATH);

    let listener = UnixListener::bind(SOCKET_PATH)?;
    let pid = std::process::id();
    std::fs::write(PID_PATH, pid.to_string().as_bytes())?;

    info!("Daemon running, listening on {}", SOCKET_PATH);

    let latest_image: Arc<RwLock<Option<Arc<RgbImage>>>> = Arc::new(RwLock::new(None));

    // Camera task
    let camera_latest = Arc::clone(&latest_image);
    tokio::spawn(async move {
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Nearest)
            .with_width(1280)
            .with_height(720);
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
                    match frame.decode_image::<image::Rgb<u8>>() {
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
            if let Some(path_str) = cmd.strip_prefix("capture ").and_then(|s| s.strip_suffix('\n')) {
                let path = PathBuf::from(path_str.trim());
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
