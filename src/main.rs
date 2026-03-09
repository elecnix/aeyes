use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dotenvy::dotenv;
use image::ImageBuffer;
use nokhwa::{Camera, CameraCloseConfiguration, CameraFormat, Cameras, Device, FormatSpecificRequestBuilder, NokhwaError, PixelFormat, RequestedFormat, RequestedFormatType, SourceBuilder};
use std::env;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixListener;
use tokio::sync::{RwLock, watch};
use tokio::time::sleep;
use tracing::{info, Level};
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
    dotenv().ok();
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
                println!("Run `aeyes --help` for usage.");
            }
        }
    }

    Ok(())
}

const SOCKET_PATH: &str = "/tmp/aeyes.sock";
const PID_PATH: &str = "/tmp/aeyes.pid";
const DEFAULT_OUTPUT: &str = "/tmp/aeyes-last.jpg";

async fn is_running() -> bool {
    match tokio::net::UnixStream::connect(SOCKET_PATH).await {
        Ok(_) => true,
        Err(_) => false,
    }
}

async fn start_daemon() -> Result<()> {
    if is_running().await {
        println!("Daemon is already running.");
        return Ok(());
    }

    let exe = env::current_exe()?;
    let mut cmd = tokio::process::Command::new(exe);
    cmd.arg("--daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let child = cmd.spawn()?;
    let pid = child.id();
    std::fs::write(PID_PATH, pid.to_string())?;

    info!("Daemon started with PID {}", pid);
    println!("Daemon started (PID: {})", pid);

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

    let mut stream = tokio::net::UnixStream::connect(SOCKET_PATH).await?;
    stream.write_all(b"capture ").await?;
    stream.write_all(output.to_string_lossy().as_bytes()).await?;
    stream.write_all(b"\n").await?;

    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;
    let response = String::from_utf8_lossy(&buf[..n]);

    if response.trim() == "OK" {
        println!("Captured to {:?}", output);
    } else {
        anyhow::bail!("Daemon error: {}", response);
    }

    Ok(())
}

async fn run_daemon() -> Result<()> {
    // Cleanup old socket/pid
    let _ = std::fs::remove_file(SOCKET_PATH);
    let _ = std::fs::remove_file(PID_PATH);

    let listener = UnixListener::bind(SOCKET_PATH)?;
    std::fs::write(PID_PATH, std::process::id().to_string())?;

    info!("Daemon running, listening on {}", SOCKET_PATH);

    // Find camera
    let devices = nokhwa::utils::available_devices()?;
    if devices.is_empty() {
        anyhow::bail!("No cameras found");
    }
    let device = devices.into_iter().next().unwrap();
    info!("Using camera: {:?}", device);

    let requested = RequestedFormat::new::<nokhwa::utils::RgbFormat>()
        .with_width(1280)
        .with_height(720)
        .build();

    let mut camera = Camera::new(device, requested)?;

    camera.open_stream()?;

    let latest_frame = Arc::new(RwLock::new(None::<nokhwa::Frame>));

    let camera_loop_latest = latest_frame.clone();
    tokio::spawn(async move {
        loop {
            match camera.capture() {
                Ok(frame) => {
                    let mut latest = camera_loop_latest.write().await;
                    *latest = Some(frame);
                }
                Err(e) => {
                    info!("Capture error: {:?}", e);
                }
            }
            sleep(Duration::from_millis(33)).await; // ~30fps
        }
    });

    loop {
        let (mut stream, _) = listener.accept().await?;
        let latest_clone = latest_frame.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            if let Ok(n) = stream.read(&mut buf).await {
                let cmd = String::from_utf8_lossy(&buf[..n]);
                if let Some((_, path_str)) = cmd.split_once("capture ") {
                    if let Some(path_str) = path_str.strip_suffix('\n') {
                        let path = PathBuf::from(path_str);
                        if let Some(frame) = latest_clone.read().await.as_ref().cloned() {
                            if let Err(e) = save_frame(&frame, &path) {
                                let _ = stream.write_all(format!("ERROR: {}", e).as_bytes()).await;
                            } else {
                                let _ = stream.write_all(b"OK").await;
                            }
                        } else {
                            let _ = stream.write_all(b"NOFRAME").await;
                        }
                    }
                }
            }
        });
    }
}

fn save_frame(frame: &nokhwa::Frame, path: &Path) -> Result<()> {
    let buf = frame.decode_image::<image::Rgb<u8>>()?;
    buf.save(path).with_context(|| format!("Failed to save to {:?}", path))?;
    Ok(())
}
