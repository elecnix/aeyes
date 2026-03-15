use anyhow::{anyhow, bail, Context, Result};
use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::{CommandFactory, Parser, Subcommand};
use image::{load_from_memory, RgbImage};
#[cfg(not(target_os = "linux"))]
use nokhwa::{
    query,
    utils::{ApiBackend, CameraIndex},
};
use std::{collections::HashMap, time::Instant};

#[cfg(target_os = "linux")]
use rscam::{Camera as RsCamera, Config as RsConfig};
#[cfg(target_os = "linux")]
use rscam::{
    Control as RsControl, CtrlData, CID_EXPOSURE_ABSOLUTE, CID_EXPOSURE_AUTO, CID_FOCUS_AUTO,
};
use serde::Serialize;
use std::{
    env, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::RwLock,
    time::sleep,
};
use tracing::{info, warn, Level};

const DEFAULT_BIND: &str = "127.0.0.1:43210";
const DEFAULT_PORT: u16 = 43210;
const DEFAULT_OUTPUT: &str = "aeyes-frame.jpg";
const FRAME_WAIT_RETRIES: usize = 50;
const FRAME_WAIT_MS: u64 = 100;
const EXPOSURE_SAMPLE_INTERVAL: u32 = 8;
const EXPOSURE_SETTLE_FRAMES: u32 = 6;
const EXPOSURE_WARMUP_FRAMES: u32 = 8;
const HIGHLIGHT_CLIP_LUMA: u8 = 250;
const SHADOW_LUMA_THRESHOLD: u8 = 40;

#[derive(Parser, Debug)]
#[command(
    name = "aeyes",
    about = "AI Eyes - non-interactive webcam daemon",
    disable_help_subcommand = true,
    arg_required_else_help = false
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the background daemon
    Start {
        /// Camera stable ID or name. Required if more than one camera is present.
        #[arg(long)]
        camera: Option<String>,
        /// Bind address for the daemon HTTP API (host:port or just host)
        #[arg(long, default_value = DEFAULT_BIND)]
        bind: String,
        /// Allowed remote IP prefixes (repeatable). Empty = allow all.
        #[arg(long)]
        allow_hosts: Vec<String>,
    },
    /// List available cameras
    Cams,
    /// Capture a frame to a file through the daemon HTTP API
    Frame {
        /// Camera stable ID or name; defaults to the daemon-selected camera.
        #[arg(long)]
        camera: Option<String>,
        /// Output file path
        #[arg(short, long, default_value = DEFAULT_OUTPUT)]
        output: PathBuf,
    },
    /// Stop the daemon
    Stop,
    /// Show daemon status
    Status,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CameraDescriptor {
    pub id: String,
    pub name: String,
    pub backend: String,
}

pub trait CameraBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn list_cameras(&self) -> Result<Vec<CameraDescriptor>>;
    fn open(&self, id: &str) -> Result<Box<dyn OpenCamera>>;
}

pub trait OpenCamera: Send {
    fn set_auto_features(&mut self) -> Result<()>;
    fn capture_jpeg(&mut self) -> Result<Vec<u8>>;
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DaemonErrorState {
    message: String,
    details: Vec<String>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct FrameLumaStats {
    average_luma: f32,
    p95_luma: u8,
    clipped_ratio: f32,
    dark_ratio: f32,
}

#[cfg(target_os = "linux")]
#[derive(Copy, Clone, Debug)]
struct ExposureController {
    minimum: i32,
    maximum: i32,
    step: i32,
    current: i32,
    frames_until_sample: u32,
    cooldown_frames: u32,
}

#[cfg(target_os = "linux")]
fn list_v4l2_cameras() -> Result<Vec<CameraDescriptor>> {
    let mut cams = vec![];
    if let Ok(entries) = fs::read_dir("/dev") {
        for entry in entries.flatten() {
            let path_str = entry.path().to_string_lossy().to_string();
            let fname = entry.file_name().to_string_lossy().to_string();
            if let Some(idx_str) = fname.strip_prefix("video") {
                if idx_str.parse::<u32>().is_ok() && device_supports_video_capture(&path_str) {
                    let output = std::process::Command::new("v4l2-ctl")
                        .arg("--device")
                        .arg(&path_str)
                        .arg("--info")
                        .output()
                        .ok();
                    let info_text = if let Some(out) = output {
                        if out.status.success() {
                            String::from_utf8_lossy(&out.stdout)
                                .lines()
                                .find(|l| l.contains("Card type"))
                                .map(|l| {
                                    l.split(':')
                                        .nth(1)
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or(fname.to_string())
                                })
                                .unwrap_or(fname.to_string())
                        } else {
                            fname.to_string()
                        }
                    } else {
                        fname.to_string()
                    };
                    let id = idx_str.to_string();
                    cams.push(CameraDescriptor {
                        id,
                        name: info_text,
                        backend: "v4l2".to_string(),
                    });
                }
            }
        }
    }
    cams.sort_by_key(|c| c.id.parse::<u32>().unwrap_or(999u32));
    Ok(cams)
}

impl DaemonErrorState {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: Vec::new(),
        }
    }

    fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.details.push(detail.into());
        self
    }
}

#[derive(Clone)]
pub struct AppState {
    selected_camera: String,
    streams: Arc<HashMap<String, CameraStreamState>>,
    cameras: Arc<Vec<CameraDescriptor>>,
    last_activity: Arc<RwLock<Instant>>,
    bind_addr: Arc<RwLock<SocketAddr>>,
    listener: Arc<RwLock<Option<TcpListener>>>,
    allow_hosts: Vec<String>,
}

#[derive(Clone)]
struct CameraStreamState {
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    last_error: Arc<RwLock<Option<DaemonErrorState>>>,
}

#[derive(Debug, Serialize)]
struct CamerasResponse {
    selected_camera: String,
    cameras: Vec<CameraDescriptor>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    details: Vec<String>,
}

pub struct NativeBackend;

impl CameraBackend for NativeBackend {
    fn name(&self) -> &'static str {
        if cfg!(target_os = "linux") {
            "v4l2"
        } else {
            "native"
        }
    }

    fn list_cameras(&self) -> Result<Vec<CameraDescriptor>> {
        #[cfg(target_os = "linux")]
        {
            list_v4l2_cameras()
        }
        #[cfg(not(target_os = "linux"))]
        {
            let cams = query(ApiBackend::Auto).context("failed to query cameras")?;
            Ok(cams
                .into_iter()
                .map(|cam| CameraDescriptor {
                    id: camera_index_to_id(cam.index()),
                    name: cam.human_name(),
                    backend: self.name().to_string(),
                })
                .collect())
        }
    }

    fn open(&self, id: &str) -> Result<Box<dyn OpenCamera>> {
        #[cfg(target_os = "linux")]
        {
            Ok(Box::new(V4l2OpenCamera::open(id)?))
        }
        #[cfg(not(target_os = "linux"))]
        {
            bail!("native frame capture backend is not implemented for this OS yet")
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn camera_index_to_id(index: &CameraIndex) -> String {
    index.as_string()
}

#[cfg(target_os = "linux")]
struct V4l2OpenCamera {
    camera: RsCamera,
    device_path: String,
    width: u32,
    height: u32,
    format: [u8; 4],
    exposure_controller: Option<ExposureController>,
}

#[cfg(target_os = "linux")]
impl V4l2OpenCamera {
    fn open(id: &str) -> Result<Self> {
        let index: u32 = id
            .parse()
            .with_context(|| format!("camera ID '{id}' must be numeric like '0'"))?;
        let device_path = format!("/dev/video{index}");
        let mut camera = RsCamera::new(&device_path)
            .with_context(|| format!("failed to open V4L2 device {device_path}"))?;

        let candidates = [
            CapturePreset {
                width: 3840,
                height: 2160,
                fps: 30,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 2560,
                height: 1440,
                fps: 30,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 1920,
                height: 1080,
                fps: 60,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 1920,
                height: 1080,
                fps: 30,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 1280,
                height: 720,
                fps: 60,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 1280,
                height: 720,
                fps: 30,
                format: *b"MJPG",
            },
            CapturePreset {
                width: 1920,
                height: 1080,
                fps: 30,
                format: *b"YUYV",
            },
            CapturePreset {
                width: 1280,
                height: 720,
                fps: 30,
                format: *b"YUYV",
            },
            CapturePreset {
                width: 640,
                height: 480,
                fps: 30,
                format: *b"MJPG",
            },
        ];

        let mut last_error = None;
        for preset in candidates {
            let config = RsConfig {
                interval: (1, preset.fps),
                resolution: (preset.width, preset.height),
                format: &preset.format,
                ..Default::default()
            };
            match camera.start(&config) {
                Ok(()) => {
                    return Ok(Self {
                        camera,
                        device_path,
                        width: preset.width,
                        height: preset.height,
                        format: preset.format,
                        exposure_controller: None,
                    });
                }
                Err(err) => {
                    last_error = Some(format!(
                        "{}x{}@{} {} rejected by {}: {}",
                        preset.width,
                        preset.height,
                        preset.fps,
                        String::from_utf8_lossy(&preset.format),
                        device_path,
                        err
                    ));
                }
            }
        }

        bail!(
            "failed to start V4L2 stream on {}. Last attempted mode: {}",
            device_path,
            last_error.unwrap_or_else(|| "no capture modes attempted".to_string())
        )
    }

    fn configure_exposure_controller(&mut self) {
        if let Err(err) = self.camera.set_control(CID_EXPOSURE_AUTO, &1i32) {
            info!(?err, device = %self.device_path, "failed to enable manual exposure mode");
        }

        match self.camera.get_control(CID_EXPOSURE_ABSOLUTE) {
            Ok(control) => {
                self.exposure_controller = ExposureController::from_control(control);
                if self.exposure_controller.is_none() {
                    info!(device = %self.device_path, "camera exposure control is not an integer range");
                }
            }
            Err(err) => {
                info!(?err, device = %self.device_path, "failed to inspect exposure controls");
            }
        }
    }

    fn warm_up_exposure(&mut self) {
        if self.exposure_controller.is_none() {
            return;
        }

        for _ in 0..EXPOSURE_WARMUP_FRAMES {
            let frame = match self.camera.capture() {
                Ok(frame) => frame,
                Err(err) => {
                    info!(?err, device = %self.device_path, "failed to capture warm-up frame");
                    break;
                }
            };

            if let Err(err) = self.observe_exposure_from_frame(&frame, true) {
                info!(?err, device = %self.device_path, "failed to warm up exposure");
                break;
            }
        }
    }

    fn observe_exposure_from_frame(&mut self, frame: &[u8], force: bool) -> Result<()> {
        let Some(controller) = self.exposure_controller.as_mut() else {
            return Ok(());
        };
        if !controller.should_sample(force) {
            return Ok(());
        }

        let stats = if self.format == *b"MJPG" {
            analyze_mjpeg_frame(frame).with_context(|| {
                format!("failed to decode MJPEG frame from {}", self.device_path)
            })?
        } else if self.format == *b"YUYV" {
            analyze_yuyv_frame(frame)?
        } else {
            return Ok(());
        };

        let Some(next) = self
            .exposure_controller
            .as_ref()
            .and_then(|controller| controller.proposed_value(stats))
        else {
            return Ok(());
        };

        self.camera
            .set_control(CID_EXPOSURE_ABSOLUTE, &next)
            .with_context(|| format!("failed to set exposure {} on {}", next, self.device_path))?;

        if let Some(controller) = self.exposure_controller.as_mut() {
            controller.record_applied(next);
        }

        info!(
            device = %self.device_path,
            exposure = next,
            average_luma = stats.average_luma,
            p95_luma = stats.p95_luma,
            clipped_ratio = stats.clipped_ratio,
            "adjusted exposure for highlight preservation"
        );
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl OpenCamera for V4l2OpenCamera {
    fn set_auto_features(&mut self) -> Result<()> {
        if let Err(err) = self.camera.set_control(CID_FOCUS_AUTO, &1i32) {
            info!(?err, device = %self.device_path, "failed to enable autofocus");
        }
        self.configure_exposure_controller();
        self.warm_up_exposure();
        Ok(())
    }

    fn capture_jpeg(&mut self) -> Result<Vec<u8>> {
        let frame = self
            .camera
            .capture()
            .with_context(|| format!("failed to capture frame from {}", self.device_path))?;

        if let Err(err) = self.observe_exposure_from_frame(&frame, false) {
            warn!(?err, device = %self.device_path, "failed to adapt exposure from captured frame");
        }

        if self.format == *b"MJPG" {
            return Ok(frame.to_vec());
        }
        if self.format == *b"YUYV" {
            return yuyv_to_jpeg(self.width, self.height, &frame)
                .with_context(|| format!("failed to encode YUYV frame from {}", self.device_path));
        }

        bail!(
            "unsupported frame format '{}' from {}",
            String::from_utf8_lossy(&self.format),
            self.device_path
        )
    }
}

#[cfg(target_os = "linux")]
#[derive(Copy, Clone)]
struct CapturePreset {
    width: u32,
    height: u32,
    fps: u32,
    format: [u8; 4],
}

#[cfg(target_os = "linux")]
impl ExposureController {
    fn from_control(control: RsControl) -> Option<Self> {
        match control.data {
            CtrlData::Integer {
                value,
                minimum,
                maximum,
                step,
                ..
            } => Some(Self {
                minimum,
                maximum,
                step: step.max(1),
                current: value.clamp(minimum, maximum),
                frames_until_sample: 0,
                cooldown_frames: 0,
            }),
            _ => None,
        }
    }

    fn should_sample(&mut self, force: bool) -> bool {
        if force {
            return true;
        }
        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
            return false;
        }
        if self.frames_until_sample == 0 {
            self.frames_until_sample = EXPOSURE_SAMPLE_INTERVAL;
            return true;
        }
        self.frames_until_sample -= 1;
        false
    }

    fn proposed_value(&self, stats: FrameLumaStats) -> Option<i32> {
        recommend_exposure_value(self.minimum, self.maximum, self.step, self.current, stats)
    }

    fn record_applied(&mut self, value: i32) {
        self.current = value;
        self.cooldown_frames = EXPOSURE_SETTLE_FRAMES;
        self.frames_until_sample = EXPOSURE_SAMPLE_INTERVAL;
    }
}

#[cfg(target_os = "linux")]
fn device_supports_video_capture(path: &str) -> bool {
    let Ok(output) = std::process::Command::new("v4l2-ctl")
        .args(["-D", "-d", path])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.contains("Video Capture")
}

pub fn parse_bind_addr(s: &str) -> Result<SocketAddr> {
    // Try parsing as full socket address first
    if let Ok(addr) = s.parse::<SocketAddr>() {
        return Ok(addr);
    }
    // Try parsing as IP only, add default port
    if let Ok(ip) = s.parse::<std::net::IpAddr>() {
        return Ok(SocketAddr::new(ip, DEFAULT_PORT));
    }
    bail!("invalid bind address '{s}'; use HOST:PORT or just HOST (defaults to port {DEFAULT_PORT})")
}

pub fn print_help() -> Result<()> {
    Cli::command().print_help()?;
    println!();
    Ok(())
}

pub async fn run_cli() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Start {
            camera,
            bind,
            allow_hosts,
        }) => {
            let addr = parse_bind_addr(&bind)?;
            start_daemon(camera, addr, allow_hosts).await
        }
        Some(Commands::Cams) => list_cameras_cmd(),
        Some(Commands::Frame { camera, output }) => frame_cmd(camera, &output).await,
        Some(Commands::Stop) => stop_daemon().await,
        Some(Commands::Status) => status_cmd().await,
        None => print_help(),
    }
}

pub fn runtime_dir() -> PathBuf {
    env::temp_dir().join("aeyes")
}

fn pid_path() -> PathBuf {
    runtime_dir().join("daemon.pid")
}

fn addr_path() -> PathBuf {
    runtime_dir().join("daemon.addr")
}

async fn daemon_addr() -> Result<SocketAddr> {
    let value = fs::read_to_string(addr_path()).context("daemon address file missing")?;
    value.trim().parse().context("invalid daemon address")
}

pub fn choose_camera(
    cameras: &[CameraDescriptor],
    requested: Option<&str>,
) -> Result<CameraDescriptor> {
    if let Some(requested) = requested {
        cameras
            .iter()
            .find(|cam| cam.id == requested || cam.name == requested)
            .cloned()
            .with_context(|| format!("camera '{requested}' not found"))
    } else if cameras.len() == 1 {
        Ok(cameras[0].clone())
    } else if cameras.is_empty() {
        bail!("no cameras found")
    } else {
        let mut msg = String::from("multiple cameras found; rerun with --camera <id-or-name>\n");
        for cam in cameras {
            msg.push_str(&format!("- {} ({})\n", cam.id, cam.name));
        }
        bail!(msg.trim_end().to_string())
    }
}

pub fn list_cameras_with_backend(backend: &dyn CameraBackend) -> Result<Vec<CameraDescriptor>> {
    backend.list_cameras()
}

fn list_cameras_cmd() -> Result<()> {
    let cams = list_cameras_with_backend(&NativeBackend)?;
    if cams.is_empty() {
        println!("No cameras found.");
    } else {
        for cam in cams {
            println!("{}\t{}\t{}", cam.id, cam.name, cam.backend);
        }
    }
    Ok(())
}

pub async fn start_daemon(
    requested_camera: Option<String>,
    bind: SocketAddr,
    allow_hosts: Vec<String>,
) -> Result<()> {
    fs::create_dir_all(runtime_dir())?;
    if let Ok(addr) = daemon_addr().await {
        if daemon_responding(addr).await {
            println!("Daemon already running at http://{addr}");
            return Ok(());
        }
    }

    let backend = NativeBackend;
    let cams = list_cameras_with_backend(&backend)?;
    let chosen = choose_camera(&cams, requested_camera.as_deref())?;

    let exe = env::current_exe()?;
    let mut cmd = tokio::process::Command::new(exe);
    cmd.env("AEYES_DAEMON", "1")
        .env("AEYES_CAMERA", &chosen.id)
        .env("AEYES_BIND", bind.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    if !allow_hosts.is_empty() {
        cmd.env("AEYES_ALLOW_HOSTS", allow_hosts.join(","));
    }
    let child = cmd.spawn().context("failed to spawn daemon")?;
    let pid = child.id().context("missing daemon pid")?;
    fs::write(pid_path(), pid.to_string())?;
    fs::write(addr_path(), bind.to_string())?;
    println!(
        "Daemon started with PID {pid} at http://{bind} using camera {} ({})",
        chosen.id, chosen.name
    );
    Ok(())
}

pub async fn stop_daemon() -> Result<()> {
    if let Ok(addr) = daemon_addr().await {
        let _ = http_get_bytes(addr, "/shutdown").await;
        for _ in 0..20 {
            if !daemon_responding(addr).await {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    if let Ok(pid_str) = fs::read_to_string(pid_path()) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            #[cfg(unix)]
            {
                let _ = tokio::process::Command::new("kill")
                    .arg(pid.to_string())
                    .status()
                    .await;
            }
            #[cfg(windows)]
            {
                let _ = tokio::process::Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/F"])
                    .status()
                    .await;
            }
        }
    }
    let _ = fs::remove_file(pid_path());
    let _ = fs::remove_file(addr_path());
    println!("Daemon stopped.");
    Ok(())
}

pub async fn status_cmd() -> Result<()> {
    match daemon_addr().await {
        Ok(addr) if daemon_responding(addr).await => println!("Daemon running at http://{addr}"),
        _ => println!("Daemon not running."),
    }
    Ok(())
}

async fn daemon_responding(addr: SocketAddr) -> bool {
    TcpStream::connect(addr).await.is_ok()
}

pub async fn frame_cmd(camera: Option<String>, output: &Path) -> Result<()> {
    let addr = match daemon_addr().await {
        Ok(addr) if daemon_responding(addr).await => addr,
        _ => {
            let backend = NativeBackend;
            let cams = backend.list_cameras().context("no cameras found")?;
            let chosen = choose_camera(&cams, camera.as_deref()).context(
                "could not auto-select camera (specify --camera <id-or-name> if multiple)",
            )?;
            let bind_str = env::var("AEYES_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string());
            let bind: SocketAddr = bind_str.parse().context("invalid AEYES_BIND env")?;
            start_daemon(Some(chosen.id.clone()), bind, vec![])
                .await
                .context("failed to auto-start daemon")?;
            let mut retries = 100;
            let mut ready_addr = None;
            while retries > 0 {
                match daemon_addr().await {
                    Ok(addr) if daemon_responding(addr).await => {
                        ready_addr = Some(addr);
                        break;
                    }
                    _ => {
                        sleep(Duration::from_millis(100)).await;
                        retries -= 1;
                    }
                }
            }
            ready_addr.context("daemon failed to become ready")?
        }
    };
    let path = if let Some(cam) = &camera {
        format!("/cams/{}/frame", cam)
    } else {
        "/cams/default/frame".to_string()
    };
    let bytes = http_get_bytes(addr, &path).await?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, bytes)?;
    println!("Saved frame to {}", output.display());
    Ok(())
}

async fn http_get_bytes(addr: SocketAddr, path: &str) -> Result<Vec<u8>> {
    let mut stream = TcpStream::connect(addr).await?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await?;
    parse_http_response(&buf)
}

pub fn parse_http_response(buf: &[u8]) -> Result<Vec<u8>> {
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .ok_or_else(|| anyhow!("invalid HTTP response"))?;
    let headers = String::from_utf8_lossy(&buf[..split]);
    let status_line = headers.lines().next().unwrap_or("HTTP error").to_string();
    let body = &buf[split..];
    if !(headers.starts_with("HTTP/1.1 200") || headers.starts_with("HTTP/1.0 200")) {
        let detailed = parse_error_response_body(body).unwrap_or_default();
        if detailed.is_empty() {
            bail!(status_line);
        }
        bail!("{status_line}: {detailed}");
    }
    Ok(body.to_vec())
}

fn parse_error_response_body(body: &[u8]) -> Option<String> {
    let json: serde_json::Value = serde_json::from_slice(body).ok()?;
    let error = json.get("error")?.as_str()?.to_string();
    let details = json
        .get("details")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .unwrap_or_default();
    if details.is_empty() {
        Some(error)
    } else {
        Some(format!("{error} [{details}]"))
    }
}

pub async fn run_daemon_from_env() -> Result<()> {
    fs::create_dir_all(runtime_dir())?;
    let bind: SocketAddr = env::var("AEYES_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND.to_string())
        .parse()
        .context("invalid AEYES_BIND")?;
    let selected_camera = env::var("AEYES_CAMERA").context("AEYES_CAMERA missing")?;
    let allow_hosts: Vec<String> = env::var("AEYES_ALLOW_HOSTS")
        .ok()
        .map(|s| s.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();
    run_daemon(bind, selected_camera, allow_hosts, Box::new(NativeBackend)).await
}

pub async fn run_daemon(
    bind: SocketAddr,
    selected_camera: String,
    allow_hosts: Vec<String>,
    backend: Box<dyn CameraBackend>,
) -> Result<()> {
    fs::create_dir_all(runtime_dir())?;
    let backend: Arc<dyn CameraBackend> = Arc::from(backend);
    let cameras = backend.list_cameras()?;
    let chosen = choose_camera(&cameras, Some(&selected_camera))?;
    let last_activity: Arc<RwLock<Instant>> = Arc::new(RwLock::new(Instant::now()));
    let mut streams = HashMap::new();

    for camera in &cameras {
        let latest_jpeg: Arc<RwLock<Option<Vec<u8>>>> = Arc::new(RwLock::new(None));
        let last_error: Arc<RwLock<Option<DaemonErrorState>>> = Arc::new(RwLock::new(None));
        let stream_state = CameraStreamState {
            latest_jpeg: latest_jpeg.clone(),
            last_error: last_error.clone(),
        };
        streams.insert(camera.id.clone(), stream_state);

        let backend = backend.clone();
        let camera_id = camera.id.clone();
        tokio::spawn(async move {
            if let Err(err) = camera_loop(backend, camera_id.clone(), latest_jpeg, last_error).await
            {
                warn!(?err, camera = %camera_id, "camera loop exited");
            }
        });
    }

    let idle_timeout_secs: u64 = env::var("AEYES_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3600);
    {
        let last_activity = last_activity.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(idle_timeout_secs)).await;
                if last_activity.read().await.elapsed() > Duration::from_secs(idle_timeout_secs) {
                    info!("daemon idle timeout, auto-stopping");
                    std::process::exit(0);
                }
            }
        });
    }

    let listener = TcpListener::bind(bind).await?;
    let resolved_addr = listener.local_addr()?;
    fs::write(addr_path(), resolved_addr.to_string())?;
    fs::write(pid_path(), std::process::id().to_string())?;

    let state = AppState {
        selected_camera: chosen.id.clone(),
        streams: Arc::new(streams),
        cameras: Arc::new(cameras),
        last_activity,
        bind_addr: Arc::new(RwLock::new(resolved_addr)),
        listener: Arc::new(RwLock::new(Some(listener))),
        allow_hosts,
    };
    let router = Router::new()
        .route("/cams", get(list_cams_http))
        .route("/cams/{id}/frame", get(frame_http))
        .route("/health", get(health_handler))
        .route("/shutdown", get(shutdown_handler))
        .route("/rebind", get(rebind_handler))
        .with_state(state.clone());

    serve_forever(state, router).await
}

async fn serve_forever(state: AppState, router: Router) -> Result<()> {
    loop {
        // Take the listener out of state to use it (rebind puts a new one back in)
        let listener = {
            let mut guard = state.listener.write().await;
            match guard.take() {
                Some(l) => l,
                None => break,
            }
        };
        let addr = *state.bind_addr.read().await;
        info!("listening on http://{addr}");

        loop {
            // Check if rebind was requested before accepting
            if state.listener.read().await.is_some() {
                info!("rebind detected, switching listener");
                break;
            }

            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((tcp_stream, remote_addr)) => {
                            let state = state.clone();
                            let router = router.clone();
                            let allow_hosts = state.allow_hosts.clone();

                            if !allow_hosts.is_empty() {
                                let ip_str = remote_addr.ip().to_string();
                                if !allow_hosts.iter().any(|prefix| ip_str.starts_with(prefix)) {
                                    info!("rejected connection from {remote_addr} (not in allow_hosts)");
                                    continue;
                                }
                            }

                            tokio::spawn(async move {
                                let hyper_service =
                                    hyper_util::service::TowerToHyperService::new(router.into_service());
                                let io = hyper_util::rt::TokioIo::new(tcp_stream);
                                if let Err(err) = hyper_util::server::conn::auto::Builder::new(
                                    hyper_util::rt::TokioExecutor::new(),
                                )
                                .serve_connection(io, hyper_service)
                                .await
                                {
                                    warn!(?err, %remote_addr, "HTTP connection error");
                                }
                            });
                        }
                        Err(err) => {
                            warn!(?err, "listener accept error");
                            // Check if we should switch to a new listener (rebind)
                            if state.listener.read().await.is_some() {
                                info!("switching to new listener after accept error");
                                break;
                            }
                            return Err(err.into());
                        }
                    }
                }
                _ = sleep(Duration::from_millis(200)) => {
                    // Timeout - loop back to check for rebind
                    continue;
                }
            }
        }
    }
    Ok(())
}

async fn camera_loop(
    backend: Arc<dyn CameraBackend>,
    selected_camera: String,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    last_error: Arc<RwLock<Option<DaemonErrorState>>>,
) -> Result<()> {
    let mut camera = match backend.open(&selected_camera) {
        Ok(camera) => camera,
        Err(err) => {
            let state = DaemonErrorState::new("failed to open selected camera")
                .with_detail(format!("camera id: {selected_camera}"))
                .with_detail(format!("error: {err:#}"));
            *last_error.write().await = Some(state);
            return Err(err);
        }
    };

    if let Err(err) = camera.set_auto_features() {
        let state = DaemonErrorState::new("failed to configure autofocus/auto-exposure")
            .with_detail(format!("camera id: {selected_camera}"))
            .with_detail(format!("error: {err:#}"));
        *last_error.write().await = Some(state);
        warn!(?err, "failed to set auto features");
    }

    loop {
        match camera.capture_jpeg() {
            Ok(bytes) => {
                *latest_jpeg.write().await = Some(bytes);
                *last_error.write().await = None;
            }
            Err(err) => {
                let state = DaemonErrorState::new("failed to capture frame from camera")
                    .with_detail(format!("camera id: {selected_camera}"))
                    .with_detail(format!("error: {err:#}"));
                *last_error.write().await = Some(state);
                warn!(?err, "failed to capture frame");
            }
        }
        sleep(Duration::from_millis(30)).await;
    }
}

async fn list_cams_http(State(state): State<AppState>) -> Json<CamerasResponse> {
    *state.last_activity.write().await = Instant::now();
    Json(CamerasResponse {
        selected_camera: state.selected_camera.clone(),
        cameras: (*state.cameras).clone(),
    })
}

async fn frame_http(AxumPath(id): AxumPath<String>, State(state): State<AppState>) -> Response {
    *state.last_activity.write().await = Instant::now();
    let requested = if id == "default" {
        state.selected_camera.clone()
    } else {
        id
    };

    let Some(stream) = state.streams.get(&requested).cloned() else {
        return error_response(
            StatusCode::NOT_FOUND,
            DaemonErrorState::new(format!(
                "camera '{requested}' is not managed by this daemon"
            ))
            .with_detail(format!(
                "available cameras: {}",
                state.streams.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
            .with_detail("request one of the IDs returned by GET /cams"),
        );
    };

    for _ in 0..FRAME_WAIT_RETRIES {
        if let Some(bytes) = stream.latest_jpeg.read().await.clone() {
            return Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "image/jpeg")
                .body(Body::from(bytes))
                .unwrap();
        }
        sleep(Duration::from_millis(FRAME_WAIT_MS)).await;
    }

    let error = stream.last_error.read().await.clone().unwrap_or_else(|| {
        DaemonErrorState::new("camera has not produced a frame yet")
            .with_detail(format!("camera id: {requested}"))
            .with_detail("no frame was available before the request timeout expired")
            .with_detail(format!(
                "waited approximately {} ms",
                FRAME_WAIT_RETRIES as u64 * FRAME_WAIT_MS
            ))
    });
    error_response(StatusCode::SERVICE_UNAVAILABLE, error)
}

async fn health_handler(State(state): State<AppState>) -> &'static str {
    *state.last_activity.write().await = Instant::now();
    "ok"
}

async fn shutdown_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    *state.last_activity.write().await = Instant::now();
    tokio::spawn(async move {
        sleep(Duration::from_millis(50)).await;
        std::process::exit(0);
    });
    Json(serde_json::json!({ "status": "stopping" }))
}

async fn rebind_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    *state.last_activity.write().await = Instant::now();
    let Some(addr_str) = params.get("addr") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            DaemonErrorState::new("missing 'addr' query parameter")
                .with_detail("usage: GET /rebind?addr=0.0.0.0:43210"),
        );
    };
    let new_addr: SocketAddr = match addr_str.parse() {
        Ok(a) => a,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                DaemonErrorState::new(format!("invalid address '{addr_str}'"))
                    .with_detail(format!("parse error: {e}")),
            );
        }
    };

    let new_listener = match TcpListener::bind(new_addr).await {
        Ok(l) => l,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                DaemonErrorState::new(format!("failed to bind to {new_addr}"))
                    .with_detail(format!("error: {e}")),
            );
        }
    };
    let resolved_addr = new_listener
        .local_addr()
        .unwrap_or(new_addr);

    // Replace the listener (drops old one, existing connections finish gracefully)
    {
        let mut guard = state.listener.write().await;
        *guard = Some(new_listener);
    }
    *state.bind_addr.write().await = resolved_addr;
    if let Err(e) = fs::write(addr_path(), resolved_addr.to_string()) {
        warn!(?e, "failed to write new daemon address file");
    }

    Json(serde_json::json!({
        "status": "rebound",
        "old_addr": params.get("addr").cloned().unwrap_or_default(),
        "new_addr": resolved_addr.to_string()
    }))
    .into_response()
}

fn error_response(status: StatusCode, error: DaemonErrorState) -> Response {
    let body = Json(ErrorResponse {
        error: error.message,
        details: error.details,
    });
    (status, body).into_response()
}

pub fn encode_rgb_to_jpeg(width: u32, height: u32, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let img = RgbImage::from_raw(width, height, bytes).context("invalid RGB buffer")?;
    let mut out = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95);
    encoder.encode_image(&img)?;
    Ok(out)
}

fn analyze_rgb_frame(bytes: &[u8]) -> FrameLumaStats {
    let mut histogram = [0u32; 256];
    let mut samples = 0u64;
    let mut total_luma = 0u64;
    let mut clipped = 0u64;
    let mut dark = 0u64;

    for pixel in bytes.chunks_exact(3).step_by(4) {
        let r = pixel[0] as u64;
        let g = pixel[1] as u64;
        let b = pixel[2] as u64;
        let luma = ((54 * r + 183 * g + 19 * b) >> 8) as u8;
        histogram[luma as usize] += 1;
        samples += 1;
        total_luma += luma as u64;
        if luma >= HIGHLIGHT_CLIP_LUMA {
            clipped += 1;
        }
        if luma <= SHADOW_LUMA_THRESHOLD {
            dark += 1;
        }
    }

    if samples == 0 {
        return FrameLumaStats::default();
    }

    let target = ((samples as f32) * 0.95).ceil() as u64;
    let mut cumulative = 0u64;
    let mut p95_luma = 0u8;
    for (index, count) in histogram.iter().enumerate() {
        cumulative += *count as u64;
        if cumulative >= target {
            p95_luma = index as u8;
            break;
        }
    }

    FrameLumaStats {
        average_luma: total_luma as f32 / samples as f32,
        p95_luma,
        clipped_ratio: clipped as f32 / samples as f32,
        dark_ratio: dark as f32 / samples as f32,
    }
}

#[cfg(target_os = "linux")]
fn analyze_mjpeg_frame(bytes: &[u8]) -> Result<FrameLumaStats> {
    let rgb = load_from_memory(bytes)
        .context("invalid MJPEG buffer")?
        .to_rgb8();
    Ok(analyze_rgb_frame(rgb.as_raw()))
}

#[cfg(target_os = "linux")]
fn analyze_yuyv_frame(bytes: &[u8]) -> Result<FrameLumaStats> {
    if !bytes.len().is_multiple_of(2) {
        bail!("invalid YUYV buffer length: {}", bytes.len());
    }

    let mut histogram = [0u32; 256];
    let mut samples = 0u64;
    let mut total_luma = 0u64;
    let mut clipped = 0u64;
    let mut dark = 0u64;

    for chunk in bytes.chunks_exact(4) {
        for y in [chunk[0], chunk[2]] {
            histogram[y as usize] += 1;
            samples += 1;
            total_luma += y as u64;
            if y >= HIGHLIGHT_CLIP_LUMA {
                clipped += 1;
            }
            if y <= SHADOW_LUMA_THRESHOLD {
                dark += 1;
            }
        }
    }

    if samples == 0 {
        return Ok(FrameLumaStats::default());
    }

    let target = ((samples as f32) * 0.95).ceil() as u64;
    let mut cumulative = 0u64;
    let mut p95_luma = 0u8;
    for (index, count) in histogram.iter().enumerate() {
        cumulative += *count as u64;
        if cumulative >= target {
            p95_luma = index as u8;
            break;
        }
    }

    Ok(FrameLumaStats {
        average_luma: total_luma as f32 / samples as f32,
        p95_luma,
        clipped_ratio: clipped as f32 / samples as f32,
        dark_ratio: dark as f32 / samples as f32,
    })
}

fn recommend_exposure_value(
    minimum: i32,
    maximum: i32,
    step: i32,
    current: i32,
    stats: FrameLumaStats,
) -> Option<i32> {
    let step = step.max(1);

    let decrease_ratio = if stats.clipped_ratio >= 0.10 || stats.p95_luma >= 252 {
        0.35
    } else if stats.clipped_ratio >= 0.03 || stats.p95_luma >= 248 {
        0.22
    } else if stats.clipped_ratio >= 0.008 || stats.p95_luma >= 242 {
        0.12
    } else {
        0.0
    };
    if decrease_ratio > 0.0 {
        let delta = ((current as f32) * decrease_ratio).round() as i32;
        let next = quantize_exposure_value(minimum, maximum, step, current - delta.max(step));
        return (next != current).then_some(next);
    }

    let increase_ratio =
        if stats.average_luma <= 18.0 && stats.dark_ratio >= 0.85 && stats.p95_luma <= 110 {
            0.35
        } else if stats.average_luma <= 30.0 && stats.dark_ratio >= 0.70 && stats.p95_luma <= 150 {
            0.20
        } else if stats.average_luma <= 45.0
            && stats.dark_ratio >= 0.55
            && stats.p95_luma <= 175
            && stats.clipped_ratio < 0.002
        {
            0.12
        } else {
            0.0
        };
    if increase_ratio > 0.0 {
        let delta = ((current as f32) * increase_ratio).round() as i32;
        let next = quantize_exposure_value(minimum, maximum, step, current + delta.max(step));
        return (next != current).then_some(next);
    }

    None
}

fn quantize_exposure_value(minimum: i32, maximum: i32, step: i32, value: i32) -> i32 {
    let step = step.max(1);
    let clamped = value.clamp(minimum, maximum);
    let offset = clamped - minimum;
    minimum + ((offset + (step / 2)) / step) * step
}

pub fn yuyv_to_jpeg(width: u32, height: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 2;
    if bytes.len() != expected {
        bail!(
            "invalid YUYV buffer length: expected {expected} bytes for {width}x{height}, got {}",
            bytes.len()
        );
    }

    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for chunk in bytes.chunks_exact(4) {
        let y0 = chunk[0] as f32;
        let u = chunk[1] as f32 - 128.0;
        let y1 = chunk[2] as f32;
        let v = chunk[3] as f32 - 128.0;
        push_yuv_pixel(&mut rgb, y0, u, v);
        push_yuv_pixel(&mut rgb, y1, u, v);
    }
    encode_rgb_to_jpeg(width, height, rgb)
}

fn push_yuv_pixel(rgb: &mut Vec<u8>, y: f32, u: f32, v: f32) {
    let r = (y + 1.402 * v).round().clamp(0.0, 255.0) as u8;
    let g = (y - 0.344_136 * u - 0.714_136 * v)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * u).round().clamp(0.0, 255.0) as u8;
    rgb.extend_from_slice(&[r, g, b]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{header, StatusCode};
    #[cfg(not(target_os = "linux"))]
    use nokhwa::utils::CameraIndex;
    use reqwest::StatusCode as ReqwestStatus;
    use serial_test::serial;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct FakeBackend {
        cameras: Vec<CameraDescriptor>,
        frame: Vec<u8>,
        fail_open: Option<String>,
        fail_capture: Option<String>,
    }

    struct FakeOpenCamera {
        frame: Vec<u8>,
        fail_capture: Option<String>,
    }

    impl Default for FakeOpenCamera {
        fn default() -> Self {
            Self {
                frame: vec![0xff, 0xd8],
                fail_capture: None,
            }
        }
    }

    impl CameraBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn list_cameras(&self) -> Result<Vec<CameraDescriptor>> {
            Ok(self.cameras.clone())
        }
        fn open(&self, id: &str) -> Result<Box<dyn OpenCamera>> {
            if let Some(message) = &self.fail_open {
                bail!(message.clone());
            }
            if self.cameras.iter().any(|c| c.id == id) {
                Ok(Box::new(FakeOpenCamera {
                    frame: self.frame.clone(),
                    fail_capture: self.fail_capture.clone(),
                }))
            } else {
                bail!("camera not found")
            }
        }
    }

    impl OpenCamera for FakeOpenCamera {
        fn set_auto_features(&mut self) -> Result<()> {
            Ok(())
        }
        fn capture_jpeg(&mut self) -> Result<Vec<u8>> {
            if let Some(message) = &self.fail_capture {
                bail!(message.clone());
            }
            Ok(self.frame.clone())
        }
    }

    fn fake_cameras() -> Vec<CameraDescriptor> {
        vec![
            CameraDescriptor {
                id: "cam-a".into(),
                name: "Front Cam".into(),
                backend: "fake".into(),
            },
            CameraDescriptor {
                id: "cam-b".into(),
                name: "Rear Cam".into(),
                backend: "fake".into(),
            },
        ]
    }

    fn default_state() -> AppState {
        let mut streams = HashMap::new();
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
            },
        );
        streams.insert(
            "cam-b".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
            },
        );
        AppState {
            selected_camera: "cam-a".to_string(),
            streams: Arc::new(streams),
            cameras: Arc::new(fake_cameras()),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            bind_addr: Arc::new(RwLock::new("127.0.0.1:0".parse().unwrap())),
            listener: Arc::new(RwLock::new(None)),
            allow_hosts: vec![],
        }
    }

    #[test]
    fn test_choose_camera_single() {
        let cams = vec![fake_cameras()[0].clone()];
        let cam = choose_camera(&cams, None).unwrap();
        assert_eq!(cam.id, "cam-a");
    }

    #[test]
    fn test_choose_camera_empty() {
        let err = choose_camera(&[], None).unwrap_err().to_string();
        assert_eq!(err, "no cameras found");
    }

    #[test]
    fn test_choose_camera_requested_not_found() {
        let err = choose_camera(&fake_cameras(), Some("missing"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_choose_camera_requires_explicit_choice_for_multiple() {
        let err = choose_camera(&fake_cameras(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple cameras found"));
        assert!(err.contains("cam-a"));
    }

    #[test]
    fn test_choose_camera_accepts_id_or_name() {
        let cams = fake_cameras();
        assert_eq!(
            choose_camera(&cams, Some("cam-b")).unwrap().name,
            "Rear Cam"
        );
        assert_eq!(choose_camera(&cams, Some("Front Cam")).unwrap().id, "cam-a");
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_camera_index_to_id() {
        assert_eq!(
            camera_index_to_id(&CameraIndex::String("test".to_string())),
            "test"
        );
        assert_eq!(camera_index_to_id(&CameraIndex::Index(42)), "42");
    }

    #[test]
    fn test_parse_http_response_http10() {
        let body = parse_http_response(b"HTTP/1.0 200 OK\r\n\r\nabc").unwrap();
        assert_eq!(body, b"abc");
    }

    #[test]
    fn test_parse_http_response_no_boundary() {
        let err = parse_http_response(b"HTTP/1.1 200 OK\r\nabc")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid HTTP response"));
    }

    #[test]
    fn test_parse_http_response_extracts_body() {
        let body = parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc").unwrap();
        assert_eq!(body, b"abc");
    }

    #[test]
    fn test_parse_http_response_rejects_non_200_with_details() {
        let response = b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\r\n{\"error\":\"failed to capture\",\"details\":[\"camera id: cam-a\",\"device busy\"]}";
        let err = parse_http_response(response).unwrap_err().to_string();
        assert!(err.contains("503"));
        assert!(err.contains("failed to capture"));
        assert!(err.contains("device busy"));
    }

    #[test]
    fn test_parse_error_response_body_simple() {
        let parsed = parse_error_response_body(b"{\"error\":\"simple\"}").unwrap();
        assert_eq!(parsed, "simple");
    }

    #[test]
    fn test_parse_error_response_body_details() {
        let parsed =
            parse_error_response_body(b"{\"error\":\"err\",\"details\":[\"d1\",\"d2\"]}").unwrap();
        assert_eq!(parsed, "err [d1 | d2]");
    }

    #[test]
    fn test_encode_rgb_to_jpeg_invalid_size() {
        let err = encode_rgb_to_jpeg(2, 1, vec![255, 0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid RGB buffer"));
    }

    #[test]
    fn test_encode_rgb_to_jpeg_encodes() {
        let jpeg = encode_rgb_to_jpeg(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn test_analyze_rgb_frame_detects_bright_highlights() {
        let stats = analyze_rgb_frame(&[
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20,
        ]);
        assert!(stats.clipped_ratio > 0.45);
        assert!(stats.p95_luma >= 250);
    }

    #[test]
    fn test_recommend_exposure_value_reduces_blown_highlights() {
        let next = recommend_exposure_value(
            2,
            40000,
            1,
            400,
            FrameLumaStats {
                average_luma: 170.0,
                p95_luma: 252,
                clipped_ratio: 0.08,
                dark_ratio: 0.10,
            },
        )
        .unwrap();
        assert!(next < 400);
    }

    #[test]
    fn test_recommend_exposure_value_increases_dark_scene() {
        let next = recommend_exposure_value(
            2,
            40000,
            1,
            100,
            FrameLumaStats {
                average_luma: 16.0,
                p95_luma: 80,
                clipped_ratio: 0.0,
                dark_ratio: 0.92,
            },
        )
        .unwrap();
        assert!(next > 100);
    }

    #[test]
    fn test_recommend_exposure_value_keeps_balanced_frame() {
        let next = recommend_exposure_value(
            2,
            40000,
            1,
            120,
            FrameLumaStats {
                average_luma: 88.0,
                p95_luma: 210,
                clipped_ratio: 0.001,
                dark_ratio: 0.15,
            },
        );
        assert!(next.is_none());
    }

    #[test]
    fn test_yuyv_to_jpeg_invalid_length() {
        let err = yuyv_to_jpeg(2, 1, &[80, 90, 81]).unwrap_err().to_string();
        assert!(err.contains("invalid YUYV buffer length"));
    }

    #[test]
    fn test_yuyv_to_jpeg_encodes() {
        let jpeg = yuyv_to_jpeg(2, 1, &[80, 90, 81, 240]).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn test_print_help_works() {
        print_help().unwrap();
    }

    #[test]
    fn test_parse_bind_addr_full_address() {
        let addr = parse_bind_addr("0.0.0.0:8080").unwrap();
        assert_eq!(addr, "0.0.0.0:8080".parse().unwrap());
    }

    #[test]
    fn test_parse_bind_addr_ip_only_uses_default_port() {
        let addr = parse_bind_addr("0.0.0.0").unwrap();
        assert_eq!(addr, "0.0.0.0:43210".parse().unwrap());
    }

    #[test]
    fn test_parse_bind_addr_ipv6() {
        let addr = parse_bind_addr("[::1]:8080").unwrap();
        assert_eq!(addr, "[::1]:8080".parse().unwrap());
    }

    #[test]
    fn test_parse_bind_addr_ipv6_only_uses_default_port() {
        let addr = parse_bind_addr("::1").unwrap();
        assert_eq!(addr, "[::1]:43210".parse().unwrap());
    }

    #[test]
    fn test_parse_bind_addr_invalid() {
        assert!(parse_bind_addr("not-an-address").is_err());
    }

    #[test]
    fn test_list_cameras_cmd_works() {
        list_cameras_cmd().unwrap();
    }

    #[test]
    fn test_native_backend_name_linux() {
        let backend = NativeBackend;
        #[cfg(target_os = "linux")]
        assert_eq!(backend.name(), "v4l2");
        #[cfg(not(target_os = "linux"))]
        assert_eq!(backend.name(), "native");
    }

    #[test]
    fn test_native_backend_list_cameras_works() {
        let backend = NativeBackend;
        let _cams = backend.list_cameras();
    }

    #[test]
    fn test_runtime_dir_contains_aeyes() {
        let dir = runtime_dir();
        assert!(dir.to_string_lossy().contains("aeyes"));
    }

    #[test]
    fn test_pid_path() {
        let path = pid_path();
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("daemon.pid")
        );
    }

    #[test]
    fn test_addr_path() {
        let path = addr_path();
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("daemon.addr")
        );
    }

    #[test]
    fn test_error_response_builds() {
        let error = DaemonErrorState::new("test").with_detail("detail");
        let _resp = error_response(StatusCode::BAD_REQUEST, error);
    }

    #[tokio::test]
    async fn test_list_cams_http() {
        let mut streams = HashMap::new();
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
            },
        );
        let state = AppState {
            selected_camera: "cam-a".to_string(),
            streams: Arc::new(streams),
            cameras: Arc::new(vec![CameraDescriptor {
                id: "cam-a".to_string(),
                name: "Test Cam".to_string(),
                backend: "test".to_string(),
            }]),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            bind_addr: Arc::new(RwLock::new("127.0.0.1:0".parse().unwrap())),
            listener: Arc::new(RwLock::new(None)),
            allow_hosts: vec![],
        };
        let Json(response) = list_cams_http(State(state)).await;
        assert_eq!(response.selected_camera, "cam-a");
        assert_eq!(response.cameras.len(), 1);
    }

    #[tokio::test]
    async fn test_frame_http_success_default() {
        let jpeg = vec![0xff, 0xd8, 0xff];
        let mut streams = HashMap::new();
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(Some(jpeg.clone()))),
                last_error: Arc::new(RwLock::new(None)),
            },
        );
        let state = AppState {
            selected_camera: "cam-a".to_string(),
            streams: Arc::new(streams),
            cameras: Arc::new(vec![]),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            bind_addr: Arc::new(RwLock::new("127.0.0.1:0".parse().unwrap())),
            listener: Arc::new(RwLock::new(None)),
            allow_hosts: vec![],
        };
        let resp = frame_http(AxumPath("default".to_string()), State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/jpeg")
        );
    }

    #[tokio::test]
    async fn test_frame_http_other_camera_success() {
        let state = default_state();
        {
            let stream = state.streams.get("cam-b").unwrap();
            *stream.latest_jpeg.write().await = Some(vec![0xff, 0xd8, 0xff]);
        }
        let resp = frame_http(AxumPath("cam-b".to_string()), State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_frame_http_unknown_camera_not_found() {
        let state = default_state();
        let resp = frame_http(AxumPath("cam-z".to_string()), State(state)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_serves_cams_and_frames_with_fake_backend() {
        let bind: SocketAddr = "127.0.0.1:43219".parse().unwrap();
        let frame =
            encode_rgb_to_jpeg(2, 2, vec![0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec![],
            Box::new(backend),
        ));

        let client = reqwest::Client::new();
        let mut ready = false;
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/cams")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    ready = true;
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
        assert!(ready, "daemon did not become ready in time");

        let cams = client
            .get(format!("http://{bind}/cams"))
            .send()
            .await
            .unwrap();
        assert_eq!(cams.status(), ReqwestStatus::OK);
        let cams_json: serde_json::Value = cams.json().await.unwrap();
        assert_eq!(cams_json["selected_camera"], "cam-a");

        let frame_resp = client
            .get(format!("http://{bind}/cams/default/frame"))
            .send()
            .await
            .unwrap();
        assert_eq!(frame_resp.status(), ReqwestStatus::OK);
        assert_eq!(frame_resp.headers()["content-type"], "image/jpeg");
        let bytes = frame_resp.bytes().await.unwrap();
        assert!(bytes.starts_with(&[0xFF, 0xD8, 0xFF]));

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_fails_to_open_camera() {
        let bind: SocketAddr = "127.0.0.1:43221".parse().unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame: vec![],
            fail_open: Some("simulated open failure".into()),
            fail_capture: None,
        };
        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec![],
            Box::new(backend),
        ));
        tokio::time::sleep(Duration::from_millis(500)).await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{bind}/cams/default/frame"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), ReqwestStatus::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"], "failed to open selected camera");
        assert!(body.to_string().contains("simulated open failure"));
        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_reports_detailed_capture_errors() {
        let bind: SocketAddr = "127.0.0.1:43220".parse().unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame: vec![],
            fail_open: None,
            fail_capture: Some("simulated capture failure".into()),
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec![],
            Box::new(backend),
        ));
        sleep(Duration::from_millis(400)).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{bind}/cams/default/frame"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), ReqwestStatus::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"], "failed to capture frame from camera");
        assert!(body.to_string().contains("simulated capture failure"));

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_shutdown_handler_stops_daemon() {
        let bind: SocketAddr = "127.0.0.1:43222".parse().unwrap();
        let frame = encode_rgb_to_jpeg(1, 1, vec![0, 0, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec![],
            Box::new(backend),
        ));
        let client = reqwest::Client::new();
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }

        let resp = client
            .get(format!("http://{bind}/shutdown"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_rebind_changes_listener_address() {
        let bind: SocketAddr = "127.0.0.1:43223".parse().unwrap();
        let new_bind: SocketAddr = "127.0.0.1:43224".parse().unwrap();
        let frame = encode_rgb_to_jpeg(1, 1, vec![0, 0, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec![],
            Box::new(backend),
        ));
        let client = reqwest::Client::new();

        // Wait for original server
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }

        // Verify original works
        let resp = client.get(format!("http://{bind}/health")).send().await.unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);

        // Rebind to new address
        let resp = client
            .get(format!("http://{bind}/rebind?addr={new_bind}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "rebound");

        // Wait for new server to be ready
        sleep(Duration::from_millis(300)).await;

        // Verify new address works
        let resp = client.get(format!("http://{new_bind}/health")).send().await.unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);

        // Verify camera stream still works through new address
        let resp = client
            .get(format!("http://{new_bind}/cams"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_allow_hosts_allows_matching_prefix() {
        let bind: SocketAddr = "127.0.0.1:43225".parse().unwrap();
        let frame = encode_rgb_to_jpeg(1, 1, vec![0, 0, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        // Allow connections from 127.0.0.x
        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            vec!["127.0.0".to_string()],
            Box::new(backend),
        ));
        let client = reqwest::Client::new();

        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }

        // Localhost (127.0.0.1) should be allowed
        let resp = client.get(format!("http://{bind}/health")).send().await.unwrap();
        assert_eq!(resp.status(), ReqwestStatus::OK);

        handle.abort();
        let _ = handle.await;
    }
}
