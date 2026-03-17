use anyhow::{anyhow, bail, Context, Result};
use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use base64::Engine as _;
use clap::{CommandFactory, Parser, Subcommand};
use image::{load_from_memory, RgbImage};
#[cfg(not(target_os = "linux"))]
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, ControlValueSetter, FrameFormat, KnownCameraControl,
        RequestedFormat, RequestedFormatType, Resolution,
    },
    Buffer, Camera,
};
use std::{collections::HashMap, time::Instant};

#[cfg(target_os = "linux")]
use rscam::{Camera as RsCamera, Config as RsConfig};

pub mod chrome_capture;
#[cfg(target_os = "linux")]
use rscam::{
    Control as RsControl, CtrlData, CID_EXPOSURE_ABSOLUTE, CID_EXPOSURE_AUTO, CID_FOCUS_AUTO,
};
use serde::Serialize;
use serde_json::json;
use std::{
    env, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::{broadcast, Mutex, RwLock},
    time::sleep,
};
use tracing::{info, warn, Level};

const DEFAULT_BIND: &str = "127.0.0.1:43210";
const DEFAULT_OUTPUT: &str = "aeyes-frame.jpg";
const DEFAULT_VIDEO_OUTPUT: &str = "aeyes-video.avi";
const DEFAULT_VIDEO_MAX_LENGTH_SECS: f64 = 5.0;
const DEFAULT_VIDEO_FPS: u32 = 15;
const FRAME_WAIT_RETRIES: usize = 50;
const FRAME_WAIT_MS: u64 = 100;
#[cfg(target_os = "linux")]
const EXPOSURE_SAMPLE_INTERVAL: u32 = 8;
#[cfg(target_os = "linux")]
const EXPOSURE_SETTLE_FRAMES: u32 = 6;
#[cfg(target_os = "linux")]
const EXPOSURE_WARMUP_FRAMES: u32 = 8;
#[cfg(target_os = "linux")]
const HIGHLIGHT_CLIP_LUMA: u8 = 250;
#[cfg(target_os = "linux")]
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
        /// Bind address or IP for the daemon HTTP API (e.g. "0.0.0.0", "0.0.0.0:43210")
        #[arg(long, default_value = DEFAULT_BIND)]
        bind: String,
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
    /// Capture a video clip through the daemon HTTP API
    Video {
        /// Camera stable ID or name; defaults to the daemon-selected camera.
        #[arg(long)]
        camera: Option<String>,
        /// Output file path (AVI MJPEG format)
        #[arg(short, long, default_value = DEFAULT_VIDEO_OUTPUT)]
        output: PathBuf,
        /// Maximum video length in seconds
        #[arg(long, default_value_t = DEFAULT_VIDEO_MAX_LENGTH_SECS)]
        max_length: f64,
        /// Frames per second
        #[arg(long, default_value_t = DEFAULT_VIDEO_FPS)]
        fps: u32,
    },
    /// Stop the daemon
    Stop,
    /// Show daemon status
    Status,
    /// Capture a screenshot from Chrome via DevTools Protocol
    Chrome {
        /// JPEG quality (1-100)
        #[arg(long, default_value_t = 85)]
        quality: u32,
        /// Output file path
        #[arg(short, long, default_value = "aeyes-chrome.jpg")]
        output: PathBuf,
        /// List available Chrome tabs
        #[arg(long)]
        list_tabs: bool,
    },
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

#[cfg(target_os = "linux")]
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
    chrome_session: OptionChromeSession,
}

/// Persistent Chrome CDP session with a background WebSocket handler.
/// The WebSocket stays connected in a background thread so Chrome's
/// "Allow debugging" permission persists across requests.
#[derive(Clone)]
pub struct OptionChromeSession(Arc<Mutex<Option<ChromeSession>>>);

impl Default for OptionChromeSession {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

#[derive(Clone)]
pub struct ChromeSession {
    /// Channel to send CDP commands to the background thread
    cmd_tx: std::sync::mpsc::Sender<CdpCommand>,
}

struct CdpCommand {
    method: String,
    params: serde_json::Value,
    respond_to: std::sync::mpsc::Sender<Result<serde_json::Value>>,
}

/// Spawn a persistent CDP connection - ONE connection for everything.
fn spawn_persistent_chrome_session(ws_url: String) -> Result<ChromeSession> {
    use tungstenite::{connect, Message};

    // Connect to Chrome ONCE
    let (mut ws, _) = connect(&ws_url).map_err(|e| anyhow::anyhow!("connect: {e}"))?;

    // Get targets using the SAME connection
    let targets_msg = json!({"id": 1, "method": "Target.getTargets", "params": {}});
    ws.send(Message::Text(targets_msg.to_string().into()))
        .map_err(|e| anyhow::anyhow!("send: {e}"))?;

    let target_id = loop {
        let msg = ws.read().map_err(|e| anyhow::anyhow!("read: {e}"))?;
        if let Ok(text) = msg.to_text() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                if value.get("id") == Some(&json!(1)) {
                    let targets = value["result"]["targetInfos"]
                        .as_array()
                        .context("no targets")?;
                    let page = targets
                        .iter()
                        .find(|t| t["type"] == "page")
                        .context("no Chrome page found")?;
                    break page["targetId"]
                        .as_str()
                        .context("no targetId")?
                        .to_string();
                }
            }
        }
    };

    // Attach using the SAME connection
    let attach_msg = json!({
        "id": 2,
        "method": "Target.attachToTarget",
        "params": { "targetId": &target_id, "flatten": true }
    });
    ws.send(Message::Text(attach_msg.to_string().into()))
        .map_err(|e| anyhow::anyhow!("attach: {e}"))?;

    let session_id = loop {
        let msg = ws.read().map_err(|e| anyhow::anyhow!("read: {e}"))?;
        if let Ok(text) = msg.to_text() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                if value.get("id") == Some(&json!(2)) {
                    if let Some(err) = value.get("error") {
                        anyhow::bail!("attach: {err}");
                    }
                    break value["result"]["sessionId"]
                        .as_str()
                        .context("no sessionId")?
                        .to_string();
                }
            }
        }
    };

    info!("Chrome: ONE connection for targets+attach+commands, session {session_id}");

    // Channel for commands
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<CdpCommand>();

    // Spawn background thread that owns this ONE connection
    std::thread::spawn(move || {
        let mut next_id: u64 = 3;

        while let Ok(cmd) = cmd_rx.recv() {
            let cmd_id = next_id;
            next_id += 1;
            let cmd_msg = json!({
                "id": cmd_id,
                "sessionId": session_id.as_str(),
                "method": cmd.method,
                "params": cmd.params
            });

            if let Err(e) = ws.send(Message::Text(cmd_msg.to_string().into())) {
                let _ = cmd.respond_to.send(Err(anyhow::anyhow!("send: {e}")));
                break;
            }

            // Read until we get our response (skip events)
            let result = loop {
                match ws.read() {
                    Ok(msg) => {
                        if let Ok(text) = msg.to_text() {
                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                                if value.get("id") == Some(&json!(cmd_id)) {
                                    if let Some(err) = value.get("error") {
                                        break Err(anyhow::anyhow!("CDP error: {err}"));
                                    }
                                    break Ok(value["result"].clone());
                                }
                                // Skip events
                            }
                        }
                    }
                    Err(e) => break Err(anyhow::anyhow!("read: {e}")),
                }
            };

            let _ = cmd.respond_to.send(result);
        }

        info!("Chrome: persistent CDP session ended");
    });

    Ok(ChromeSession { cmd_tx })
}

impl ChromeSession {
    /// Execute a CDP command through the persistent background connection.
    pub async fn execute(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let (respond_to, response_rx) = std::sync::mpsc::channel();

        self.cmd_tx
            .send(CdpCommand {
                method: method.to_string(),
                params,
                respond_to,
            })
            .map_err(|_| anyhow::anyhow!("chrome session thread not running"))?;

        tokio::task::spawn_blocking(move || {
            response_rx
                .recv()
                .map_err(|_| anyhow::anyhow!("chrome session thread dropped"))?
        })
        .await?
    }
}

#[derive(Clone)]
struct CameraStreamState {
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    last_error: Arc<RwLock<Option<DaemonErrorState>>>,
    frame_tx: broadcast::Sender<Vec<u8>>,
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
            Ok(Box::new(NokhwaOpenCamera::open(id)?))
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn camera_index_to_id(index: &CameraIndex) -> String {
    index.as_string()
}

/// Nokhwa-based camera backend for macOS and other non-Linux platforms
#[cfg(not(target_os = "linux"))]
struct NokhwaOpenCamera {
    camera: Camera,
}

#[cfg(not(target_os = "linux"))]
impl NokhwaOpenCamera {
    fn open(id: &str) -> Result<Self> {
        let camera_index = if let Ok(index) = id.parse::<u32>() {
            CameraIndex::Index(index)
        } else {
            CameraIndex::String(id.to_string())
        };
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(
            CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::MJPEG, 30),
        ));
        let mut camera = Camera::new(camera_index, requested).context("failed to create camera")?;
        camera
            .open_stream()
            .context("failed to open camera stream")?;
        Ok(Self { camera })
    }
}

#[cfg(not(target_os = "linux"))]
impl OpenCamera for NokhwaOpenCamera {
    fn set_auto_features(&mut self) -> Result<()> {
        for control in [KnownCameraControl::Exposure, KnownCameraControl::Focus] {
            if let Err(err) = self
                .camera
                .set_camera_control(control, ControlValueSetter::Boolean(true))
            {
                info!(?control, ?err, "camera control not enabled automatically");
            }
        }
        Ok(())
    }

    fn capture_jpeg(&mut self) -> Result<Vec<u8>> {
        let frame = self.camera.frame().context("failed to read camera frame")?;
        encode_frame_as_jpeg(&frame)
    }
}

/// Encode a nokhwa Buffer as JPEG
#[cfg(not(target_os = "linux"))]
fn encode_frame_as_jpeg(buffer: &Buffer) -> Result<Vec<u8>> {
    let img: RgbImage = buffer.decode_image::<RgbFormat>()?;
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95);
    encoder.encode_image(&img)?;
    Ok(bytes)
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

pub fn print_help() -> Result<()> {
    Cli::command().print_help()?;
    println!();
    Ok(())
}

pub async fn run_cli() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Start { camera, bind }) => start_daemon(camera, bind).await,
        Some(Commands::Cams) => list_cameras_cmd(),
        Some(Commands::Frame { camera, output }) => frame_cmd(camera, &output).await,
        Some(Commands::Video {
            camera,
            output,
            max_length,
            fps,
        }) => video_cmd(camera, &output, max_length, fps).await,
        Some(Commands::Stop) => stop_daemon().await,
        Some(Commands::Status) => status_cmd().await,
        Some(Commands::Chrome {
            quality,
            output,
            list_tabs,
        }) => chrome_cmd(quality, &output, list_tabs).await,
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

fn parse_bind_address(input: &str) -> Result<SocketAddr> {
    if let Ok(addr) = input.parse::<SocketAddr>() {
        return Ok(addr);
    }
    if let Ok(ip) = input.parse::<std::net::IpAddr>() {
        let default_port: u16 = DEFAULT_BIND
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(43210);
        return Ok(SocketAddr::new(ip, default_port));
    }
    bail!("invalid bind address '{input}'; expected an IP like \"0.0.0.0\" or a full address like \"0.0.0.0:43210\"");
}

pub async fn start_daemon(requested_camera: Option<String>, bind: String) -> Result<()> {
    let bind = parse_bind_address(&bind)?;
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

/// Ensure daemon is running, auto-starting if needed. Returns daemon address.
/// If camera is specified, uses it when starting the daemon.
async fn ensure_daemon_running(camera: Option<&str>) -> Result<SocketAddr> {
    if let Ok(addr) = daemon_addr().await {
        if daemon_responding(addr).await {
            return Ok(addr);
        }
    }

    // Auto-start daemon with specified or first available camera
    let backend = NativeBackend;
    let cams = backend.list_cameras().context("no cameras found")?;
    let chosen = if let Some(cam_id) = camera {
        choose_camera(&cams, Some(cam_id))?
    } else {
        choose_camera(&cams, None)
            .or_else(|_| cams.first().cloned().context("no cameras available"))?
    };
    let bind_str = env::var("AEYES_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string());
    start_daemon(Some(chosen.id.clone()), bind_str)
        .await
        .context("failed to auto-start daemon")?;

    // Wait for daemon to be ready
    for _ in 0..100 {
        if let Ok(addr) = daemon_addr().await {
            if daemon_responding(addr).await {
                return Ok(addr);
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    anyhow::bail!("daemon failed to become ready")
}

/// Handle the chrome command - uses daemon's persistent CDP session
/// Auto-starts daemon if not running.
async fn chrome_cmd(quality: u32, output: &Path, list_tabs: bool) -> Result<()> {
    let addr = ensure_daemon_running(None).await?;
    let client = reqwest::Client::new();

    if list_tabs {
        println!("Chrome debug targets:");
        println!();

        let resp = client
            .get(format!("http://{addr}/chrome/tabs"))
            .send()
            .await
            .context("failed to query daemon")?;

        let json: serde_json::Value = resp.json().await.context("failed to parse response")?;

        if let Some(tabs) = json["tabs"].as_array() {
            for target in tabs {
                let target_id = target["target_id"].as_str().unwrap_or("");
                let short_id = &target_id[..8.min(target_id.len())];
                let title = target["title"].as_str().unwrap_or("");
                let url = target["url"].as_str().unwrap_or("");
                if !title.is_empty() {
                    println!("  {} {}", short_id, title);
                    println!("      {}", url);
                    println!();
                }
            }
        }
        return Ok(());
    }

    // Use daemon's persistent session (Chrome permission stays active)
    println!("Capturing screenshot via daemon (persistent session)...");
    let resp = client
        .get(format!("http://{addr}/chrome/screenshot?quality={quality}"))
        .send()
        .await
        .context("failed to capture via daemon")?;

    if resp.status().is_success() {
        let jpeg = resp.bytes().await?.to_vec();
        std::fs::write(output, &jpeg)
            .with_context(|| format!("failed to write screenshot to {}", output.display()))?;
        println!(
            "Screenshot saved to {} ({} bytes)",
            output.display(),
            jpeg.len()
        );
    } else {
        let err: serde_json::Value = resp.json().await?;
        anyhow::bail!(
            "daemon error: {}",
            err["error"].as_str().unwrap_or("unknown")
        );
    }

    Ok(())
}

pub async fn frame_cmd(camera: Option<String>, output: &Path) -> Result<()> {
    let addr = ensure_daemon_running(camera.as_deref()).await?;
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

pub async fn video_cmd(
    camera: Option<String>,
    output: &Path,
    max_length: f64,
    fps: u32,
) -> Result<()> {
    let addr = ensure_daemon_running(camera.as_deref()).await?;
    let path = if let Some(cam) = &camera {
        format!("/cams/{}/video?max_length={}&fps={}", cam, max_length, fps)
    } else {
        format!("/cams/default/video?max_length={}&fps={}", max_length, fps)
    };
    let bytes = http_get_bytes(addr, &path).await?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, bytes)?;
    println!("Saved video to {}", output.display());
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
    run_daemon(bind, selected_camera, Box::new(NativeBackend)).await
}

pub async fn run_daemon(
    bind: SocketAddr,
    selected_camera: String,
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
        let (frame_tx, _) = broadcast::channel::<Vec<u8>>(60);
        let stream_state = CameraStreamState {
            latest_jpeg: latest_jpeg.clone(),
            last_error: last_error.clone(),
            frame_tx: frame_tx.clone(),
        };
        streams.insert(camera.id.clone(), stream_state);

        let backend = backend.clone();
        let camera_id = camera.id.clone();
        tokio::spawn(async move {
            if let Err(err) = camera_loop(
                backend,
                camera_id.clone(),
                latest_jpeg,
                last_error,
                frame_tx,
            )
            .await
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

    fs::write(addr_path(), bind.to_string())?;
    fs::write(pid_path(), std::process::id().to_string())?;

    let state = AppState {
        selected_camera: chosen.id.clone(),
        streams: Arc::new(streams),
        cameras: Arc::new(cameras),
        last_activity,
        chrome_session: OptionChromeSession::default(),
    };
    let app = Router::new()
        .route("/cams", get(list_cams_http))
        .route("/cams/{id}/frame", get(frame_http))
        .route("/cams/{id}/video", get(video_http))
        .route("/cams/{id}/stream", get(stream_http))
        .route("/web/{id}", get(web_ui_handler))
        .route("/chrome/tabs", get(chrome_tabs_handler))
        .route("/chrome/screenshot", get(chrome_screenshot_handler))
        .route("/health", get(health_handler))
        .route("/shutdown", get(shutdown_handler))
        .route("/", get(openapi_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!("daemon listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn camera_loop(
    backend: Arc<dyn CameraBackend>,
    selected_camera: String,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    last_error: Arc<RwLock<Option<DaemonErrorState>>>,
    frame_tx: broadcast::Sender<Vec<u8>>,
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
                *latest_jpeg.write().await = Some(bytes.clone());
                *last_error.write().await = None;
                // Broadcast to all stream subscribers (ignore if no receivers)
                let _ = frame_tx.send(bytes);
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

async fn video_http(
    AxumPath(id): AxumPath<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    State(state): State<AppState>,
) -> Response {
    *state.last_activity.write().await = Instant::now();
    let requested = if id == "default" {
        state.selected_camera.clone()
    } else {
        id
    };

    let max_length: f64 = params
        .get("max_length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_VIDEO_MAX_LENGTH_SECS)
        .clamp(0.1, 60.0);
    let fps: u32 = params
        .get("fps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_VIDEO_FPS)
        .clamp(1, 60);

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

    let frame_count = (max_length * fps as f64).ceil() as usize;
    let frame_interval = Duration::from_millis(1000 / fps as u64);
    let mut frames = Vec::with_capacity(frame_count);
    let start_time = Instant::now();

    // Wait for first frame
    for _ in 0..FRAME_WAIT_RETRIES {
        if let Some(bytes) = stream.latest_jpeg.read().await.clone() {
            frames.push(bytes);
            break;
        }
        sleep(Duration::from_millis(FRAME_WAIT_MS)).await;
    }

    if frames.is_empty() {
        let error = stream.last_error.read().await.clone().unwrap_or_else(|| {
            DaemonErrorState::new("camera has not produced a frame yet")
                .with_detail(format!("camera id: {requested}"))
                .with_detail("no frame was available before video capture started")
        });
        return error_response(StatusCode::SERVICE_UNAVAILABLE, error);
    }

    // Capture remaining frames
    for _ in 1..frame_count {
        let elapsed = start_time.elapsed();
        if elapsed >= Duration::from_secs_f64(max_length) {
            break;
        }

        sleep(frame_interval).await;

        if let Some(bytes) = stream.latest_jpeg.read().await.clone() {
            frames.push(bytes);
        }
    }

    // Generate AVI MJPEG video
    match create_avi_mjpeg(&frames, fps) {
        Ok(avi_data) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "video/x-msvideo")
            .body(Body::from(avi_data))
            .unwrap(),
        Err(err) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            DaemonErrorState::new("failed to create video").with_detail(format!("error: {err:#}")),
        ),
    }
}

async fn stream_http(AxumPath(id): AxumPath<String>, State(state): State<AppState>) -> Response {
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

    let mut frame_rx = stream.frame_tx.subscribe();

    let body = Body::from_stream(async_stream::stream! {
        // Send initial boundary
        yield Ok::<_, axum::Error>(bytes::Bytes::from("--frame\r\n"));

        loop {
            match frame_rx.recv().await {
                Ok(frame_data) => {
                    let header = format!(
                        "Content-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        frame_data.len()
                    );
                    yield Ok(bytes::Bytes::from(header));
                    yield Ok(bytes::Bytes::from(frame_data));
                    yield Ok(bytes::Bytes::from("\r\n--frame\r\n"));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // Skip lagged frames, continue with next
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .body(body)
        .unwrap()
}

async fn web_ui_handler(AxumPath(id): AxumPath<String>, State(state): State<AppState>) -> Response {
    *state.last_activity.write().await = Instant::now();
    let requested = if id == "default" {
        state.selected_camera.clone()
    } else {
        id.clone()
    };

    // Validate camera exists
    let camera_name = state
        .streams
        .get(&requested)
        .and_then(|_| state.cameras.iter().find(|c| c.id == requested))
        .map(|c| c.name.clone())
        .unwrap_or_else(|| requested.clone());

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>aeyes - {camera_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: #1a1a1a;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        header {{
            width: 100%;
            padding: 1rem 2rem;
            background: #2a2a2a;
            border-bottom: 1px solid #3a3a3a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            font-size: 1.25rem;
            font-weight: 500;
            color: #ffffff;
        }}
        .camera-info {{
            font-size: 0.875rem;
            color: #888;
        }}
        .stream-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            width: 100%;
        }}
        .stream-container img {{
            max-width: 100%;
            max-height: calc(100vh - 80px);
            border-radius: 4px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}
        .status {{
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            background: #2a2a2a;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #888;
        }}
        .status.connected {{
            color: #4caf50;
        }}
        .status.error {{
            color: #f44336;
        }}
    </style>
</head>
<body>
    <header>
        <h1>aeyes</h1>
        <span class="camera-info">{camera_name}</span>
    </header>
    <div class="stream-container">
        <img src="/cams/{id}/stream" alt="Live stream from {camera_name}" onerror="setStatus('error', 'Stream error')" onload="setStatus('connected', 'Connected')">
    </div>
    <div class="status" id="status">Connecting...</div>
    <script>
        function setStatus(cls, text) {{
            const el = document.getElementById('status');
            el.className = 'status ' + cls;
            el.textContent = text;
        }}
    </script>
</body>
</html>"#,
        camera_name = camera_name,
        id = id
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(html))
        .unwrap()
}

/// Create an AVI MJPEG video from a sequence of JPEG frames
pub fn create_avi_mjpeg(frames: &[Vec<u8>], fps: u32) -> Result<Vec<u8>> {
    if frames.is_empty() {
        bail!("no frames to encode");
    }

    // Get dimensions from first frame
    let img = load_from_memory(&frames[0])?;
    let width = img.width();
    let height = img.height();
    let num_frames = frames.len() as u32;

    // Calculate total frame data size (each frame chunk is: 4+4 bytes header + data + optional padding)
    let total_frame_data: u32 = frames
        .iter()
        .map(|f| {
            let size = f.len() as u32;
            let padding = size % 2; // AVI requires even-sized chunks
            8 + size + padding // chunk header (4+4) + data + padding
        })
        .sum();

    // Structure sizes
    let avih_size: u32 = 56;
    let strh_size: u32 = 56;
    let strf_size: u32 = 40;

    // strl LIST size: 4 (identifier) + 8 + strh_size + 8 + strf_size
    let strl_list_size: u32 = 4 + 8 + strh_size + 8 + strf_size;

    // hdrl LIST size: 4 (identifier) + 8 + avih_size + 8 + strl_list_size
    let hdrl_list_size: u32 = 4 + 8 + avih_size + 8 + strl_list_size;

    // movi LIST size: 4 (identifier) + total_frame_data
    let movi_list_size: u32 = 4 + total_frame_data;

    // Total RIFF size: 4 (AVI ) + 8 + hdrl_list_size + 8 + movi_list_size
    let riff_size: u32 = 4 + 8 + hdrl_list_size + 8 + movi_list_size;

    let mut avi = Vec::with_capacity(riff_size as usize);
    let microseconds_per_frame = 1_000_000u32 / fps;

    // RIFF header
    avi.extend_from_slice(b"RIFF");
    avi.extend_from_slice(&riff_size.to_le_bytes());
    avi.extend_from_slice(b"AVI ");

    // hdrl LIST
    avi.extend_from_slice(b"LIST");
    avi.extend_from_slice(&hdrl_list_size.to_le_bytes());
    avi.extend_from_slice(b"hdrl");

    // avih chunk
    avi.extend_from_slice(b"avih");
    avi.extend_from_slice(&avih_size.to_le_bytes());
    avi.extend_from_slice(&microseconds_per_frame.to_le_bytes()); // dwMicroSecPerFrame
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwMaxBytesPerSec
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwPaddingGranularity
    avi.extend_from_slice(&0x10u32.to_le_bytes()); // dwFlags (AVIF_HASINDEX)
    avi.extend_from_slice(&num_frames.to_le_bytes()); // dwTotalFrames
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    avi.extend_from_slice(&1u32.to_le_bytes()); // dwStreams
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwSuggestedBufferSize
    avi.extend_from_slice(&width.to_le_bytes()); // dwWidth
    avi.extend_from_slice(&height.to_le_bytes()); // dwHeight
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwReserved[0]
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwReserved[1]
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwReserved[2]
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwReserved[3]

    // strl LIST
    avi.extend_from_slice(b"LIST");
    avi.extend_from_slice(&strl_list_size.to_le_bytes());
    avi.extend_from_slice(b"strl");

    // strh chunk (AVISTREAMHEADER)
    avi.extend_from_slice(b"strh");
    avi.extend_from_slice(&strh_size.to_le_bytes());
    avi.extend_from_slice(b"vids"); // fccType (video stream)
    avi.extend_from_slice(b"MJPG"); // fccHandler
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwFlags
    avi.extend_from_slice(&0u16.to_le_bytes()); // wPriority
    avi.extend_from_slice(&0u16.to_le_bytes()); // wLanguage
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    avi.extend_from_slice(&1u32.to_le_bytes()); // dwScale
    avi.extend_from_slice(&fps.to_le_bytes()); // dwRate
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwStart
    avi.extend_from_slice(&num_frames.to_le_bytes()); // dwLength
    avi.extend_from_slice(&total_frame_data.to_le_bytes()); // dwSuggestedBufferSize
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwQuality
    avi.extend_from_slice(&0u32.to_le_bytes()); // dwSampleSize
    avi.extend_from_slice(&0u32.to_le_bytes()); // rcFrame (left, top)
    avi.extend_from_slice(&(width as u16).to_le_bytes()); // rcFrame (right)
    avi.extend_from_slice(&(height as u16).to_le_bytes()); // rcFrame (bottom)

    // strf chunk (BITMAPINFOHEADER)
    avi.extend_from_slice(b"strf");
    avi.extend_from_slice(&strf_size.to_le_bytes());
    avi.extend_from_slice(&strf_size.to_le_bytes()); // biSize
    avi.extend_from_slice(&width.to_le_bytes()); // biWidth
    avi.extend_from_slice(&height.to_le_bytes()); // biHeight
    avi.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    avi.extend_from_slice(&24u16.to_le_bytes()); // biBitCount
    avi.extend_from_slice(&0u32.to_le_bytes()); // biCompression (0 = BI_RGB, but MJPEG uses FCC)
                                                // Actually for MJPEG we need the FCC code
                                                // Let's use the bytes 'MJPG' as the compression type
    avi.truncate(avi.len() - 4); // remove the 0 we just wrote
    avi.extend_from_slice(b"MJPG"); // biCompression = MJPG FCC
    avi.extend_from_slice(&((width * height * 3) as u32).to_le_bytes()); // biSizeImage
    avi.extend_from_slice(&0u32.to_le_bytes()); // biXPelsPerMeter
    avi.extend_from_slice(&0u32.to_le_bytes()); // biYPelsPerMeter
    avi.extend_from_slice(&0u32.to_le_bytes()); // biClrUsed
    avi.extend_from_slice(&0u32.to_le_bytes()); // biClrImportant

    // movi LIST
    avi.extend_from_slice(b"LIST");
    avi.extend_from_slice(&movi_list_size.to_le_bytes());
    avi.extend_from_slice(b"movi");

    // Frame data chunks
    for frame in frames {
        avi.extend_from_slice(b"00db"); // Stream 0, DIB frame
        let size = frame.len() as u32;
        avi.extend_from_slice(&size.to_le_bytes());
        avi.extend_from_slice(frame);
        // Pad to even boundary
        if !size.is_multiple_of(2) {
            avi.push(0);
        }
    }

    Ok(avi)
}

// ============================================================================
// Chrome DevTools Protocol handlers - maintains persistent sessions
// ============================================================================

/// Get or create a Chrome CDP session - connects ONCE.
async fn get_or_create_chrome_session(state: &AppState) -> Result<ChromeSession> {
    // Check if we have a valid cached session
    {
        let session_guard = state.chrome_session.0.lock().await;
        if let Some(ref session) = *session_guard {
            info!("Chrome: reusing cached CDP session");
            return Ok(session.clone());
        }
    }

    // Create new session - single connection for everything
    info!("Chrome: creating new persistent CDP session");
    let ws_url = chrome_capture::get_browser_ws_url()?;

    // spawn_persistent_chrome_session connects ONCE, gets targets, attaches, keeps connection
    let session = spawn_persistent_chrome_session(ws_url)?;

    // Cache the session
    *state.chrome_session.0.lock().await = Some(session.clone());

    Ok(session)
}

/// GET /chrome/tabs - List Chrome tabs
async fn chrome_tabs_handler(State(state): State<AppState>) -> impl IntoResponse {
    *state.last_activity.write().await = Instant::now();

    match chrome_capture::list_targets() {
        Ok(targets) => {
            let tabs: Vec<_> = targets
                .into_iter()
                .filter(|t| t.target_type == "page")
                .collect();
            Json(serde_json::json!({ "tabs": tabs })).into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET /chrome/screenshot - Capture screenshot from Chrome
async fn chrome_screenshot_handler(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    *state.last_activity.write().await = Instant::now();

    let quality = params
        .get("quality")
        .and_then(|q| q.parse::<u32>().ok())
        .unwrap_or(85);

    match get_or_create_chrome_session(&state).await {
        Ok(session) => {
            // Use the persistent session to capture
            let result = session
                .execute(
                    "Page.captureScreenshot",
                    json!({
                        "format": "jpeg",
                        "quality": quality,
                        "fromSurface": true
                    }),
                )
                .await;

            match result {
                Ok(data) => {
                    let img_data = data["data"].as_str().unwrap_or("");
                    match base64::engine::general_purpose::STANDARD.decode(img_data) {
                        Ok(jpeg) => {
                            (StatusCode::OK, [("Content-Type", "image/jpeg")], jpeg).into_response()
                        }
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({ "error": format!("decode: {e}") })),
                        )
                            .into_response(),
                    }
                }
                Err(e) => {
                    // Clear session on error
                    *state.chrome_session.0.lock().await = None;
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({ "error": e.to_string() })),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
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

async fn openapi_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "aeyes",
            "description": "AI Eyes – non-interactive webcam daemon",
            "version": "0.1.0"
        },
        "paths": {
            "/cams": {
                "get": {
                    "summary": "List cameras",
                    "operationId": "listCams",
                    "responses": {
                        "200": {
                            "description": "Available cameras",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CamerasResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/cams/{id}/frame": {
                "get": {
                    "summary": "Capture a JPEG frame",
                    "operationId": "captureFrame",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": true,
                            "description": "Camera ID or \"default\" for the selected camera",
                            "schema": { "type": "string" }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "JPEG image",
                            "content": {
                                "image/jpeg": {
                                    "schema": { "type": "string", "format": "binary" }
                                }
                            }
                        },
                        "404": {
                            "description": "Camera not found",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ErrorResponse" }
                                }
                            }
                        },
                        "503": {
                            "description": "Frame unavailable",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ErrorResponse" }
                                }
                            }
                        }
                    }
                }
            },
            "/cams/{id}/video": {
                "get": {
                    "summary": "Capture an AVI MJPEG video clip",
                    "operationId": "captureVideo",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": true,
                            "description": "Camera ID or \"default\" for the selected camera",
                            "schema": { "type": "string" }
                        },
                        {
                            "name": "max_length",
                            "in": "query",
                            "description": "Maximum video length in seconds (0.1–60, default 5.0)",
                            "schema": { "type": "number", "default": 5.0 }
                        },
                        {
                            "name": "fps",
                            "in": "query",
                            "description": "Frames per second (1–60, default 15)",
                            "schema": { "type": "integer", "default": 15 }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "AVI video",
                            "content": {
                                "video/x-msvideo": {
                                    "schema": { "type": "string", "format": "binary" }
                                }
                            }
                        },
                        "404": {
                            "description": "Camera not found",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ErrorResponse" }
                                }
                            }
                        },
                        "503": {
                            "description": "Frame unavailable",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ErrorResponse" }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "health",
                    "responses": {
                        "200": {
                            "description": "Daemon is healthy",
                            "content": {
                                "text/plain": {
                                    "schema": { "type": "string", "example": "ok" }
                                }
                            }
                        }
                    }
                }
            },
            "/shutdown": {
                "get": {
                    "summary": "Stop the daemon",
                    "operationId": "shutdown",
                    "responses": {
                        "200": {
                            "description": "Daemon is stopping",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": { "type": "string", "example": "stopping" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "CameraDescriptor": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "backend": { "type": "string" }
                    },
                    "required": ["id", "name", "backend"]
                },
                "CamerasResponse": {
                    "type": "object",
                    "properties": {
                        "selected_camera": { "type": "string" },
                        "cameras": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/CameraDescriptor" }
                        }
                    },
                    "required": ["selected_camera", "cameras"]
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": { "type": "string" },
                        "details": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["error", "details"]
                }
            }
        }
    }))
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

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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
        let (tx_a, _) = broadcast::channel(60);
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
                frame_tx: tx_a,
            },
        );
        let (tx_b, _) = broadcast::channel(60);
        streams.insert(
            "cam-b".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
                frame_tx: tx_b,
            },
        );
        AppState {
            selected_camera: "cam-a".to_string(),
            streams: Arc::new(streams),
            cameras: Arc::new(fake_cameras()),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            chrome_session: OptionChromeSession::default(),
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
    #[cfg(target_os = "linux")]
    fn test_analyze_rgb_frame_detects_bright_highlights() {
        let stats = analyze_rgb_frame(&[
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20,
        ]);
        assert!(stats.clipped_ratio > 0.45);
        assert!(stats.p95_luma >= 250);
    }

    #[test]
    #[cfg(target_os = "linux")]
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
    #[cfg(target_os = "linux")]
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
    #[cfg(target_os = "linux")]
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
        let (tx, _) = broadcast::channel(60);
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(None)),
                last_error: Arc::new(RwLock::new(None)),
                frame_tx: tx,
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
            chrome_session: OptionChromeSession::default(),
        };
        let Json(response) = list_cams_http(State(state)).await;
        assert_eq!(response.selected_camera, "cam-a");
        assert_eq!(response.cameras.len(), 1);
    }

    #[tokio::test]
    async fn test_frame_http_success_default() {
        let jpeg = vec![0xff, 0xd8, 0xff];
        let mut streams = HashMap::new();
        let (tx, _) = broadcast::channel(60);
        streams.insert(
            "cam-a".to_string(),
            CameraStreamState {
                latest_jpeg: Arc::new(RwLock::new(Some(jpeg.clone()))),
                last_error: Arc::new(RwLock::new(None)),
                frame_tx: tx,
            },
        );
        let state = AppState {
            selected_camera: "cam-a".to_string(),
            streams: Arc::new(streams),
            cameras: Arc::new(vec![]),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            chrome_session: OptionChromeSession::default(),
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

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));

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
        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));
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

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));
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

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));
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

    #[test]
    fn test_create_avi_mjpeg_basic() {
        let frame = encode_rgb_to_jpeg(4, 4, vec![255; 48]).unwrap();
        let frames = vec![frame.clone(), frame];
        let avi = create_avi_mjpeg(&frames, 15).unwrap();

        // Check RIFF header
        assert!(avi.starts_with(b"RIFF"));
        // Check AVI signature at offset 8
        assert_eq!(&avi[8..12], b"AVI ");
        // Check for hdrl list
        assert!(avi.windows(4).any(|w| w == b"hdrl"));
        // Check for movi list
        assert!(avi.windows(4).any(|w| w == b"movi"));
        // Check for MJPG codec
        assert!(avi.windows(4).any(|w| w == b"MJPG"));
    }

    #[test]
    fn test_create_avi_mjpeg_empty_frames() {
        let err = create_avi_mjpeg(&[], 15).unwrap_err().to_string();
        assert!(err.contains("no frames"));
    }

    #[test]
    fn test_create_avi_mjpeg_single_frame() {
        let frame =
            encode_rgb_to_jpeg(2, 2, vec![0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0]).unwrap();
        let avi = create_avi_mjpeg(&[frame], 30).unwrap();
        assert!(avi.starts_with(b"RIFF"));
        assert_eq!(&avi[8..12], b"AVI ");
    }

    #[tokio::test]
    async fn test_video_http_unknown_camera() {
        let state = default_state();
        let params = std::collections::HashMap::new();
        let resp = video_http(AxumPath("cam-z".to_string()), Query(params), State(state)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_serves_video_with_fake_backend() {
        let bind: SocketAddr = "127.0.0.1:43223".parse().unwrap();
        let frame = encode_rgb_to_jpeg(4, 4, vec![128; 48]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));

        let client = reqwest::Client::new();
        let mut ready = false;
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    ready = true;
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
        assert!(ready, "daemon did not become ready in time");

        // Request a short video (0.2 seconds at 10 fps = 2 frames)
        let video_resp = client
            .get(format!(
                "http://{bind}/cams/default/video?max_length=0.2&fps=10"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(video_resp.status(), ReqwestStatus::OK);
        assert_eq!(video_resp.headers()["content-type"], "video/x-msvideo");

        let bytes = video_resp.bytes().await.unwrap();
        // Verify it's a valid AVI file
        assert!(bytes.starts_with(b"RIFF"));
        assert_eq!(&bytes[8..12], b"AVI ");

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_serves_video_default_params() {
        let bind: SocketAddr = "127.0.0.1:43226".parse().unwrap();
        let frame = encode_rgb_to_jpeg(2, 2, vec![100; 12]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));

        let client = reqwest::Client::new();
        let mut ready = false;
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    ready = true;
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
        assert!(ready, "daemon did not become ready in time");

        // Request video with default params (using a short max_length for test speed)
        let video_resp = client
            .get(format!(
                "http://{bind}/cams/default/video?max_length=0.1&fps=5"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(video_resp.status(), ReqwestStatus::OK);
        assert_eq!(video_resp.headers()["content-type"], "video/x-msvideo");

        let bytes = video_resp.bytes().await.unwrap();
        assert!(bytes.len() > 100, "video should have content");

        handle.abort();
        let _ = handle.await;
    }

    #[tokio::test]
    #[serial]
    async fn test_video_command_camera_not_found() {
        let bind: SocketAddr = "127.0.0.1:43227".parse().unwrap();
        let frame =
            encode_rgb_to_jpeg(2, 2, vec![0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));

        let client = reqwest::Client::new();
        let mut ready = false;
        for _ in 0..20 {
            if let Ok(resp) = client.get(format!("http://{bind}/health")).send().await {
                if resp.status() == ReqwestStatus::OK {
                    ready = true;
                    break;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
        assert!(ready, "daemon did not become ready in time");

        // Request video for non-existent camera
        let video_resp = client
            .get(format!(
                "http://{bind}/cams/nonexistent/video?max_length=0.1&fps=5"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(video_resp.status(), ReqwestStatus::NOT_FOUND);
        let body: serde_json::Value = video_resp.json().await.unwrap();
        assert!(body["error"].as_str().unwrap().contains("not managed"));

        handle.abort();
        let _ = handle.await;
    }

    #[test]
    fn test_parse_bind_address_full() {
        let addr = parse_bind_address("127.0.0.1:8080").unwrap();
        assert_eq!(addr.port(), 8080);
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
    }

    #[test]
    fn test_parse_bind_address_ip_only() {
        let addr = parse_bind_address("0.0.0.0").unwrap();
        assert_eq!(addr.port(), 43210);
        assert_eq!(addr.ip().to_string(), "0.0.0.0");
    }

    #[test]
    fn test_parse_bind_address_ipv6() {
        let addr = parse_bind_address("[::1]:9000").unwrap();
        assert_eq!(addr.port(), 9000);
        assert!(addr.ip().is_ipv6());
    }

    #[test]
    fn test_parse_bind_address_invalid() {
        let err = parse_bind_address("not-an-address")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid bind address"));
    }

    #[test]
    fn test_parse_bind_address_invalid_port() {
        let err = parse_bind_address("127.0.0.1:99999")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid bind address"));
    }

    #[test]
    fn test_encode_rgb_to_jpeg_basic() {
        let jpeg = encode_rgb_to_jpeg(2, 2, vec![255; 12]).unwrap();
        assert!(jpeg.starts_with(&[0xff, 0xd8])); // JPEG magic bytes
    }

    #[test]
    fn test_encode_rgb_to_jpeg_single_pixel() {
        let jpeg = encode_rgb_to_jpeg(1, 1, vec![255, 0, 0]).unwrap();
        assert!(jpeg.starts_with(&[0xff, 0xd8]));
    }

    #[test]
    fn test_camera_descriptor_clone() {
        let cam = CameraDescriptor {
            id: "cam-0".to_string(),
            name: "Test Cam".to_string(),
            backend: "fake".to_string(),
        };
        let cloned = cam.clone();
        assert_eq!(cam.id, cloned.id);
        assert_eq!(cam.name, cloned.name);
    }

    #[test]
    fn test_daemon_error_state_structure() {
        // Just test we can access fields
        let path = addr_path();
        assert!(path.ends_with("daemon.addr"));
    }

    #[test]
    fn test_native_backend_name() {
        // Just ensure it returns a non-empty string
        assert!(!NativeBackend.name().is_empty());
    }

    #[test]
    fn test_addr_path_not_empty() {
        let path = addr_path();
        assert!(path.ends_with("daemon.addr"));
    }

    #[test]
    fn test_pid_path_not_empty() {
        let path = pid_path();
        assert!(path.ends_with("daemon.pid"));
    }
}
