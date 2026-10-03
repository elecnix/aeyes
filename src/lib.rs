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
use rscam::{
    Camera as RsCamera, Config as RsConfig, Control as RsControl, CtrlData, ResolutionInfo,
    CID_EXPOSURE_ABSOLUTE, CID_EXPOSURE_AUTO, CID_FOCUS_AUTO,
};

pub mod chrome_capture;
use serde::Serialize;
use serde_json::json;
use std::{
    env, fs,
    net::{IpAddr, Ipv4Addr, SocketAddr},
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
        /// Force capture resolution (e.g. "320x320") instead of the device native/default mode
        #[arg(long)]
        resolution: Option<String>,
        /// Force capture format (MJPG/YUYV/YU12/YV12/NV12/RGB3/BGR3) instead of the device native/default mode
        #[arg(long)]
        format: Option<String>,
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
        /// Force capture resolution (e.g. "320x320"); applied when the daemon is started/auto-started
        #[arg(long)]
        resolution: Option<String>,
        /// Force capture format (MJPG/YUYV/YU12/YV12/NV12/RGB3/BGR3); applied when the daemon is started/auto-started
        #[arg(long)]
        format: Option<String>,
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
        /// Force capture resolution (e.g. "320x320"); applied when the daemon is started/auto-started
        #[arg(long)]
        resolution: Option<String>,
        /// Force capture format (MJPG/YUYV/YU12/YV12/NV12/RGB3/BGR3); applied when the daemon is started/auto-started
        #[arg(long)]
        format: Option<String>,
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
    fn open(&self, id: &str, options: &CameraOpenOptions) -> Result<Box<dyn OpenCamera>>;
}

/// Explicit capture overrides applied when opening a camera.
///
/// Both fields are optional; when unset the backend picks the device's
/// native/current mode and falls back to its preset list (see
/// `plan_capture_presets`). Only formats aeyes can encode (MJPG, YUYV,
/// YU12, YV12, NV12, RGB3, BGR3) are accepted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraOpenOptions {
    /// Explicit resolution override as (width, height), e.g. `Some((320, 320))`.
    pub resolution: Option<(u32, u32)>,
    /// Explicit fourcc format override, e.g. `Some(*b"YUYV")`.
    pub format: Option<[u8; 4]>,
}

/// Parse a `WxH` resolution string (e.g. `"320x320"` or `"1280X720"`).
pub fn parse_resolution(input: &str) -> Result<(u32, u32)> {
    let (width, height) = input
        .split_once(['x', 'X'])
        .with_context(|| format!("invalid resolution '{input}'; expected WxH like 320x320"))?;
    let width: u32 = width
        .trim()
        .parse()
        .with_context(|| format!("invalid width in '{input}'; expected WxH like 320x320"))?;
    let height: u32 = height
        .trim()
        .parse()
        .with_context(|| format!("invalid height in '{input}'; expected WxH like 320x320"))?;
    if width == 0 || height == 0 {
        bail!("invalid resolution '{input}'; width and height must be non-zero");
    }
    Ok((width, height))
}

/// Parse a fourcc format string, restricted to the formats aeyes can encode.
pub fn parse_fourcc(input: &str) -> Result<[u8; 4]> {
    let upper = input.trim().to_ascii_uppercase();
    match upper.as_str() {
        "MJPG" | "YUYV" | "YU12" | "YV12" | "NV12" | "RGB3" | "BGR3" => {
            let mut fourcc = [0u8; 4];
            fourcc.copy_from_slice(upper.as_bytes());
            Ok(fourcc)
        }
        _ => bail!(
            "unsupported format '{input}'; aeyes can only encode MJPG, YUYV, YU12, YV12, NV12, RGB3, or BGR3"
        ),
    }
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

    fn open(&self, id: &str, options: &CameraOpenOptions) -> Result<Box<dyn OpenCamera>> {
        #[cfg(target_os = "linux")]
        {
            Ok(Box::new(V4l2OpenCamera::open(id, options)?))
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(Box::new(NokhwaOpenCamera::open(id, options)?))
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
    fn open(id: &str, options: &CameraOpenOptions) -> Result<Self> {
        let camera_index = if let Ok(index) = id.parse::<u32>() {
            CameraIndex::Index(index)
        } else {
            CameraIndex::String(id.to_string())
        };
        // Non-Linux backend keeps MJPEG encoding; only the resolution override
        // is honored (the format override maps to FrameFormat on other backends
        // and is intentionally ignored here to keep the fallback deterministic).
        let (width, height) = options.resolution.unwrap_or((1920, 1080));
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(
            CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, 30),
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
    fn open(id: &str, options: &CameraOpenOptions) -> Result<Self> {
        let index: u32 = id
            .parse()
            .with_context(|| format!("camera ID '{id}' must be numeric like '0'"))?;
        let device_path = format!("/dev/video{index}");
        let mut camera = RsCamera::new(&device_path)
            .with_context(|| format!("failed to open V4L2 device {device_path}"))?;

        // Ask the device what it actually supports instead of assuming the
        // fixed preset list. With v4l2loopback (exclusive_caps=1) the device
        // only advertises the producer's native mode, so the old 640x480-only
        // attempt could never succeed (elecnix/aeyes#26).
        let supported = enumerate_supported_formats(&camera);
        let native = query_native_format(&device_path);
        let candidates = plan_capture_presets(&supported, native, options);

        let mut attempted: Vec<String> = Vec::new();
        for preset in candidates {
            for (num, den) in candidate_intervals(preset.fps) {
                let config = RsConfig {
                    interval: (num, den),
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
                        let attempt = format!(
                            "{}x{}@{}fps {} rejected by {}: {}",
                            preset.width,
                            preset.height,
                            den,
                            String::from_utf8_lossy(&preset.format),
                            device_path,
                            err
                        );
                        if !attempted.contains(&attempt) {
                            attempted.push(attempt);
                        }
                    }
                }
            }
        }

        let last = attempted
            .last()
            .cloned()
            .unwrap_or_else(|| "no capture modes attempted".to_string());
        bail!(
            "failed to start V4L2 stream on {}\nattempted modes: {}\nlast attempted mode: {}\ndevice supports: {}",
            device_path,
            if attempted.is_empty() {
                "(none)".to_string()
            } else {
                attempted.join("; ")
            },
            last,
            summarize_supported(&supported)
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

        match &self.format {
            b"MJPG" => Ok(frame.to_vec()),
            b"YUYV" => yuyv_to_jpeg(self.width, self.height, &frame)
                .with_context(|| format!("failed to encode YUYV frame from {}", self.device_path)),
            b"YU12" => yuv420_to_jpeg(self.width, self.height, &frame, false)
                .with_context(|| format!("failed to encode YU12 frame from {}", self.device_path)),
            b"YV12" => yuv420_to_jpeg(self.width, self.height, &frame, true)
                .with_context(|| format!("failed to encode YV12 frame from {}", self.device_path)),
            b"NV12" => nv12_to_jpeg(self.width, self.height, &frame)
                .with_context(|| format!("failed to encode NV12 frame from {}", self.device_path)),
            b"RGB3" => rgb24_to_jpeg(self.width, self.height, &frame)
                .with_context(|| format!("failed to encode RGB3 frame from {}", self.device_path)),
            b"BGR3" => bgr24_to_jpeg(self.width, self.height, &frame)
                .with_context(|| format!("failed to encode BGR3 frame from {}", self.device_path)),
            other => bail!(
                "unsupported frame format '{}' from {}",
                String::from_utf8_lossy(other),
                self.device_path
            ),
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CapturePreset {
    width: u32,
    height: u32,
    fps: u32,
    format: [u8; 4],
}

/// A format advertised by the device (VIDIOC_ENUM_FMT + VIDIOC_ENUM_FRAMESIZES).
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct SupportedFormat {
    format: [u8; 4],
    resolutions: Vec<(u32, u32)>,
}

/// The device's current/active mode as reported by `v4l2-ctl --get-fmt-video`.
#[cfg(target_os = "linux")]
#[derive(Copy, Clone, Debug)]
struct NativeFormat {
    width: u32,
    height: u32,
    format: [u8; 4],
}

/// Formats aeyes can hand back as JPEG, in preference order: MJPG is passed
/// through untouched, YUYV / YU12 / YV12 / NV12 / RGB3 / BGR3 are converted.
#[cfg(target_os = "linux")]
fn encodable_formats() -> [[u8; 4]; 7] {
    [
        *b"MJPG", *b"YUYV", *b"RGB3", *b"BGR3", *b"YU12", *b"YV12", *b"NV12",
    ]
}

/// Formats aeyes can hand back as JPEG (MJPEG passthrough, the rest converted).
#[cfg(target_os = "linux")]
fn is_encodable_format(format: &[u8; 4]) -> bool {
    encodable_formats().contains(format)
}

/// Common sizes used as fallbacks when a device advertises a stepwise frame
/// size range or an explicit --format without a matching enumerated entry.
#[cfg(target_os = "linux")]
fn common_resolutions() -> [(u32, u32); 7] {
    [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1280, 720),
        (640, 480),
        (320, 320),
        (320, 240),
    ]
}

/// The historical quality-first candidate list, kept as a last-resort fallback
/// for devices whose enumeration returns nothing useful.
#[cfg(target_os = "linux")]
const FIXED_CAPTURE_PRESETS: [CapturePreset; 10] = [
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
        format: *b"YUYV",
    },
    CapturePreset {
        width: 640,
        height: 480,
        fps: 30,
        format: *b"MJPG",
    },
];

/// Enumerate the formats and discrete frame sizes a device advertises via
/// VIDIOC_ENUM_FMT / VIDIOC_ENUM_FRAMESIZES.
#[cfg(target_os = "linux")]
fn enumerate_supported_formats(camera: &RsCamera) -> Vec<SupportedFormat> {
    let mut out = Vec::new();
    for fmt in camera.formats() {
        let Ok(fmt) = fmt else { continue };
        let resolutions = match camera.resolutions(&fmt.format) {
            Ok(ResolutionInfo::Discretes(list)) => list,
            Ok(ResolutionInfo::Stepwise { min, max, .. }) => {
                // Stepwise range: expose the bounds plus common sizes inside it.
                let mut list = vec![min, max];
                for (width, height) in common_resolutions() {
                    if width >= min.0 && width <= max.0 && height >= min.1 && height <= max.1 {
                        list.push((width, height));
                    }
                }
                list.sort_unstable();
                list.dedup();
                list
            }
            Err(_) => Vec::new(),
        };
        out.push(SupportedFormat {
            format: fmt.format,
            resolutions,
        });
    }
    out
}

/// Query the device's current/active mode via `v4l2-ctl --get-fmt-video`.
#[cfg(target_os = "linux")]
fn query_native_format(device_path: &str) -> Option<NativeFormat> {
    let output = std::process::Command::new("v4l2-ctl")
        .arg("--device")
        .arg(device_path)
        .arg("--get-fmt-video")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut width = 0u32;
    let mut height = 0u32;
    let mut format: Option<[u8; 4]> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Width/Height") {
            if let Some(value) = rest.split(':').nth(1) {
                let mut it = value.trim().split('/');
                if let (Some(w), Some(h)) = (it.next(), it.next()) {
                    width = w.trim().parse().ok()?;
                    height = h.trim().parse().ok()?;
                }
            }
        }
        if let Some(rest) = line.strip_prefix("Pixel Format") {
            // e.g. "Pixel Format      : 'YUYV'" (optionally followed by a
            // human-readable description, e.g. "'MJPG' (Motion-JPEG)").
            if let Some(quote) = rest.find('\'') {
                let bytes: Vec<u8> = rest[quote + 1..].chars().take(4).map(|c| c as u8).collect();
                if bytes.len() == 4 {
                    format = Some([bytes[0], bytes[1], bytes[2], bytes[3]]);
                }
            }
        }
    }
    if width == 0 || height == 0 {
        return None;
    }
    Some(NativeFormat {
        width,
        height,
        format: format?,
    })
}

/// Interval candidates tried for each preset, in preference order. Some devices
/// (and v4l2loopback producers) only accept a specific frame rate, so a single
/// 30fps attempt is not always enough.
#[cfg(target_os = "linux")]
fn candidate_intervals(preferred_fps: u32) -> Vec<(u32, u32)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for fps in [preferred_fps, 15, 10, 5] {
        if fps == 0 || !seen.insert(fps) {
            continue;
        }
        out.push((1, fps));
    }
    out
}

/// Decide which capture modes to attempt, in order. Pure and unit-tested.
///
/// Order of preference:
/// 1. explicit `--resolution`/`--format` overrides (exactly what was asked);
/// 2. the device's native/current mode (if aeyes can encode it);
/// 3. enumerated encodable modes (MJPG → YUYV → RGB3 → BGR3 → YU12 → YV12 → NV12,
///    larger first);
/// 4. the historical fixed preset list, deduplicated against the above.
#[cfg(target_os = "linux")]
fn plan_capture_presets(
    supported: &[SupportedFormat],
    native: Option<NativeFormat>,
    options: &CameraOpenOptions,
) -> Vec<CapturePreset> {
    let mut presets: Vec<CapturePreset> = Vec::new();
    let mut seen: std::collections::HashSet<([u8; 4], u32, u32)> = std::collections::HashSet::new();
    let mut push = |preset: CapturePreset| {
        if seen.insert((preset.format, preset.width, preset.height)) {
            presets.push(preset);
        }
    };

    match (options.resolution, options.format) {
        (Some((width, height)), Some(format)) => {
            push(CapturePreset {
                width,
                height,
                fps: 30,
                format,
            });
            return presets;
        }
        (Some((width, height)), None) => {
            for format in encodable_formats() {
                push(CapturePreset {
                    width,
                    height,
                    fps: 30,
                    format,
                });
            }
            return presets;
        }
        (None, Some(format)) => {
            if let Some(native) = native.filter(|native| native.format == format) {
                push(CapturePreset {
                    width: native.width,
                    height: native.height,
                    fps: 30,
                    format,
                });
            }
            if let Some(supported) = supported.iter().find(|sf| sf.format == format) {
                let mut resolutions = supported.resolutions.clone();
                resolutions.sort_by_key(|(w, h)| std::cmp::Reverse(w * h));
                for (width, height) in resolutions {
                    push(CapturePreset {
                        width,
                        height,
                        fps: 30,
                        format,
                    });
                }
            }
            for (width, height) in common_resolutions() {
                push(CapturePreset {
                    width,
                    height,
                    fps: 30,
                    format,
                });
            }
            return presets;
        }
        (None, None) => {
            if let Some(native) = native {
                if is_encodable_format(&native.format) {
                    push(CapturePreset {
                        width: native.width,
                        height: native.height,
                        fps: 30,
                        format: native.format,
                    });
                }
            }
            for format in encodable_formats() {
                let mut resolutions: Vec<(u32, u32)> = Vec::new();
                for sf in supported {
                    if sf.format == format {
                        resolutions.extend(sf.resolutions.iter().copied());
                    }
                }
                resolutions.sort_by_key(|(w, h)| std::cmp::Reverse(w * h));
                for (width, height) in resolutions {
                    push(CapturePreset {
                        width,
                        height,
                        fps: 30,
                        format,
                    });
                }
            }
            for preset in FIXED_CAPTURE_PRESETS {
                push(preset);
            }
        }
    }

    presets
}

/// Human-readable summary of what the device advertises, for error reporting.
#[cfg(target_os = "linux")]
fn summarize_supported(supported: &[SupportedFormat]) -> String {
    if supported.is_empty() {
        return "(no formats enumerated)".to_string();
    }
    supported
        .iter()
        .map(|sf| {
            let res = sf
                .resolutions
                .iter()
                .map(|(w, h)| format!("{w}x{h}"))
                .collect::<Vec<_>>()
                .join(", ");
            if sf.resolutions.is_empty() {
                String::from_utf8_lossy(&sf.format).to_string()
            } else {
                format!("{} ({})", String::from_utf8_lossy(&sf.format), res)
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
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
    stdout
        .lines()
        .skip_while(|l| !l.contains("Device Caps"))
        .take(10)
        .any(|l| l.contains("Video Capture"))
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
            resolution,
            format,
        }) => start_daemon(camera, bind, cli_open_options(resolution, format)?).await,
        Some(Commands::Cams) => list_cameras_cmd(),
        Some(Commands::Frame {
            camera,
            output,
            resolution,
            format,
        }) => frame_cmd(camera, &output, cli_open_options(resolution, format)?).await,
        Some(Commands::Video {
            camera,
            output,
            max_length,
            fps,
            resolution,
            format,
        }) => {
            video_cmd(
                camera,
                &output,
                max_length,
                fps,
                cli_open_options(resolution, format)?,
            )
            .await
        }
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

/// Build `CameraOpenOptions` from the CLI `--resolution`/`--format` flags.
fn cli_open_options(
    resolution: Option<String>,
    format: Option<String>,
) -> Result<CameraOpenOptions> {
    Ok(CameraOpenOptions {
        resolution: resolution.as_deref().map(parse_resolution).transpose()?,
        format: format.as_deref().map(parse_fourcc).transpose()?,
    })
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

pub async fn start_daemon(
    requested_camera: Option<String>,
    bind: String,
    options: CameraOpenOptions,
) -> Result<()> {
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
    if let Some((width, height)) = options.resolution {
        cmd.env("AEYES_RESOLUTION", format!("{width}x{height}"));
    }
    if let Some(format) = options.format {
        cmd.env("AEYES_FORMAT", String::from_utf8_lossy(&format).to_string());
    }
    let child = cmd.spawn().context("failed to spawn daemon")?;
    let pid = child.id().context("missing daemon pid")?;
    // The child owns the runtime registry: it publishes its pid/addr only after
    // `TcpListener::bind` has succeeded. Writing them here would be a guess that
    // races the child, and a child that loses the port would leave the registry
    // naming a dead pid instead of the daemon that won it.
    println!(
        "Daemon started with PID {pid} at http://{bind} using camera {} ({})",
        chosen.id, chosen.name
    );
    Ok(())
}

pub async fn stop_daemon() -> Result<()> {
    // Prefer the registered address, but fall back to probing the bind address.
    // A daemon started outside `aeyes start` serves its port without ever
    // publishing the registry, and `stop` must still reach it instead of
    // announcing victory over a port it never touched.
    //
    // Both branches ask the peer *who it is* (`aeyes_responding`) before this
    // function sends anything at it. Reachability is the wrong question here:
    // the consequence of a wrong answer is a `GET /shutdown` delivered to an
    // unrelated process.
    let registered = daemon_addr().await.ok();
    let mut unregistered = false;
    let target = match registered {
        // Identity here too, and not only on the fallback branch: this function
        // acts destructively on whatever `target` names — it sends `/shutdown`
        // and reports success. The registry file was written by a daemon that
        // has since died, and the OS is happy to hand its port to anyone, so a
        // bare connect here would aim the same `/shutdown` at a stranger. If the
        // identity check fails we fall through to the pid path below, which
        // still stops a daemon of ours that has gone unresponsive.
        //
        // Falling through is not a hole in that check: the pid path classifies
        // the pid with `classify_daemon_pid` before it signals anything, and
        // refuses on every verdict except `Ours` — including `Unverified`, which
        // is what a platform with no identity check (Windows, non-Linux unix)
        // produces for any pid at all. A stranger holding the registered port
        // therefore cannot be reached by that fall-through either, and a
        // genuine daemon of ours that has gone unresponsive still can be, on
        // Linux.
        Some(addr) if aeyes_responding(addr).await => Some(addr),
        _ => match fallback_probe_addr() {
            Some(addr) if aeyes_responding(addr).await => {
                unregistered = true;
                Some(addr)
            }
            _ => None,
        },
    };

    if let Some(addr) = target {
        let _ = http_get_bytes(addr, "/shutdown").await;
        for _ in 0..20 {
            if !daemon_responding(addr).await {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        if unregistered {
            // We proved this is an aeyes daemon and stopped it, but no registry
            // entry names its pid, so nothing here can be attributed to a
            // process — including any pid/addr pair the files happen to hold,
            // which may name a *different* live daemon. Leave them alone: a
            // later `stop` re-classifies that pid and clears the files only when
            // there is genuinely nothing left to lose.
            println!(
                "Daemon stopped at http://{addr} (found by probing the bind address; it had no runtime registry entry, so it was not started by `aeyes start`). The runtime registry was left as it was, since no pid could be attributed to the daemon that was stopped."
            );
            return Ok(());
        }
    }

    // `refused` records a daemon we deliberately declined to signal, together
    // with the reason. Reporting success for a kill that never happened is
    // worse than reporting nothing: it tells the user the daemon is gone while
    // it is still serving. The reason travels with it because the reasons are
    // genuinely different — "I checked and it is a different executable" and
    // "there is no way to check on this platform" call for different advice,
    // and telling a user the first when the second is what happened would be a
    // lie.
    let mut refused: Option<(u32, &'static str)> = None;
    if let Ok(pid_str) = fs::read_to_string(pid_path()) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            match action_for_verdict(classify_daemon_pid(pid)) {
                PidAction::SkipSignal(note) => {
                    // Stale or recycled pid: nothing is running there to signal,
                    // and clearing the registry below is still correct.
                    warn!(pid, "pid file names no live process; skipping kill: {note}");
                }
                PidAction::Signal => {
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
                PidAction::Refuse(reason) => {
                    warn!(
                        pid,
                        "pid file names a process aeyes declined to signal: {reason}"
                    );
                    refused = Some((pid, reason));
                }
            }
        }
    }
    if let Some((pid, reason)) = refused {
        // Refusal keeps the registry. The pid file is the only handle this
        // process has on a daemon it just declined to signal, and clearing it
        // here would leave that daemon running with nothing left to stop it —
        // the next `stop` would find no registry at all and fall back to
        // probing the bind address.
        println!(
            "Did NOT stop PID {pid}: {reason}. Signalling it could kill an unrelated process, so no signal was sent and the runtime registry has been left in place. If that process really is an aeyes daemon, shut it down through its own GET /shutdown endpoint, or end it yourself."
        );
        return Ok(());
    }
    // Either we signalled the pid, or `PidAction::SkipSignal` proved nothing is
    // running there. Both satisfy the command's contract: no daemon of ours is
    // left serving, and the registry names nothing live any more.
    let _ = fs::remove_file(pid_path());
    let _ = fs::remove_file(addr_path());
    println!("Daemon stopped.");
    Ok(())
}

/// Publish this process in the runtime registry.
///
/// Only the process that actually owns the listener may write these files, and
/// only once `TcpListener::bind` has succeeded: writing them earlier would leave
/// a pid/addr pair behind describing a daemon that never came up.
///
/// `AEYES_DAEMON` is set to `1` by `start_daemon` on the child it spawns, and
/// `main` checks for exactly that value before handing off to `run_daemon_from_env`,
/// so the spawned daemon is the only process that publishes. In-process callers
/// must not touch the registry — notably the test suite, which binds throwaway
/// ports and would otherwise clobber the pid/addr files of a daemon actually
/// running on this machine.
///
/// The flip side is that a daemon started any other way — `run_daemon_from_env`
/// invoked directly, or under a supervisor — binds its port but registers
/// nothing. That is intended: an unregistered process must not be able to point
/// `status` and `stop` at a pid we never verified. `status` and `stop` cover the
/// gap by probing the fallback bind address (`fallback_probe_addr`) and saying
/// plainly that the daemon they found has no registry entry.
fn publish_runtime_registry(bind: SocketAddr) -> Result<()> {
    if env::var("AEYES_DAEMON").ok().as_deref() != Some("1") {
        return Ok(());
    }
    fs::create_dir_all(runtime_dir())?;
    fs::write(addr_path(), bind.to_string())?;
    fs::write(pid_path(), std::process::id().to_string())?;
    Ok(())
}

/// What `stop_daemon` could establish about the pid named in the registry.
///
/// Each platform only ever produces some of these — see `classify_daemon_pid` —
/// so the unused ones are silenced per target rather than crate-wide.
#[cfg_attr(target_os = "linux", allow(dead_code))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum PidVerdict {
    /// No such process.
    Absent,
    /// The process is running this very executable.
    Ours,
    /// The process is alive and is provably running a *different* executable.
    Foreign,
    /// The process is alive but its executable could not be read, so there is
    /// no way to tell ours from a stranger's.
    Unreadable,
    /// The process may well be running this very executable, but nothing on
    /// this platform can establish that: there is no cheap identity check
    /// (Windows, and unix other than Linux). Liveness, when it can even be
    /// established, is not identity — a recycled pid is alive too. This is the
    /// verdict that keeps `stop_daemon` from aiming `taskkill` at whatever
    /// process the OS has since put at that pid.
    Unverified,
}

/// Why a pid was not signalled, in the words shown to the user.
///
/// These are separate constants because they say different things and a user
/// needs different advice for each. Collapsing them into one "not ours" message
/// would be false for `UNREADABLE` (we could not read it) and for `UNVERIFIED`
/// (we had no way to read it), and each falsehood sends the user looking for a
/// problem they do not have.
const FOREIGN_REASON: &str =
    "aeyes read its executable and it is provably a different program from this one";
const UNREADABLE_REASON: &str = "aeyes could not read the executable of the process at that pid, so it cannot be confirmed as this one";
const UNVERIFIED_REASON: &str = "this platform offers no way to read which executable a pid is running — Windows, or unix other than Linux — so the pid cannot be confirmed as an aeyes daemon, and `aeyes stop` here shuts a daemon down over the daemon's own /shutdown endpoint, which has already been tried";

/// What `stop_daemon` does about a classified pid.
///
/// Split out of `stop_daemon` so the routing is testable on any target: the
/// interesting question is what happens to an `Unverified` verdict, and only
/// Linux CI can run the code that produces one.
#[derive(Debug)]
enum PidAction {
    /// Nothing is running at that pid, so there is nothing to signal and the
    /// registry may be cleared. The note says why we believe nothing is there.
    SkipSignal(&'static str),
    /// The pid is ours. Signalling it cannot hit a bystander.
    Signal,
    /// Not ours, or not provably ours: do not signal, keep the registry, and
    /// tell the user which of the two reasons applies.
    Refuse(&'static str),
}

/// Route a classified pid to an action.
///
/// The only verdict that may be signalled is [`PidVerdict::Ours`]. Everything
/// else either means "nothing is running there" (safe to clear the registry)
/// or means "we cannot tell" (must not signal, must keep the registry).
#[cfg_attr(target_os = "linux", allow(dead_code))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn action_for_verdict(verdict: PidVerdict) -> PidAction {
    match verdict {
        PidVerdict::Absent => PidAction::SkipSignal("the pid file names no live process"),
        PidVerdict::Ours => PidAction::Signal,
        PidVerdict::Foreign => PidAction::Refuse(FOREIGN_REASON),
        PidVerdict::Unreadable => PidAction::Refuse(UNREADABLE_REASON),
        PidVerdict::Unverified => PidAction::Refuse(UNVERIFIED_REASON),
    }
}

/// How much a platform with no identity check could learn about a pid.
///
/// Windows and non-Linux unix can ask *something*, but never *who*: `kill -0`
/// proves a process exists without naming its executable, and Windows offers no
/// probe worth the name here. The three cases stay distinct because only one of
/// them permits `Absent`, and `Absent` is the verdict that clears the registry.
#[cfg_attr(target_os = "linux", allow(dead_code))]
enum PidLiveness {
    /// A probe ran and said a process is there.
    Alive,
    /// A probe ran and said no process is there.
    Gone,
    /// Nothing could be asked — the probe is missing or could not be spawned.
    Unprobed,
}

/// The verdict a platform with no cheap identity check may reach.
///
/// Only a probe that actually ran and denied the process may produce
/// [`PidVerdict::Absent`]. "There is something there" and "we could not ask"
/// both produce [`PidVerdict::Unverified`], which refuses to signal: an
/// unverified pid may be a real aeyes daemon, or may be an unrelated process
/// that inherited the number when the daemon died.
///
/// This used to answer `Alive` for the first two cases and fail open on the
/// theory that refusing would leave a daemon with no way to stop it. That no
/// longer holds: `stop_daemon` shuts a daemon down over HTTP — an
/// identity-checked `GET /shutdown` — *before* it ever reaches this code, so
/// the signal is only the fallback for a daemon too unresponsive to answer.
/// Refusing here removes the ability to kill an arbitrary process, not the
/// ability to stop a daemon.
#[cfg_attr(target_os = "linux", allow(dead_code))]
fn verdict_without_identity_check(probes: PidLiveness) -> PidVerdict {
    match probes {
        PidLiveness::Gone => PidVerdict::Absent,
        PidLiveness::Alive | PidLiveness::Unprobed => PidVerdict::Unverified,
    }
}

/// Classify the pid named by the runtime registry.
///
/// The registry goes stale whenever a daemon dies, fails to bind, or loses a
/// start-up race, and the OS recycles stale pids. Signalling a recycled pid
/// would take down an unrelated process, so `stop_daemon` asks this first. It
/// deliberately errs towards "do not kill": the HTTP `/shutdown` path runs
/// beforehand and handles the normal case, so a declined force-kill is far
/// cheaper than killing a bystander.
///
/// The strength of the answer is platform-dependent, because a dependency-free
/// identity check only exists on Linux. **This function does not always answer
/// the identity question**, and on every platform that cannot answer it the
/// answer is "I do not know", never "probably fine":
///
/// * Linux — a real identity check. `/proc/<pid>/exe` must resolve to the very
///   executable we are running, so a shell running `aeyes stop`, an editor with
///   the checkout open, or a build under a directory that merely contains
///   "aeyes" all resolve elsewhere and are rejected as `Foreign`. The two paths
///   are compared canonically rather than byte-for-byte (`same_executable`),
///   because `read_link` returns a kernel-resolved path while `current_exe` is
///   derived from `/proc/self/exe` and the two spellings diverge as soon as a
///   symlinked directory sits between them — which would turn the real daemon
///   into the very `Foreign` verdict this check exists to avoid.
///   An unreadable `/proc/<pid>/exe` — the process is alive but owned by another
///   user — reads as `Unreadable`, i.e.
///   fail-closed *and* registry-preserving. Mapping it to `Absent` would be a
///   lie that costs the user their only handle on a daemon they cannot signal
///   anyway. A genuinely dead pid reads as `Absent`, which does clear the
///   registry. A `hidepid` mount defeats the distinction entirely; see
///   `verdict_for_unreadable_exe` for that limit.
/// * Other unix — **no identity check at all.** `kill -0` answers only "does a
///   process exist at this pid", which is not a statement about which program
///   it is: a recycled pid is just as alive as the daemon that used to own it.
///   So a pid the probe says is alive reads as `Unverified`, and `stop_daemon`
///   refuses to signal it. Only a probe that ran *and* denied the process
///   produces `Absent`; if `kill` cannot be spawned at all the pid reads as
///   `Unverified` too, because "we could not ask" is not "nothing is there".
/// * Windows — **neither an identity check nor a liveness probe.** There is no
///   dependency-free way to ask what program a pid is running, so this returns
///   `Unverified` for *every* pid, live, dead or recycled, and `stop_daemon`
///   sends no signal and keeps the registry.
///   Two consequences are worth stating plainly rather than leaving to be
///   discovered: a stale pid file is never cleaned up on Windows, because this
///   function cannot distinguish a stale pid from a live one; and on Windows
///   `aeyes stop` is the HTTP `/shutdown` path only — a daemon too
///   unresponsive to answer `/shutdown` has to be ended by hand.
///
/// Why the non-Linux platforms refuse rather than signal: the HTTP branch of
/// `stop_daemon` runs first and only ever talks to a peer that proved it is
/// aeyes (`aeyes_responding`), so a daemon that answers at all is already
/// stopped before this function is consulted. The pid signal exists for the
/// daemon that no longer answers, and in that state the pid may name anything.
/// Trading "cannot stop an unresponsive daemon" for "never kills a bystander"
/// is the right trade, and it is a *narrower* trade than it looks: a pid file
/// is written at bind time and outlives the daemon by however long the registry
/// survives, so every second spent on Windows with an unverified pid is a
/// second in which that pid may belong to someone else entirely.
///
/// Closing the gap for real needs a mechanism, not a doc comment: a
/// pid/creation-time pair published at bind time, or a platform process-query
/// dependency. Neither is in scope here, and until one is, "unverified" is the
/// honest answer and this function gives it.
fn classify_daemon_pid(pid: u32) -> PidVerdict {
    #[cfg(target_os = "linux")]
    {
        // An unresolvable /proc/<pid>/exe means either the pid is gone or the
        // process is not ours to inspect. Those are not the same answer and
        // must not get the same treatment: the first leaves nothing running,
        // the second leaves a live process whose only handle is the pid file.
        let Ok(theirs) = fs::read_link(format!("/proc/{pid}/exe")) else {
            return verdict_for_unreadable_exe(Path::new(&format!("/proc/{pid}")).exists());
        };
        let Ok(ours) = env::current_exe() else {
            return PidVerdict::Unreadable;
        };
        if same_executable(&ours, &theirs) {
            PidVerdict::Ours
        } else {
            PidVerdict::Foreign
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // Liveness, not identity. There is no cheap way to read the executable
        // of another process here, so a pid that answers "alive" cannot be
        // distinguished from a recycled one and must not be signalled.
        let probes = match std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
        {
            Ok(status) if status.success() => PidLiveness::Alive,
            Ok(_) => PidLiveness::Gone,
            // `kill` missing or not executable: we did not get an answer, and
            // an unanswered question is not the same as a negative one.
            Err(_) => PidLiveness::Unprobed,
        };
        verdict_without_identity_check(probes)
    }
    #[cfg(not(unix))]
    {
        // No probe and no identity check on Windows: every pid is unverified,
        // including pids that no longer exist.
        let _ = pid;
        verdict_without_identity_check(PidLiveness::Unprobed)
    }
}

/// What an unreadable `/proc/<pid>/exe` means.
///
/// `read_link` fails both when the pid is gone and when the process belongs to
/// another user (or `/proc` is mounted `hidepid`). Those must not collapse into
/// one verdict: the first leaves nothing running, the second leaves a live
/// process whose only handle is the pid file, and clearing that file is exactly
/// the mistake this module exists to avoid.
///
/// **Known limit:** on a `hidepid` mount the `/proc/<pid>` directory of another
/// user's process is hidden as well, so `proc_entry_exists` is false and the
/// verdict collapses back to `Absent` — the pre-`Unreadable` behaviour, registry
/// cleared and all. Nothing inside this process can do better: `kill -0` is not a
/// usable tiebreaker because a shell reports EPERM and ESRCH with the same exit
/// code, so an invisible-but-live pid is indistinguishable from a dead one.
/// `hidepid` is off by default, but on a host that enables it, cross-user
/// `aeyes stop` falls back to the old behaviour. That is why the doc on
/// `classify_daemon_pid` names `hidepid` as an additional trigger for
/// `Unreadable` rather than promising it.
///
/// Split out so the mapping is testable — an unprivileged test binary cannot
/// arrange for a process it may not inspect.
#[cfg(target_os = "linux")]
fn verdict_for_unreadable_exe(proc_entry_exists: bool) -> PidVerdict {
    if proc_entry_exists {
        PidVerdict::Unreadable
    } else {
        PidVerdict::Absent
    }
}

/// Do `ours` and `theirs` name the same executable file?
///
/// Identity belongs to the file, not to the spelling of its path.
/// `/proc/<pid>/exe` is already kernel-resolved while `env::current_exe()` is
/// derived from `/proc/self/exe`; the two agree byte-for-byte on a plain path but
/// not once a symlinked directory is involved (a `cargo install` prefix reached
/// through a symlink, a version-manager shim, a bind-mounted home). Comparing
/// them literally then reports the real daemon as `Foreign`, and the refusal
/// that follows leaves a live aeyes daemon unstoppable.
///
/// So compare device+inode when the kernel will tell us — that is exact and
/// immune to path spelling — then fall back to canonical paths for the cases
/// where `stat` is unavailable. The literal comparison is the *first* line, not
/// a last resort: if neither `stat` nor `canonicalize` can say anything and the
/// two strings already differed, there is no evidence left either way, and the
/// answer is "not ours".
///
/// Linux-only: this exists solely for the `/proc/<pid>/exe` check in
/// `classify_daemon_pid`, and gating it keeps the other targets free of the
/// dead code that `--all-features` clippy flags there.
#[cfg(target_os = "linux")]
fn same_executable(ours: &Path, theirs: &Path) -> bool {
    if ours == theirs {
        return true;
    }
    use std::os::unix::fs::MetadataExt;
    if let (Ok(a), Ok(b)) = (fs::metadata(ours), fs::metadata(theirs)) {
        if a.dev() == b.dev() && a.ino() == b.ino() {
            return true;
        }
    }
    match (fs::canonicalize(ours), fs::canonicalize(theirs)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The address to probe when the runtime registry names nothing usable.
///
/// Probing it lets `status` and `stop` see a daemon that never published the
/// registry, because the registry alone cannot distinguish "nothing is running"
/// from "something is running that we have no record of". It honours `AEYES_BIND`
/// — the same variable `run_daemon_from_env` and `ensure_daemon_running` read —
/// so a daemon a supervisor started on a non-default port is still reachable.
/// An unspecified bind address is normalised to loopback, since connecting to
/// `0.0.0.0` is not portable.
fn fallback_probe_addr() -> Option<SocketAddr> {
    let addr: SocketAddr = env::var("AEYES_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND.to_string())
        .parse()
        .ok()?;
    if addr.ip().is_unspecified() {
        return Some(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            addr.port(),
        ));
    }
    Some(addr)
}

pub async fn status_cmd() -> Result<()> {
    if let Ok(addr) = daemon_addr().await {
        if daemon_responding(addr).await {
            println!("Daemon running at http://{addr}");
            return Ok(());
        }
    }
    // No registry, or the registered address is dead. A daemon started outside
    // `aeyes start` never publishes the registry, so probe the address it would
    // have bound before reporting that nothing is running — "not running" for a
    // port that is listening is a false negative the user cannot act on. The
    // probe asks for identity, not merely reachability, because the sentence it
    // prints ("Daemon running at …") is a claim about aeyes specifically.
    if let Some(addr) = fallback_probe_addr() {
        if aeyes_responding(addr).await {
            println!(
                "Daemon running at http://{addr} (not in the runtime registry — it was not started by `aeyes start`)"
            );
            return Ok(());
        }
    }
    println!("Daemon not running.");
    Ok(())
}

/// Can we open a TCP connection to `addr`?
///
/// This is a *reachability* check and nothing more: it does not establish that
/// the peer speaks HTTP, let alone that it is aeyes. That is enough where we
/// are asking about something this process owns or merely reporting on it — the
/// address this process wrote into the registry after its own successful bind,
/// and the readiness loops around `ensure_daemon_running`.
///
/// It is not enough where we act on the peer. `stop_daemon` sends
/// `GET /shutdown`, so both of its target-selection branches use
/// [`aeyes_responding`] instead: a daemon can die between writing the registry
/// and being stopped, and the OS will hand its port to anything.
async fn daemon_responding(addr: SocketAddr) -> bool {
    TcpStream::connect(addr).await.is_ok()
}

/// The exact body `health_handler` returns for a live aeyes daemon.
const AEYES_HEALTH_BODY: &[u8] = b"ok";

/// Budget for each request of the identity probe.
///
/// A bystander on the probe port may accept the connection and then never say
/// anything, so the probe has to be bounded — `http_get_bytes` reads until EOF
/// and would otherwise hang `stop` forever on a socket that is not a daemon.
const IDENTITY_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Does the OpenAPI document served at `/` identify an aeyes daemon?
///
/// Split out from [`aeyes_responding`] so the decision is testable without a
/// socket: what matters is the parsed `info.title`, not the byte layout, so
/// whitespace and key order cannot make a real daemon look like an impostor.
fn openapi_identifies_aeyes(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|doc| {
            doc.get("info")
                .and_then(|info| info.get("title"))
                .and_then(|title| title.as_str())
                .map(str::to_owned)
        })
        .as_deref()
        == Some("aeyes")
}

/// Whether the peer listening on `addr` is a running aeyes daemon.
///
/// [`daemon_responding`] only proves that *something* accepts TCP there, which
/// is enough for the registered address — this process wrote that file itself
/// after its own successful bind — and nowhere near enough for the fallback
/// probe. `stop_daemon` acts destructively on whatever the fallback finds: it
/// sends `/shutdown` and reports success. A bare connect there would aim that at
/// any unrelated service holding the port, so the fallback must first establish
/// that the peer speaks HTTP *and* is aeyes.
///
/// Two requests, because either alone is weak: `/health` returning `ok` is
/// suggestive but not distinctive, and `/` returns this daemon's OpenAPI
/// document, whose `info.title` is `aeyes`. Anything that is not us fails one or
/// both — a non-HTTP listener fails to parse, a different HTTP server 404s, and
/// a lookalike health endpoint fails the title check.
async fn aeyes_responding(addr: SocketAddr) -> bool {
    match tokio::time::timeout(IDENTITY_PROBE_TIMEOUT, http_get_bytes(addr, "/health")).await {
        Ok(Ok(body)) if body.as_slice() == AEYES_HEALTH_BODY => {}
        _ => return false,
    }
    match tokio::time::timeout(IDENTITY_PROBE_TIMEOUT, http_get_bytes(addr, "/")).await {
        Ok(Ok(body)) => openapi_identifies_aeyes(&body),
        _ => false,
    }
}

/// Ensure daemon is running, auto-starting if needed. Returns daemon address.
/// If camera is specified, uses it when starting the daemon.
async fn ensure_daemon_running(
    camera: Option<&str>,
    options: &CameraOpenOptions,
) -> Result<SocketAddr> {
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
    start_daemon(Some(chosen.id.clone()), bind_str, options.clone())
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
    let options = CameraOpenOptions::default();
    let addr = ensure_daemon_running(None, &options).await?;
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

pub async fn frame_cmd(
    camera: Option<String>,
    output: &Path,
    options: CameraOpenOptions,
) -> Result<()> {
    let addr = ensure_daemon_running(camera.as_deref(), &options).await?;
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
    options: CameraOpenOptions,
) -> Result<()> {
    let addr = ensure_daemon_running(camera.as_deref(), &options).await?;
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
    let options = CameraOpenOptions {
        resolution: env::var("AEYES_RESOLUTION")
            .ok()
            .and_then(|v| parse_resolution(&v).ok()),
        format: env::var("AEYES_FORMAT")
            .ok()
            .and_then(|v| parse_fourcc(&v).ok()),
    };
    run_daemon(bind, selected_camera, options, Box::new(NativeBackend)).await
}

pub async fn run_daemon(
    bind: SocketAddr,
    selected_camera: String,
    options: CameraOpenOptions,
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
        let options = options.clone();
        tokio::spawn(async move {
            if let Err(err) = camera_loop(
                backend,
                camera_id.clone(),
                options,
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
    // Only now does this process own the socket, so only now may it publish.
    publish_runtime_registry(bind)?;
    info!("daemon listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn camera_loop(
    backend: Arc<dyn CameraBackend>,
    selected_camera: String,
    options: CameraOpenOptions,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    last_error: Arc<RwLock<Option<DaemonErrorState>>>,
    frame_tx: broadcast::Sender<Vec<u8>>,
) -> Result<()> {
    let mut camera = match backend.open(&selected_camera, &options) {
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
        <img src="/cams/{id}/stream" alt="Live stream from {camera_name}" onerror="reconnect()" onload="setStatus('connected', 'Connected')">
        <script>
            function reconnect() {{
                setStatus('error', 'Reconnecting…');
                setTimeout(() => {{
                    const img = document.querySelector('img');
                    if (img) img.src = '/cams/{id}/stream?t=' + Date.now();
                }}, 2000);
            }}
        </script>
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

    for pixel in bytes.as_chunks::<3>().0.iter().step_by(4) {
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

    for chunk in bytes.as_chunks::<4>().0 {
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
    for chunk in bytes.as_chunks::<4>().0 {
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

/// Convert a planar YUV 4:2:0 frame to JPEG.
///
/// Covers YU12 (I420: Y plane, then U, then V) and YV12 (Y plane, then V,
/// then U) via the `uv_swapped` plane-order flag. Odd sizes round the chroma
/// planes up (`div_ceil`), matching V4L2's behavior.
pub fn yuv420_to_jpeg(width: u32, height: u32, bytes: &[u8], uv_swapped: bool) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y_len = w * h;
    let uv_len = cw * ch;
    let expected = y_len + 2 * uv_len;
    if bytes.len() != expected {
        bail!(
            "invalid YUV420 buffer length: expected {expected} bytes for {width}x{height}, got {}",
            bytes.len()
        );
    }
    let y_plane = &bytes[..y_len];
    let (u_plane, v_plane) = if uv_swapped {
        (&bytes[y_len + uv_len..], &bytes[y_len..y_len + uv_len])
    } else {
        (&bytes[y_len..y_len + uv_len], &bytes[y_len + uv_len..])
    };
    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col] as f32;
            let chroma = (row / 2) * cw + col / 2;
            push_yuv_pixel(
                &mut rgb,
                y,
                u_plane[chroma] as f32 - 128.0,
                v_plane[chroma] as f32 - 128.0,
            );
        }
    }
    encode_rgb_to_jpeg(width, height, rgb)
}

/// Convert a bi-planar NV12 frame (full-resolution Y plane followed by an
/// interleaved U/V plane at half resolution) to JPEG.
pub fn nv12_to_jpeg(width: u32, height: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y_len = w * h;
    let expected = y_len + cw * ch * 2;
    if bytes.len() != expected {
        bail!(
            "invalid NV12 buffer length: expected {expected} bytes for {width}x{height}, got {}",
            bytes.len()
        );
    }
    let y_plane = &bytes[..y_len];
    let uv_plane = &bytes[y_len..];
    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col] as f32;
            let uv_off = (row / 2) * (cw * 2) + (col / 2) * 2;
            push_yuv_pixel(
                &mut rgb,
                y,
                uv_plane[uv_off] as f32 - 128.0,
                uv_plane[uv_off + 1] as f32 - 128.0,
            );
        }
    }
    encode_rgb_to_jpeg(width, height, rgb)
}

/// Convert a tightly-packed 24-bit RGB (RGB3) frame to JPEG.
pub fn rgb24_to_jpeg(width: u32, height: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if bytes.len() != expected {
        bail!(
            "invalid RGB buffer length: expected {expected} bytes for {width}x{height}, got {}",
            bytes.len()
        );
    }
    encode_rgb_to_jpeg(width, height, bytes.to_vec())
}

/// Convert a tightly-packed 24-bit BGR (BGR3) frame to JPEG.
pub fn bgr24_to_jpeg(width: u32, height: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if bytes.len() != expected {
        bail!(
            "invalid BGR buffer length: expected {expected} bytes for {width}x{height}, got {}",
            bytes.len()
        );
    }
    let mut rgb = Vec::with_capacity(expected);
    for chunk in bytes.as_chunks::<3>().0 {
        rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
    }
    encode_rgb_to_jpeg(width, height, rgb)
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
        fn open(&self, id: &str, _options: &CameraOpenOptions) -> Result<Box<dyn OpenCamera>> {
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
    fn test_yuv420_to_jpeg_encodes() {
        // 4x4 frame: solid Y=80, U=90, V=240 planes (both plane orders).
        let mut bytes = vec![80u8; 16];
        bytes.extend([90u8; 4]);
        bytes.extend([240u8; 4]);
        assert_eq!(bytes.len(), 24);
        for uv_swapped in [false, true] {
            let jpeg = yuv420_to_jpeg(4, 4, &bytes, uv_swapped).unwrap();
            assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
        }
    }

    #[test]
    fn test_yuv420_to_jpeg_invalid_length() {
        let err = yuv420_to_jpeg(4, 4, &[0u8; 20], false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid YUV420 buffer length"));
    }

    #[test]
    fn test_yuv420_to_jpeg_colorspace_sanity() {
        // Solid 2x2 YU12 frame encoding a blue-ish pixel (Y=82, U=240, V=90).
        let mut bytes = vec![82u8; 4];
        bytes.push(240); // U plane (1x1 chroma)
        bytes.push(90); // V plane
        let jpeg = yuv420_to_jpeg(2, 2, &bytes, false).unwrap();
        let img = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (2, 2));
        let px = img.get_pixel(0, 0).0;
        assert!(px[2] > 180 && px[0] < 80, "expected blue-ish, got {px:?}");
    }

    #[test]
    fn test_nv12_to_jpeg_encodes() {
        let mut bytes = vec![80u8; 16]; // Y plane 4x4
        bytes.extend([90u8; 8]); // interleaved U/V 2x2
        let jpeg = nv12_to_jpeg(4, 4, &bytes).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn test_nv12_to_jpeg_invalid_length() {
        let err = nv12_to_jpeg(4, 4, &[0u8; 23]).unwrap_err().to_string();
        assert!(err.contains("invalid NV12 buffer length"));
    }

    #[test]
    fn test_nv12_to_jpeg_colorspace_sanity() {
        let mut bytes = vec![82u8; 4]; // Y plane 2x2
        bytes.extend_from_slice(&[240, 90]); // U, V
        let jpeg = nv12_to_jpeg(2, 2, &bytes).unwrap();
        let img = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (2, 2));
        let px = img.get_pixel(0, 0).0;
        assert!(px[2] > 180 && px[0] < 80, "expected blue-ish, got {px:?}");
    }

    #[test]
    fn test_rgb24_to_jpeg_encodes() {
        let jpeg = rgb24_to_jpeg(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn test_rgb24_to_jpeg_invalid_length() {
        let err = rgb24_to_jpeg(2, 2, &[0u8; 11]).unwrap_err().to_string();
        assert!(err.contains("invalid RGB buffer length"));
    }

    #[test]
    fn test_rgb24_to_jpeg_roundtrip_colors() {
        // 1x3 frame: red, green, blue.
        let jpeg = rgb24_to_jpeg(1, 3, &[255, 0, 0, 0, 255, 0, 0, 0, 255]).unwrap();
        let img = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (1, 3));
        let px = |x: u32, y: u32| img.get_pixel(x, y).0;
        assert!(
            px(0, 0)[0] > 200 && px(0, 0)[1] < 60 && px(0, 0)[2] < 60,
            "red: {:?}",
            px(0, 0)
        );
        assert!(
            px(0, 1)[1] > 200 && px(0, 1)[0] < 60 && px(0, 1)[2] < 60,
            "green: {:?}",
            px(0, 1)
        );
        assert!(
            px(0, 2)[2] > 200 && px(0, 2)[0] < 60 && px(0, 2)[1] < 60,
            "blue: {:?}",
            px(0, 2)
        );
    }

    #[test]
    fn test_bgr24_to_jpeg_encodes() {
        // BGR bytes for red, green, blue, white.
        let jpeg = bgr24_to_jpeg(2, 2, &[0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255]).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn test_bgr24_to_jpeg_invalid_length() {
        let err = bgr24_to_jpeg(2, 2, &[0u8; 10]).unwrap_err().to_string();
        assert!(err.contains("invalid BGR buffer length"));
    }

    #[test]
    fn test_bgr24_to_jpeg_roundtrip_colors() {
        // BGR bytes for red, green, blue.
        let jpeg = bgr24_to_jpeg(1, 3, &[0, 0, 255, 0, 255, 0, 255, 0, 0]).unwrap();
        let img = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (1, 3));
        let px = |x: u32, y: u32| img.get_pixel(x, y).0;
        assert!(
            px(0, 0)[0] > 200 && px(0, 0)[1] < 60 && px(0, 0)[2] < 60,
            "red: {:?}",
            px(0, 0)
        );
        assert!(
            px(0, 1)[1] > 200 && px(0, 1)[0] < 60 && px(0, 1)[2] < 60,
            "green: {:?}",
            px(0, 1)
        );
        assert!(
            px(0, 2)[2] > 200 && px(0, 2)[0] < 60 && px(0, 2)[1] < 60,
            "blue: {:?}",
            px(0, 2)
        );
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

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
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
            CameraOpenOptions::default(),
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
            CameraOpenOptions::default(),
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
            CameraOpenOptions::default(),
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

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            Box::new(backend),
        ));

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

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            Box::new(backend),
        ));

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

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            Box::new(backend),
        ));

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

    // ---------------------------------------------------------------------
    // V4L2 capture-mode planning (issue #26: native-format fallback)
    // ---------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    #[allow(clippy::type_complexity)]
    fn supported_formats(entries: &[([u8; 4], &[(u32, u32)])]) -> Vec<SupportedFormat> {
        entries
            .iter()
            .map(|(format, resolutions)| SupportedFormat {
                format: *format,
                resolutions: resolutions.to_vec(),
            })
            .collect()
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_prefers_native_mode_first() {
        // v4l2loopback exclusive_caps=1: only the producer's mode is advertised.
        let supported = supported_formats(&[(*b"YUYV", &[(320, 320)])]);
        let native = Some(NativeFormat {
            width: 320,
            height: 320,
            format: *b"YUYV",
        });
        let presets = plan_capture_presets(&supported, native, &CameraOpenOptions::default());
        assert_eq!(presets[0].format, *b"YUYV");
        assert_eq!((presets[0].width, presets[0].height), (320, 320));
        // Native + enumerated are the same mode, so it appears exactly once.
        assert_eq!(
            presets
                .iter()
                .filter(|p| p.format == *b"YUYV" && (p.width, p.height) == (320, 320))
                .count(),
            1
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_uses_enumerated_resolutions_desc() {
        let supported = supported_formats(&[(*b"MJPG", &[(640, 480), (1920, 1080), (1280, 720)])]);
        let presets = plan_capture_presets(&supported, None, &CameraOpenOptions::default());
        // Enumerated sizes come first, largest first.
        let sizes: Vec<(u32, u32)> = presets.iter().map(|p| (p.width, p.height)).collect();
        assert_eq!(&sizes[..3], &[(1920, 1080), (1280, 720), (640, 480)]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_uses_nv12_native_first() {
        // Producer writes NV12 (v4l2loopback converts it to YU12/YV12/RGB3/BGR3,
        // but a real device may advertise NV12 natively): now encodable, so the
        // native mode is tried first and the fixed MJPG/YUYV list is only a
        // later fallback.
        let supported = supported_formats(&[(*b"NV12", &[(320, 320)])]);
        let native = Some(NativeFormat {
            width: 320,
            height: 320,
            format: *b"NV12",
        });
        let presets = plan_capture_presets(&supported, native, &CameraOpenOptions::default());
        assert_eq!(presets[0].format, *b"NV12");
        assert_eq!((presets[0].width, presets[0].height), (320, 320));
        assert!(presets.iter().any(|p| p.format == *b"MJPG"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_explicit_resolution_and_format() {
        let supported = supported_formats(&[(*b"YUYV", &[(320, 320)])]);
        let options = CameraOpenOptions {
            resolution: Some((320, 320)),
            format: Some(*b"YUYV"),
        };
        let presets = plan_capture_presets(&supported, None, &options);
        assert_eq!(presets.len(), 1);
        assert_eq!(presets[0].format, *b"YUYV");
        assert_eq!((presets[0].width, presets[0].height), (320, 320));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_explicit_resolution_tries_encodable_formats() {
        let options = CameraOpenOptions {
            resolution: Some((320, 320)),
            format: None,
        };
        let presets = plan_capture_presets(&[], None, &options);
        assert_eq!(presets.len(), 7);
        assert_eq!(presets[0].format, *b"MJPG");
        assert_eq!(presets[1].format, *b"YUYV");
        assert_eq!(presets[2].format, *b"RGB3");
        assert_eq!(presets[3].format, *b"BGR3");
        assert_eq!(presets[4].format, *b"YU12");
        assert_eq!(presets[5].format, *b"YV12");
        assert_eq!(presets[6].format, *b"NV12");
        for p in &presets {
            assert_eq!((p.width, p.height), (320, 320));
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_explicit_format_uses_native_res_first() {
        let supported = supported_formats(&[(*b"YUYV", &[(640, 480), (320, 320)])]);
        let native = Some(NativeFormat {
            width: 320,
            height: 320,
            format: *b"YUYV",
        });
        let options = CameraOpenOptions {
            resolution: None,
            format: Some(*b"YUYV"),
        };
        let presets = plan_capture_presets(&supported, native, &options);
        assert_eq!((presets[0].width, presets[0].height), (320, 320));
        assert_eq!(presets[1].width * presets[1].height, 640 * 480);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_deduplicates_against_fixed_list() {
        let supported = supported_formats(&[(*b"MJPG", &[(1920, 1080)])]);
        let presets = plan_capture_presets(&supported, None, &CameraOpenOptions::default());
        let mut seen = std::collections::HashSet::new();
        for p in &presets {
            assert!(
                seen.insert((p.format, p.width, p.height)),
                "duplicate preset {}x{} {:?}",
                p.width,
                p.height,
                p.format
            );
        }
        assert_eq!(presets.len(), 8); // 1 enumerated + 7 unique fixed (3 dupes removed)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn candidate_intervals_fall_back_ladder() {
        let intervals = candidate_intervals(30);
        assert_eq!(intervals, vec![(1, 30), (1, 15), (1, 10), (1, 5)]);
        // Preferred fps is tried first and not duplicated.
        let intervals = candidate_intervals(15);
        assert_eq!(intervals, vec![(1, 15), (1, 10), (1, 5)]);
    }

    #[test]
    fn parse_resolution_accepts_common_formats() {
        assert_eq!(parse_resolution("320x320").unwrap(), (320, 320));
        assert_eq!(parse_resolution("1280X720").unwrap(), (1280, 720));
        assert_eq!(parse_resolution(" 640 x 480 ").unwrap(), (640, 480));
        assert!(parse_resolution("abc").is_err());
        assert!(parse_resolution("0x10").is_err());
        assert!(parse_resolution("320x").is_err());
    }

    #[test]
    fn parse_fourcc_restricts_to_encodable() {
        assert_eq!(parse_fourcc("MJPG").unwrap(), *b"MJPG");
        assert_eq!(parse_fourcc("yuyv").unwrap(), *b"YUYV");
        assert_eq!(parse_fourcc("YU12").unwrap(), *b"YU12");
        assert_eq!(parse_fourcc("yv12").unwrap(), *b"YV12");
        assert_eq!(parse_fourcc("NV12").unwrap(), *b"NV12");
        assert_eq!(parse_fourcc("RGB3").unwrap(), *b"RGB3");
        assert_eq!(parse_fourcc("bgr3").unwrap(), *b"BGR3");
        assert!(parse_fourcc("H264").is_err());
        assert!(parse_fourcc("XVID").is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn encodable_formats_covers_new_fourccs() {
        for format in [
            *b"MJPG", *b"YUYV", *b"YU12", *b"YV12", *b"NV12", *b"RGB3", *b"BGR3",
        ] {
            assert!(
                is_encodable_format(&format),
                "{format:?} should be encodable"
            );
        }
        for format in [*b"H264", *b"MP4V", *b"XVID"] {
            assert!(
                !is_encodable_format(&format),
                "{format:?} should not be encodable"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn plan_enumerates_all_encodable_formats_in_preference_order() {
        // A device advertising several formats: MJPG first, then YUYV, then
        // the new converted formats, each largest-first.
        let supported = supported_formats(&[
            (*b"NV12", &[(320, 320)]),
            (*b"YU12", &[(320, 320)]),
            (*b"MJPG", &[(1920, 1080)]),
            (*b"RGB3", &[(640, 480)]),
        ]);
        let presets = plan_capture_presets(&supported, None, &CameraOpenOptions::default());
        let formats: Vec<[u8; 4]> = presets.iter().map(|p| p.format).collect();
        let first_four = &formats[..4];
        assert_eq!(first_four, &[*b"MJPG", *b"RGB3", *b"YU12", *b"NV12"]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn summarize_supported_reports_modes() {
        let supported = supported_formats(&[(*b"YUYV", &[(320, 320)])]);
        let summary = summarize_supported(&supported);
        assert!(summary.contains("YUYV"));
        assert!(summary.contains("320x320"));
    }

    // ---- identity probe (`aeyes_responding`) and executable identity ----

    /// The OpenAPI document this daemon serves at `/`.
    fn aeyes_openapi_doc() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "openapi": "3.0.3",
            "info": { "title": "aeyes", "version": "0.1.0" }
        }))
        .expect("serialize openapi")
    }

    /// Serve `handler(path) -> Option<body>` on an ephemeral loopback port.
    ///
    /// The identity probe exists to tell three peers apart — one that is not
    /// HTTP at all, one that is HTTP but not aeyes, and aeyes — which cannot be
    /// checked without a real socket.
    fn serve_http<F>(handler: F) -> SocketAddr
    where
        F: Fn(&str) -> Option<Vec<u8>> + Send + Sync + 'static,
    {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
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
                let response = match handler(&path) {
                    Some(body) => {
                        let mut r = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        r.extend_from_slice(&body);
                        r
                    }
                    None => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_vec()
                    }
                };
                let _ = std::io::Write::write_all(&mut stream, &response);
                let _ = std::io::Write::flush(&mut stream);
            }
        });
        addr
    }

    /// The identity probe must accept a peer that really is an aeyes daemon —
    /// otherwise `stop` would never reach the unregistered daemons it exists for.
    #[tokio::test]
    async fn identity_probe_accepts_a_peer_that_answers_as_aeyes() {
        let addr = serve_http(|path| match path {
            "/health" => Some(b"ok".to_vec()),
            "/" => Some(aeyes_openapi_doc()),
            _ => None,
        });

        assert!(
            aeyes_responding(addr).await,
            "a peer serving this daemon's own routes was not recognised as aeyes"
        );
    }

    /// A different HTTP service that happens to answer `/health` with `ok` must
    /// not be mistaken for a daemon — it would then be handed a `/shutdown`.
    #[tokio::test]
    async fn identity_probe_rejects_an_http_peer_that_is_not_aeyes() {
        let addr = serve_http(|path| match path {
            "/health" => Some(b"ok".to_vec()),
            // Right route, wrong document.
            "/" => Some(br#"{"openapi":"3.0.3","info":{"title":"somethingelse"}}"#.to_vec()),
            _ => None,
        });

        assert!(
            !aeyes_responding(addr).await,
            "a foreign HTTP server answering /health with 'ok' was accepted as aeyes"
        );
    }

    /// Most of what actually listens on a developer's port 43210 is not an aeyes
    /// daemon and has no `/health` at all.
    #[tokio::test]
    async fn identity_probe_rejects_an_http_peer_without_health() {
        let addr = serve_http(|_| None);

        assert!(
            !aeyes_responding(addr).await,
            "a peer with no /health route was accepted as aeyes"
        );
    }

    /// A socket that accepts the connection and then says nothing must not hang
    /// `stop` forever: the probe is bounded, so it gives up rather than blocking
    /// the command.
    #[tokio::test]
    async fn identity_probe_gives_up_on_a_peer_that_never_answers() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            // Hold the connections open without ever writing a response.
            for stream in listener.incoming() {
                std::thread::spawn(move || {
                    if let Ok(mut s) = stream {
                        let mut sink = Vec::new();
                        let _ = std::io::Read::read_to_end(&mut s, &mut sink);
                        std::thread::sleep(std::time::Duration::from_secs(30));
                    }
                });
            }
        });

        assert!(
            !aeyes_responding(addr).await,
            "the identity probe waited on a silent peer instead of giving up"
        );
    }

    /// The OpenAPI check must key on the parsed title, not on a byte pattern, so
    /// key order and pretty-printing cannot reject the real daemon.
    #[test]
    fn openapi_identity_keys_on_the_title_not_the_bytes() {
        assert!(openapi_identifies_aeyes(&aeyes_openapi_doc()));
        assert!(openapi_identifies_aeyes(
            br#"{ "info" : { "title" : "aeyes" } }"#.as_slice()
        ));

        assert!(!openapi_identifies_aeyes(b"not json at all"));
        assert!(!openapi_identifies_aeyes(b""));
        assert!(!openapi_identifies_aeyes(
            br#"{"info":{"title":"aeyes-ctl"}}"#
        ));
        assert!(!openapi_identifies_aeyes(
            br#"{"info":{"version":"1.0.1"}}"#
        ));
        // The title has to be the title, not merely present somewhere in the
        // document.
        assert!(!openapi_identifies_aeyes(
            br#"{"info":{"title":"other"},"paths":{"/":{"summary":"aeyes"}}}"#
        ));
    }

    /// `/proc/<pid>/exe` comes back kernel-resolved while `current_exe()` comes
    /// from `/proc/self/exe`, and the two spellings diverge as soon as a
    /// symlinked directory sits between them. Comparing them literally reports
    /// the real daemon as `Foreign` and leaves it unstoppable — the exact
    /// failure the identity check exists to prevent.
    #[test]
    #[cfg(target_os = "linux")]
    fn same_executable_survives_a_symlinked_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("releases").join("v1.0.0");
        fs::create_dir_all(&real).expect("create real dir");
        let binary = real.join("aeyes");
        fs::write(&binary, b"#!/bin/true\n").expect("write binary");

        let link = dir.path().join("current");
        std::os::unix::fs::symlink(dir.path().join("releases"), &link).expect("symlink dir");
        let via_link = link.join("v1.0.0").join("aeyes");

        // The premise: the two paths genuinely differ byte-for-byte...
        assert_ne!(
            binary, via_link,
            "test premise broken: the symlinked path resolved to the same string"
        );
        // ...while naming one and the same executable file.
        assert!(
            same_executable(&binary, &via_link),
            "same_executable called one file two ways a pair of different files"
        );
        assert!(
            same_executable(&via_link, &binary),
            "same_executable is not symmetric"
        );
    }

    /// The other half of the contract: canonicalising must not make every path
    /// match. A different file is still `Foreign`.
    #[test]
    #[cfg(target_os = "linux")]
    fn same_executable_still_separates_different_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let one = dir.path().join("aeyes-one");
        let other = dir.path().join("aeyes-two");
        fs::write(&one, b"one").expect("write one");
        fs::write(&other, b"two").expect("write other");

        assert!(
            !same_executable(&one, &other),
            "same_executable called two distinct files identical"
        );
        assert!(
            !same_executable(&one, &dir.path().join("absent")),
            "same_executable matched a file that does not exist"
        );
    }

    /// Our own pid must classify as `Ours` — the positive half of the identity
    /// check, which a regression in the canonicalisation would break.
    #[test]
    #[cfg(target_os = "linux")]
    fn classify_daemon_pid_recognises_this_process() {
        assert!(
            matches!(classify_daemon_pid(std::process::id()), PidVerdict::Ours),
            "the running aeyes executable was not recognised as our own"
        );
    }

    /// A pid that is definitely gone must read as `Absent`, which is the only
    /// verdict that lets `stop` clear the registry.
    #[test]
    #[cfg(target_os = "linux")]
    fn classify_daemon_pid_reports_a_dead_pid_as_absent() {
        // `u32::MAX` as a pid is well past `pid_max` on any sane kernel.
        assert!(
            matches!(classify_daemon_pid(u32::MAX), PidVerdict::Absent),
            "a pid with no /proc entry was not classified Absent"
        );
    }

    /// An unreadable `/proc/<pid>/exe` must not be confused with a pid that is
    /// not there. The first is a live process we refuse to signal and keep a
    /// handle on; the second is nothing, and the registry may be cleared.
    #[test]
    #[cfg(target_os = "linux")]
    fn unreadable_exe_is_not_the_same_as_no_such_process() {
        assert!(
            matches!(verdict_for_unreadable_exe(true), PidVerdict::Unreadable),
            "a live process with an unreadable exe was classified as if nothing were running"
        );
        assert!(
            matches!(verdict_for_unreadable_exe(false), PidVerdict::Absent),
            "a pid with no /proc entry was not classified Absent"
        );
    }

    /// The whole point of the non-Linux platforms' answer: a pid nobody can
    /// identify must never reach a signal.
    ///
    /// Windows and non-Linux unix cannot run this test — the code that produces
    /// `Unverified` is `#[cfg]`-ed out there, and CI cannot execute Windows unit
    /// tests anyway. What this test *does* exercise is the routing that decides
    /// what `stop_daemon` does with the verdict, on every target including the
    /// two that cannot produce it on Linux. That is the half of the fix that was
    /// actually broken: the platform branch produced `Alive` and the match
    /// treated `Alive` as "signal it", and this assertion is on the match. The
    /// platform branch itself is pinned by
    /// `platform_without_an_identity_check_refuses_to_signal` below, which
    /// calls the exact helper the Windows and non-Linux-unix arms call.
    #[test]
    fn an_unverified_pid_is_never_signalled() {
        assert!(
            matches!(
                action_for_verdict(PidVerdict::Unverified),
                PidAction::Refuse(_)
            ),
            "an unverified pid was routed to a signal; that is how a recycled pid gets a bystander killed"
        );
        assert!(
            !matches!(
                action_for_verdict(PidVerdict::Unverified),
                PidAction::SkipSignal(_)
            ),
            "an unverified pid was routed to SkipSignal, which clears the registry and drops the user's only handle on the daemon"
        );
    }

    /// `Ours` is the only verdict that may be signalled. This is the invariant
    /// the whole module exists to protect, stated once so a future verdict
    /// cannot be added without deciding its place.
    #[test]
    fn only_a_proven_ours_pid_is_signalled() {
        let signal = |v| matches!(action_for_verdict(v), PidAction::Signal);
        assert!(signal(PidVerdict::Ours), "Ours must be signalable");
        for (verdict, what) in [
            (PidVerdict::Unverified, "Unverified"),
            (PidVerdict::Foreign, "Foreign"),
            (PidVerdict::Unreadable, "Unreadable"),
        ] {
            assert!(
                !signal(verdict),
                "{what} must not be signalled: nothing proves that pid is our own daemon"
            );
        }
        assert!(
            matches!(
                action_for_verdict(PidVerdict::Absent),
                PidAction::SkipSignal(_)
            ),
            "Absent must not be signalled; there is nothing there to signal"
        );
    }

    /// The refusal message must say what actually happened.
    ///
    /// "Different executable" and "could not tell" are different facts with
    /// different remedies: the first means the user has two installs of aeyes
    /// and should stop the right one, the second means this platform cannot
    /// answer the question at all. Printing the first for the second is a lie
    /// about the very check the message exists to explain.
    #[test]
    fn refusal_reasons_do_not_claim_more_than_they_know() {
        assert!(
            FOREIGN_REASON.contains("provably a different program"),
            "the Foreign reason no longer says the executable was actually read and found different"
        );
        assert!(
            UNREADABLE_REASON.contains("could not read"),
            "the Unreadable reason no longer says the read failed"
        );
        assert!(
            !UNREADABLE_REASON.contains("different program"),
            "the Unreadable reason accuses the process of being a different program; nobody knows that"
        );
        let unverified = match action_for_verdict(PidVerdict::Unverified) {
            PidAction::Refuse(reason) => reason,
            other => panic!("Unverified did not refuse: {other:?}"),
        };
        assert_eq!(
            unverified, UNVERIFIED_REASON,
            "the Unverified route reports a different reason than the one the constant defines"
        );
        assert!(
            unverified.contains("no way to read"),
            "the Unverified reason does not say that no identity check exists on this platform: {unverified}"
        );
        assert!(
            unverified.contains("/shutdown"),
            "the Unverified reason does not point the user at the path that does work here: {unverified}"
        );
        assert!(
            !unverified.contains("different program"),
            "the Unverified reason claims the executable was read and found different; nothing was read: {unverified}"
        );
    }

    /// The decision a platform with no identity check reaches, for all three
    /// things such a platform could possibly learn.
    ///
    /// Windows calls this with `Unprobed` unconditionally and non-Linux unix
    /// calls it with whatever `kill -0` could establish, so pinning the mapping
    /// here pins both branches — see
    /// `platform_without_an_identity_check_refuses_to_signal`.
    #[test]
    fn platform_without_an_identity_check_refuses_to_signal() {
        assert!(
            matches!(
                verdict_without_identity_check(PidLiveness::Alive),
                PidVerdict::Unverified
            ),
            "a pid a liveness probe could confirm exists was not marked unverified; a recycled pid looks exactly this alive"
        );
        assert!(
            matches!(
                verdict_without_identity_check(PidLiveness::Unprobed),
                PidVerdict::Unverified
            ),
            "a pid nobody could ask about was not marked unverified; an unanswered question is not a negative answer"
        );
        assert!(
            matches!(
                verdict_without_identity_check(PidLiveness::Gone),
                PidVerdict::Absent
            ),
            "a probe that actually ran and denied the process did not produce Absent, so a stale pid file could never be cleaned up on macOS"
        );
    }
}
