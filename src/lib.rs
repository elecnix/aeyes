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
pub mod motion;
use serde::Serialize;
use serde_json::json;
use std::{
    env, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
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
/// How many verdicts a motion subscriber may fall behind before the broadcaster
/// starts dropping them. Same shape as the frame channel.
const MOTION_CHANNEL_CAPACITY: usize = 60;
/// Verdicts per second. The expensive part of a verdict is the full-resolution
/// JPEG decode, not the detector, so this throttles the decode rate.
pub const DEFAULT_MOTION_HZ: u32 = 10;
/// Maximum verdicts per second the daemon will run.
pub const MAX_MOTION_HZ: u32 = 60;
/// Width of the analysis frame. The decoded frame is downscaled to this width
/// (never upscaled), preserving aspect ratio, so detector cost is independent
/// of camera resolution.
pub const DEFAULT_MOTION_WIDTH: u32 = 320;
/// Narrowest analysis frame the daemon accepts (a narrower one is clamped up).
pub const MIN_MOTION_WIDTH: u32 = 16;
/// Widest analysis frame the daemon accepts (a wider one is clamped down).
pub const MAX_MOTION_WIDTH: u32 = 4096;
/// Edge threshold handed to the detector. Same default as
/// `motion::LightingInvariantConfig::default()`.
pub const DEFAULT_MOTION_THRESHOLD: u8 = 25;
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
    /// Most recent motion verdict, read by `GET /cams/{id}/motion`.
    latest_motion: Arc<RwLock<Option<MotionVerdict>>>,
    /// Verdict broadcast, subscribed to by `GET /cams/{id}/events`.
    motion_tx: broadcast::Sender<MotionVerdict>,
    /// Number of clients currently subscribed to this camera's verdicts.
    /// Detection runs only while this is non-zero, so an idle daemon does not
    /// decode full-resolution frames for nobody.
    motion_watchers: Arc<AtomicUsize>,
}

/// How a motion verdict was produced.
///
/// `detected: false` alone cannot distinguish "the scene was still" from "the
/// frame could not be analysed", which is exactly how the old JPEG/RGB length
/// mismatch stayed invisible. A verdict always states which of the two
/// happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionStatus {
    /// The frame was decoded, downscaled and scanned.
    Ok,
    /// The published JPEG could not be decoded.
    DecodeError,
    /// The frame decoded but does not match the geometry the detector scanned.
    GeometryMismatch,
}

/// One frame's motion measurement, as published by the daemon.
///
/// This is the daemon's raw measurement: `changed_pixels` and `bbox` are in
/// **analysis** pixels (see [`MotionConfig::width`]), while `frame_width`/
/// `frame_height` are the camera's real decoded resolution. Clients apply their
/// own policy (for example `--min-area`) to the published number instead of
/// re-running a detector, so two clients can filter one stream differently
/// without either changing what the other measures.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct MotionVerdict {
    /// Camera the verdict belongs to.
    pub camera: String,
    /// Monotonic per-camera verdict counter, so a client can tell "no new
    /// verdict" from "no motion".
    pub sequence: u64,
    /// True when the frame was analysed and at least one pixel changed.
    pub detected: bool,
    /// Number of pixels flagged as motion, in analysis pixels.
    pub changed_pixels: usize,
    /// Bounds of the flagged pixels, in analysis pixels; `None` when nothing
    /// moved (or the frame could not be analysed).
    pub bbox: Option<motion::MotionBox>,
    /// Real decoded frame dimensions, in camera pixels.
    pub frame_width: u32,
    pub frame_height: u32,
    /// Dimensions of the analysis frame the detector ran on. Stated rather
    /// than assumed: the detector geometry is the camera's real resolution
    /// downscaled, never a hardcoded 640x480.
    pub detector_width: usize,
    pub detector_height: usize,
    /// Wall-clock time the verdict was published, in milliseconds since the
    /// Unix epoch. The detector itself has no clock; the daemon's loop owns it.
    pub timestamp_ms: u64,
    /// How the verdict was produced.
    pub status: MotionStatus,
}

/// Daemon-side motion detection settings.
///
/// These change what the measurement *is*, so they are daemon configuration
/// rather than per-request parameters: a client asking for a different
/// threshold would silently get every other client's detector instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionConfig {
    /// Verdicts per second, clamped to `1..=MAX_MOTION_HZ`.
    pub hz: u32,
    /// Analysis frame width, clamped to `MIN_MOTION_WIDTH..=MAX_MOTION_WIDTH`.
    /// The decoded frame is downscaled to this width preserving aspect ratio
    /// and is never upscaled.
    pub width: u32,
    /// Detector edge threshold.
    pub threshold: u8,
    /// Run the Sobel stage.
    pub use_sobel: bool,
    /// Run the LBP stage.
    pub use_lbp: bool,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            hz: DEFAULT_MOTION_HZ,
            width: DEFAULT_MOTION_WIDTH,
            threshold: DEFAULT_MOTION_THRESHOLD,
            use_sobel: true,
            use_lbp: true,
        }
    }
}

impl MotionConfig {
    /// The config with every value forced into its documented range, so an
    /// out-of-range flag or environment variable cannot produce a detector
    /// that never fires or a loop that spins.
    pub fn clamped(self) -> Self {
        Self {
            hz: self.hz.clamp(1, MAX_MOTION_HZ),
            width: self.width.clamp(MIN_MOTION_WIDTH, MAX_MOTION_WIDTH),
            ..self
        }
    }

    /// Detector tuning derived from the daemon settings.
    fn detector_config(self) -> motion::LightingInvariantConfig {
        motion::LightingInvariantConfig {
            edge_threshold: self.threshold,
            use_sobel: self.use_sobel,
            use_lbp: self.use_lbp,
            ..motion::LightingInvariantConfig::default()
        }
    }
}

/// Keeps a camera's watcher count raised for as long as it is alive.
///
/// Held by a streaming response for the life of its body, so the count is
/// decremented on drop even when the client disconnects mid-stream.
struct MotionWatcher {
    watchers: Arc<AtomicUsize>,
}

impl MotionWatcher {
    fn register(watchers: Arc<AtomicUsize>) -> Self {
        watchers.fetch_add(1, Ordering::SeqCst);
        Self { watchers }
    }
}

impl Drop for MotionWatcher {
    fn drop(&mut self) {
        self.watchers.fetch_sub(1, Ordering::SeqCst);
    }
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
    run_daemon(
        bind,
        selected_camera,
        options,
        motion_config_from_env(),
        Box::new(NativeBackend),
    )
    .await
}

/// Daemon motion settings: `AEYES_MOTION_*` environment variables, falling
/// back to the documented defaults.
///
/// Invalid values fall back to the default rather than failing the daemon,
/// matching `AEYES_IDLE_TIMEOUT_SECS`.
fn motion_config_from_env() -> MotionConfig {
    let env_u32 = |name: &str, default: u32| {
        env::var(name)
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(default)
    };
    let env_flag = |name: &str, default: bool| {
        env::var(name)
            .ok()
            .map(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(default)
    };
    MotionConfig {
        hz: env_u32("AEYES_MOTION_HZ", DEFAULT_MOTION_HZ),
        width: env_u32("AEYES_MOTION_WIDTH", DEFAULT_MOTION_WIDTH),
        threshold: env_u32(
            "AEYES_MOTION_THRESHOLD",
            u32::from(DEFAULT_MOTION_THRESHOLD),
        ) as u8,
        use_sobel: env_flag("AEYES_MOTION_SOBEL", true),
        use_lbp: env_flag("AEYES_MOTION_LBP", true),
    }
    .clamped()
}

pub async fn run_daemon(
    bind: SocketAddr,
    selected_camera: String,
    options: CameraOpenOptions,
    motion: MotionConfig,
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
        let latest_motion: Arc<RwLock<Option<MotionVerdict>>> = Arc::new(RwLock::new(None));
        let (motion_tx, _) = broadcast::channel::<MotionVerdict>(MOTION_CHANNEL_CAPACITY);
        let motion_watchers = Arc::new(AtomicUsize::new(0));
        let stream_state = CameraStreamState {
            latest_jpeg: latest_jpeg.clone(),
            last_error: last_error.clone(),
            frame_tx: frame_tx.clone(),
            latest_motion: latest_motion.clone(),
            motion_tx: motion_tx.clone(),
            motion_watchers: motion_watchers.clone(),
        };
        streams.insert(camera.id.clone(), stream_state.clone());

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

        let camera_id = camera.id.clone();
        let latest_jpeg = stream_state.latest_jpeg.clone();
        tokio::spawn(async move {
            motion_loop(
                camera_id,
                latest_jpeg,
                latest_motion,
                motion_tx,
                motion_watchers,
                motion,
            )
            .await;
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
        .route("/cams/{id}/motion", get(motion_http))
        .route("/cams/{id}/events", get(events_http))
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

/// Geometry of the frame the detector scans: the decoded frame downscaled to
/// `target_width`, preserving aspect ratio, and never upscaled.
///
/// Returns `(0, 0)` for a degenerate frame, which no detector can scan.
fn analysis_dimensions(frame_width: u32, frame_height: u32, target_width: u32) -> (usize, usize) {
    if frame_width == 0 || frame_height == 0 {
        return (0, 0);
    }
    let target = target_width.clamp(MIN_MOTION_WIDTH, MAX_MOTION_WIDTH);
    if frame_width <= target {
        return (frame_width as usize, frame_height as usize);
    }
    let scale = f64::from(target) / f64::from(frame_width);
    let height = ((f64::from(frame_height) * scale).round() as usize).max(1);
    (target as usize, height)
}

/// The detector plus the analysis geometry it was built for.
struct MotionDetectorState {
    width: usize,
    height: usize,
    detector: motion::LightingInvariantDetector,
}

impl MotionDetectorState {
    fn new(width: usize, height: usize, config: MotionConfig) -> Self {
        Self {
            width,
            height,
            detector: motion::LightingInvariantDetector::new(
                width,
                height,
                config.detector_config(),
            ),
        }
    }
}

/// Everything one analysed frame produced.
struct MotionAnalysis {
    status: MotionStatus,
    changed_pixels: usize,
    bbox: Option<motion::MotionBox>,
    frame_width: u32,
    frame_height: u32,
    detector_width: usize,
    detector_height: usize,
    /// Why the frame could not be analysed, when it could not be. Logged once.
    problem: Option<String>,
}

impl MotionAnalysis {
    /// A frame that was not analysed. The measurement is empty, but `status`
    /// says so, which `detected: false` alone cannot.
    fn unanalysed(status: MotionStatus, problem: String) -> Self {
        Self {
            status,
            changed_pixels: 0,
            bbox: None,
            frame_width: 0,
            frame_height: 0,
            detector_width: 0,
            detector_height: 0,
            problem: Some(problem),
        }
    }
}

/// Decode, downscale and detect one JPEG frame.
///
/// The daemon has no raw RGB: `OpenCamera::capture_jpeg` is the only capture
/// shape, and both `latest_jpeg` and `frame_tx` carry JPEG. So this decodes,
/// exactly as `analyze_mjpeg_frame` already does for the exposure path, which
/// makes it work for MJPEG passthrough and for re-encoded YUYV/NV12/RGB3
/// alike.
///
/// The decode is full-resolution and CPU-bound, so callers run this on the
/// blocking pool and hold no lock across it.
fn analyze_motion_frame(
    state: &mut Option<MotionDetectorState>,
    jpeg: &[u8],
    config: MotionConfig,
) -> MotionAnalysis {
    let decoded = match load_from_memory(jpeg) {
        Ok(image) => image.to_rgb8(),
        Err(err) => {
            return MotionAnalysis::unanalysed(
                MotionStatus::DecodeError,
                format!("motion: failed to decode the published JPEG frame: {err}"),
            )
        }
    };
    let (frame_width, frame_height) = decoded.dimensions();
    let (detector_width, detector_height) =
        analysis_dimensions(frame_width, frame_height, config.width);

    // A frame too small to have an interior pixel is not "still", it is
    // unanalysable; say so instead of reporting no motion forever.
    if detector_width < motion::MIN_SCAN_DIMENSION || detector_height < motion::MIN_SCAN_DIMENSION {
        return MotionAnalysis::unanalysed(
            MotionStatus::GeometryMismatch,
            format!(
                "motion: decoded {frame_width}x{frame_height} frame yields a 
                 {detector_width}x{detector_height} analysis frame, which is smaller than the 
                 {}x{} the detector needs",
                motion::MIN_SCAN_DIMENSION,
                motion::MIN_SCAN_DIMENSION
            ),
        );
    }

    let rgb = if (detector_width, detector_height) == (frame_width as usize, frame_height as usize)
    {
        decoded.into_raw()
    } else {
        image::imageops::resize(
            &decoded,
            detector_width as u32,
            detector_height as u32,
            image::imageops::FilterType::Triangle,
        )
        .into_raw()
    };

    // The detector silently ignores a buffer whose length is not exactly
    // `width * height * 3`, so a mismatch here must be reported, not scanned.
    if rgb.len() != detector_width * detector_height * 3 {
        return MotionAnalysis::unanalysed(
            MotionStatus::GeometryMismatch,
            format!(
                "motion: analysis frame is {}x{} but its RGB buffer is {} bytes, expected {}",
                detector_width,
                detector_height,
                rgb.len(),
                detector_width * detector_height * 3
            ),
        );
    }

    // The detector geometry comes from the frame itself, so the hardcoded
    // geometry the client used to assume cannot come back. A change of
    // resolution rebuilds the detector (and loses its previous-frame state,
    // which is why a camera that alternates resolutions will not detect
    // motion across the change).
    if state.as_ref().map(|s| (s.width, s.height)) != Some((detector_width, detector_height)) {
        *state = Some(MotionDetectorState::new(
            detector_width,
            detector_height,
            config,
        ));
    }
    let Some(detector_state) = state.as_mut() else {
        return MotionAnalysis::unanalysed(
            MotionStatus::GeometryMismatch,
            "motion: detector state was not constructed".to_string(),
        );
    };

    let result = detector_state.detector.detect(&rgb);

    MotionAnalysis {
        status: MotionStatus::Ok,
        changed_pixels: result.changed_pixels,
        bbox: result.bbox,
        frame_width,
        frame_height,
        detector_width,
        detector_height,
        problem: None,
    }
}

/// Run one analyse step on the blocking pool, handing the detector state back.
///
/// Returns `None` only if the blocking task itself failed (a panic or a
/// shutting-down runtime); the detector is then rebuilt from scratch on the
/// next frame rather than reporting a stale measurement.
async fn run_motion_analysis(
    state: Option<MotionDetectorState>,
    jpeg: Vec<u8>,
    config: MotionConfig,
) -> Option<(Option<MotionDetectorState>, MotionAnalysis)> {
    tokio::task::spawn_blocking(move || {
        let mut state = state;
        let analysis = analyze_motion_frame(&mut state, &jpeg, config);
        (state, analysis)
    })
    .await
    .ok()
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Per-camera motion detection.
///
/// A task of its own, not inline in `camera_loop`: the decode and the scan are
/// CPU-bound, and running them here means capture is never stalled by them (and
/// no lock is ever held across them - the JPEG is cloned under a short read
/// guard, and the analysis runs on the blocking pool).
///
/// Detection is gated on a subscriber count so an idle daemon does no decoding
/// work, and throttled to `config.hz` because the full-resolution decode, not
/// the analysis, is the expensive part.
async fn motion_loop(
    camera_id: String,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    latest_motion: Arc<RwLock<Option<MotionVerdict>>>,
    motion_tx: broadcast::Sender<MotionVerdict>,
    watchers: Arc<AtomicUsize>,
    config: MotionConfig,
) {
    let config = config.clamped();
    let tick = Duration::from_millis(u64::from(1000 / config.hz).max(1));
    let mut state: Option<MotionDetectorState> = None;
    let mut sequence: u64 = 0;
    let mut reported_problem: Option<String> = None;

    loop {
        sleep(tick).await;

        if watchers.load(Ordering::SeqCst) == 0 {
            continue;
        }

        // Short read lock: clone the current JPEG and release before touching
        // it. The clone is one memcpy; the decode below is not.
        let Some(jpeg) = latest_jpeg.read().await.clone() else {
            continue;
        };

        let Some((state_back, analysis)) = run_motion_analysis(state.take(), jpeg, config).await
        else {
            warn!(camera = %camera_id, "motion analysis task failed; rebuilding the detector");
            continue;
        };
        state = state_back;

        match &analysis.problem {
            // Once per distinct problem: a frame that cannot be analysed must
            // not look like a quiet scene forever, and must not flood the log
            // either.
            Some(problem) if reported_problem.as_deref() != Some(problem.as_str()) => {
                warn!(camera = %camera_id, "{problem}");
                reported_problem = Some(problem.clone());
            }
            Some(_) => {}
            None => reported_problem = None,
        }

        sequence += 1;
        let verdict = MotionVerdict {
            camera: camera_id.clone(),
            sequence,
            detected: analysis.status == MotionStatus::Ok && analysis.changed_pixels > 0,
            changed_pixels: analysis.changed_pixels,
            bbox: analysis.bbox,
            frame_width: analysis.frame_width,
            frame_height: analysis.frame_height,
            detector_width: analysis.detector_width,
            detector_height: analysis.detector_height,
            timestamp_ms: unix_millis(),
            status: analysis.status,
        };

        {
            // Both are set under one lock acquisition, so a subscriber that
            // wakes on the broadcast can never find a `latest_motion` that
            // disagrees with the verdict it was handed.
            let mut guard = latest_motion.write().await;
            *guard = Some(verdict.clone());
            let _ = motion_tx.send(verdict);
        }
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

/// Query parameters of `GET /cams/{id}/events`.
#[derive(Debug, serde::Deserialize)]
struct EventsQuery {
    /// Only emit verdicts whose `changed_pixels` is at least this value.
    ///
    /// This is an *emission* filter, not a measurement: the daemon computes
    /// and publishes the same number either way, so two clients filtering
    /// differently on one camera do not disagree about anything. It exists so a
    /// watcher waiting on a threshold does not have to receive every quiet
    /// frame; omitting it emits every verdict.
    #[serde(default)]
    min_area: usize,
}

/// How often an open `/events` stream refreshes `last_activity`.
///
/// The idle watchdog reads that field and calls `process::exit(0)` when it has
/// gone stale, but today it is only written at handler entry, so a long-lived
/// stream can be killed mid-flight. The refresh period is well below any
/// useful idle timeout, so a connected client keeps the daemon alive.
const EVENTS_ACTIVITY_REFRESH: Duration = Duration::from_secs(1);

/// `GET /cams/{id}/motion` - the latest verdict for a camera.
///
/// The verdict is one the detector produced *after* this request arrived, not
/// whatever happens to be cached: detection only runs while someone is
/// watching, so serving a cached verdict would report the state of the scene
/// at some earlier subscription. Asking also registers the caller as a watcher
/// for the life of the request, which is what makes the detector run at all.
async fn motion_http(AxumPath(id): AxumPath<String>, State(state): State<AppState>) -> Response {
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

    // Asking for the latest verdict is a subscription for the duration of the
    // request: detection only runs while someone is watching, so without this
    // a verdict would exist only when some *other* client happened to be
    // streaming.
    let _watcher = MotionWatcher::register(stream.motion_watchers.clone());

    // A verdict published before this request (or before this request's
    // watcher registration) describes an earlier moment and must not be served
    // as "what is happening now". `sequence` is the daemon's monotonic counter,
    // so a greater value is by construction a verdict produced later.
    let baseline = stream
        .latest_motion
        .read()
        .await
        .as_ref()
        .map(|v| v.sequence);

    for _ in 0..FRAME_WAIT_RETRIES {
        if let Some(verdict) = stream.latest_motion.read().await.clone() {
            let is_fresh = match baseline {
                Some(previous) => verdict.sequence > previous,
                None => true,
            };
            if is_fresh {
                return Json(verdict).into_response();
            }
        }
        sleep(Duration::from_millis(FRAME_WAIT_MS)).await;
    }

    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        DaemonErrorState::new("camera has not produced a motion verdict yet")
            .with_detail(format!("camera id: {requested}"))
            .with_detail("no verdict was available before the request timeout expired")
            .with_detail(format!(
                "waited approximately {} ms",
                FRAME_WAIT_RETRIES as u64 * FRAME_WAIT_MS
            )),
    )
}

/// `GET /cams/{id}/events` - the verdict stream for a camera, as NDJSON.
///
async fn events_http(
    AxumPath(id): AxumPath<String>,
    Query(params): Query<EventsQuery>,
    State(state): State<AppState>,
) -> Response {
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

    // Registered here rather than inside the stream so the detector resumes
    // before the response starts flowing, and moved into the stream so a client
    // that disconnects mid-stream still releases the subscription.
    let watcher = MotionWatcher::register(stream.motion_watchers.clone());
    let mut motion_rx = stream.motion_tx.subscribe();
    let last_activity = state.last_activity.clone();
    let min_area = params.min_area;

    let body = Body::from_stream(async_stream::stream! {
        let _watcher = watcher;

        loop {
            match tokio::time::timeout(EVENTS_ACTIVITY_REFRESH, motion_rx.recv()).await {
                Ok(Ok(verdict)) => {
                    *last_activity.write().await = Instant::now();
                    if verdict.changed_pixels < min_area {
                        continue;
                    }
                    if let Ok(mut line) = serde_json::to_string(&verdict) {
                        line.push('\n');
                        yield Ok::<_, axum::Error>(bytes::Bytes::from(line));
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(_))) => {
                    // A slow client missed verdicts; `sequence` gap tells it so.
                    *last_activity.write().await = Instant::now();
                    continue;
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => break,
                Err(_) => {
                    *last_activity.write().await = Instant::now();
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-store")
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
    use futures_util::StreamExt;
    #[cfg(not(target_os = "linux"))]
    use nokhwa::utils::CameraIndex;
    use reqwest::StatusCode as ReqwestStatus;
    use serial_test::serial;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct FakeBackend {
        cameras: Vec<CameraDescriptor>,
        frame: Vec<u8>,
        /// When non-empty, each `capture_jpeg` returns the next frame in turn,
        /// so a detector watching the stream sees a change.
        frames: Vec<Vec<u8>>,
        fail_open: Option<String>,
        fail_capture: Option<String>,
    }

    struct FakeOpenCamera {
        frame: Vec<u8>,
        frames: Vec<Vec<u8>>,
        next_frame: usize,
        fail_capture: Option<String>,
    }

    impl Default for FakeOpenCamera {
        fn default() -> Self {
            Self {
                frame: vec![0xff, 0xd8],
                frames: Vec::new(),
                next_frame: 0,
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
                    frames: self.frames.clone(),
                    next_frame: 0,
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
            if self.frames.is_empty() {
                return Ok(self.frame.clone());
            }
            let frame = self.frames[self.next_frame % self.frames.len()].clone();
            self.next_frame += 1;
            Ok(frame)
        }
    }

    /// A camera stream state with no motion history, for tests that only care
    /// about the frame path.
    fn stream_state(latest_jpeg: Option<Vec<u8>>) -> CameraStreamState {
        let (frame_tx, _) = broadcast::channel(60);
        let (motion_tx, _) = broadcast::channel(MOTION_CHANNEL_CAPACITY);
        CameraStreamState {
            latest_jpeg: Arc::new(RwLock::new(latest_jpeg)),
            last_error: Arc::new(RwLock::new(None)),
            frame_tx,
            latest_motion: Arc::new(RwLock::new(None)),
            motion_tx,
            motion_watchers: Arc::new(AtomicUsize::new(0)),
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
        streams.insert("cam-a".to_string(), stream_state(None));
        streams.insert("cam-b".to_string(), stream_state(None));
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
        streams.insert("cam-a".to_string(), stream_state(None));
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
        streams.insert("cam-a".to_string(), stream_state(Some(jpeg.clone())));
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: Some("simulated open failure".into()),
            fail_capture: None,
        };
        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: Some("simulated capture failure".into()),
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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
            frames: Vec::new(),
            fail_open: None,
            fail_capture: None,
        };

        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            MotionConfig::default(),
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

    // =====================================================================
    // Motion detection: the daemon-side pipeline, the watcher gate, the
    // verdict publication and the two endpoints. None of these need a camera.
    // =====================================================================

    /// A `width`x`height` JPEG of a solid mid-grey frame. Decoding the same
    /// bytes twice yields identical pixels, so a detector sees no change.
    fn uniform_jpeg(width: u32, height: u32) -> Vec<u8> {
        encode_rgb_to_jpeg(width, height, vec![128u8; (width * height * 3) as usize]).unwrap()
    }

    /// A `width`x`height` JPEG with a 20x20 dark block whose top-left corner is
    /// at (`x`, `y`), on a lighter background.
    fn block_jpeg(width: u32, height: u32, x: usize, y: usize) -> Vec<u8> {
        let mut rgb = vec![150u8; (width * height * 3) as usize];
        for row in y..(y + 20).min(height as usize) {
            for col in x..(x + 20).min(width as usize) {
                let idx = (row * width as usize + col) * 3;
                rgb[idx] = 20;
                rgb[idx + 1] = 20;
                rgb[idx + 2] = 20;
            }
        }
        encode_rgb_to_jpeg(width, height, rgb).unwrap()
    }

    fn test_motion_config() -> MotionConfig {
        MotionConfig {
            hz: 50,
            width: 64,
            ..MotionConfig::default()
        }
    }

    /// Run `motion_loop` for this test and hand back the stream state it feeds.
    fn spawn_motion_loop(jpeg: Option<Vec<u8>>) -> CameraStreamState {
        let stream = stream_state(jpeg);
        tokio::spawn(motion_loop(
            "cam-a".to_string(),
            stream.latest_jpeg.clone(),
            stream.latest_motion.clone(),
            stream.motion_tx.clone(),
            stream.motion_watchers.clone(),
            test_motion_config(),
        ));
        stream
    }

    #[test]
    fn analysis_dimensions_downscales_preserving_aspect_ratio() {
        assert_eq!(analysis_dimensions(1920, 1080, 320), (320, 180));
        assert_eq!(analysis_dimensions(640, 480, 320), (320, 240));
        assert_eq!(analysis_dimensions(1280, 720, 320), (320, 180));
        // A 4:3 frame that rounds to an odd height must still be a real height.
        assert_eq!(analysis_dimensions(1000, 333, 320), (320, 107));
    }

    #[test]
    fn analysis_dimensions_never_upscales_and_is_clamped() {
        // Smaller than the target: the frame's own size is used, not the target.
        assert_eq!(analysis_dimensions(100, 50, 320), (100, 50));
        assert_eq!(analysis_dimensions(320, 240, 320), (320, 240));
        // Out-of-range targets are clamped by the same function the config uses.
        assert_eq!(
            analysis_dimensions(1000, 500, 0),
            (MIN_MOTION_WIDTH as usize, 8)
        );
        assert_eq!(
            analysis_dimensions(10_000, 10_000, 100_000),
            (MAX_MOTION_WIDTH as usize, MAX_MOTION_WIDTH as usize)
        );
    }

    #[test]
    fn analysis_dimensions_rejects_degenerate_frames() {
        assert_eq!(analysis_dimensions(0, 0, 320), (0, 0));
        assert_eq!(analysis_dimensions(0, 480, 320), (0, 0));
        assert_eq!(analysis_dimensions(640, 0, 320), (0, 0));
    }

    #[tokio::test]
    async fn analyze_motion_frame_reports_a_decode_error_instead_of_no_motion() {
        let mut state = None;
        let analysis = tokio::task::spawn_blocking(move || {
            analyze_motion_frame(&mut state, b"this is not a JPEG", test_motion_config())
        })
        .await
        .unwrap();

        // A frame nobody could decode must not be reported as "the scene was
        // still": `detected: false` is indistinguishable from a quiet scene.
        assert_eq!(analysis.status, MotionStatus::DecodeError);
        assert_eq!(analysis.changed_pixels, 0);
        assert_eq!(analysis.bbox, None);
        assert!(analysis.problem.is_some(), "the reason must be loggable");
    }

    #[tokio::test]
    async fn analyze_motion_frame_reports_a_frame_too_small_to_scan() {
        // A 2x2 frame is smaller than the detector's minimum scan dimension; a
        // detector constructed for it would silently report no motion forever.
        let jpeg = uniform_jpeg(2, 2);
        let mut state = None;
        let analysis = tokio::task::spawn_blocking(move || {
            analyze_motion_frame(&mut state, &jpeg, test_motion_config())
        })
        .await
        .unwrap();

        assert_eq!(analysis.status, MotionStatus::GeometryMismatch);
        assert!(analysis.problem.is_some());
    }

    #[tokio::test]
    async fn analyze_motion_frame_uses_the_frames_real_geometry() {
        // 400x300 decoded, analysis width 100: the detector geometry is derived
        // from the frame, not from any hardcoded 640x480.
        let jpeg = uniform_jpeg(400, 300);
        let mut state = None;
        let analysis = tokio::task::spawn_blocking(move || {
            let mut config = test_motion_config();
            config.width = 100;
            analyze_motion_frame(&mut state, &jpeg, config)
        })
        .await
        .unwrap();

        assert_eq!(analysis.status, MotionStatus::Ok);
        assert_eq!((analysis.frame_width, analysis.frame_height), (400, 300));
        assert_eq!(
            (analysis.detector_width, analysis.detector_height),
            (100, 75)
        );
    }

    #[tokio::test]
    async fn analyze_motion_frame_detects_a_changed_frame_and_locates_it() {
        // Two frames that differ only by a block moving 5 pixels: the fixture
        // guarantees motion, and the bbox must be inside the frame.
        let first = block_jpeg(64, 64, 8, 8);
        let second = block_jpeg(64, 64, 13, 13);
        let analysis = tokio::task::spawn_blocking(move || {
            let mut state = None;
            analyze_motion_frame(&mut state, &first, test_motion_config());
            let mut analysis = analyze_motion_frame(&mut state, &second, test_motion_config());
            analysis.problem = None;
            analysis
        })
        .await
        .unwrap();

        assert_eq!(analysis.status, MotionStatus::Ok);
        assert!(
            analysis.changed_pixels > 0,
            "a moved block must register as motion"
        );
        let bbox = analysis.bbox.expect("motion must carry bounds");
        assert!(bbox.x + bbox.width <= 64 && bbox.y + bbox.height <= 64);
    }

    #[tokio::test]
    async fn analyze_motion_frame_rebuilds_the_detector_for_a_new_geometry() {
        let first = uniform_jpeg(64, 64);
        let second = uniform_jpeg(128, 64);
        let analysis = tokio::task::spawn_blocking(move || {
            let mut config = test_motion_config();
            config.width = 128;
            let mut state = None;
            analyze_motion_frame(&mut state, &first, config);
            let analysis = analyze_motion_frame(&mut state, &second, config);
            (state, analysis)
        })
        .await
        .unwrap();

        let (state, analysis) = analysis;
        assert_eq!(
            (analysis.detector_width, analysis.detector_height),
            (128, 64)
        );
        let state = state.expect("a detector must exist");
        assert_eq!((state.width, state.height), (128, 64));
    }

    #[tokio::test]
    async fn motion_watcher_guard_counts_subscribers() {
        let watchers = Arc::new(AtomicUsize::new(0));
        let first = MotionWatcher::register(watchers.clone());
        assert_eq!(watchers.load(Ordering::SeqCst), 1);
        let second = MotionWatcher::register(watchers.clone());
        assert_eq!(watchers.load(Ordering::SeqCst), 2);

        drop(first);
        assert_eq!(watchers.load(Ordering::SeqCst), 1);
        drop(second);
        assert_eq!(watchers.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn motion_loop_does_no_work_without_a_watcher() {
        let stream = spawn_motion_loop(Some(uniform_jpeg(64, 64)));
        let mut rx = stream.motion_tx.subscribe();

        // Several ticks with nobody watching: no verdict is produced at all.
        sleep(Duration::from_millis(200)).await;

        assert!(
            stream.latest_motion.read().await.is_none(),
            "an unwatched camera must not spend CPU decoding frames"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(50), rx.recv())
                .await
                .is_err(),
            "no verdict may be broadcast while nobody is subscribed"
        );
        assert_eq!(stream.motion_watchers.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn motion_loop_publishes_while_watched_and_keeps_both_copies_in_step() {
        let stream = spawn_motion_loop(Some(uniform_jpeg(64, 64)));
        let mut rx = stream.motion_tx.subscribe();
        stream.motion_watchers.store(1, Ordering::SeqCst);

        let broadcast = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a verdict must be published within the timeout")
            .expect("the channel must stay open");

        assert_eq!(broadcast.status, MotionStatus::Ok);
        assert_eq!((broadcast.frame_width, broadcast.frame_height), (64, 64));
        assert_eq!(
            (broadcast.detector_width, broadcast.detector_height),
            (64, 64),
            "the detector geometry must be the frame's, not an assumed size"
        );
        assert!(broadcast.sequence >= 1);

        // The broadcast copy and the stored copy describe the same verdict;
        // they are written under one lock acquisition.
        let stored = stream
            .latest_motion
            .read()
            .await
            .clone()
            .expect("the latest verdict must be stored");
        assert_eq!(stored.sequence, broadcast.sequence);
        assert_eq!(stored, broadcast);
        assert!(stored.timestamp_ms > 0);

        // The sequence is monotonic across verdicts.
        let next = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a second verdict must be published")
            .expect("the channel must stay open");
        assert!(
            next.sequence > stored.sequence,
            "sequence must increase so a client can tell 'no new verdict' from 'no motion'"
        );
    }

    #[tokio::test]
    async fn motion_loop_stops_when_the_last_watcher_leaves() {
        let stream = spawn_motion_loop(Some(uniform_jpeg(64, 64)));
        stream.motion_watchers.store(1, Ordering::SeqCst);

        let mut observed = None;
        for _ in 0..200 {
            if let Some(verdict) = stream.latest_motion.read().await.clone() {
                observed = Some(verdict);
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
        let observed = observed.expect("a verdict must be published while watched");

        stream.motion_watchers.store(0, Ordering::SeqCst);
        sleep(Duration::from_millis(200)).await;

        let after = stream
            .latest_motion
            .read()
            .await
            .clone()
            .expect("the last verdict stays readable");
        assert_eq!(
            after.sequence, observed.sequence,
            "detection must stop when the last subscriber leaves"
        );
    }

    #[tokio::test]
    async fn motion_loop_reports_an_undecodable_frame_instead_of_going_silent() {
        let stream = spawn_motion_loop(Some(b"not a jpeg frame".to_vec()));
        stream.motion_watchers.store(1, Ordering::SeqCst);

        let mut verdict = None;
        for _ in 0..200 {
            if let Some(current) = stream.latest_motion.read().await.clone() {
                verdict = Some(current);
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
        let verdict = verdict.expect("a verdict must still be published");

        assert_eq!(verdict.status, MotionStatus::DecodeError);
        assert!(!verdict.detected);
        assert_eq!(verdict.changed_pixels, 0);
    }

    #[tokio::test]
    async fn motion_http_serves_a_verdict_produced_after_the_request_arrived() {
        let state = default_state();
        let stream = state.streams.get("cam-a").unwrap().clone();
        let stale = MotionVerdict {
            camera: "cam-a".to_string(),
            sequence: 7,
            detected: true,
            changed_pixels: 42,
            bbox: Some(motion::MotionBox {
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            }),
            frame_width: 640,
            frame_height: 480,
            detector_width: 320,
            detector_height: 240,
            timestamp_ms: 1_700_000_000_000,
            status: MotionStatus::Ok,
        };
        *stream.latest_motion.write().await = Some(stale.clone());

        // Stand in for the detection loop: a newer verdict arrives while the
        // request is waiting.
        let expected = MotionVerdict {
            sequence: 8,
            changed_pixels: 99,
            ..stale.clone()
        };
        let publisher = stream.clone();
        let published = expected.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(50)).await;
            *publisher.latest_motion.write().await = Some(published);
        });

        // The request registers itself as a watcher, which is what makes the
        // detector run when no other client is subscribed.
        let watchers = stream.motion_watchers.clone();
        let resp = motion_http(AxumPath("default".to_string()), State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            watchers.load(Ordering::SeqCst),
            0,
            "the subscription ends with the request"
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: MotionVerdict = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed, expected);
        assert_ne!(
            parsed.sequence, stale.sequence,
            "a verdict from before the request must not be served as the current one"
        );
    }

    #[tokio::test]
    async fn motion_http_unknown_camera_is_not_found() {
        let state = default_state();
        let resp = motion_http(AxumPath("cam-z".to_string()), State(state)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn events_http_streams_ndjson_and_holds_a_watcher_subscription() {
        let state = default_state();
        let verdict = MotionVerdict {
            camera: "cam-a".to_string(),
            sequence: 1,
            detected: true,
            changed_pixels: 9,
            bbox: None,
            frame_width: 64,
            frame_height: 64,
            detector_width: 64,
            detector_height: 64,
            timestamp_ms: 1_700_000_000_000,
            status: MotionStatus::Ok,
        };

        let resp = events_http(
            AxumPath("cam-a".to_string()),
            Query(EventsQuery { min_area: 0 }),
            State(state.clone()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/x-ndjson")
        );

        // Serving this response is what turns detection on.
        let watchers = state.streams.get("cam-a").unwrap().motion_watchers.clone();
        assert_eq!(watchers.load(Ordering::SeqCst), 1);

        let mut body = resp.into_body().into_data_stream();
        state
            .streams
            .get("cam-a")
            .unwrap()
            .motion_tx
            .send(verdict.clone())
            .unwrap();

        let chunk = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("a verdict must be streamed")
            .expect("the body must yield")
            .unwrap();

        // One verdict per line: NDJSON, not an array and not a frame.
        let text = std::str::from_utf8(&chunk).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(text.matches('\n').count(), 1);
        let parsed: MotionVerdict = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(parsed, verdict);

        // Dropping the response (a client that went away) releases the
        // subscription, so the detector stops when the last watcher leaves.
        drop(body);
        assert_eq!(watchers.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn events_min_area_filters_emissions_not_the_measurement() {
        let state = default_state();
        let quiet = MotionVerdict {
            camera: "cam-a".to_string(),
            sequence: 1,
            detected: false,
            changed_pixels: 0,
            bbox: None,
            frame_width: 64,
            frame_height: 64,
            detector_width: 64,
            detector_height: 64,
            timestamp_ms: 1_700_000_000_000,
            status: MotionStatus::Ok,
        };
        let moving = MotionVerdict {
            sequence: 2,
            detected: true,
            changed_pixels: 12,
            ..quiet.clone()
        };

        let resp = events_http(
            AxumPath("cam-a".to_string()),
            Query(EventsQuery { min_area: 5 }),
            State(state.clone()),
        )
        .await;
        let mut body = resp.into_body().into_data_stream();

        // Both verdicts exist and are published; only the one at or above
        // `min_area` is emitted, in order.
        let tx = state.streams.get("cam-a").unwrap().motion_tx.clone();
        tx.send(quiet).unwrap();
        tx.send(moving.clone()).unwrap();

        let chunk = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("a verdict must be streamed")
            .expect("the body must yield")
            .unwrap();
        let parsed: MotionVerdict =
            serde_json::from_str(std::str::from_utf8(&chunk).unwrap().trim_end()).unwrap();
        assert_eq!(
            parsed.sequence, 2,
            "the below-threshold verdict must not be emitted"
        );
    }

    #[tokio::test]
    async fn events_http_unknown_camera_is_not_found() {
        let state = default_state();
        let resp = events_http(
            AxumPath("cam-z".to_string()),
            Query(EventsQuery { min_area: 0 }),
            State(state),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    #[serial]
    async fn test_daemon_serves_motion_verdicts_and_events_with_fake_backend() {
        let bind: SocketAddr = "127.0.0.1:43231".parse().unwrap();
        // Alternating frames, so the detector sees the block move.
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame: vec![],
            frames: vec![block_jpeg(64, 64, 8, 8), block_jpeg(64, 64, 20, 20)],
            fail_open: None,
            fail_capture: None,
        };
        let handle = tokio::spawn(run_daemon(
            bind,
            "cam-a".into(),
            CameraOpenOptions::default(),
            test_motion_config(),
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

        // Polling `/motion` is itself a subscription, so the detector runs even
        // though no other client is streaming.
        let mut verdict = None;
        for _ in 0..50 {
            let resp = client
                .get(format!("http://{bind}/cams/default/motion"))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), ReqwestStatus::OK);
            let parsed: MotionVerdict = resp.json().await.unwrap();
            if parsed.detected {
                verdict = Some(parsed);
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        let verdict = verdict.expect("the moving block must be detected");
        assert_eq!(verdict.camera, "cam-a");
        assert_eq!(verdict.status, MotionStatus::Ok);
        assert_eq!((verdict.frame_width, verdict.frame_height), (64, 64));
        assert_eq!((verdict.detector_width, verdict.detector_height), (64, 64));

        // The event stream carries the same verdict shape over the wire: this
        // is the contract the CLI deserialises.
        let mut events = client
            .get(format!("http://{bind}/cams/default/events"))
            .send()
            .await
            .unwrap();
        assert_eq!(events.headers()["content-type"], "application/x-ndjson");
        let mut buffer = Vec::new();
        let line = loop {
            let chunk = events.chunk().await.unwrap().expect("stream ended early");
            buffer.extend_from_slice(&chunk);
            if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                break String::from_utf8(buffer[..newline].to_vec()).unwrap();
            }
        };
        let streamed: MotionVerdict = serde_json::from_str(&line).unwrap();
        assert_eq!(streamed.camera, "cam-a");
        assert!(streamed.sequence >= 1);

        handle.abort();
        let _ = handle.await;
    }
}
