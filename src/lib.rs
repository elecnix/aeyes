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
use image::RgbImage;
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, ControlValueSetter, FrameFormat, KnownCameraControl,
        RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera, Frame,
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
    net::TcpStream,
    sync::RwLock,
    time::sleep,
};
use tracing::{info, warn, Level};

#[derive(Parser, Debug)]
#[command(
    name = "aeyes",
    about = "AI Eyes - non-interactive webcam daemon",
    disable_help_subcommand = true
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
        /// Bind address for the daemon HTTP API
        #[arg(long, default_value = "127.0.0.1:43210")]
        bind: SocketAddr,
    },
    /// List available cameras
    Cams,
    /// Capture a frame to a file through the daemon HTTP API
    Frame {
        /// Camera stable ID or name; defaults to the daemon-selected camera.
        #[arg(long)]
        camera: Option<String>,
        /// Output file path
        #[arg(short, long, default_value = "aeyes-frame.jpg")]
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

pub struct NokhwaBackend;

impl CameraBackend for NokhwaBackend {
    fn name(&self) -> &'static str {
        "native"
    }

    fn list_cameras(&self) -> Result<Vec<CameraDescriptor>> {
        let api = ApiBackend::Auto;
        let cams = query(api).context("failed to query cameras")?;
        Ok(cams
            .into_iter()
            .map(|cam| CameraDescriptor {
                id: cam.index().as_string(),
                name: cam.human_name(),
                backend: self.name().to_string(),
            })
            .collect())
    }

    fn open(&self, id: &str) -> Result<Box<dyn OpenCamera>> {
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
        Ok(Box::new(NokhwaOpenCamera { camera }))
    }
}

struct NokhwaOpenCamera {
    camera: Camera,
}

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

#[derive(Clone)]
pub struct AppState {
    selected_camera: String,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
    cameras: Arc<Vec<CameraDescriptor>>,
}

#[derive(Debug, Serialize)]
struct CamerasResponse {
    selected_camera: String,
    cameras: Vec<CameraDescriptor>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
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
    let cams = list_cameras_with_backend(&NokhwaBackend)?;
    if cams.is_empty() {
        println!("No cameras found.");
    } else {
        for cam in cams {
            println!("{}\t{}\t{}", cam.id, cam.name, cam.backend);
        }
    }
    Ok(())
}

pub async fn start_daemon(requested_camera: Option<String>, bind: SocketAddr) -> Result<()> {
    fs::create_dir_all(runtime_dir())?;
    if let Ok(addr) = daemon_addr().await {
        if daemon_responding(addr).await {
            println!("Daemon already running at http://{addr}");
            return Ok(());
        }
    }

    let backend = NokhwaBackend;
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
    let addr = daemon_addr()
        .await
        .context("daemon is not running; run 'aeyes start' first")?;
    let path = if let Some(camera) = camera {
        format!("/cams/{camera}/frame")
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
    let status_ok = headers.starts_with("HTTP/1.1 200") || headers.starts_with("HTTP/1.0 200");
    if !status_ok {
        bail!(headers.lines().next().unwrap_or("HTTP error"));
    }
    Ok(buf[split..].to_vec())
}

pub async fn run_daemon_from_env() -> Result<()> {
    fs::create_dir_all(runtime_dir())?;
    let bind: SocketAddr = env::var("AEYES_BIND")
        .unwrap_or_else(|_| "127.0.0.1:43210".to_string())
        .parse()
        .context("invalid AEYES_BIND")?;
    let selected_camera = env::var("AEYES_CAMERA").context("AEYES_CAMERA missing")?;
    run_daemon(bind, selected_camera, Box::new(NokhwaBackend)).await
}

pub async fn run_daemon(
    bind: SocketAddr,
    selected_camera: String,
    backend: Box<dyn CameraBackend>,
) -> Result<()> {
    let cameras = backend.list_cameras()?;
    let chosen = choose_camera(&cameras, Some(&selected_camera))?;
    let latest_jpeg: Arc<RwLock<Option<Vec<u8>>>> = Arc::new(RwLock::new(None));
    let latest_jpeg_task = latest_jpeg.clone();
    let chosen_id = chosen.id.clone();
    tokio::spawn(async move {
        if let Err(err) = camera_loop(backend, chosen_id, latest_jpeg_task).await {
            warn!(?err, "camera loop exited");
        }
    });

    fs::write(addr_path(), bind.to_string())?;
    fs::write(pid_path(), std::process::id().to_string())?;

    let state = AppState {
        selected_camera: chosen.id.clone(),
        latest_jpeg,
        cameras: Arc::new(cameras),
    };
    let app = Router::new()
        .route("/cams", get(list_cams_http))
        .route("/cams/:id/frame", get(frame_http))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!("daemon listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn camera_loop(
    backend: Box<dyn CameraBackend>,
    selected_camera: String,
    latest_jpeg: Arc<RwLock<Option<Vec<u8>>>>,
) -> Result<()> {
    let mut camera = backend.open(&selected_camera)?;
    camera.set_auto_features()?;
    loop {
        match camera.capture_jpeg() {
            Ok(bytes) => {
                *latest_jpeg.write().await = Some(bytes);
            }
            Err(err) => warn!(?err, "failed to capture frame"),
        }
        sleep(Duration::from_millis(30)).await;
    }
}

async fn list_cams_http(State(state): State<AppState>) -> Json<CamerasResponse> {
    Json(CamerasResponse {
        selected_camera: state.selected_camera,
        cameras: (*state.cameras).clone(),
    })
}

async fn frame_http(AxumPath(id): AxumPath<String>, State(state): State<AppState>) -> Response {
    let requested = if id == "default" {
        state.selected_camera.clone()
    } else {
        id
    };
    if requested != state.selected_camera {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("camera '{requested}' is not active in this daemon"),
            }),
        )
            .into_response();
    }
    match state.latest_jpeg.read().await.clone() {
        Some(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/jpeg")
            .body(Body::from(bytes))
            .unwrap(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "camera has not produced a frame yet".to_string(),
            }),
        )
            .into_response(),
    }
}

pub fn encode_frame_as_jpeg(frame: &Frame) -> Result<Vec<u8>> {
    let img: RgbImage = frame.decode_image::<RgbFormat>()?;
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95);
    encoder.encode_image(&img)?;
    Ok(bytes)
}

pub fn encode_rgb_to_jpeg(width: u32, height: u32, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let img = RgbImage::from_raw(width, height, bytes).context("invalid RGB buffer")?;
    let mut out = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95);
    encoder.encode_image(&img)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode as ReqwestStatus;
    use serial_test::serial;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    struct FakeBackend {
        cameras: Vec<CameraDescriptor>,
        frame: Vec<u8>,
    }

    struct FakeOpenCamera {
        frame: Vec<u8>,
    }

    impl CameraBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn list_cameras(&self) -> Result<Vec<CameraDescriptor>> {
            Ok(self.cameras.clone())
        }
        fn open(&self, id: &str) -> Result<Box<dyn OpenCamera>> {
            if self.cameras.iter().any(|c| c.id == id) {
                Ok(Box::new(FakeOpenCamera {
                    frame: self.frame.clone(),
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

    #[test]
    fn choose_camera_requires_explicit_choice_for_multiple() {
        let err = choose_camera(&fake_cameras(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple cameras found"));
        assert!(err.contains("cam-a"));
    }

    #[test]
    fn choose_camera_accepts_id_or_name() {
        let cams = fake_cameras();
        assert_eq!(
            choose_camera(&cams, Some("cam-b")).unwrap().name,
            "Rear Cam"
        );
        assert_eq!(choose_camera(&cams, Some("Front Cam")).unwrap().id, "cam-a");
    }

    #[test]
    fn parse_http_response_extracts_body() {
        let body = parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc").unwrap();
        assert_eq!(body, b"abc");
    }

    #[test]
    fn parse_http_response_rejects_non_200() {
        let err = parse_http_response(b"HTTP/1.1 404 Not Found\r\n\r\nnope")
            .unwrap_err()
            .to_string();
        assert!(err.contains("404"));
    }

    #[test]
    fn rgb_to_jpeg_encodes() {
        let jpeg = encode_rgb_to_jpeg(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[tokio::test]
    #[serial]
    async fn daemon_serves_cams_and_frames_with_fake_backend() {
        let _guard = TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("TMPDIR", dir.path());
        let bind: SocketAddr = "127.0.0.1:43219".parse().unwrap();
        let frame =
            encode_rgb_to_jpeg(2, 2, vec![0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0]).unwrap();
        let backend = FakeBackend {
            cameras: fake_cameras(),
            frame,
        };

        let handle = tokio::spawn(run_daemon(bind, "cam-a".into(), Box::new(backend)));
        sleep(Duration::from_millis(250)).await;

        let client = reqwest::Client::new();
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
    }
}
