//! Chrome DevTools Protocol screenshot capture.
//!
//! Supports multiple discovery methods:
//! 1. DevToolsActivePort file (standard Chrome approach)
//! 2. Direct port probing with /json/version fallback
//! 3. Common debugging ports (9222, 9223, 9224)

use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{json, Value};

/// Common Chrome debugging ports to try.
const COMMON_PORTS: &[u16] = &[9222, 9223, 9224];

/// Try to get WebSocket URL from DevToolsActivePort file.
/// Returns None if file doesn't exist or is invalid.
fn try_devtools_active_port() -> Option<String> {
    let port_file = if cfg!(target_os = "macos") {
        dirs::home_dir()?.join("Library/Application Support/Google/Chrome/DevToolsActivePort")
    } else {
        dirs::home_dir()?.join(".config/google-chrome/DevToolsActivePort")
    };

    let content = std::fs::read_to_string(&port_file).ok()?;
    let mut lines = content.lines();
    let port = lines.next()?;
    let path = lines.next()?;

    // Validate port is a number
    port.parse::<u16>().ok()?;

    Some(format!("ws://127.0.0.1:{port}{path}"))
}

/// Try to get WebSocket URL by probing a port's /json/version endpoint.
fn try_port_json_version(port: u16) -> Option<String> {
    let url = format!("http://127.0.0.1:{port}/json/version");
    let resp = ureq::get(&url).call().ok()?;
    let json = resp.into_body().read_json::<Value>().ok()?;
    let ws_url = json["webSocketDebuggerUrl"].as_str()?;
    Some(ws_url.to_string())
}

/// Get Chrome's DevTools WebSocket URL.
///
/// Tries multiple discovery methods:
/// 1. DevToolsActivePort file
/// 2. Common debugging ports (9222, 9223, 9224)
fn get_browser_ws_url() -> Result<String> {
    // Method 1: Try DevToolsActivePort file first
    if let Some(ws_url) = try_devtools_active_port() {
        return Ok(ws_url);
    }

    // Method 2: Probe common ports
    for port in COMMON_PORTS {
        if let Some(ws_url) = try_port_json_version(*port) {
            return Ok(ws_url);
        }
    }

    // Nothing worked
    anyhow::bail!(
        "could not find Chrome debugging endpoint. \
         Make sure Chrome is running with remote debugging enabled.\n\
         \n\
         To enable:\n\
         - Linux/macOS: Open chrome://inspect/#remote-debugging and enable\n\
         - Or launch Chrome with: google-chrome --remote-debugging-port=9222\n\
         \n\
         Tried:\n\
         - DevToolsActivePort file ({})\n\
         - Ports: {}",
        if cfg!(target_os = "macos") {
            "~/Library/Application Support/Google/Chrome/DevToolsActivePort"
        } else {
            "~/.config/google-chrome/DevToolsActivePort"
        },
        COMMON_PORTS.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")
    );
}

/// Send a CDP command and receive the response.
fn cdp_send(
    ws: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
    method: &str,
    params: Value,
) -> Result<Value> {
    use tungstenite::Message;

    let msg = json!({
        "id": 1,
        "method": method,
        "params": params
    });

    ws.send(Message::Text(msg.to_string().into()))?;

    loop {
        let response = ws.read()?;
        let text = response.to_text()?;
        let value: Value = serde_json::from_str(text)?;

        if value.get("id") == Some(&json!(1)) {
            return Ok(value);
        }
    }
}

/// Capture a JPEG screenshot from the first available Chrome tab.
///
/// Chrome must be running with remote debugging enabled.
///
/// # Example
/// ```no_run
/// let jpeg = aeyes::chrome_capture::capture_screenshot(85).unwrap();
/// std::fs::write("screenshot.jpg", jpeg).unwrap();
/// ```
pub fn capture_screenshot(quality: u32) -> Result<Vec<u8>> {
    use tungstenite::connect;
    use tungstenite::Message;

    let ws_url = get_browser_ws_url()?;
    let (mut ws, _) = connect(&ws_url).context("failed to connect to Chrome DevTools")?;

    // Get list of targets
    let response = cdp_send(&mut ws, "Target.getTargets", json!({}))?;
    let targets = response["result"]["targetInfos"]
        .as_array()
        .context("no targets in response")?;

    // Find first page target
    let page = targets
        .iter()
        .find(|t| t["type"] == "page")
        .context("no Chrome page found")?;

    let target_id = page["targetId"].as_str().context("targetId not a string")?;

    // Attach to target
    let attach_response = cdp_send(
        &mut ws,
        "Target.attachToTarget",
        json!({
            "targetId": target_id,
            "flatten": true
        }),
    )?;

    let session_id = attach_response["result"]["sessionId"]
        .as_str()
        .context("sessionId not found")?;

    // Send screenshot command
    let msg = json!({
        "id": 2,
        "method": "Page.captureScreenshot",
        "sessionId": session_id,
        "params": {
            "format": "jpeg",
            "quality": quality,
            "fromSurface": true
        }
    });

    ws.send(Message::Text(msg.to_string().into()))?;

    loop {
        let response = ws.read()?;
        let text = response.to_text()?;
        let value: Value = serde_json::from_str(text)?;

        if value.get("id") == Some(&json!(2)) {
            if let Some(err) = value.get("error") {
                anyhow::bail!("CDP error: {}", err);
            }

            let data = value["result"]["data"]
                .as_str()
                .context("no screenshot data in response")?;

            return base64::engine::general_purpose::STANDARD
                .decode(data)
                .context("failed to decode base64 screenshot");
        }
    }
}

/// Information about a Chrome debug target.
#[derive(Debug, Clone)]
pub struct TargetInfo {
    pub target_id: String,
    pub title: String,
    pub url: String,
    pub target_type: String,
}

/// List available Chrome debug targets (pages, extensions, etc).
pub fn list_targets() -> Result<Vec<TargetInfo>> {
    use tungstenite::connect;

    let ws_url = get_browser_ws_url()?;
    let (mut ws, _) = connect(&ws_url).context("failed to connect to Chrome DevTools")?;

    let response = cdp_send(&mut ws, "Target.getTargets", json!({}))?;
    let targets = response["result"]["targetInfos"]
        .as_array()
        .context("no targets in response")?;

    Ok(targets
        .iter()
        .filter_map(|t| {
            Some(TargetInfo {
                target_id: t["targetId"].as_str()?.to_string(),
                title: t["title"].as_str().unwrap_or("").to_string(),
                url: t["url"].as_str().unwrap_or("").to_string(),
                target_type: t["type"].as_str().unwrap_or("").to_string(),
            })
        })
        .collect())
}
