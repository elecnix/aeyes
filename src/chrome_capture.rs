//! Chrome DevTools Protocol screenshot capture.
//!
//! Uses the same approach as Chrome's own DevTools: reads the
//! DevToolsActivePort file and connects via WebSocket.

use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{json, Value};

/// Get Chrome's DevTools WebSocket URL from DevToolsActivePort file.
///
/// On Linux: ~/.config/google-chrome/DevToolsActivePort
/// On macOS: ~/Library/Application Support/Google/Chrome/DevToolsActivePort
fn get_browser_ws_url() -> Result<String> {
    let port_file = if cfg!(target_os = "macos") {
        dirs::home_dir()
            .context("cannot find home directory")?
            .join("Library/Application Support/Google/Chrome/DevToolsActivePort")
    } else {
        dirs::home_dir()
            .context("cannot find home directory")?
            .join(".config/google-chrome/DevToolsActivePort")
    };

    let content = std::fs::read_to_string(&port_file)
        .with_context(|| format!("failed to read {}", port_file.display()))?;

    let mut lines = content.lines();
    let port = lines.next().context("DevToolsActivePort missing port")?;
    let path = lines.next().context("DevToolsActivePort missing path")?;

    Ok(format!("ws://127.0.0.1:{port}{path}"))
}

/// Send a CDP command and receive the response.
fn cdp_send(
    ws: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
    method: &str,
    params: Value,
) -> Result<Value> {
    use tungstenite::Message;

    // Generate session-less command (browser-level)
    let msg = json!({
        "id": 1,
        "method": method,
        "params": params
    });

    ws.send(Message::Text(msg.to_string().into()))?;

    // Read responses until we get id=1
    loop {
        let response = ws.read()?;
        let text = response.to_text()?;
        let value: Value = serde_json::from_str(text)?;

        if value.get("id") == Some(&json!(1)) {
            return Ok(value);
        }
        // Skip events (no id field)
    }
}

/// Capture a JPEG screenshot from the first available Chrome tab.
///
/// Chrome must be running with remote debugging enabled.
///
/// # Arguments
/// * `quality` - JPEG quality 1-100
///
/// # Example
/// ```no_run
/// let jpeg = aeyes::chrome_capture::capture_screenshot(85).unwrap();
/// std::fs::write("screenshot.jpg", jpeg).unwrap();
/// ```
pub fn capture_screenshot(quality: u32) -> Result<Vec<u8>> {
    use tungstenite::connect;
    use tungstenite::Message;

    // 1. Connect to Chrome's browser target
    let ws_url = get_browser_ws_url()?;
    let (mut ws, _) = connect(&ws_url).context("failed to connect to Chrome DevTools")?;

    // 2. Get list of targets
    let response = cdp_send(&mut ws, "Target.getTargets", json!({}))?;
    let targets = response["result"]["targetInfos"]
        .as_array()
        .context("no targets in response")?;

    // 3. Find first page target
    let page = targets
        .iter()
        .find(|t| t["type"] == "page")
        .context("no Chrome page found")?;

    let target_id = page["targetId"].as_str().context("targetId not a string")?;

    // 4. Attach to target
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

    // 5. Send screenshot command to the target session
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

    // 6. Read response
    loop {
        let response = ws.read()?;
        let text = response.to_text()?;
        let value: Value = serde_json::from_str(text)?;

        if value.get("id") == Some(&json!(2)) {
            // Check for errors
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
