//! Chrome DevTools Protocol transport and endpoint discovery.
//!
//! This module owns the request/response half of CDP. Endpoint discovery
//! tries, in order:
//! 1. DevToolsActivePort file (standard Chrome approach)
//! 2. Direct port probing with /json/version fallback
//! 3. Common debugging ports (9222, 9223, 9224)
//!
//! Connection *lifetime* is deliberately not owned here: the daemon keeps one
//! persistent session open and drives [`cdp_request`] over that socket, so
//! Chrome's "Allow debugging" permission is held across requests.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::Duration;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

/// Common Chrome debugging ports to try.
const COMMON_PORTS: &[u16] = &[9222, 9223, 9224];

/// How long a single CDP handshake may block before it is treated as failed.
///
/// Chrome never stalls mid-handshake, so a slow answer means the endpoint is
/// wedged. Without this bound a stalled connect would park the caller forever
/// and, because the daemon creates its session under a lock, wedge every later
/// Chrome request behind it.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// The socket a CDP session runs over: plain TCP for a `ws://` endpoint.
pub type CdpSocket = WebSocket<MaybeTlsStream<std::net::TcpStream>>;

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
/// Tries DevToolsActivePort first, then common ports.
///
/// This blocks: call it from a blocking task, not a tokio worker.
pub fn get_browser_ws_url() -> Result<String> {
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
        COMMON_PORTS
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Bound how long the next read may block.
///
/// Only a plain TCP stream is re-timed; a TLS stream is left as-is rather than
/// silently mistimed. Failures are advisory: an endpoint that ignores the
/// timeout still works, it just blocks.
pub fn set_read_timeout(ws: &CdpSocket, timeout: Option<Duration>) {
    if let MaybeTlsStream::Plain(stream) = ws.get_ref() {
        let _ = stream.set_read_timeout(timeout);
    }
}

/// Send one CDP command and read until the response with the same `id` comes
/// back, discarding the unsolicited events Chrome interleaves.
///
/// `session_id` scopes the command to an attached target; `None` sends it as a
/// browser-level command. Returns the whole decoded message so callers can
/// read either `result` or `error` via [`cdp_result`].
///
/// Every failure here is a *transport* failure (send, read, decode). A command
/// Chrome understood and refused arrives as `Ok` and becomes an `Err` only at
/// [`cdp_result`], which is what lets a caller tell "the session is broken"
/// apart from "Chrome said no".
pub fn cdp_request(
    ws: &mut CdpSocket,
    id: u64,
    method: &str,
    params: Value,
    session_id: Option<&str>,
) -> Result<Value> {
    let mut msg = json!({ "id": id, "method": method, "params": params });
    if let Some(session_id) = session_id {
        msg["sessionId"] = json!(session_id);
    }

    ws.send(Message::Text(msg.to_string().into()))
        .with_context(|| format!("cdp send: {method}"))?;

    loop {
        let text = ws
            .read()
            .with_context(|| format!("cdp read: {method}"))?
            .into_text()
            .with_context(|| format!("cdp text: {method}"))?;
        let value: Value =
            serde_json::from_str(&text).with_context(|| format!("cdp decode: {method}"))?;

        if value.get("id").and_then(Value::as_u64) == Some(id) {
            return Ok(value);
        }
        // Anything else is an event we did not ask for - keep reading.
    }
}

/// Unwrap a CDP response into its `result`, turning a protocol-level `error`
/// object into an `Err`.
pub fn cdp_result(message: Value) -> Result<Value> {
    if let Some(err) = message.get("error") {
        bail!("CDP error: {err}");
    }

    message
        .get("result")
        .cloned()
        .context("CDP response carried no result")
}

/// Pick the first `page` target out of a `Target.getTargets` result.
pub fn first_page_target(result: &Value) -> Result<String> {
    result["targetInfos"]
        .as_array()
        .context("no targets in response")?
        .iter()
        .find(|target| target["type"] == "page")
        .and_then(|target| target["targetId"].as_str())
        .map(str::to_string)
        .context("no Chrome page found")
}

/// Information about a Chrome debug target.
#[derive(Debug, Clone, Serialize)]
pub struct TargetInfo {
    pub target_id: String,
    pub title: String,
    pub url: String,
    pub target_type: String,
}

impl TargetInfo {
    /// Describe one entry of a `Target.getTargets` result, or `None` if it has
    /// no target id.
    pub fn from_value(target: &Value) -> Option<Self> {
        Some(Self {
            target_id: target["targetId"].as_str()?.to_string(),
            title: target["title"].as_str().unwrap_or_default().to_string(),
            url: target["url"].as_str().unwrap_or_default().to_string(),
            target_type: target["type"].as_str().unwrap_or_default().to_string(),
        })
    }

    /// Map a `Target.getTargets` result into target descriptions.
    pub fn list_from_result(result: &Value) -> Vec<Self> {
        result["targetInfos"]
            .as_array()
            .map(|targets| targets.iter().filter_map(Self::from_value).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets_result(json: Value) -> Value {
        json!({ "targetInfos": json })
    }

    #[test]
    fn cdp_result_unwraps_a_successful_response() {
        let message = json!({ "id": 3, "result": { "data": "abc" } });
        assert_eq!(cdp_result(message).unwrap()["data"], json!("abc"));
    }

    #[test]
    fn cdp_result_turns_a_protocol_error_into_err() {
        let message = json!({ "id": 3, "error": { "code": -32000, "message": "nope" } });
        let err = cdp_result(message).unwrap_err().to_string();
        assert!(err.contains("CDP error"), "unexpected error: {err}");
        assert!(err.contains("nope"), "unexpected error: {err}");
    }

    #[test]
    fn cdp_result_rejects_a_response_without_a_result() {
        let message = json!({ "id": 3 });
        assert!(cdp_result(message).is_err());
    }

    #[test]
    fn first_page_target_picks_the_first_page() {
        let result = targets_result(json!([
            { "type": "service_worker", "targetId": "sw" },
            { "type": "page", "targetId": "page-a", "title": "A" },
            { "type": "page", "targetId": "page-b" },
        ]));

        assert_eq!(first_page_target(&result).unwrap(), "page-a");
    }

    #[test]
    fn first_page_target_reports_a_browser_with_no_pages() {
        let result = targets_result(json!([{ "type": "service_worker", "targetId": "sw" }]));
        let err = first_page_target(&result).unwrap_err().to_string();
        assert!(err.contains("no Chrome page found"), "unexpected: {err}");
    }

    #[test]
    fn list_from_result_maps_targets_and_skips_malformed_ones() {
        let result = targets_result(json!([
            { "type": "page", "targetId": "page-a", "title": "A", "url": "https://a" },
            { "type": "page", "title": "no id" },
        ]));

        let listed = TargetInfo::list_from_result(&result);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].target_id, "page-a");
        assert_eq!(listed[0].title, "A");
        assert_eq!(listed[0].url, "https://a");
        assert_eq!(listed[0].target_type, "page");
    }

    #[test]
    fn list_from_result_defaults_absent_fields() {
        let result = targets_result(json!([{ "type": "page", "targetId": "page-a" }]));
        let listed = TargetInfo::list_from_result(&result);
        assert_eq!(listed[0].title, "");
        assert_eq!(listed[0].url, "");
    }

    #[test]
    fn list_from_result_tolerates_a_missing_target_list() {
        assert!(TargetInfo::list_from_result(&json!({})).is_empty());
    }

    #[test]
    fn list_from_result_preserves_the_json_shape_the_cli_reads() {
        let result = targets_result(json!([{
            "type": "page", "targetId": "abcdef0123", "title": "T", "url": "u"
        }]));

        let serialized = serde_json::to_value(TargetInfo::list_from_result(&result)[0].clone())
            .expect("TargetInfo serializes");

        assert_eq!(serialized["target_id"], json!("abcdef0123"));
        assert_eq!(serialized["title"], json!("T"));
        assert_eq!(serialized["url"], json!("u"));
        assert_eq!(serialized["target_type"], json!("page"));
    }
}
