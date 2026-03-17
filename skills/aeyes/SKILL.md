---
name: aeyes
description: Capture webcam frames and video clips using the aeyes daemon. Use when asked to take a photo, snap a picture, grab a frame, record video, view live webcam stream, or inspect webcam feed. Supports multiple cameras and live streaming via web UI.
---

# Webcam capture with aeyes

Use this skill when the user wants to capture an image or video from their webcam, or view a live stream.

## What is aeyes?

**A-Eyes** (AI's eyes) is a CLI daemon that keeps your webcam open for **instant captures** without auto-exposure/focus delays. It maintains a ~30 FPS frame buffer, so captures complete in under 100ms.

- GitHub: https://github.com/elecnix/aeyes
- Crate: https://crates.io/crates/aeyes
- License: MIT

## Locating the binary

Check if `aeyes` is already installed:

```bash
which aeyes
# or
command -v aeyes
```

If not found, check common cargo install locations:

```bash
ls ~/.cargo/bin/aeyes 2>/dev/null
ls /usr/local/bin/aeyes 2>/dev/null
```

## Installing

Install from crates.io (recommended):

```bash
cargo install aeyes
```

Or build from source:

```bash
git clone https://github.com/elecnix/aeyes.git /tmp/aeyes
cd /tmp/aeyes
cargo install --path .
```

This installs the binary to `~/.cargo/bin/aeyes`. Ensure `~/.cargo/bin` is in your PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

**Prerequisites**: Rust toolchain (rustc 1.75+) must be installed. Check with:

```bash
rustc --version
```

If Rust is not installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

## Available commands

```
start   Start the background daemon
cams    List available cameras
frame   Capture a frame to a file through the daemon HTTP API
video   Capture a video clip through the daemon HTTP API
stop    Stop the daemon
status  Show daemon status
```

## Capturing a webcam image

### 1. Start the daemon (if not running)

```bash
aeyes start
```

This forks to the background and opens the webcam (~30 FPS frame buffer).

### 2. Check daemon status

```bash
aeyes status
```

### 3. Capture a frame

```bash
# Custom output path (required - no default path)
aeyes frame -o /tmp/my-photo.jpg
```

### 4. List available cameras

```bash
aeyes cams
```

### 5. Capture a video clip

```bash
aeyes video -o /tmp/my-clip.avi
```

### 6. Stop the daemon (optional)

```bash
aeyes stop
```

## Quick capture workflow

```bash
# One-liner: ensure daemon is running, capture, return path
aeyes start 2>/dev/null || true
sleep 1
aeyes frame -o /tmp/aeyes-capture.jpg
echo "Image saved to: /tmp/aeyes-capture.jpg"
```

## Live webcam streaming

You can view the live webcam feed directly in your browser while the daemon is running. This is useful for inspecting the webcam or showing the feed to the user.

### Web UI URLs

| Camera | URL |
|--------|-----|
| Camera 0 | http://localhost:43210/web/0 |
| Camera 1 | http://localhost:43210/web/1 |
| Default camera | http://localhost:43210/web/default |

### Opening the live stream

Share the URL with the user so they can click it, or open it yourself in a browser:

```bash
# Display URL for user
echo "Live webcam view: http://localhost:43210/web/0"

# Or open in Chrome (if available)
google-chrome http://localhost:43210/web/0 2>/dev/null &
```

### Features of the web UI

- Dark-themed interface
- Real-time MJPEG stream
- Status indicator (Connecting/Connected/Stream error)
- Responsive layout
- Multiple clients can view the same stream simultaneously

### MJPEG stream endpoint

For programmatic access to the stream:

```bash
# Get the MJPEG multipart stream
curl http://localhost:43210/cams/default/stream
```

## HTTP API reference

```bash
# List cameras
curl http://localhost:43210/cams

# Get a frame
curl -o frame.jpg http://localhost:43210/cams/default/frame

# Capture a video (query params: max_length, fps)
curl -o video.avi http://localhost:43210/cams/default/video?max_length=5&fps=15

# Live MJPEG stream
curl http://localhost:43210/cams/default/stream
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `aeyes` not found | Not in PATH | Add `~/.cargo/bin` to PATH |
| Build fails | Missing Rust | Install rustup (see above) |
| Daemon fails to start | Webcam in use or port 43210 occupied | Close other apps using camera, check `aeyes status` |
| Empty/black capture | Wrong device | Check `/dev/video*` devices |
| Permission denied | No video group access | `sudo usermod -aG video $USER` then logout/login |
| Web UI not loading | Daemon not running | Run `aeyes start` first |

## Platform notes

- **Linux**: Primary supported platform (uses V4L2/nokhwa)
- **macOS**: Supported via nokhwa/AVFoundation backend
- **Windows**: Supported via nokhwa/MSMF backend
- **Chrome screenshots**: Works on all platforms with Chrome remote debugging enabled

## Chrome screenshot capture

A-Eyes can capture screenshots from Chrome tabs using the Chrome DevTools Protocol.

### Requirements

Chrome must have remote debugging enabled:
- Open `chrome://inspect/#remote-debugging` and toggle the switch
- Or launch Chrome with `--remote-debugging-port=9222`

### Capturing Chrome screenshots

```bash
# List all open tabs
aeyes chrome --list-tabs

# Capture screenshot from first available tab
aeyes chrome -o /tmp/chrome-shot.jpg

# Capture with custom quality (1-100)
aeyes chrome --quality 95 -o /tmp/chrome-shot.jpg
```

### Alternative: chrome-cdp skill

For more advanced Chrome interaction (clicking, typing, evaluating JS, accessibility snapshots), install the chrome-cdp skill:

```bash
# Install chrome-cdp skill for Pi/Claude Code
pip install pi  # or install via pi-mono
# Then install chrome-cdp from: https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent
```

The chrome-cdp skill provides:
- Lightweight Chrome DevTools Protocol CLI
- Direct WebSocket connection — no Puppeteer
- Works with 100+ tabs, instant connection
- Screenshot capture, accessibility tree snapshots, JavaScript evaluation
- Click, type, and navigate commands

**Important**: Only interact with Chrome when explicitly approved by the user after being asked to inspect, debug, or interact with a page open in Chrome.

## IPC details

- HTTP API: `http://127.0.0.1:43210`
- The daemon maintains a latest-frame buffer, so captures are instant
- Adaptive exposure control keeps bright screens readable in dark rooms
- Live streaming via MJPEG multipart (supports multiple concurrent viewers)
