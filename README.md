# A-Eyes 👀

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/elecnix/aeyes)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-blue.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/badge/crates.io-aeyes-orange.svg)](https://crates.io/crates/aeyes)

**A-Eyes** (AI's eyes) is a CLI daemon that keeps your webcam open for **instant captures** without auto-exposure/focus delays. Perfect for AI agents needing quick "eyes".

## For AI Agents

A-Eyes is published as a skill on [skills.sh](https://skills.sh), making it easy to add webcam capabilities to your AI agent.

### Install the skill

```bash
npx skills add elecnix/aeyes
```

This installs the skill for popular AI coding agents (Pi, Claude Code, Cursor, Copilot, Cline, Codex, and more). Once installed, your agent will know how to use `aeyes` for capturing photos, recording videos, and viewing live webcam streams.

## Features
- Daemon mode: Open webcam continuously (~30 FPS latest frame buffer).
- Fast CLI capture: Save latest frame in <100ms.
- **Video capture**: Record short video clips in AVI MJPEG format.
- **Live streaming**: Real-time MJPEG stream via HTTP with web UI.
- **Chrome screenshots**: Capture any Chrome tab via DevTools Protocol (no Puppeteer).
- Multiple clients can stream from the same camera simultaneously.
- Multiple webcams can stream in parallel.
- Adaptive Linux exposure control to keep bright screens readable in dark rooms.
- **Motion detection**: `aeyes motion` watches the stream with a lighting-invariant detector.
- HTTP API for frame, video, and stream capture.
- Linux V4L2 backend. macOS support via nokhwa/AVFoundation.

## Installation

### From crates.io (recommended)
```bash
cargo install aeyes
```

### From source
```bash
git clone https://github.com/elecnix/aeyes.git
cd aeyes
cargo install --path .
```

## Usage

### Start the daemon
```bash
aeyes start                    # Start with auto-selected camera
aeyes start --camera 0         # Start with specific camera
aeyes start --bind 0.0.0.0:43210  # Bind to all interfaces
```

### Capture a frame
```bash
aeyes frame                    # Saves to aeyes-frame.jpg
aeyes frame -o img.jpg         # Custom output path
aeyes frame --camera 0         # Specific camera
```

### Capture a video clip
```bash
aeyes video                                    # 5 second clip at 15 FPS (default)
aeyes video -o clip.avi                        # Custom output path
aeyes video --max-length 10 --fps 30           # 10 second clip at 30 FPS
aeyes video --camera 0 --max-length 3 --fps 24 # Specific camera
```

### HTTP API
```bash
# List cameras
curl http://localhost:43210/cams

# Get a frame
curl -o frame.jpg http://localhost:43210/cams/default/frame

# Capture a video (query params: max_length, fps)
curl -o video.avi http://localhost:43210/cams/default/video?max_length=5&fps=15

# Live stream (MJPEG multipart)
curl http://localhost:43210/cams/default/stream
```

### Live Web UI
Open `http://localhost:43210/web/0` in a browser to see a live view from camera 0.
Use `http://localhost:43210/web/default` for the daemon-selected camera.

Multiple clients can view the same stream simultaneously without affecting each other.

### Inspecting the Webcam
You can view the live webcam feed directly in your browser while the daemon is running:

| Camera | URL |
|--------|-----|
| Camera 0 | http://localhost:43210/web/0 |
| Camera 1 | http://localhost:43210/web/1 |
| Default camera | http://localhost:43210/web/default |

Click any link above or ask me to open the browser for you.

### Status & stop
```bash
aeyes status
aeyes stop
```

### Chrome screenshot

Capture screenshots from any open Chrome tab using the Chrome DevTools Protocol. No Puppeteer or heavy dependencies — just WebSocket.

```bash
aeyes chrome --list-tabs              # List all Chrome tabs
aeyes chrome -o screenshot.jpg        # Capture screenshot (JPEG)
aeyes chrome --quality 95 -o img.jpg  # High quality capture
```

**Requirements**: Chrome must be running with remote debugging enabled:
- **Linux**: Open `chrome://inspect/#remote-debugging` and toggle the switch
- **macOS**: Enabled by default when Chrome is launched via terminal

## Motion detection

`aeyes motion` watches the camera and reports when something changes. The CLI auto-starts the daemon if it is not running, then polls the selected camera's latest frame and runs the detector on every frame it receives.

```bash
aeyes motion                                  # One detection pass, fail if nothing moved
aeyes motion --wait --timeout 60              # Keep sampling for up to 60s
aeyes motion --wait --min-area 200 -o m.jpg   # Require 200 changed pixels, save a visualisation
aeyes motion --threshold 60                   # Less sensitive (higher edge threshold)
aeyes motion --no-lbp                         # Skip the Local Binary Pattern stage
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-o`, `--output <PATH>` | *(none)* | Write a visualisation of the detection to `PATH`. Detected pixels are painted red over the frame. The file is only written when motion is actually detected. |
| `--threshold <0-255>` | `25` | Edge-detection threshold. Higher values mean only strong edges count as motion. |
| `--wait` | off | Keep sampling frames until motion is detected, instead of doing a single pass. |
| `--timeout <SECS>` | `30` | Maximum time to spend waiting when `--wait` is given. Without `--wait` the flag is ignored. |
| `--min-area <PIXELS>` | `10` | Minimum number of detected pixels required before the command reports motion. |
| `--no-sobel` | off (Sobel is on) | Disable the Sobel gradient-magnitude stage. |
| `--no-lbp` | off (LBP is on) | Disable the Local Binary Pattern stage. Faster, less robust to lighting. |

Exit behaviour:

- **Without `--wait`**: one frame is captured and analysed. If at least `--min-area` pixels are detected, the count is printed and the visualisation is written (if `--output` was given), and the command exits `0`. Otherwise the command exits non-zero. A failed frame fetch also exits non-zero.
- **With `--wait`**: the capture/analyse cycle repeats every 100 ms until either `--min-area` pixels are detected (exit `0`) or `--timeout` seconds elapse (exit non-zero, `motion detection timeout after N seconds`).

### How detection works

`src/motion.rs` converts each frame to luminance and then combines three signals:

1. **Temporal edge movement** (Sobel). A pixel is motion if its edge map changed by more than the minimum edge movement since the previous frame *and* its edge magnitude is above the threshold. Edges survive brightness changes, so this stays quiet under changing light.
2. **Local pattern change** (Local Binary Pattern). A pixel is motion if the Hamming distance between its current and previous local binary pattern exceeds the contrast threshold. This catches texture moving through the frame, which matters for scrolling screens.
3. **Adaptive per-region threshold.** The frame is split into 32x32-pixel regions, each with its own sensitivity multiplier that is damped and then decayed after every frame, floored at a minimum sensitivity. A region that keeps triggering therefore gets progressively less sensitive instead of firing forever.

Together these are designed to ignore gradual global brightness changes, local shadows, camera exposure changes and flickering lights, while still reporting objects entering or leaving the scene, people or animals moving, and on-screen content changing.

### Known limitations

- **No pre/post-roll buffer.** Sampling starts when the command is invoked, so there is no recording of the seconds *before* motion was detected and no automatic capture of the seconds after.
- **The detector is not wired to the frame endpoint yet.** `LightingInvariantDetector::detect` expects raw RGB24 at the configured resolution, but `/cams/{id}/frame` serves a JPEG, so the buffer length never matches and the detector returns no detections. In practice `--wait` will run until `--timeout` and then fail. Related work is tracked in [the motion redesign issue](https://github.com/elecnix/aeyes/issues).
- **Resolution is assumed.** The detector is currently constructed for 640x480 regardless of the camera's real resolution.

## Video Format

Videos are saved as **AVI with MJPEG encoding**. This format:
- Works with most video players (VLC, mpv, etc.)
- Requires no external dependencies for encoding
- Can be converted to MP4 with ffmpeg if needed:
  ```bash
  ffmpeg -i input.avi -c:v libx264 output.mp4
  ```

## Testing
```bash
# Run all tests (42 tests)
cargo test --lib

# Run with single thread (avoids port conflicts)
cargo test --lib -- --test-threads=1
```

Tests use a `FakeBackend` that simulates camera capture without requiring actual webcam hardware.

## License
MIT
