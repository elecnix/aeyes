# A-Eyes 👀

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
- Multiple clients can stream from the same camera simultaneously.
- Multiple webcams can stream in parallel.
- Adaptive Linux exposure control to keep bright screens readable in dark rooms.
- HTTP API for frame, video, and stream capture.
- Linux V4L2 backend. Pluggable for Windows/Mac.

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
