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
- **Motion detection in the daemon**: one detector per camera, shared by every client.
- Adaptive Linux exposure control to keep bright screens readable in dark rooms.
- HTTP API for frame, video, stream, motion verdicts and motion events.
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

Motion detection settings are daemon settings, because they change what is
measured rather than what a client is told:

```bash
aeyes start --motion-hz 10          # verdicts per second (1-60, default 10)
aeyes start --motion-width 320      # width of the analysis frame (16-4096, default 320)
aeyes start --motion-threshold 25   # edge threshold 0-255 (default 25)
aeyes start --motion-no-sobel       # disable the Sobel (edge magnitude) stage
aeyes start --motion-no-lbp         # disable the LBP (local texture) stage
```

The same settings can be supplied as environment variables, which is what the
daemon process itself reads: `AEYES_MOTION_HZ`, `AEYES_MOTION_WIDTH`,
`AEYES_MOTION_THRESHOLD`, `AEYES_MOTION_SOBEL`, `AEYES_MOTION_LBP`.

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

# Latest motion verdict (JSON)
curl http://localhost:43210/cams/default/motion

# Motion verdict stream (one JSON verdict per line, NDJSON)
curl -N http://localhost:43210/cams/default/events
curl -N "http://localhost:43210/cams/default/events?min_area=20"   # only verdicts at or above 20 changed pixels
```

`GET /cams/{id}/motion` waits for and returns a verdict produced *after* the
request arrived, and registers the caller as a watcher for the duration of the
request. `GET /cams/{id}/events` streams every verdict as it is published, plus
an optional server-side emission filter: `?min_area=N` skips verdicts below `N`
changed pixels (`0`, the default, emits everything). Both endpoints accept
`default` for the daemon-selected camera, exactly like `/frame`, `/video` and
`/stream`.

A verdict looks like this:

```json
{"camera":"0","sequence":42,"detected":true,"changed_pixels":318,
 "bbox":{"x":96,"y":74,"width":41,"height":63},
 "frame_width":1280,"frame_height":720,"detector_width":320,"detector_height":180,
 "timestamp_ms":1767225600123,"status":"ok"}
```

`sequence` increases with every verdict, so a client can tell "no new verdict"
from "no motion". `bbox` and `changed_pixels` are in **analysis** pixels
(`detector_width` x `detector_height`), while `frame_width`/`frame_height` are
the camera's real decoded resolution. `status` is `ok`, `decode_error` or
`geometry_mismatch`: a frame that could not be analysed is reported as such
rather than as `detected: false`.

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

### Watch for motion
```bash
aeyes motion                           # one measurement of the current scene
aeyes motion --wait --timeout 60        # keep watching for up to a minute
aeyes motion --wait --min-area 20       # only counts as motion at 20+ changed pixels
aeyes motion --wait -o still.jpg        # save a still, with the motion bounds outlined
aeyes motion --wait --clip event.avi    # 2s before and 2s after the change
aeyes motion --wait --clip event.avi --pre 5 --post 3
aeyes motion --wait --frames out/       # one JPEG per frame around the change
```

The command is a thin client: the daemon owns the detector, and `aeyes motion`
only reads its verdicts. `--min-area` is applied locally, over the
`changed_pixels` the daemon published, so two clients watching the same camera
with different `--min-area` values do not need to agree.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Motion passed the filter |
| 1 | No motion within the budget (a fulfilled contract, not an error) |
| >= 2 | The command failed (daemon unreachable, stream died, file unwritable) |

```bash
aeyes motion --wait --timeout 30 -o still.jpg && echo "something moved"
```

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

## Video Format

Videos are saved as **AVI with MJPEG encoding**. This format:
- Works with most video players (VLC, mpv, etc.)
- Requires no external dependencies for encoding
- Can be converted to MP4 with ffmpeg if needed:
  ```bash
  ffmpeg -i input.avi -c:v libx264 output.mp4
  ```

## Known limitations

Detection runs only while at least one client is subscribed to a camera's
verdicts, so an idle daemon does no decoding work at all. `GET /cams/{id}/motion`
counts as a subscription for the length of its request and `GET
/cams/{id}/events` for the life of its response, so the CLI always wakes the
detector; a daemon with no clients keeps its last verdict rather than updating
it.

`aeyes motion --clip` (and `--frames`) records a pre/post-roll around a
*detected change*, not a continuously running video. Pre-roll is best-effort:
the client keeps the frames the daemon's stream actually delivered, so a
backpressured stream shortens the effective pre-roll window. The ring is capped
at 240 frames and 64 MiB, so on a high-resolution or high-frame-rate camera the
effective pre-roll is shorter than `--pre` seconds - it fails safe with a
shorter clip rather than by growing the process.

`--min-area` is in **analysis** pixels, not camera pixels, and the analysis
frame is `--motion-width` wide (320 by default). Downscaling a 1280x720 camera
to 320x180 shrinks the count for the same motion by roughly 16x, so a value
tuned for full resolution will under-trigger. Check the verdict's
`detector_width`/`detector_height` to see what the number is measured in.

Motion detection is a lighting-invariant heuristic, not a classifier: it reports
where pixels changed, not what changed. Detector accuracy on decoded JPEG frames
is unmeasured - the thresholds were tuned against raw RGB, and JPEG is lossy
with chroma subsampling that can perturb edges.

## Testing
```bash
# Run all tests
cargo test

# Run with single thread (avoids port conflicts)
cargo test --lib -- --test-threads=1
```

Tests use a `FakeBackend` that simulates camera capture without requiring actual webcam hardware.

## License
MIT
