# A-Eyes 🖼️

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-blue.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/badge/crates.io-aeyes-orange.svg)](https://crates.io/crates/aeyes)

**A-Eyes** (AI's eyes) is a CLI daemon that keeps your webcam open for **instant captures** without auto-exposure/focus delays. Perfect for AI agents needing quick "eyes".

## Features
- Daemon mode: Open webcam continuously (~30 FPS latest frame buffer).
- Fast CLI capture: Save latest frame in &lt;100ms.
- Adaptive Linux exposure control to keep bright screens readable in dark rooms.
- Unix socket IPC (/tmp/aeyes.sock).
- Linux V4L2 (nokhwa). Pluggable for Windows/Mac.

## Installation
```bash
git clone https://github.com/elecnix/aeyes.git
cd aeyes
cargo install --path .
```

## Usage
```bash
# Start daemon (forks to background)
aeyes start

# Capture latest frame
aeyes capture          # Saves to /tmp/aeyes-last.jpg
aeyes capture -o img.jpg

# Status & stop
aeyes status
aeyes stop
```

## Testing
```bash
cargo run -- start &amp;
sleep 2
cargo run -- capture /tmp/test.jpg
ls -la /tmp/test.jpg  # Verify image saved
```

## Roadmap
- Unit/integration tests (&gt;80% coverage) [#1](https://github.com/elecnix/aeyes/issues/1) [#2](https://github.com/elecnix/aeyes/issues/2)
- GitHub Actions CI [#3](https://github.com/elecnix/aeyes/issues/3)
- Pluggable backends (Win/Mac) [#4](https://github.com/elecnix/aeyes/issues/4)
- Cross-platform [#5](https://github.com/elecnix/aeyes/issues/5)

## License
MIT
