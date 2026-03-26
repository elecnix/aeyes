#!/usr/bin/env bash
# Run CDnet 2014 evaluation for motion detection
#
# Environment variables:
# - CDNET_DOWNLOAD: set to 1 to download dataset if not cached (default: 1)
# - CDNET_CACHE_DIR: path to cache directory (default: ./data)
# - CDNET_OUTPUT_DIR: path for results (default: ./results)
# - CDNET_MAX_FRAMES: limit frames per sequence for quick testing (default: 0 = all)
#
# Example:
#   CDNET_MAX_FRAMES=100 ./run.sh   # quick test with 100 frames per sequence

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default to download if not explicitly disabled
: "${CDNET_DOWNLOAD:=1}"

cargo run --release --bin cdnet-evaluation
