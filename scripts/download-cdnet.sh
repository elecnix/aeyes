#!/usr/bin/env bash
#
# Fetch the CDnet 2014 change-detection dataset used by the motion-detection
# evaluation in tests/cdnet_evaluation.
#
# The dataset is third-party benchmark data (3.65 GiB compressed, ~4 GiB
# unpacked), so no part of it is stored in this repository. This script
# downloads the official archive, checks it against a pinned checksum, and
# unpacks it into the cache directory the evaluation reads. The pre-converted
# raw/ frames that the evaluation can consume are a deterministic re-encoding
# of the input/ frames inside this archive, so they need not be stored either.
#
# Usage:
#   scripts/download-cdnet.sh [--dest DIR] [--force]
#
# Environment:
#   CDNET_CACHE_DIR   destination directory (default: tests/cdnet_evaluation/data)
#
# Idempotent: exits 0 without touching the network when the dataset is already
# unpacked and the archive still matches the pinned checksum. Pass --force to
# re-download and re-extract anyway.
#
set -euo pipefail

usage() {
	cat <<'EOF'
Fetch the CDnet 2014 dataset used by tests/cdnet_evaluation.

Usage:
  scripts/download-cdnet.sh [--dest DIR] [--force]

Options:
  --dest DIR   cache directory (default: tests/cdnet_evaluation/data,
               or $CDNET_CACHE_DIR when set)
  --force      re-download and re-extract even if the dataset is already there
  -h, --help   show this help

The archive is verified against a pinned SHA-256 before it is unpacked.
EOF
}

# Official archive, published 2014-01-15: 11 categories, 3,918,467,615 bytes.
# The second URL is the same file on the lab's other host.
CDNET_URLS=(
	"https://changedetection.vitalab.usherbrooke.ca/static/dataset/dataset2014.zip"
	"https://changedetection.net/static/dataset/dataset2014.zip"
)
CDNET_SIZE=3918467615
CDNET_SHA256=05c6d0dd64858b094cb0d309e5b63be42de52f2b5c5adff4a65e6a640dfd2a84

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dest="${CDNET_CACHE_DIR:-$repo_root/tests/cdnet_evaluation/data}"
force=0

while [ $# -gt 0 ]; do
	case "$1" in
		--dest)
			[ $# -ge 2 ] || { echo "error: --dest needs a directory" >&2; exit 2; }
			dest="$2"
			shift 2
			;;
		--force)
			force=1
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			echo "error: unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

zip="$dest/dataset2014.zip"
extracted="$dest/CDnet2014/dataset"

sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | cut -d' ' -f1
	else
		shasum -a 256 "$1" | cut -d' ' -f1
	fi
}

# Cheap check first (existence + size), then the full checksum.
archive_ok() {
	[ -f "$zip" ] || return 1
	[ "$(wc -c <"$zip")" -eq "$CDNET_SIZE" ] || return 1
	[ "$(sha256_of "$zip")" = "$CDNET_SHA256" ]
}

if [ "$force" -eq 0 ] && [ -d "$extracted" ] && archive_ok; then
	echo "✓ CDnet 2014 already present: $extracted"
	echo "  archive verified: $zip"
	exit 0
fi

mkdir -p "$dest"

if [ "$force" -eq 1 ] || ! archive_ok; then
	downloaded=0
	for url in "${CDNET_URLS[@]}"; do
		echo "-- downloading $url"
		# -C - resumes a partial file, so a re-run does not start from zero.
		if curl -fL --retry 3 --retry-delay 5 -C - -o "$zip" "$url"; then
			downloaded=1
			break
		fi
		echo "   download failed, trying the next mirror" >&2
	done
	if [ "$downloaded" -eq 0 ]; then
		echo "error: could not download the CDnet 2014 archive." >&2
		echo "       Download dataset2014.zip from https://changedetection.net/dataset2014" >&2
		echo "       by hand and place it at $zip, then re-run this script." >&2
		exit 1
	fi
fi

actual_size=$(wc -c <"$zip")
if [ "$actual_size" -ne "$CDNET_SIZE" ]; then
	echo "error: $zip is $actual_size bytes, expected $CDNET_SIZE." >&2
	rm -f "$zip"
	exit 1
fi
actual_sha=$(sha256_of "$zip")
if [ "$actual_sha" != "$CDNET_SHA256" ]; then
	echo "error: checksum mismatch for $zip" >&2
	echo "       expected $CDNET_SHA256" >&2
	echo "       got      $actual_sha" >&2
	rm -f "$zip"
	exit 1
fi
echo "✓ archive checksum verified ($CDNET_SHA256)"

if [ "$force" -eq 1 ] || [ ! -d "$extracted" ]; then
	echo "-- extracting into $dest/CDnet2014"
	if command -v unzip >/dev/null 2>&1; then
		unzip -q -o "$zip" -d "$dest/CDnet2014"
	elif command -v 7z >/dev/null 2>&1; then
		7z x -y -o"$dest/CDnet2014" "$zip" >/dev/null
	else
		echo "error: neither unzip nor 7z is available to extract $zip" >&2
		exit 1
	fi
fi

if [ ! -d "$extracted" ]; then
	echo "error: extraction finished but $extracted is missing." >&2
	exit 1
fi

categories=$(find "$extracted" -mindepth 1 -maxdepth 1 -type d | wc -l)
sequences=$(find "$extracted" -mindepth 2 -maxdepth 2 -type d | wc -l)
echo "✓ CDnet 2014 ready: $categories categories, $sequences sequences"
echo "  dataset: $extracted"
echo "  archive: $zip (kept, so the evaluation does not download it again)"
echo
echo "Next steps:"
echo "  tests/cdnet_evaluation/run.sh                       # evaluate on the input/ frames"
echo "  CDNET_PRECONVERT=1 tests/cdnet_evaluation/run.sh   # same, after building the raw/ cache"
