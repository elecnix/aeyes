# CDnet 2014 Motion Detection Evaluation

This test suite evaluates the `LightingInvariantDetector` on the standard [CDnet 2014](http://jacarini.dinf.usherbrooke.ca/dataset2014) change detection benchmark.

## Dataset

- **Download**: `scripts/download-cdnet.sh` fetches the official [CDnet 2014](https://changedetection.net/dataset2014) archive (`dataset2014.zip`, 3,918,467,615 bytes, SHA-256 `05c6d0dd64858b094cb0d309e5b63be42de52f2b5c5adff4a65e6a640dfd2a84`)
- **Categories**: baseline, cameraJitter, dynamicBackground, intermittentObjectMotion, shadow, thermal
- **Ground truth**: Pixel-wise binary masks for each frame

No part of the dataset is committed: it is third-party benchmark data, and every frame in it is reproducible from the official archive, so the repository only carries the code that consumes it. Fetch the data once with:

```bash
scripts/download-cdnet.sh
```

The script downloads the archive, verifies its checksum and unpacks it into `tests/cdnet_evaluation/data` (override with `--dest DIR` or `$CDNET_CACHE_DIR`). It is idempotent, and it keeps the archive beside the extracted frames so the evaluation below does not download it again.

When `tests/cdnet_evaluation/data/dataset2014.zip` is missing, the evaluation downloads the archive itself (`CDNET_DOWNLOAD=1`, the default), so `./run.sh` still works on a fresh checkout without the script.

### Pre-converted raw frames

Reading frames straight from the JPEGs is I/O bound, and that dominated the benchmark runs recorded in `results/cdnet_report.json`. Running the evaluation once with `CDNET_PRECONVERT=1` writes a headerless RGB copy of every `input/*.jpg` into a `raw/` directory next to it and uses those from then on. `raw/` is a deterministic function of the archive content and is not committed either; it lives under `tests/cdnet_evaluation/data`, which is git-ignored.

## Usage

### Run full evaluation

```bash
cd tests/cdnet_evaluation
./run.sh
```

Or directly:

```bash
cargo run --release --bin cdnet-evaluation
```

### Quick test (subset)

To run on a limited number of frames (e.g., first 100 frames of each sequence) for faster iteration:

```bash
CDNET_MAX_FRAMES=100 ./run.sh
```

### Custom cache/output directories

```bash
export CDNET_CACHE_DIR=/path/to/cache
export CDNET_OUTPUT_DIR=/path/to/results
./run.sh
```

### Force re-download

Delete the `data` directory, run `scripts/download-cdnet.sh --force`, or set `CDNET_DOWNLOAD=1` (it will re-download if the zip is missing).

## Output

- **Report**: `tests/cdnet_evaluation/results/cdnet_report.json` contains:
  - Per-category precision, recall, F1-score
  - Per-sequence metrics
  - Overall metrics
  - Configuration used
- **Console summary**: Human-readable table with F1, precision, recall

## Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-score**: 2 * P * R / (P + R)

Pixels are evaluated independently against ground truth binary masks.

## CI/CD

A GitHub Actions workflow (`.github/workflows/cdnet_evaluation.yml`) runs this evaluation:
- Manual trigger available via "Run workflow"
- Runs on the self-hosted Hetzner ARC runner pool (label `arc-runners`)
- The archive is fetched by `scripts/download-cdnet.sh` and cached between runs (key `cdnet-dataset-v1`)
- Results are uploaded as artifacts

## Notes

- The detector uses default configuration. Future experiments can adjust parameters via code.
- Evaluation processes ~70,000 annotated frames. Expect ~1-2 hours on typical hardware.
- Ensure sufficient disk space (~5 GB for dataset + results).
