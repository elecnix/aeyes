# CDnet 2014 Motion Detection Evaluation

This test suite evaluates the `LightingInvariantDetector` on the standard [CDnet 2014](http://jacarini.dinf.usherbrooke.ca/dataset2014) change detection benchmark.

## Dataset

- **Download**: [CDnet 2014](http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip) (3.6 GB)
- **Categories**: baseline, cameraJitter, dynamicBackground, intermittentObjectMotion, shadow, thermal
- **Ground truth**: Pixel-wise binary masks for each frame

The dataset is automatically downloaded and cached when first run (set `CDNET_DOWNLOAD=1`). The cache is stored in `tests/cdnet_evaluation/data` by default.

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

Delete the `data` directory or set `CDNET_DOWNLOAD=1` (it will re-download if the zip is missing).

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
- Dataset is cached between runs (key `cdnet-dataset-v1`)
- Results are uploaded as artifacts

## Notes

- The detector uses default configuration. Future experiments can adjust parameters via code.
- Evaluation processes ~70,000 annotated frames. Expect ~1-2 hours on typical hardware.
- Ensure sufficient disk space (~5 GB for dataset + results).
