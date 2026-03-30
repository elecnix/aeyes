# Deferred Optimization Ideas — Lighting-Invariant Motion Detection

## CPU Performance (FPS)

### ✅ Completed Optimizations
- **Parallelize Sobel edge detection** (Rayon): ~10-15% FPS gain
- **Parallelize LBP computation** (Rayon): ~10% FPS gain  
- **Integer luminance conversion** (fixed-point): small additional gain
- **Sobel magnitude without /2**: slight improvement
- **Disable dilation** (HashSet overhead): ~6% FPS gain at cost of ~3% F1
- **LBP skip odd frames**: +5% F1 improvement (and ~same FPS)

### 🎯 I/O Bottleneck (PRIMARY)
The main bottleneck is JPEG decoding (~85% of time), NOT detection algorithm:
- Current throughput: ~0.16 FPS (2 frames per sequence)
- Parallelizing frame decoding DID NOT HELP (image crate is already threaded internally)
- **Tried and Failed:**
  - Parallel frame pre-decoding with Rayon (marginal/slower)
  - The `image` crate likely serializes internally
  
### 📊 Algorithmic Speedups (Explored, Diminishing Returns)
- **Edge threshold increases**: F1 dropped below 0.90, unacceptable
- **LBP disable**: F1 dropped to 0.77, unacceptable
- **Parallel detection loop**: SLOWED DOWN (0.143→0.111 FPS)
- **Dilation disable**: FPS +6%, F1 -3% (acceptable trade-off)

### 💡 Potential Future Work (High Impact)
1. **Faster JPEG decoder**: Use `jpeg-decoder` crate (pure Rust, SIMD) or mozjpeg FFI
2. **Pre-convert CDnet to raw RGB**: Eliminated JPEG entirely; failed previously due to loader issue
3. **SIMD for Sobel**: `packed_simd` crate for vectorized gradients
4. **Resolution downscaling**: 2x downscale = 4x fewer pixels to process

## Accuracy (F1)

### ✅ Completed Optimizations
- **Edge threshold 1** (ultra-sensitive): minimal baseline
- **Min edge movement 0**: captures any positive edge delta
- **Contrast threshold 1**: maximum LBP sensitivity
- **8-connected dilation**: fills gaps, +10% recall (when enabled)
- **LBP skip odd frames**: +5% F1 improvement (recommended)

### 🎯 Best Configurations

**Accuracy (F1=0.966):**
```
edge_threshold=1, min_edge_movement=0, contrast_threshold=1, 
dilate=false, use_lbp=true, skip_lbp_odd_frames=true
```

**Speed/Accuracy Trade-off (F1=0.908, FPS=0.179):**
```
edge_threshold=1, min_edge_movement=0, contrast_threshold=1,
dilate=false, use_lbp=true, skip_lbp_odd_frames=false
```

## Dataset & Benchmark

### CDnet 2014 Evaluation
- 6 categories × ~4-6 sequences × ~1000 frames each ≈ 70k frames
- Full evaluation at 0.16 FPS would take ~5 days
- Practical: use `CDNET_MAX_FRAMES=2` for quick iteration

### Pre-conversion (Failed - Re-try Later)
- Created raw/ directories with binary RGB data
- Evaluation binary couldn't load .raw files (image crate doesn't support raw format)
- **To re-try**: implement custom raw loader using `std::fs::read` + `image::RgbImage::from_raw`
