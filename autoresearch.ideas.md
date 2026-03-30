# Deferred Optimization Ideas — Lighting-Invariant Motion Detection

## CPU Performance (FPS)

### ✅ Completed Optimizations
- **Parallelize Sobel edge detection** (Rayon): ~10-15% FPS gain
- **Parallelize LBP computation** (Rayon): ~10% FPS gain  
- **Integer luminance conversion** (fixed-point): small additional gain
- **Sobel magnitude without /2**: slight improvement
- **Disable dilation** (HashSet overhead): ~6% FPS gain at cost of ~3% F1

### 🎯 I/O Bottleneck (PRIMARY)
The main bottleneck is JPEG decoding, not detection algorithm:
- Current throughput: ~0.16 FPS (2 frames per sequence)
- JPEG decoding dominates wall-clock time
- **Solutions (deferred):**
  - Pre-convert CDnet dataset to raw RGB binary format (3x smaller, instant load)
  - Use RAM disk for dataset during evaluation
  - Use faster JPEG decoder (mozjpeg via FFI, or `jpeg-decoder` crate)
  - For production use: daemon captures raw RGB directly (no JPEG overhead)

### 📊 Algorithmic Speedups (Explored)
- **Skip LBP on odd frames**: Surprisingly IMPROVED F1 (0.913→0.966), FPS slightly lower
  - Theory: skipping reduces texture noise false positives on alternating frames
  - Trade-off: may miss very fast texture changes between skipped frames
  - **Recommended for accuracy over speed**
  
- **Parallel detection loop**: SLOWED DOWN (0.143→0.111 FPS)
  - Parallel mask approach had too much overhead for small frames
  - Sequential detection is cache-friendly; parallelizing adds overhead

- **Edge threshold increases**: F1 dropped below 0.90, unacceptable

- **LBP disable**: F1 dropped to 0.77, unacceptable

### 💡 Potential Future Work
1. **Batch frame decoding**: Pre-load N frames in parallel, then process sequentially
2. **SIMD acceleration**: Use `std::simd` or `packed_simd` for Sobel kernels
3. **Resolution scaling**: Downscale for detection, upscale for output
4. **Adaptive skip**: Skip frames when motion is low, process more when high

## Accuracy (F1)

### ✅ Completed Optimizations
- **Edge threshold 1** (ultra-sensitive): minimal baseline
- **Min edge movement 0**: captures any positive edge delta
- **Contrast threshold 1**: maximum LBP sensitivity
- **8-connected dilation**: fills gaps, +10% recall
- **LBP every odd frame**: +5% F1 improvement

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
- For production benchmarking: pre-convert to raw, use RAM disk

### Pre-conversion (Failed Attempt)
- Created raw/ directories with binary RGB data
- Evaluation binary couldn't load .raw files (image crate doesn't support raw format)
- **Deferred**: implement proper raw loader with dimensions metadata
