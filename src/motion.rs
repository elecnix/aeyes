/// Lighting-Invariant Motion Detection for aeyes
///
/// Detects motion while being robust to global and local lighting changes.
/// Key techniques:
/// 1. Edge detection (Sobel) - edges stable across brightness changes
/// 2. Local contrast (Local Binary Patterns) - invariant to monotonic lighting
/// 3. Temporal edge tracking - motion = edge movement
/// 4. Adaptive thresholding - per-region baselines
///
/// Robust to:
/// - Gradual global brightness changes (sun moving)
/// - Local shadows (clouds, objects passing)
/// - Camera exposure changes
/// - Flickering lights
///
/// Still detects:
/// - Objects entering/leaving scene
/// - People/animals moving
/// - Screen changes (pixel-level motion)
use std::cmp;

/// Smallest frame dimension that still has an interior pixel whose full 3x3
/// neighbourhood lies inside the image. Both the Sobel and the LBP stage scan
/// `1..(dim - 1)`, which underflows on a zero dimension, and there is nothing
/// to scan below this size anyway.
const MIN_SCAN_DIMENSION: usize = 3;

#[derive(Clone, Debug)]
pub struct LightingInvariantConfig {
    /// Edge detection threshold (0-255)
    /// Higher = only strong edges trigger motion
    pub edge_threshold: u8,

    /// Minimum edge movement distance (pixels)
    /// Prevents noise from triggering detections
    pub min_edge_movement: u8,

    /// Local contrast threshold, in **Hamming-distance units over the 8-bit
    /// LBP pattern**, i.e. the valid range is `0..=LocalBinaryPattern::PATTERN_BITS`
    /// (0..=8).
    ///
    /// A pixel counts as texture motion when the Hamming distance between its
    /// current and previous local binary pattern is *greater* than this value.
    /// Because the distance can never exceed the pattern width, a threshold
    /// above `PATTERN_BITS` makes the whole LBP stage unreachable.
    pub contrast_threshold: u8,

    /// Decay rate for edge maps (same as luminance)
    pub decay_rate: f32,

    /// Floor on each region's adaptive gain, in `0.0..=1.0`.
    ///
    /// The gain multiplies into the detection bar as
    /// `effective_threshold = edge_threshold / gain`: a gain of `1.0` is the
    /// unadapted base threshold, and a gain *below* `1.0` makes the region
    /// progressively **less** sensitive, which is what stops a busy region
    /// from firing forever. Raising this floor therefore makes the detector
    /// **more** sensitive, and lowering it lets a region quiet down further —
    /// it only ever bounds how far adaptation may move the threshold away
    /// from the base threshold.
    pub min_sensitivity: f32,

    /// Enable temporal edge tracking
    pub use_temporal_edges: bool,

    /// Enable local binary patterns (more robust but slower)
    pub use_lbp: bool,

    /// Enable gradient magnitude (Sobel edges)
    pub use_sobel: bool,
}

impl Default for LightingInvariantConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 25,
            min_edge_movement: 2,
            contrast_threshold: 4,
            decay_rate: 0.96,
            min_sensitivity: 0.3,
            use_temporal_edges: true,
            use_lbp: true,
            use_sobel: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecayMetrics {
    pub sensitivity: f32,
    pub current_threshold: u8,
    pub edge_count: u64,
    pub detection_count: u64,
}

/// Sobel edge detector - extremely fast, lighting invariant
/// Detects pixel intensity gradients (edges)
pub struct SobelEdgeDetector {
    width: usize,
    height: usize,
    edges: Vec<u8>,
    prev_edges: Vec<u8>,
}

impl SobelEdgeDetector {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            edges: vec![0; size],
            prev_edges: vec![0; size],
        }
    }

    /// Compute horizontal gradient (Sobel Gx)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn sobel_gx(
        _topleft: u8,
        _top: u8,
        topright: u8,
        left: u8,
        right: u8,
        botleft: u8,
        _bot: u8,
        botright: u8,
    ) -> i16 {
        -(_topleft as i16) + topright as i16 - 2 * (left as i16) + 2 * (right as i16)
            - (botleft as i16)
            + botright as i16
    }

    /// Compute vertical gradient (Sobel Gy)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn sobel_gy(
        topleft: u8,
        top: u8,
        topright: u8,
        _left: u8,
        _right: u8,
        botleft: u8,
        bot: u8,
        botright: u8,
    ) -> i16 {
        topleft as i16 + 2 * top as i16 + topright as i16
            - botleft as i16
            - 2 * bot as i16
            - botright as i16
    }

    /// Detect edges in luminance frame
    /// Returns edge magnitude map
    pub fn detect(&mut self, lum_frame: &[u8]) -> &[u8] {
        if lum_frame.len() != self.width * self.height {
            return &self.edges;
        }

        self.prev_edges.copy_from_slice(&self.edges);

        // Degenerate frames have no interior pixel whose 3x3 neighbourhood is
        // fully inside the image; bail out before the loop bounds below would
        // underflow on a zero width or height.
        if self.width < MIN_SCAN_DIMENSION || self.height < MIN_SCAN_DIMENSION {
            return &self.edges;
        }

        // Process interior pixels (skip borders)
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let idx = y * self.width + x;

                // 3x3 neighborhood
                let tl = lum_frame[(y - 1) * self.width + (x - 1)];
                let tm = lum_frame[(y - 1) * self.width + x];
                let tr = lum_frame[(y - 1) * self.width + (x + 1)];

                let ml = lum_frame[y * self.width + (x - 1)];
                let mr = lum_frame[y * self.width + (x + 1)];

                let bl = lum_frame[(y + 1) * self.width + (x - 1)];
                let bm = lum_frame[(y + 1) * self.width + x];
                let br = lum_frame[(y + 1) * self.width + (x + 1)];

                // Sobel gradients
                let gx = Self::sobel_gx(tl, tm, tr, ml, mr, bl, bm, br);
                let gy = Self::sobel_gy(tl, tm, tr, ml, mr, bl, bm, br);

                // Magnitude: sqrt(gx² + gy²) ≈ |gx| + |gy| (faster)
                let magnitude = (gx.abs() + gy.abs()) / 2;
                self.edges[idx] = cmp::min(255, magnitude as u8);
            }
        }

        &self.edges
    }

    /// Get previous edge map for temporal comparison
    pub fn prev_edges(&self) -> &[u8] {
        &self.prev_edges
    }
}

/// Local Binary Pattern - texture descriptor invariant to monotonic lighting
/// Compares each pixel to neighbors: bright neighbor = 1 bit
pub struct LocalBinaryPattern {
    width: usize,
    height: usize,
    patterns: Vec<u8>,
    prev_patterns: Vec<u8>,
}

impl LocalBinaryPattern {
    /// Width of a single LBP code, in bits. This is the largest value
    /// [`LocalBinaryPattern::hamming_distance`] can ever return, and therefore
    /// the largest meaningful value for `LightingInvariantConfig::contrast_threshold`.
    pub const PATTERN_BITS: u8 = 8;

    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            patterns: vec![0; size],
            prev_patterns: vec![0; size],
        }
    }

    /// Compute LBP for single pixel (8-bit pattern)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn compute_lbp(
        center: u8,
        tl: u8,
        tm: u8,
        tr: u8,
        ml: u8,
        mr: u8,
        bl: u8,
        bm: u8,
        br: u8,
    ) -> u8 {
        let mut pattern = 0u8;
        pattern |= if tl > center { 1 << 0 } else { 0 };
        pattern |= if tm > center { 1 << 1 } else { 0 };
        pattern |= if tr > center { 1 << 2 } else { 0 };
        pattern |= if mr > center { 1 << 3 } else { 0 };
        pattern |= if br > center { 1 << 4 } else { 0 };
        pattern |= if bm > center { 1 << 5 } else { 0 };
        pattern |= if bl > center { 1 << 6 } else { 0 };
        pattern |= if ml > center { 1 << 7 } else { 0 };
        pattern
    }

    /// Compute LBP for entire frame
    pub fn compute(&mut self, lum_frame: &[u8]) {
        if lum_frame.len() != self.width * self.height {
            return;
        }

        self.prev_patterns.copy_from_slice(&self.patterns);

        // Degenerate frames have no interior pixel whose 3x3 neighbourhood is
        // fully inside the image; bail out before the loop bounds below would
        // underflow on a zero width or height.
        if self.width < MIN_SCAN_DIMENSION || self.height < MIN_SCAN_DIMENSION {
            return;
        }

        // Process interior pixels
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let idx = y * self.width + x;
                let center = lum_frame[idx];

                let tl = lum_frame[(y - 1) * self.width + (x - 1)];
                let tm = lum_frame[(y - 1) * self.width + x];
                let tr = lum_frame[(y - 1) * self.width + (x + 1)];

                let ml = lum_frame[y * self.width + (x - 1)];
                let mr = lum_frame[y * self.width + (x + 1)];

                let bl = lum_frame[(y + 1) * self.width + (x - 1)];
                let bm = lum_frame[(y + 1) * self.width + x];
                let br = lum_frame[(y + 1) * self.width + (x + 1)];

                self.patterns[idx] = Self::compute_lbp(center, tl, tm, tr, ml, mr, bl, bm, br);
            }
        }
    }

    /// Get pattern at pixel
    #[inline]
    pub fn pattern(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.patterns[y * self.width + x]
        } else {
            0
        }
    }

    /// Get previous pattern at pixel
    #[inline]
    pub fn prev_pattern(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.prev_patterns[y * self.width + x]
        } else {
            0
        }
    }

    /// Hamming distance between two patterns (how different)
    ///
    /// Bounded by [`LocalBinaryPattern::PATTERN_BITS`] because a pattern is a
    /// single byte: at most 8 bits can differ.
    #[inline]
    fn hamming_distance(p1: u8, p2: u8) -> u8 {
        debug_assert!(
            (p1 ^ p2).count_ones() <= u32::from(Self::PATTERN_BITS),
            "LBP patterns are {PATTERN_BITS}-bit codes",
            PATTERN_BITS = Self::PATTERN_BITS
        );
        (p1 ^ p2).count_ones() as u8
    }
}

/// Adaptive threshold per region
/// Different areas can have different base lighting
pub struct AdaptiveThreshold {
    width: usize,
    region_size: usize,
    sensitivities: Vec<f32>,
    decay_rate: f32,
    min_sensitivity: f32,
}

impl AdaptiveThreshold {
    pub fn new(
        width: usize,
        height: usize,
        region_size: usize,
        decay_rate: f32,
        min_sensitivity: f32,
    ) -> Self {
        let regions_x = width.div_ceil(region_size);
        let regions_y = height.div_ceil(region_size);

        Self {
            width,
            region_size,
            sensitivities: vec![1.0; regions_x * regions_y],
            decay_rate,
            min_sensitivity,
        }
    }

    /// Get threshold for pixel location.
    ///
    /// `sensitivities[region]` is a *gain*, not a threshold ratio:
    /// `effective_threshold = base_threshold / gain`. A gain of `1.0` is the
    /// unadapted base threshold; damping (`register_detection`) and decay
    /// (`apply_decay`) push the gain below `1.0`, which raises the bar and
    /// makes a repeatedly-firing region progressively *less* sensitive.
    /// `min_sensitivity` is the floor on that gain, so it bounds how far
    /// adaptation may move the threshold up from the base threshold - it can
    /// never push the threshold below the base value.
    pub fn threshold_at(&self, x: usize, y: usize, base_threshold: u8) -> u8 {
        let region_x = x / self.region_size;
        let region_y = y / self.region_size;

        let regions_x = self.width.div_ceil(self.region_size);
        let idx = region_y * regions_x + region_x;

        // Keep the floor inside `0.0..=1.0` so an out-of-range config value can
        // neither divide by zero nor invert the gain into a multiplier that
        // would *lower* the bar below the base threshold.
        let floor = self.min_sensitivity.clamp(f32::EPSILON, 1.0);
        let sensitivity = self
            .sensitivities
            .get(idx)
            .copied()
            .unwrap_or(1.0)
            .clamp(floor, 1.0);

        let adjusted = (base_threshold as f32 / sensitivity) as u8;
        adjusted.max(base_threshold)
    }

    /// Register detection in region
    ///
    /// Damping pushes the region's gain below `1.0`, which raises its
    /// effective threshold and makes it less sensitive, so a region that keeps
    /// firing stops firing forever. `min_sensitivity` floors the gain: it is an
    /// `f32`-to-`f32` comparison on the same scale the gain lives in, so the
    /// gain can never fall below it.
    pub fn register_detection(&mut self, x: usize, y: usize) {
        let region_x = x / self.region_size;
        let region_y = y / self.region_size;

        let regions_x = self.width.div_ceil(self.region_size);
        let idx = region_y * regions_x + region_x;

        if let Some(threshold) = self.sensitivities.get_mut(idx) {
            *threshold *= 0.85; // Damping: less sensitive after a detection
            *threshold = threshold.max(self.min_sensitivity);
        }
    }

    /// Mean gain across all regions, or `None` when there are no regions.
    ///
    /// The per-region gains are the detector's real sensitivity state, so
    /// there is no mean to report for a zero-sized frame: a `0.0` here would
    /// be indistinguishable from a fully damped detector, and inventing a
    /// number would misreport it. Callers that need a value must choose what
    /// "no regions" means for them.
    pub fn mean_sensitivity(&self) -> Option<f32> {
        if self.sensitivities.is_empty() {
            return None;
        }
        Some(self.sensitivities.iter().sum::<f32>() / self.sensitivities.len() as f32)
    }

    /// Apply decay to all regions
    ///
    /// Same floor as `register_detection`: the decay pulls each gain back up
    /// towards `min_sensitivity` so adaptation never becomes unbounded.
    pub fn apply_decay(&mut self) {
        for threshold in self.sensitivities.iter_mut() {
            *threshold *= self.decay_rate; // Gain relaxes back towards 1.0
            *threshold = threshold.max(self.min_sensitivity);
        }
    }
}

/// Lighting-invariant motion detector
pub struct LightingInvariantDetector {
    width: usize,
    height: usize,
    config: LightingInvariantConfig,

    // Image processing
    lum_frame: Vec<u8>,
    prev_lum_frame: Vec<u8>,

    // Edge detection
    sobel: SobelEdgeDetector,

    // Texture analysis
    lbp: LocalBinaryPattern,

    // Adaptive thresholding
    adaptive_threshold: AdaptiveThreshold,

    // Metrics
    frame_count: u64,
    detection_count: u64,
}

impl LightingInvariantDetector {
    pub fn new(width: usize, height: usize, config: LightingInvariantConfig) -> Self {
        Self {
            width,
            height,
            sobel: SobelEdgeDetector::new(width, height),
            lbp: LocalBinaryPattern::new(width, height),
            adaptive_threshold: AdaptiveThreshold::new(
                width,
                height,
                32, // 32x32 pixel regions
                config.decay_rate,
                config.min_sensitivity,
            ),
            lum_frame: vec![0; width * height],
            prev_lum_frame: vec![0; width * height],
            config,
            frame_count: 0,
            detection_count: 0,
        }
    }

    /// Convert RGB to luminance
    #[inline]
    fn rgb_to_lum(r: u8, g: u8, b: u8) -> u8 {
        (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8
    }

    /// Row stride of the frames this detector accepts, in pixels.
    ///
    /// `detect` returns no detections unless the frame is exactly
    /// `width * height * 3` bytes, so this is the width of every frame whose
    /// detection coordinates can be used to index that frame.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Detect motion in RGB frame
    #[allow(clippy::cast_possible_truncation)]
    pub fn detect(&mut self, frame_rgb: &[u8]) -> Vec<(usize, usize)> {
        // A zero-sized frame is trivially "no motion": there is nothing to scan
        // and the interior loop bounds below would underflow. Guard it before
        // anything else so a degenerate frame is inert rather than a panic.
        if self.width < MIN_SCAN_DIMENSION || self.height < MIN_SCAN_DIMENSION {
            return Vec::new();
        }

        if frame_rgb.len() != self.width * self.height * 3 {
            return Vec::new();
        }

        // Convert to luminance
        for i in 0..self.width * self.height {
            let rgb_idx = i * 3;
            self.lum_frame[i] = Self::rgb_to_lum(
                frame_rgb[rgb_idx],
                frame_rgb[rgb_idx + 1],
                frame_rgb[rgb_idx + 2],
            );
        }

        let mut detections = Vec::new();

        // Run detectors
        let edges = if self.config.use_sobel {
            self.sobel.detect(&self.lum_frame).to_vec()
        } else {
            vec![0; self.width * self.height]
        };

        if self.config.use_lbp {
            self.lbp.compute(&self.lum_frame);
        }

        // Analyze detections by method
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let idx = y * self.width + x;

                let mut is_motion = false;

                // METHOD 1: Temporal edge movement
                if self.config.use_temporal_edges {
                    let curr_edge = edges[idx];
                    let prev_edge = self.sobel.prev_edges()[idx];
                    let edge_delta = (curr_edge as i16 - prev_edge as i16).unsigned_abs() as u8;

                    if edge_delta > self.config.min_edge_movement
                        && curr_edge > self.config.edge_threshold
                    {
                        is_motion = true;
                    }
                }

                // METHOD 2: Local pattern change (texture movement).
                // `hamming_distance` is bounded by `PATTERN_BITS`, so a
                // configured `contrast_threshold` above that bound can never
                // be exceeded and this stage is inert.
                if self.config.use_lbp && !is_motion {
                    let curr_pattern = self.lbp.pattern(x, y);
                    let prev_pattern = self.lbp.prev_pattern(x, y);
                    let pattern_distance =
                        LocalBinaryPattern::hamming_distance(curr_pattern, prev_pattern);

                    if pattern_distance > self.config.contrast_threshold {
                        is_motion = true;
                    }
                }

                if is_motion {
                    let threshold =
                        self.adaptive_threshold
                            .threshold_at(x, y, self.config.edge_threshold);

                    // Double-check with edge magnitude
                    if edges[idx] > threshold {
                        detections.push((x, y));
                        self.adaptive_threshold.register_detection(x, y);
                        self.detection_count += 1;
                    }
                }
            }
        }

        // Apply per-region decay
        self.adaptive_threshold.apply_decay();

        // Update previous frame
        self.prev_lum_frame.copy_from_slice(&self.lum_frame);
        self.frame_count += 1;

        detections
    }

    pub fn metrics(&self) -> DecayMetrics {
        DecayMetrics {
            // The real sensitivity state is the mean of the per-region gains.
            // A zero-sized frame has no regions and therefore no mean; it
            // falls back to the unadapted base gain of 1.0, which is also the
            // value every region holds before any adaptation has run, so a
            // caller reading `metrics()` on a degenerate detector sees the
            // documented starting point rather than an invented number.
            sensitivity: self.adaptive_threshold.mean_sensitivity().unwrap_or(1.0),
            current_threshold: self.config.edge_threshold,
            edge_count: self.frame_count,
            detection_count: self.detection_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_edge_detection() {
        let mut detector = SobelEdgeDetector::new(10, 10);

        // Create frame with vertical edge
        let mut frame = vec![100u8; 10 * 10];
        for y in 0..10 {
            for x in 5..10 {
                frame[y * 10 + x] = 200;
            }
        }

        let edges = detector.detect(&frame);

        // Check that edges are detected around x=4-5
        let mut edge_found = false;
        for y in 1..9 {
            let idx = y * 10 + 5;
            if edges[idx] > 50 {
                edge_found = true;
                break;
            }
        }

        assert!(edge_found, "Should detect vertical edge");
    }

    #[test]
    fn test_lbp_texture() {
        let mut lbp = LocalBinaryPattern::new(10, 10);

        let frame1 = vec![100u8; 10 * 10];
        lbp.compute(&frame1);
        let pattern1 = lbp.pattern(5, 5);

        // Create slightly different frame (some pixels brighter)
        let mut frame2 = vec![100u8; 10 * 10];
        frame2[4 * 10 + 4] = 120; // Top-left neighbor
        lbp.compute(&frame2);
        let pattern2 = lbp.pattern(5, 5);

        let distance = LocalBinaryPattern::hamming_distance(pattern1, pattern2);
        assert!(distance > 0, "Should detect pattern change");
    }

    #[test]
    fn test_global_lighting_change() {
        let config = LightingInvariantConfig::default();
        let mut detector = LightingInvariantDetector::new(100, 100, config);

        // Create frame with uniform brightness
        let frame1 = vec![128u8; 100 * 100 * 3];
        let _detections1 = detector.detect(&frame1);

        // Create frame with same content but 50% brighter
        let frame2: Vec<u8> = frame1
            .iter()
            .map(|&p| (p as f32 * 1.5).min(255.0) as u8)
            .collect();
        let detections2 = detector.detect(&frame2);

        // Global lighting change should produce minimal detections
        assert!(
            detections2.len() < 100,
            "Global lighting change should not produce many detections, got {}",
            detections2.len()
        );
    }

    #[test]
    fn test_local_shadow() {
        let config = LightingInvariantConfig::default();
        let mut detector = LightingInvariantDetector::new(100, 100, config);

        // Create initial frame with pattern
        let mut frame1 = vec![150u8; 100 * 100 * 3];
        for i in 0..(100 * 100) {
            if i % 2 == 0 {
                frame1[i * 3] = 100;
            }
        }

        detector.detect(&frame1);

        // Add shadow (local darkening)
        let mut frame2 = frame1.clone();
        for y in 25..75 {
            for x in 25..75 {
                let idx = (y * 100 + x) * 3;
                frame2[idx] = (frame2[idx] as f32 * 0.7) as u8;
                frame2[idx + 1] = (frame2[idx + 1] as f32 * 0.7) as u8;
                frame2[idx + 2] = (frame2[idx + 2] as f32 * 0.7) as u8;
            }
        }

        let detections = detector.detect(&frame2);

        // Should detect shadow boundary as motion (edges move)
        assert!(!detections.is_empty(), "Should detect shadow edges");
    }

    #[test]
    fn test_object_motion() {
        let config = LightingInvariantConfig::default();
        let mut detector = LightingInvariantDetector::new(100, 100, config);

        // Create frame with object
        let mut frame1 = vec![150u8; 100 * 100 * 3];
        for y in 40..60 {
            for x in 40..60 {
                let idx = (y * 100 + x) * 3;
                frame1[idx] = 50;
                frame1[idx + 1] = 50;
                frame1[idx + 2] = 50;
            }
        }

        detector.detect(&frame1);

        // Move object
        let mut frame2 = vec![150u8; 100 * 100 * 3];
        for y in 45..65 {
            for x in 45..65 {
                let idx = (y * 100 + x) * 3;
                frame2[idx] = 50;
                frame2[idx + 1] = 50;
                frame2[idx + 2] = 50;
            }
        }

        let detections = detector.detect(&frame2);

        // Should detect object movement
        assert!(!detections.is_empty(), "Should detect object motion");
    }

    #[test]
    fn test_static_scene_no_detections_after_warmup() {
        let config = LightingInvariantConfig::default();
        let mut detector = LightingInvariantDetector::new(64, 64, config);

        let frame = vec![128u8; 64 * 64 * 3];

        // First detection warms up prev_lum
        detector.detect(&frame);

        // Second detection on same frame should have no motion
        let detections = detector.detect(&frame);
        assert!(
            detections.is_empty(),
            "Static scene should have no detections after warmup"
        );
    }

    #[test]
    fn test_detector_reset() {
        let config = LightingInvariantConfig::default();
        let mut detector = LightingInvariantDetector::new(64, 64, config);

        let frame1 = vec![100u8; 64 * 64 * 3];
        let mut frame2 = vec![100u8; 64 * 64 * 3];
        frame2[0] = 200; // small change

        detector.detect(&frame1);
        detector.detect(&frame2);

        // Reset
        detector.prev_lum_frame.fill(0);
        detector.frame_count = 0;
        detector.detection_count = 0;

        let metrics = detector.metrics();
        // DecayMetrics uses edge_count to store frame_count
        assert_eq!(metrics.edge_count, 0);
        assert_eq!(metrics.detection_count, 0);
    }

    // ---------------------------------------------------------------------
    // Bug: `metrics()` hardcoded `sensitivity: 1.0`, so it kept claiming the
    // unadapted base gain long after the per-region adaptation loop had
    // damped every region's gain.
    // ---------------------------------------------------------------------

    #[test]
    fn test_metrics_sensitivity_starts_at_base_gain() {
        let detector = LightingInvariantDetector::new(64, 64, LightingInvariantConfig::default());

        // No frame seen yet: every region still sits at the unadapted gain.
        assert_eq!(detector.metrics().sensitivity, 1.0);
    }

    #[test]
    fn test_metrics_sensitivity_reports_adapted_gain() {
        let mut detector =
            LightingInvariantDetector::new(100, 100, LightingInvariantConfig::default());

        // Warm up, then move an object so at least one detection registers and
        // damps + decays its region's gain below 1.0.
        detector.detect(&vec![100u8; 100 * 100 * 3]);
        let mut frame2 = vec![150u8; 100 * 100 * 3];
        for y in 45..65 {
            for x in 45..65 {
                let idx = (y * 100 + x) * 3;
                frame2[idx] = 50;
                frame2[idx + 1] = 50;
                frame2[idx + 2] = 50;
            }
        }
        let detections = detector.detect(&frame2);
        assert!(!detections.is_empty(), "fixture must drive detections");

        let gains = &detector.adaptive_threshold.sensitivities;
        assert!(
            gains.iter().any(|g| *g < 1.0),
            "fixture must adapt at least one region away from the base gain"
        );
        let expected: f32 = gains.iter().sum::<f32>() / gains.len() as f32;

        let reported = detector.metrics().sensitivity;
        assert_ne!(
            reported, 1.0,
            "metrics() must not keep reporting a hardcoded 1.0"
        );
        assert!(
            (reported - expected).abs() < 1e-6,
            "metrics() must report the mean region gain: expected {expected}, got {reported}"
        );
    }

    #[test]
    fn test_mean_sensitivity_is_none_without_regions() {
        // A zero-sized frame yields no regions at all, so there is no gain to
        // average; the mean is genuinely undefined rather than zero.
        let adaptive = AdaptiveThreshold::new(0, 0, 32, 0.96, 0.3);
        assert_eq!(adaptive.mean_sensitivity(), None);
    }

    // ---------------------------------------------------------------------
    // Bug 1: the LBP branch can never fire because `contrast_threshold`
    // defaults to a value outside the range `hamming_distance` can return.
    // ---------------------------------------------------------------------

    #[test]
    fn test_default_contrast_threshold_is_within_lbp_hamming_range() {
        let config = LightingInvariantConfig::default();

        // An LBP pattern is `LocalBinaryPattern::PATTERN_BITS` bits wide, so the
        // Hamming distance between two patterns can only ever be
        // 0..=PATTERN_BITS. A `contrast_threshold` above that makes
        // `pattern_distance > contrast_threshold` permanently false, which
        // silently turns the whole LBP stage into dead code.
        assert!(
            config.contrast_threshold <= LocalBinaryPattern::PATTERN_BITS,
            "default contrast_threshold {} is outside the range hamming_distance can \
             return (0..={}); the LBP stage can never fire",
            config.contrast_threshold,
            LocalBinaryPattern::PATTERN_BITS
        );
    }

    #[test]
    fn test_lbp_stage_alone_detects_moving_texture() {
        // Temporal edges are disabled so that the ONLY way a pixel can be
        // flagged is the LBP branch. With an unreachable `contrast_threshold`
        // this returns an empty list, proving the stage is dead.
        let config = LightingInvariantConfig {
            edge_threshold: 10,
            use_temporal_edges: false,
            ..LightingInvariantConfig::default()
        };
        let mut detector = LightingInvariantDetector::new(64, 64, config);

        // 2x2 block checkerboard: strong Sobel gradients at every block edge
        // and an 8-bit LBP pattern that changes for most pixels when the
        // texture shifts by one pixel.
        let texture = |x: usize, y: usize| -> u8 {
            if ((x / 2) + (y / 2)).is_multiple_of(2) {
                255
            } else {
                0
            }
        };
        let frame1: Vec<u8> = (0..64 * 64)
            .flat_map(|i| {
                let (x, y) = (i % 64, i / 64);
                [texture(x, y); 3]
            })
            .collect();
        let frame2: Vec<u8> = (0..64 * 64)
            .flat_map(|i| {
                let (x, y) = (i % 64, i / 64);
                [texture(x + 1, y); 3]
            })
            .collect();

        detector.detect(&frame1);
        let detections = detector.detect(&frame2);

        assert!(
            !detections.is_empty(),
            "shifting a high-contrast texture by one pixel must be caught by the LBP stage"
        );
    }

    // ---------------------------------------------------------------------
    // Bug 2: the adaptive-threshold floor is applied to the threshold scale
    // instead of bounding the sensitivity multiplier, and nothing stops an
    // out-of-range `min_sensitivity` from pushing the effective threshold
    // ABOVE the base threshold.
    // ---------------------------------------------------------------------

    #[test]
    fn test_effective_threshold_never_falls_below_base_threshold() {
        const BASE: u8 = 25;

        for min_sensitivity in [0.0f32, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, -1.0] {
            let adaptive = AdaptiveThreshold::new(64, 64, 32, 0.96, min_sensitivity);
            let effective = adaptive.threshold_at(5, 5, BASE);
            assert!(
                effective >= BASE,
                "min_sensitivity {min_sensitivity} produced threshold {effective}, which is \
                 below the base threshold {BASE}; adaptation must only ever raise the bar, \
                 never lower it"
            );
        }
    }

    #[test]
    fn test_effective_threshold_is_bounded_by_the_sensitivity_floor() {
        const BASE: u8 = 25;

        // A damped region sits at the floor, so its effective threshold is the
        // furthest the adaptation is allowed to move: base / floor. The cast
        // to u8 saturates, so an extreme floor must not wrap or panic.
        for (min_sensitivity, expected_cap) in [(0.3f32, 83u32), (0.5, 50), (1.0, 25)] {
            let mut adaptive = AdaptiveThreshold::new(64, 64, 32, 0.96, min_sensitivity);
            for _ in 0..10 {
                adaptive.register_detection(5, 5);
            }
            let effective = u32::from(adaptive.threshold_at(5, 5, BASE));
            assert!(
                effective <= expected_cap,
                "min_sensitivity {min_sensitivity} produced threshold {effective}, above the \
                 cap {expected_cap} implied by base / floor"
            );
            assert!(effective >= u32::from(BASE));
        }

        let mut high = AdaptiveThreshold::new(64, 64, 32, 0.96, 0.9);
        for _ in 0..10 {
            high.register_detection(5, 5);
        }
        let high_bar = high.threshold_at(5, 5, BASE);

        // With no floor at all, adaptation is allowed to push the bar all the
        // way towards u8::MAX. The float->u8 cast saturates, so this must not
        // wrap around to a tiny threshold.
        let mut unfloored = AdaptiveThreshold::new(64, 64, 32, 0.96, 0.0);
        for _ in 0..10 {
            unfloored.register_detection(5, 5);
        }
        let unfloored_bar = unfloored.threshold_at(5, 5, BASE);
        assert!(
            unfloored_bar >= BASE,
            "an unfloored sensitivity must never drop below BASE, got {unfloored_bar}"
        );
        assert!(
            unfloored_bar > high_bar,
            "removing the sensitivity floor must let the region quiet down further: \
             {unfloored_bar} vs {high_bar}"
        );
    }

    #[test]
    fn test_higher_sensitivity_floor_never_raises_the_detection_bar() {
        const BASE: u8 = 25;

        // A region that keeps triggering gets damped, so its stored gain sits
        // at the floor. Asking for MORE sensitivity (a higher floor) must never
        // produce a HIGHER effective threshold.
        let mut low = AdaptiveThreshold::new(64, 64, 32, 0.96, 0.3);
        for _ in 0..10 {
            low.register_detection(5, 5);
        }

        let mut high = AdaptiveThreshold::new(64, 64, 32, 0.96, 0.9);
        for _ in 0..10 {
            high.register_detection(5, 5);
        }

        let low_sensitivity_bar = low.threshold_at(5, 5, BASE);
        let high_sensitivity_bar = high.threshold_at(5, 5, BASE);

        assert!(
            high_sensitivity_bar < low_sensitivity_bar,
            "raising the sensitivity floor from 0.3 to 0.9 changed the effective threshold \
             from {low_sensitivity_bar} to {high_sensitivity_bar}; more sensitivity must mean \
             a lower detection bar"
        );
    }

    // ---------------------------------------------------------------------
    // Bug 3: `self.height - 1` / `self.width - 1` underflow at zero.
    // ---------------------------------------------------------------------

    #[test]
    fn test_zero_sized_frames_return_no_detections() {
        for (width, height) in [(0usize, 0usize), (0, 5), (5, 0), (1, 1), (2, 2)] {
            let mut detector =
                LightingInvariantDetector::new(width, height, LightingInvariantConfig::default());
            let frame = vec![0u8; width * height * 3];
            assert!(
                detector.detect(&frame).is_empty(),
                "{width}x{height} frame must produce no detections"
            );
        }
    }

    #[test]
    fn test_zero_sized_frames_do_not_panic_in_stage_types() {
        for (width, height) in [(0usize, 0usize), (0, 5), (5, 0)] {
            let mut sobel = SobelEdgeDetector::new(width, height);
            assert_eq!(sobel.detect(&[]).len(), width * height);

            let mut lbp = LocalBinaryPattern::new(width, height);
            lbp.compute(&[]);
        }
    }
}
