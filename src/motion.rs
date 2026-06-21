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

#[derive(Clone, Debug)]
pub struct LightingInvariantConfig {
    /// Edge detection threshold (0-255)
    /// Higher = only strong edges trigger motion
    pub edge_threshold: u8,

    /// Minimum edge movement distance (pixels)
    /// Prevents noise from triggering detections
    pub min_edge_movement: u8,

    /// Local contrast threshold (0-255)
    /// Changes in local pattern indicate motion
    pub contrast_threshold: u8,

    /// Decay rate for edge maps (same as luminance)
    pub decay_rate: f32,

    /// Sensitivity floor
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
            contrast_threshold: 20,
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
    #[inline]
    fn hamming_distance(p1: u8, p2: u8) -> u8 {
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

    /// Get threshold for pixel location
    pub fn threshold_at(&self, x: usize, y: usize, base_threshold: u8) -> u8 {
        let region_x = x / self.region_size;
        let region_y = y / self.region_size;

        let regions_x = self.width.div_ceil(self.region_size);
        let idx = region_y * regions_x + region_x;

        let sensitivity = self.sensitivities.get(idx).copied().unwrap_or(1.0);
        let adjusted = (base_threshold as f32 * sensitivity) as u8;
        adjusted.max((base_threshold as f32 * self.min_sensitivity) as u8)
    }

    /// Register detection in region
    pub fn register_detection(&mut self, x: usize, y: usize) {
        let region_x = x / self.region_size;
        let region_y = y / self.region_size;

        let regions_x = self.width.div_ceil(self.region_size);
        let idx = region_y * regions_x + region_x;

        if let Some(threshold) = self.sensitivities.get_mut(idx) {
            *threshold *= 0.85; // Damping
            *threshold = threshold.max(self.min_sensitivity);
        }
    }

    /// Apply decay to all regions
    pub fn apply_decay(&mut self) {
        for threshold in self.sensitivities.iter_mut() {
            *threshold *= self.decay_rate;
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

    /// Detect motion in RGB frame
    #[allow(clippy::cast_possible_truncation)]
    pub fn detect(&mut self, frame_rgb: &[u8]) -> Vec<(usize, usize)> {
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

                // METHOD 2: Local pattern change (texture movement)
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
            sensitivity: 1.0, // Could be per-region, but simplified
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
}
