use std::env;
use std::path::PathBuf;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use image::GenericImageView;
use std::fs;
use std::io::Write;

use aeyes::motion::{LightingInvariantDetector, LightingInvariantConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Metrics {
    precision: f64,
    recall: f64,
    f1_score: f64,
    true_positives: u64,
    false_positives: u64,
    false_negatives: u64,
    frames_processed: u64,
    total_time_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SequenceResult {
    name: String,
    metrics: Metrics,
    frames: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CategoryResult {
    name: String,
    sequences: Vec<SequenceResult>,
    aggregate: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Report {
    dataset: String,
    timestamp: String,
    categories: Vec<CategoryResult>,
    overall: Metrics,
    config: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvaluationConfig {
    max_frames_per_sequence: u64,
}

struct Config {
    cache_dir: PathBuf,
    download: bool,
    output_dir: PathBuf,
    categories: Vec<String>,
    max_frames_per_sequence: u64,
}

impl Default for Config {
    fn default() -> Self {
        let cache_dir = env::var("CDNET_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("tests/cdnet_evaluation/data"));

        let output_dir = env::var("CDNET_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("tests/cdnet_evaluation/results"));

        Self {
            cache_dir,
            download: env::var("CDNET_DOWNLOAD").map(|v| v == "1").unwrap_or(true),
            output_dir,
            categories: vec![
                "baseline".to_string(),
                "cameraJitter".to_string(),
                "dynamicBackground".to_string(),
                "intermittentObjectMotion".to_string(),
                "shadow".to_string(),
                "thermal".to_string(),
            ],
            max_frames_per_sequence: env::var("CDNET_MAX_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        }
    }
}

fn ensure_dataset(config: &Config) -> Result<PathBuf> {
    let zip_path = config.cache_dir.join("dataset2014.zip");
    if zip_path.exists() {
        println!("✓ Dataset already cached at {:?}", zip_path);
        return Ok(zip_path);
    }
    if !config.download {
        return Err(anyhow!("Dataset not found and CDNET_DOWNLOAD not set"));
    }
    println!("Downloading CDnet 2014 dataset...");
    fs::create_dir_all(&config.cache_dir)?;
    let url = "http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip";
    let status = std::process::Command::new("curl")
        .args(&["-L", "-o", zip_path.to_str().unwrap(), url])
        .status()
        .context("failed to execute curl")?;
    if !status.success() {
        return Err(anyhow!("download failed"));
    }
    println!("✓ Download complete.");
    Ok(zip_path)
}

fn extract_dataset(zip_path: &PathBuf, config: &Config) -> Result<PathBuf> {
    let extraction_root = config.cache_dir.join("CDnet2014");
    let dataset_dir = extraction_root.join("dataset");
    if dataset_dir.exists() {
        println!("✓ Dataset already extracted at {:?}", dataset_dir);
        return Ok(dataset_dir);
    }
    if extraction_root.exists() {
        fs::remove_dir_all(&extraction_root)?;
    }
    println!("Extracting dataset...");
    fs::create_dir_all(&extraction_root)?;
    let extractors = ["unzip", "7z", "tar"];
    let mut success = false;
    for extractor in &extractors {
        let mut cmd = std::process::Command::new(extractor);
        let args = match *extractor {
            "unzip" => vec!["-q", zip_path.to_str().unwrap(), "-d", extraction_root.to_str().unwrap()],
            "7z" => vec!["x", zip_path.to_str().unwrap(), "-o", extraction_root.to_str().unwrap()],
            "tar" => vec!["-xf", zip_path.to_str().unwrap(), "-C", extraction_root.to_str().unwrap()],
            _ => vec![],
        };
        cmd.args(&args);
        if cmd.status().map(|s| s.success()).unwrap_or(false) {
            success = true;
            break;
        }
    }
    if !success {
        return Err(anyhow!("extraction failed"));
    }
    if !dataset_dir.exists() {
        return Err(anyhow!("dataset subfolder missing"));
    }
    println!("✓ Extraction complete.");
    Ok(dataset_dir)
}

fn load_ground_truth(gt_dir: &PathBuf, frame_idx: usize) -> Result<Option<Vec<u8>>> {
    let candidates = [
        format!("gt{:06}.png", frame_idx + 1),
        format!("gt{:06}.jpg", frame_idx + 1),
        format!("gt{:08}.png", frame_idx + 1),
        format!("gt{:08}.jpg", frame_idx + 1),
        format!("{:06}.png", frame_idx + 1),
        format!("{:06}.jpg", frame_idx + 1),
    ];
    for name in &candidates {
        let gt_path = gt_dir.join(name);
        if gt_path.exists() {
            let img = image::open(&gt_path)?;
            let luma = img.to_luma8();
            let binary = luma.iter().map(|&p| if p > 0 { 255 } else { 0 }).collect();
            return Ok(Some(binary));
        }
    }
    Ok(None)
}

fn evaluate_sequence(sequence_dir: &PathBuf, config: &Config) -> Result<SequenceResult> {
    let seq_name = sequence_dir.file_name().unwrap().to_string_lossy().into_owned();
    let input_dir = if sequence_dir.join("input").exists() {
        sequence_dir.join("input")
    } else {
        sequence_dir.clone()
    };
    let gt_dir = if sequence_dir.join("groundtruth").exists() {
        sequence_dir.join("groundtruth")
    } else {
        sequence_dir.clone()
    };
    let input_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("png") || ext.eq_ignore_ascii_case("bmp")
            } else { false }
        })
        .map(|e| e.path())
        .collect();
    if input_files.is_empty() {
        println!("  ⚠ No input frames in {}", seq_name);
        return Ok(SequenceResult { name: seq_name, metrics: Metrics { precision: 0.0, recall: 0.0, f1_score: 0.0, true_positives: 0, false_positives: 0, false_negatives: 0, frames_processed: 0, total_time_secs: 0.0 }, frames: 0 });
    }
    let mut input_files = input_files;
    input_files.sort_by_key(|p| p.file_name().unwrap().to_owned());
    let has_ground_truth = gt_dir.join("gt000001.png").exists() || gt_dir.join("gt000001.jpg").exists();
    if !has_ground_truth {
        println!("  ⚠ No groundtruth found in {}, skipping metrics", seq_name);
        return Ok(SequenceResult { name: seq_name, metrics: Metrics { precision: 0.0, recall: 0.0, f1_score: 0.0, true_positives: 0, false_positives: 0, false_negatives: 0, frames_processed: 0, total_time_secs: 0.0 }, frames: input_files.len() as u64 });
    }
    let first_img = image::open(&input_files[0])?;
    let (width, height) = first_img.dimensions();
    let width = width as usize;
    let height = height as usize;
    let mut detector = LightingInvariantDetector::new(width, height, LightingInvariantConfig::default());
    println!("  ▶ {}: {} frames, {}×{}", seq_name, input_files.len(), width, height);
    let mut total_tp = 0u64;
    let mut total_fp = 0u64;
    let mut total_fn = 0u64;
    let mut frames_processed = 0u64;
    let start_time = Instant::now();
    let max_frames = if config.max_frames_per_sequence > 0 {
        std::cmp::min(config.max_frames_per_sequence as usize, input_files.len())
    } else {
        input_files.len()
    };
    for (i, input_path) in input_files.iter().take(max_frames).enumerate() {
        let input_img = image::open(input_path)
            .with_context(|| format!("failed to open input frame: {:?}", input_path))?;
        let rgb = input_img.to_rgb8();
        let gt_binary = load_ground_truth(&gt_dir, i)?
            .ok_or_else(|| anyhow!("missing ground truth for frame {}", i+1))?;
        let detections = detector.detect(rgb.as_raw());
        let mut frame_tp = 0;
        let mut frame_fp = 0;
        let mut frame_fn = 0;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let is_foreground_gt = gt_binary[idx] > 0;
                let is_detected = detections.contains(&(x, y));
                if is_foreground_gt && is_detected {
                    frame_tp += 1;
                } else if !is_foreground_gt && is_detected {
                    frame_fp += 1;
                } else if is_foreground_gt && !is_detected {
                    frame_fn += 1;
                }
            }
        }
        total_tp += frame_tp;
        total_fp += frame_fp;
        total_fn += frame_fn;
        frames_processed += 1;
        if frames_processed % 100 == 0 {
            print!("    {} frames...\r", frames_processed);
            std::io::stdout().flush()?;
        }
    }
    let elapsed = start_time.elapsed();
    let precision = if total_tp + total_fp > 0 { total_tp as f64 / (total_tp + total_fp) as f64 } else { 0.0 };
    let recall = if total_tp + total_fn > 0 { total_tp as f64 / (total_tp + total_fn) as f64 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    println!("  ✓ {}: F1={:.4}, P={:.4}, R={:.4}", seq_name, f1_score, precision, recall);
    Ok(SequenceResult { name: seq_name, metrics: Metrics { precision, recall, f1_score, true_positives: total_tp, false_positives: total_fp, false_negatives: total_fn, frames_processed, total_time_secs: elapsed.as_secs_f64() }, frames: frames_processed })
}

fn evaluate_category(category_name: &str, dataset_root: &PathBuf, config: &Config) -> Result<CategoryResult> {
    println!("\n📁 Category: {}", category_name);
    let category_dir = dataset_root.join(category_name);
    if !category_dir.exists() {
        return Err(anyhow!("category directory not found: {:?}", category_dir));
    }
    let sequence_dirs: Vec<PathBuf> = fs::read_dir(&category_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    if sequence_dirs.is_empty() {
        return Err(anyhow!("no sequences found in category: {}", category_name));
    }
    println!("  Found {} sequences", sequence_dirs.len());
    let mut sequence_results = Vec::new();
    let mut agg_tp = 0u64;
    let mut agg_fp = 0u64;
    let mut agg_fn = 0u64;
    let mut total_frames = 0u64;
    let mut total_time = 0.0;
    for seq_dir in &sequence_dirs {
        match evaluate_sequence(seq_dir, config) {
            Ok(result) => {
                sequence_results.push(result.clone());
                agg_tp += result.metrics.true_positives;
                agg_fp += result.metrics.false_positives;
                agg_fn += result.metrics.false_negatives;
                total_frames += result.frames;
                total_time += result.metrics.total_time_secs;
            }
            Err(e) => {
                eprintln!("    ✗ Error: {}", e);
            }
        }
    }
    let precision = if agg_tp + agg_fp > 0 { agg_tp as f64 / (agg_tp + agg_fp) as f64 } else { 0.0 };
    let recall = if agg_tp + agg_fn > 0 { agg_tp as f64 / (agg_tp + agg_fn) as f64 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    let aggregate = Metrics { precision, recall, f1_score, true_positives: agg_tp, false_positives: agg_fp, false_negatives: agg_fn, frames_processed: total_frames, total_time_secs: total_time };
    println!("  📊 Category Summary: F1={:.4}, P={:.4}, R={:.4}", f1_score, precision, recall);
    Ok(CategoryResult { name: category_name.to_string(), sequences: sequence_results, aggregate })
}

fn save_report(report: &Report, config: &Config) -> Result<PathBuf> {
    fs::create_dir_all(&config.output_dir)?;
    let report_path = config.output_dir.join("cdnet_report.json");
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(&report_path, json)?;
    println!("\n✓ Report saved to {:?}", report_path);
    Ok(report_path)
}

fn print_summary(report: &Report) {
    println!("\n{}", "═".repeat(60));
    println!("CDnet 2014 Evaluation Summary");
    println!("{}", "═".repeat(60));
    println!("{:<20} {:>12} {:>12} {:>12}", "Category", "F1-Score", "Precision", "Recall");
    println!("{}", "─".repeat(60));
    for cat in &report.categories {
        let agg = &cat.aggregate;
        println!("{:<20} {:>12.4} {:>12.4} {:>12.4}", cat.name, agg.f1_score, agg.precision, agg.recall);
    }
    println!("{}", "─".repeat(60));
    println!("{:<20} {:>12.4} {:>12.4} {:>12.4}", "OVERALL", report.overall.f1_score, report.overall.precision, report.overall.recall);
    println!("{}", "═".repeat(60));
    println!("Frames: {}", report.overall.frames_processed);
    println!("Time: {:.2} seconds", report.overall.total_time_secs);
    println!("Config: max_frames_per_sequence={}", report.config.max_frames_per_sequence);
    println!();
    println!("METRIC overall_f1={:.6}", report.overall.f1_score);
    println!("METRIC overall_precision={:.6}", report.overall.precision);
    println!("METRIC overall_recall={:.6}", report.overall.recall);
}

fn main() -> Result<()> {
    let config = Config::default();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   CDnet 2014 Motion Detection Evaluation                 ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("Cache: {:?}", config.cache_dir);
    println!("Output: {:?}", config.output_dir);
    println!("Categories: {:?}", config.categories);
    if config.max_frames_per_sequence > 0 {
        println!("Max frames: {}", config.max_frames_per_sequence);
    }
    println!();
    let zip_path = ensure_dataset(&config)?;
    let dataset_root = extract_dataset(&zip_path, &config)?;
    let mut category_results = Vec::new();
    let mut overall_tp = 0u64;
    let mut overall_fp = 0u64;
    let mut overall_fn = 0u64;
    let mut overall_frames = 0u64;
    let mut overall_time = 0.0;
    for category in &config.categories {
        match evaluate_category(category, &dataset_root, &config) {
            Ok(result) => {
                category_results.push(result.clone());
                overall_tp += result.aggregate.true_positives;
                overall_fp += result.aggregate.false_positives;
                overall_fn += result.aggregate.false_negatives;
                overall_frames += result.aggregate.frames_processed;
                overall_time += result.aggregate.total_time_secs;
            }
            Err(e) => {
                eprintln!("✗ Failed to evaluate category {}: {}", category, e);
            }
        }
    }
    let precision = if overall_tp + overall_fp > 0 { overall_tp as f64 / (overall_tp + overall_fp) as f64 } else { 0.0 };
    let recall = if overall_tp + overall_fn > 0 { overall_tp as f64 / (overall_tp + overall_fn) as f64 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    let overall = Metrics { precision, recall, f1_score, true_positives: overall_tp, false_positives: overall_fp, false_negatives: overall_fn, frames_processed: overall_frames, total_time_secs: overall_time };
    let report = Report { dataset: "CDnet 2014".to_string(), timestamp: chrono::Utc::now().to_rfc3339(), categories: category_results, overall, config: EvaluationConfig { max_frames_per_sequence: config.max_frames_per_sequence } };
    let _report_path = save_report(&report, &config)?;
    print_summary(&report);
    Ok(())
}
