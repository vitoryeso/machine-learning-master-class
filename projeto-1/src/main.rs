use anyhow::Result;
use fastembed::{ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};
use indicatif::{ProgressBar, ProgressStyle};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp"];
const SAMPLE_SIZE: usize = 8000;
const BATCH_SIZE: usize = 64;
const K_MIN: usize = 2;
const K_MAX: usize = 15;
const KMEANS_MAX_ITER: usize = 100;
const EMB_DIM: usize = 512;
const SEED: u64 = 42;
const OUTPUT_DIR: &str = "D:/media/cluster_output";

// ─── Image Discovery ──────────────────────────────────────────────

fn collect_image_paths(root: &str) -> Vec<PathBuf> {
    let skip = ["src", "target", "cluster_output", ".fastembed_cache", ".git"];
    WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().is_file() {
                return false;
            }
            if let Ok(rel) = e.path().strip_prefix(root) {
                if let Some(first) = rel.components().next() {
                    let s = first.as_os_str().to_string_lossy();
                    if skip.iter().any(|&sk| s == sk) {
                        return false;
                    }
                }
            }
            match e.path().extension().and_then(|s| s.to_str()) {
                Some(ext) => IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()),
                None => false,
            }
        })
        .map(|e| e.into_path())
        .collect()
}

// ─── Embedding Cache ──────────────────────────────────────────────

fn save_cache(paths: &[PathBuf], embeddings: &[Vec<f32>]) -> Result<()> {
    // Save paths
    let paths_str: String = paths.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>().join("\n");
    std::fs::write(format!("{}/paths.txt", OUTPUT_DIR), &paths_str)?;

    // Save embeddings as raw f32 binary: [n:u64][dim:u64][f32 * n * dim]
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}/embeddings.bin", OUTPUT_DIR))?);
    f.write_all(&(embeddings.len() as u64).to_le_bytes())?;
    f.write_all(&(EMB_DIM as u64).to_le_bytes())?;
    for emb in embeddings {
        for &v in emb {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    println!("Cache saved ({} embeddings)", embeddings.len());
    Ok(())
}

fn load_cache() -> Result<(Vec<PathBuf>, Vec<Vec<f32>>)> {
    let paths_str = std::fs::read_to_string(format!("{}/paths.txt", OUTPUT_DIR))?;
    let paths: Vec<PathBuf> = paths_str.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect();

    let mut f = std::io::BufReader::new(std::fs::File::open(format!("{}/embeddings.bin", OUTPUT_DIR))?);
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)?;
    let dim = u64::from_le_bytes(buf8) as usize;

    let mut embeddings = Vec::with_capacity(n);
    let mut buf4 = [0u8; 4];
    for _ in 0..n {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            f.read_exact(&mut buf4)?;
            emb.push(f32::from_le_bytes(buf4));
        }
        embeddings.push(emb);
    }

    assert_eq!(paths.len(), n);
    println!("Cache loaded ({} embeddings of dim {})", n, dim);
    Ok((paths, embeddings))
}

fn cache_exists() -> bool {
    Path::new(&format!("{}/embeddings.bin", OUTPUT_DIR)).exists()
        && Path::new(&format!("{}/paths.txt", OUTPUT_DIR)).exists()
}

// ─── Metadata ─────────────────────────────────────────────────────

struct ImageMeta {
    top_folder: String,
    extension: String,
    file_size: u64,
    aspect: f32, // width/height, 0.0 if unknown
}

fn collect_metadata(paths: &[PathBuf]) -> Vec<ImageMeta> {
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} reading metadata [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let metas: Vec<ImageMeta> = paths
        .iter()
        .map(|p| {
            pb.inc(1);
            let top_folder = p
                .strip_prefix("D:/media")
                .ok()
                .and_then(|rel| {
                    if rel.components().count() <= 1 {
                        Some("root".to_string())
                    } else {
                        rel.components().next().map(|c| c.as_os_str().to_string_lossy().to_string())
                    }
                })
                .unwrap_or_else(|| "unknown".to_string());

            let extension = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            let file_size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);

            let aspect = image::image_dimensions(p)
                .map(|(w, h)| if h > 0 { w as f32 / h as f32 } else { 0.0 })
                .unwrap_or(0.0);

            ImageMeta { top_folder, extension, file_size, aspect }
        })
        .collect();

    pb.finish();
    metas
}

// ─── K-Means ──────────────────────────────────────────────────────

fn l2_normalize(data: &mut [Vec<f32>]) {
    for v in data.iter_mut() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }
}

fn euclidean_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn assign_clusters(data: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    data.iter()
        .map(|point| {
            centroids.iter().enumerate()
                .map(|(i, c)| (i, euclidean_dist_sq(point, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap().0
        })
        .collect()
}

fn compute_centroids(data: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Vec<f32>> {
    let dim = data[0].len();
    let mut sums = vec![vec![0.0f64; dim]; k];
    let mut counts = vec![0usize; k];
    for (point, &label) in data.iter().zip(labels) {
        counts[label] += 1;
        for (s, &v) in sums[label].iter_mut().zip(point) {
            *s += v as f64;
        }
    }
    sums.iter().zip(&counts).map(|(s, &c)| {
        if c == 0 { vec![0.0f32; dim] }
        else { s.iter().map(|v| (*v / c as f64) as f32).collect() }
    }).collect()
}

fn kmeans(data: &[Vec<f32>], k: usize, seed: u64) -> (Vec<usize>, Vec<Vec<f32>>, f64) {
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // K-Means init
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    use rand::Rng;
    centroids.push(data[rng.random_range(0..n)].clone());

    for _ in 1..k {
        let dists: Vec<f32> = data.iter().map(|p| {
            centroids.iter().map(|c| euclidean_dist_sq(p, c)).fold(f32::MAX, f32::min)
        }).collect();
        let total: f32 = dists.iter().sum();
        let threshold = rng.random_range(0.0..total);
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold { chosen = i; break; }
        }
        centroids.push(data[chosen].clone());
    }

    let mut labels = assign_clusters(data, &centroids);
    for _ in 0..KMEANS_MAX_ITER {
        let new_centroids = compute_centroids(data, &labels, k);
        let new_labels = assign_clusters(data, &new_centroids);
        let converged = new_labels == labels;
        centroids = new_centroids;
        labels = new_labels;
        if converged { break; }
    }

    let inertia: f64 = data.iter().zip(&labels)
        .map(|(p, &l)| euclidean_dist_sq(p, &centroids[l]) as f64).sum();
    (labels, centroids, inertia)
}

// ─── Silhouette ───────────────────────────────────────────────────

fn silhouette_score(data: &[Vec<f32>], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    let sample_size = n.min(3000);
    let indices: Vec<usize> = if sample_size < n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut StdRng::seed_from_u64(SEED));
        idx.truncate(sample_size);
        idx
    } else {
        (0..n).collect()
    };

    let mut cluster_indices: Vec<Vec<usize>> = vec![vec![]; k];
    for &i in &indices { cluster_indices[labels[i]].push(i); }

    let mut total_sil = 0.0f64;
    let mut count = 0usize;
    for &i in &indices {
        let ci = labels[i];
        let a = if cluster_indices[ci].len() <= 1 { 0.0 }
        else {
            cluster_indices[ci].iter().filter(|&&j| j != i)
                .map(|&j| euclidean_dist_sq(&data[i], &data[j]).sqrt() as f64)
                .sum::<f64>() / (cluster_indices[ci].len() - 1) as f64
        };
        let mut b = f64::MAX;
        for ck in 0..k {
            if ck == ci || cluster_indices[ck].is_empty() { continue; }
            let avg: f64 = cluster_indices[ck].iter()
                .map(|&j| euclidean_dist_sq(&data[i], &data[j]).sqrt() as f64)
                .sum::<f64>() / cluster_indices[ck].len() as f64;
            if avg < b { b = avg; }
        }
        if b == f64::MAX { continue; }
        total_sil += (b - a) / a.max(b);
        count += 1;
    }
    if count == 0 { 0.0 } else { total_sil / count as f64 }
}

// ─── PCA (reusable) ──────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct PcaResult {
    scores1: Vec<f32>,
    scores2: Vec<f32>,
    mean: Vec<f64>,
    pc1: Vec<f32>,
    pc2: Vec<f32>,
}

fn compute_pca_2d(data: &[Vec<f32>]) -> PcaResult {
    let n = data.len();
    let dim = data[0].len();

    let mut mean = vec![0.0f64; dim];
    for v in data {
        for (m, &x) in mean.iter_mut().zip(v) { *m += x as f64; }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);

    let centered: Vec<Vec<f32>> = data.iter().map(|v| {
        v.iter().zip(&mean).map(|(&x, &m)| x - m as f32).collect()
    }).collect();

    let project = |comp: &[f32], src: &[Vec<f32>]| -> Vec<f32> {
        src.iter().map(|v| dot(v, comp)).collect()
    };

    // Power iteration for PC1
    let mut pc1 = vec![0.0f32; dim];
    pc1[0] = 1.0;
    for _ in 0..50 {
        let scores = project(&pc1, &centered);
        let mut new_pc = vec![0.0f32; dim];
        for (v, &s) in centered.iter().zip(&scores) {
            for (np, &x) in new_pc.iter_mut().zip(v) { *np += x * s; }
        }
        let norm: f32 = new_pc.iter().map(|x| x * x).sum::<f32>().sqrt();
        new_pc.iter_mut().for_each(|x| *x /= norm);
        pc1 = new_pc;
    }
    let scores1 = project(&pc1, &centered);

    // Deflate
    let deflated: Vec<Vec<f32>> = centered.iter().zip(&scores1).map(|(v, &s)| {
        v.iter().zip(&pc1).map(|(&x, &p)| x - s * p).collect()
    }).collect();

    let mut pc2 = vec![0.0f32; dim];
    pc2[1] = 1.0;
    for _ in 0..50 {
        let scores = project(&pc2, &deflated);
        let mut new_pc = vec![0.0f32; dim];
        for (v, &s) in deflated.iter().zip(&scores) {
            for (np, &x) in new_pc.iter_mut().zip(v) { *np += x * s; }
        }
        let norm: f32 = new_pc.iter().map(|x| x * x).sum::<f32>().sqrt();
        new_pc.iter_mut().for_each(|x| *x /= norm);
        pc2 = new_pc;
    }
    let scores2 = project(&pc2, &centered);

    PcaResult { scores1, scores2, mean, pc1, pc2 }
}

/// Project arbitrary points (e.g. centroids) into the same PCA space
fn project_to_pca(points: &[Vec<f32>], pca: &PcaResult) -> (Vec<f32>, Vec<f32>) {
    let xs: Vec<f32> = points.iter().map(|p| {
        let centered: Vec<f32> = p.iter().zip(&pca.mean).map(|(&x, &m)| x - m as f32).collect();
        dot(&centered, &pca.pc1)
    }).collect();
    let ys: Vec<f32> = points.iter().map(|p| {
        let centered: Vec<f32> = p.iter().zip(&pca.mean).map(|(&x, &m)| x - m as f32).collect();
        dot(&centered, &pca.pc2)
    }).collect();
    (xs, ys)
}

// ─── Plotting helpers ─────────────────────────────────────────────

const PALETTE: [RGBColor; 15] = [
    RGBColor(31, 119, 180), RGBColor(255, 127, 14), RGBColor(44, 160, 44),
    RGBColor(214, 39, 40), RGBColor(148, 103, 189), RGBColor(140, 86, 75),
    RGBColor(227, 119, 194), RGBColor(127, 127, 127), RGBColor(188, 189, 34),
    RGBColor(23, 190, 207), RGBColor(174, 199, 232), RGBColor(255, 187, 120),
    RGBColor(152, 223, 138), RGBColor(255, 152, 150), RGBColor(197, 176, 213),
];

fn chart_bounds(xs: &[f32], ys: &[f32]) -> (f64, f64, f64, f64) {
    let x_min = xs.iter().cloned().fold(f32::MAX, f32::min);
    let x_max = xs.iter().cloned().fold(f32::MIN, f32::max);
    let y_min = ys.iter().cloned().fold(f32::MAX, f32::min);
    let y_max = ys.iter().cloned().fold(f32::MIN, f32::max);
    let px = (x_max - x_min) * 0.05;
    let py = (y_max - y_min) * 0.05;
    ((x_min - px) as f64, (x_max + px) as f64, (y_min - py) as f64, (y_max + py) as f64)
}

/// Scatter plot colored by categorical labels (cluster, folder, extension)
fn plot_scatter_categorical(
    filename: &str, title: &str,
    xs: &[f32], ys: &[f32],
    categories: &[String],
) -> Result<()> {
    let path = format!("{}/{}", OUTPUT_DIR, filename);
    let root = BitMapBackend::new(&path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x0, x1, y0, y1) = chart_bounds(xs, ys);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(15).x_label_area_size(35).y_label_area_size(45)
        .build_cartesian_2d(x0..x1, y0..y1)?;
    chart.configure_mesh().x_desc("PC1").y_desc("PC2").draw()?;

    // Get unique categories sorted by frequency
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for c in categories { *freq.entry(c.as_str()).or_default() += 1; }
    let mut sorted_cats: Vec<(&str, usize)> = freq.into_iter().collect();
    sorted_cats.sort_by(|a, b| b.1.cmp(&a.1));

    // Top 12 categories, rest = "other"
    let top_cats: Vec<&str> = sorted_cats.iter().take(12).map(|&(c, _)| c).collect();

    for (ci, &cat) in top_cats.iter().enumerate() {
        let color = PALETTE[ci % PALETTE.len()];
        let points: Vec<(f64, f64)> = (0..xs.len())
            .filter(|&i| categories[i] == cat)
            .map(|i| (xs[i] as f64, ys[i] as f64))
            .collect();
        chart.draw_series(points.iter().map(|&(x, y)| {
            Circle::new((x, y), 2, ShapeStyle::from(color.mix(0.6)).filled())
        }))?.label(cat).legend(move |(x, y)| Circle::new((x + 10, y), 4, color.filled()));
    }

    // "other"
    let other_points: Vec<(f64, f64)> = (0..xs.len())
        .filter(|&i| !top_cats.contains(&categories[i].as_str()))
        .map(|i| (xs[i] as f64, ys[i] as f64))
        .collect();
    if !other_points.is_empty() {
        let gray = RGBColor(200, 200, 200);
        chart.draw_series(other_points.iter().map(|&(x, y)| {
            Circle::new((x, y), 1, ShapeStyle::from(gray.mix(0.4)).filled())
        }))?.label("other").legend(move |(x, y)| Circle::new((x + 10, y), 4, gray.filled()));
    }

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    root.present()?;
    println!("Plot saved: {}", path);
    Ok(())
}

/// Scatter plot by cluster with centroid markers
fn plot_scatter_with_centroids(
    filename: &str, title: &str,
    xs: &[f32], ys: &[f32],
    categories: &[String],
    cx: &[f32], cy: &[f32],
    k: usize,
) -> Result<()> {
    let path = format!("{}/{}", OUTPUT_DIR, filename);
    let root = BitMapBackend::new(&path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x0, x1, y0, y1) = chart_bounds(xs, ys);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(15).x_label_area_size(35).y_label_area_size(45)
        .build_cartesian_2d(x0..x1, y0..y1)?;
    chart.configure_mesh().x_desc("PC1").y_desc("PC2").draw()?;

    // Draw points per cluster
    for ci in 0..k {
        let cat = format!("cluster_{}", ci);
        let color = PALETTE[ci % PALETTE.len()];
        let points: Vec<(f64, f64)> = (0..xs.len())
            .filter(|&i| categories[i] == cat)
            .map(|i| (xs[i] as f64, ys[i] as f64))
            .collect();
        chart.draw_series(points.iter().map(|&(x, y)| {
            Circle::new((x, y), 2, ShapeStyle::from(color.mix(0.5)).filled())
        }))?.label(format!("Cluster {}", ci))
            .legend(move |(x, y)| Circle::new((x + 10, y), 4, color.filled()));
    }

    // Draw centroids as large stars with black border
    for ci in 0..k {
        let color = PALETTE[ci % PALETTE.len()];
        chart.draw_series(std::iter::once(
            Circle::new((cx[ci] as f64, cy[ci] as f64), 12,
                ShapeStyle::from(color).filled())
        ))?;
        chart.draw_series(std::iter::once(
            Circle::new((cx[ci] as f64, cy[ci] as f64), 12,
                ShapeStyle::from(BLACK).stroke_width(3))
        ))?;
        // Inner white dot for visibility
        chart.draw_series(std::iter::once(
            Circle::new((cx[ci] as f64, cy[ci] as f64), 4,
                ShapeStyle::from(WHITE).filled())
        ))?;
    }

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    root.present()?;
    println!("Plot saved: {}", path);
    Ok(())
}

/// Scatter plot colored by continuous value (file size, aspect ratio)
fn plot_scatter_gradient(
    filename: &str, title: &str,
    xs: &[f32], ys: &[f32],
    values: &[f32], value_label: &str,
) -> Result<()> {
    let path = format!("{}/{}", OUTPUT_DIR, filename);
    let root = BitMapBackend::new(&path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x0, x1, y0, y1) = chart_bounds(xs, ys);

    // Use percentile clipping to handle outliers
    let mut sorted_vals: Vec<f32> = values.to_vec();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p5 = sorted_vals[sorted_vals.len() * 5 / 100];
    let p95 = sorted_vals[sorted_vals.len() * 95 / 100];
    let range = if (p95 - p5).abs() < 1e-6 { 1.0 } else { p95 - p5 };

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} (color: {})", title, value_label), ("sans-serif", 22))
        .margin(15).x_label_area_size(35).y_label_area_size(45)
        .build_cartesian_2d(x0..x1, y0..y1)?;
    chart.configure_mesh().x_desc("PC1").y_desc("PC2").draw()?;

    chart.draw_series((0..xs.len()).map(|i| {
        let t = ((values[i] - p5) / range).clamp(0.0, 1.0);
        // Blue (low) → Red (high)
        let r = (t * 255.0) as u8;
        let b = ((1.0 - t) * 255.0) as u8;
        let g = ((0.5 - (t - 0.5).abs()) * 255.0 * 2.0).max(0.0) as u8;
        Circle::new(
            (xs[i] as f64, ys[i] as f64), 2,
            ShapeStyle::from(RGBColor(r, g, b).mix(0.6)).filled(),
        )
    }))?;

    root.present()?;
    println!("Plot saved: {}", path);
    Ok(())
}

fn plot_elbow_silhouette(k_values: &[usize], inertias: &[f64], silhouettes: &[f64]) -> Result<()> {
    let path = format!("{}/elbow_silhouette.png", OUTPUT_DIR);
    let root = BitMapBackend::new(&path, (1400, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));

    {
        let max_i = inertias.iter().cloned().fold(0.0f64, f64::max) * 1.05;
        let mut chart = ChartBuilder::on(&areas[0])
            .caption("Elbow Method (Inertia / WCSS)", ("sans-serif", 22))
            .margin(15).x_label_area_size(35).y_label_area_size(60)
            .build_cartesian_2d(
                (*k_values.first().unwrap() as f64)..(*k_values.last().unwrap() as f64), 0.0..max_i,
            )?;
        chart.configure_mesh().x_desc("K").y_desc("Inertia").draw()?;
        chart.draw_series(LineSeries::new(
            k_values.iter().zip(inertias).map(|(&k, &v)| (k as f64, v)), &BLUE,
        ))?;
        chart.draw_series(k_values.iter().zip(inertias)
            .map(|(&k, &v)| Circle::new((k as f64, v), 4, BLUE.filled())))?;
    }
    {
        let max_s = silhouettes.iter().cloned().fold(0.0f64, f64::max) * 1.15;
        let min_s = silhouettes.iter().cloned().fold(f64::MAX, f64::min) * 0.85;
        let best_idx = silhouettes.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Silhouette Score", ("sans-serif", 22))
            .margin(15).x_label_area_size(35).y_label_area_size(60)
            .build_cartesian_2d(
                (*k_values.first().unwrap() as f64)..(*k_values.last().unwrap() as f64), min_s..max_s,
            )?;
        chart.configure_mesh().x_desc("K").y_desc("Silhouette").draw()?;
        chart.draw_series(LineSeries::new(
            k_values.iter().zip(silhouettes).map(|(&k, &v)| (k as f64, v)), &RED,
        ))?;
        chart.draw_series(k_values.iter().zip(silhouettes)
            .map(|(&k, &v)| Circle::new((k as f64, v), 4, RED.filled())))?;
        chart.draw_series(std::iter::once(Circle::new(
            (k_values[best_idx] as f64, silhouettes[best_idx]), 8, GREEN.filled(),
        )))?;
    }
    root.present()?;
    println!("Plot saved: {}", path);
    Ok(())
}

// ─── Cluster Report ───────────────────────────────────────────────

fn analyze_clusters(paths: &[PathBuf], labels: &[usize], k: usize, metas: &[ImageMeta]) -> Result<()> {
    let mut report = String::new();
    report.push_str(&format!("{}\nCLUSTER ANALYSIS REPORT — K={}\n{}\n\n", "=".repeat(70), k, "=".repeat(70)));

    let total = labels.len();
    for cluster in 0..k {
        let members: Vec<usize> = (0..total).filter(|&i| labels[i] == cluster).collect();
        let n = members.len();
        report.push_str(&format!("\n--- Cluster {} ({} images, {:.1}%) ---\n", cluster, n, n as f64 / total as f64 * 100.0));

        let mut folder_counts: HashMap<&str, usize> = HashMap::new();
        let mut ext_counts: HashMap<&str, usize> = HashMap::new();
        let mut sizes: Vec<u64> = Vec::new();
        let mut aspects: Vec<f32> = Vec::new();

        for &i in &members {
            *folder_counts.entry(&metas[i].top_folder).or_default() += 1;
            *ext_counts.entry(&metas[i].extension).or_default() += 1;
            if metas[i].file_size > 0 { sizes.push(metas[i].file_size); }
            if metas[i].aspect > 0.0 { aspects.push(metas[i].aspect); }
        }

        let mut folders: Vec<_> = folder_counts.into_iter().collect();
        folders.sort_by(|a, b| b.1.cmp(&a.1));
        report.push_str("  Top folders:\n");
        for (folder, count) in folders.iter().take(7) {
            report.push_str(&format!("    {}: {} ({:.1}%)\n", folder, count, *count as f64 / n as f64 * 100.0));
        }

        let mut exts: Vec<_> = ext_counts.into_iter().collect();
        exts.sort_by(|a, b| b.1.cmp(&a.1));
        report.push_str("  Extensions:\n");
        for (ext, count) in exts.iter().take(5) {
            report.push_str(&format!("    .{}: {} ({:.1}%)\n", ext, count, *count as f64 / n as f64 * 100.0));
        }

        if !sizes.is_empty() {
            sizes.sort();
            let avg = sizes.iter().sum::<u64>() as f64 / sizes.len() as f64;
            let median = sizes[sizes.len() / 2];
            report.push_str(&format!("  File size: avg={:.0}KB, median={:.0}KB, min={:.0}KB, max={:.1}MB\n",
                avg / 1024.0, median as f64 / 1024.0,
                *sizes.first().unwrap() as f64 / 1024.0,
                *sizes.last().unwrap() as f64 / 1024.0 / 1024.0));
        }

        if !aspects.is_empty() {
            aspects.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let landscape = aspects.iter().filter(|&&a| a > 1.1).count();
            let portrait = aspects.iter().filter(|&&a| a < 0.9).count();
            let square = aspects.len() - landscape - portrait;
            report.push_str(&format!("  Aspect: landscape={} ({:.0}%), portrait={} ({:.0}%), square={} ({:.0}%)\n",
                landscape, landscape as f64 / aspects.len() as f64 * 100.0,
                portrait, portrait as f64 / aspects.len() as f64 * 100.0,
                square, square as f64 / aspects.len() as f64 * 100.0));
        }

        report.push_str(&format!("  Unique top-folders: {}\n", folders.len()));
        report.push_str("  Sample files:\n");
        let mut sample_idx: Vec<usize> = members.clone();
        sample_idx.shuffle(&mut StdRng::seed_from_u64(SEED + cluster as u64));
        for &i in sample_idx.iter().take(5) {
            let name = paths[i].file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
            report.push_str(&format!("    - {}\n", name));
        }
    }

    print!("{}", report);
    std::fs::write(format!("{}/cluster_report.txt", OUTPUT_DIR), &report)?;
    println!("\nReport saved: {}/cluster_report.txt", OUTPUT_DIR);
    Ok(())
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let media_dir = "D:/media";
    std::fs::create_dir_all(OUTPUT_DIR)?;

    // 1. Embeddings: load from cache or generate
    let (valid_paths, mut embeddings) = if cache_exists() {
        println!("Found embedding cache, loading...");
        load_cache()?
    } else {
        println!("Scanning for images...");
        let mut all_paths = collect_image_paths(media_dir);
        println!("Found {} images total", all_paths.len());

        // Shuffle all paths; we'll draw from this pool until we have SAMPLE_SIZE valid embeddings
        all_paths.shuffle(&mut StdRng::seed_from_u64(SEED));

        println!("Loading CLIP ViT-B/32 model...");
        let mut model = ImageEmbedding::try_new(
            ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
        )?;
        println!("Model loaded. Generating embeddings (target: {})...", SAMPLE_SIZE);

        let pb = ProgressBar::new(SAMPLE_SIZE as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:50.cyan/blue}] {pos}/{len} valid ({eta})")?
            .progress_chars("=>-"));

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(SAMPLE_SIZE);
        let mut valid_paths: Vec<PathBuf> = Vec::with_capacity(SAMPLE_SIZE);
        let mut failed = 0usize;
        let mut idx = 0;

        while embeddings.len() < SAMPLE_SIZE && idx < all_paths.len() {
            let end = (idx + BATCH_SIZE).min(all_paths.len());
            let chunk = &all_paths[idx..end];
            idx = end;

            let str_paths: Vec<String> = chunk.iter().map(|p| p.to_string_lossy().into_owned()).collect();
            match model.embed(&str_paths, Some(BATCH_SIZE)) {
                Ok(embs) => {
                    for (path, emb) in chunk.iter().zip(embs) {
                        if embeddings.len() >= SAMPLE_SIZE { break; }
                        valid_paths.push(path.clone());
                        embeddings.push(emb);
                    }
                }
                Err(_) => {
                    // Fallback: try individually to skip corrupted
                    for path in chunk {
                        if embeddings.len() >= SAMPLE_SIZE { break; }
                        let single = vec![path.to_string_lossy().into_owned()];
                        match model.embed(&single, Some(1)) {
                            Ok(embs) => {
                                valid_paths.push(path.clone());
                                embeddings.push(embs.into_iter().next().unwrap());
                            }
                            Err(_) => failed += 1,
                        }
                    }
                }
            }
            pb.set_position(embeddings.len() as u64);
        }
        pb.finish();
        println!("{} embedded, {} failed/skipped", embeddings.len(), failed);
        drop(model);

        save_cache(&valid_paths, &embeddings)?;
        (valid_paths, embeddings)
    };

    // 2. L2-normalize
    println!("Normalizing...");
    l2_normalize(&mut embeddings);

    // 3. Collect metadata
    println!("Collecting metadata (dimensions, sizes)...");
    let metas = collect_metadata(&valid_paths);

    // 4. PCA 2D (computed once, reused for all plots)
    println!("Computing PCA...");
    let pca = compute_pca_2d(&embeddings);
    let pc1 = &pca.scores1;
    let pc2 = &pca.scores2;

    // 5. Elbow + Silhouette
    let mut k_values = Vec::new();
    let mut inertias = Vec::new();
    let mut silhouettes = Vec::new();

    println!("\nK-Means for K={}..{}:", K_MIN, K_MAX);
    for k in K_MIN..=K_MAX {
        let mut best_inertia = f64::MAX;
        let mut best_labels = vec![];
        for run in 0..3u64 {
            let (labels, _, inertia) = kmeans(&embeddings, k, SEED + run + k as u64 * 100);
            if inertia < best_inertia {
                best_inertia = inertia;
                best_labels = labels;
            }
        }
        let sil = silhouette_score(&embeddings, &best_labels, k);
        println!("  K={:2} | inertia={:12.1} | silhouette={:.4}", k, best_inertia, sil);
        k_values.push(k);
        inertias.push(best_inertia);
        silhouettes.push(sil);

        let content: String = best_labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(format!("{}/labels_k{}.txt", OUTPUT_DIR, k), content)?;
    }

    // 6. Plots
    println!("\nGenerating plots...");
    plot_elbow_silhouette(&k_values, &inertias, &silhouettes)?;

    let best_idx = silhouettes.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let best_k = k_values[best_idx];
    println!("\nBest K by silhouette: {} ({:.4})", best_k, silhouettes[best_idx]);

    // Final clustering
    println!("Final K-Means K={}...", best_k);
    let mut best_inertia = f64::MAX;
    let mut final_labels = vec![];
    let mut final_centroids = vec![];
    for run in 0..5u64 {
        let (labels, centroids, inertia) = kmeans(&embeddings, best_k, SEED + run + 1000);
        if inertia < best_inertia {
            best_inertia = inertia;
            final_labels = labels;
            final_centroids = centroids;
        }
    }

    // === ALL SCATTER PLOTS ===

    // Project centroids to PCA space
    let (cx, cy) = project_to_pca(&final_centroids, &pca);

    // a) By cluster (with centroids)
    let cluster_cats: Vec<String> = final_labels.iter().map(|l| format!("cluster_{}", l)).collect();
    plot_scatter_with_centroids("pca_by_cluster.png",
        &format!("PCA — K-Means Clusters (K={})", best_k),
        pc1, pc2, &cluster_cats, &cx, &cy, best_k)?;

    // b) By top folder (source)
    let folder_cats: Vec<String> = metas.iter().map(|m| m.top_folder.clone()).collect();
    plot_scatter_categorical("pca_by_folder.png",
        "PCA — Colored by Source Folder",
        &pc1, &pc2, &folder_cats)?;

    // c) By extension (type)
    let ext_cats: Vec<String> = metas.iter().map(|m| format!(".{}", m.extension)).collect();
    plot_scatter_categorical("pca_by_extension.png",
        "PCA — Colored by Image Type",
        &pc1, &pc2, &ext_cats)?;

    // d) By file size (gradient)
    let sizes: Vec<f32> = metas.iter().map(|m| (m.file_size as f32) / 1024.0).collect(); // KB
    plot_scatter_gradient("pca_by_filesize.png",
        "PCA", &pc1, &pc2, &sizes, "File Size (KB)")?;

    // e) By aspect ratio (gradient)
    let aspects: Vec<f32> = metas.iter().map(|m| m.aspect).collect();
    plot_scatter_gradient("pca_by_aspect.png",
        "PCA", &pc1, &pc2, &aspects, "Aspect Ratio (W/H)")?;

    // f) By aspect category
    let aspect_cats: Vec<String> = metas.iter().map(|m| {
        if m.aspect == 0.0 { "unknown".to_string() }
        else if m.aspect > 1.1 { "landscape".to_string() }
        else if m.aspect < 0.9 { "portrait".to_string() }
        else { "square".to_string() }
    }).collect();
    plot_scatter_categorical("pca_by_aspect_cat.png",
        "PCA — Colored by Aspect Ratio",
        &pc1, &pc2, &aspect_cats)?;

    // 7. Report
    analyze_clusters(&valid_paths, &final_labels, best_k, &metas)?;

    println!("\nAll done! Output in {}/", OUTPUT_DIR);
    Ok(())
}
