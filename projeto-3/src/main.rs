//! Mini-Projeto 3 -- Regressao Linear: Gradient Descent vs LMS (Widrow-Hoff)
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::fs;
use std::io::Write as _;

const SEED: u64 = 42;
const N_SAMPLES: usize = 300;
// Multi-seed usa um dataset GRANDE: o std entre seeds e dominado pelo erro
// amostral (~1/sqrt(n)), nao pelo nº de seeds. Como o dado e SINTETICO, mais
// amostras = estimativa de MSE/pesos quase de graca (Rust roda 10k instantaneo).
// A ilustracao single-seed segue em N_SAMPLES=300 (figuras legiveis).
const MS_N_SAMPLES: usize = 10_000;
const N_FEATURES: usize = 2;
const TRUE_W: [f64; N_FEATURES] = [3.0, -2.0];
const TRUE_BIAS: f64 = 1.0;
const NOISE_STD: f64 = 0.5;
const GD_LR: f64 = 0.01;
const LMS_LR: f64 = 0.001;
const MAX_ITERS: usize = 200;
// Axis padding constants for plots (in data units)
const SCATTER_Y_PAD: f64 = 1.0; // 1-unit margin so regression lines extend visually beyond data range
const PRED_PAD: f64 = 0.5;      // 0.5-unit margin around pred-vs-actual scatter range
const N_LINE_STEPS: usize = 100; // segments for regression line curves; ~12px/step at 1200px gives smooth lines
// OUTPUT_DIR: resolved at runtime. Priority:
//   1. PROJETO3_OUTPUT env var (absolute path, for running binary outside project root)
//   2. If binary lives inside a `target/` subtree, walk up to the directory containing
//      Cargo.toml and use <project-root>/output — fixes `cargo run` without PROJETO3_OUTPUT
//   3. Runtime: directory of the current executable + /output (portable across machines)
//   4. Compile-time CARGO_MANIFEST_DIR + /output (last resort; only valid on build machine)
fn output_dir() -> String {
    if let Ok(v) = std::env::var("PROJETO3_OUTPUT") {
        return v;
    }
    // If the binary is inside a target/ subtree (i.e. invoked via `cargo run`),
    // walk up until we find a directory that contains Cargo.toml and use
    // <project-root>/output.  This prevents output silently landing in
    // target/release/output/ when the user runs `cargo run` without the Makefile.
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.as_path();
        // Walk up through parent directories
        loop {
            if let Some(parent) = dir.parent() {
                // Check if this ancestor contains Cargo.toml
                if parent.join("Cargo.toml").exists() {
                    // Only use this path if the binary is inside a target/ subdir
                    // (to avoid false positives when the binary is copied elsewhere)
                    let exe_str = exe.to_string_lossy();
                    if exe_str.contains("/target/") || exe_str.contains("\\target\\") {
                        return parent.join("output").to_string_lossy().into_owned();
                    }
                    break;
                }
                dir = parent;
            } else {
                break;
            }
        }
    }
    // Use the binary's runtime location so the output goes next to the binary,
    // even when copied to another machine or directory without PROJETO3_OUTPUT set.
    if let Some(dir) = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.join("output").to_string_lossy().into_owned()))
    {
        return dir;
    }
    // Fallback: compile-time path (only valid on the original build machine)
    format!("{}/output", env!("CARGO_MANIFEST_DIR"))
}

// Path to a TrueType font available on this system
// Font paths tried in order (Debian/Ubuntu first, then Arch Linux)
const FONT_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
];
const FONT_BOLD_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
];

fn register_fonts() {
    use std::sync::Once;
    static FONT_INIT: Once = Once::new();
    FONT_INIT.call_once(|| {
    use plotters::style::register_font;
    let loaded_normal = FONT_PATHS.iter().any(|path| {
        if let Ok(bytes) = std::fs::read(path) {
            register_font("sans-serif", plotters::style::FontStyle::Normal,
                // Box::leak is intentional: plotters requires a static [u8] for
                // registered fonts; the process lifetime covers the leak.
                Box::leak(bytes.into_boxed_slice())).is_ok()
        } else {
            false
        }
    });
    if !loaded_normal {
        eprintln!("Warning: nenhuma fonte normal encontrada em {:?}, labels dos plots podem aparecer degradados", FONT_PATHS);
    }
    let loaded_bold = FONT_BOLD_PATHS.iter().any(|path| {
        if let Ok(bytes) = std::fs::read(path) {
            register_font("sans-serif", plotters::style::FontStyle::Bold,
                // Box::leak is intentional: plotters requires a static [u8] for
                // registered fonts; the process lifetime covers the leak.
                Box::leak(bytes.into_boxed_slice())).is_ok()
        } else {
            false
        }
    });
    if !loaded_bold {
        eprintln!("Warning: nenhuma fonte bold encontrada em {:?}, labels dos plots podem aparecer degradados", FONT_BOLD_PATHS);
    }
    }); // end FONT_INIT.call_once
}

fn generate_dataset(rng: &mut StdRng, n_samples: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let feat_dist = Normal::new(0.0, 1.0).unwrap();
    let noise_dist = Normal::new(0.0, NOISE_STD).unwrap();
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let xi: Vec<f64> = (0..N_FEATURES).map(|_| feat_dist.sample(rng)).collect();
        let yi = dot(&xi, &TRUE_W) + TRUE_BIAS + noise_dist.sample(rng);
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn predict(x: &[f64], w: &[f64], b: f64) -> f64 {
    dot(x, w) + b
}

fn mse(x: &[Vec<f64>], y: &[f64], w: &[f64], b: f64) -> f64 {
    let n = x.len() as f64;
    x.iter().zip(y).map(|(xi, &yi)| {
        let e = predict(xi, w, b) - yi;
        e * e
    }).sum::<f64>() / n
}

fn batch_gradient_descent(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_iters: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let n = x.len() as f64;
    let mut w = vec![0.0f64; N_FEATURES];
    let mut b = 0.0f64;
    let mut history = Vec::with_capacity(max_iters + 1);
    history.push(mse(x, y, &w, b));
    for _iter in 0..max_iters {
        let mut grad_w = vec![0.0f64; N_FEATURES];
        let mut grad_b = 0.0f64;
        for (xi, &yi) in x.iter().zip(y) {
            let err = predict(xi, &w, b) - yi;
            for (gw, &xij) in grad_w.iter_mut().zip(xi) {
                *gw += err * xij;
            }
            grad_b += err;
        }
        for (wj, gw) in w.iter_mut().zip(&grad_w) {
            *wj -= lr * gw / n;
        }
        b -= lr * grad_b / n;
        history.push(mse(x, y, &w, b));
    }
    (w, b, history)
}

fn lms_widrow_hoff(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_iters: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let mut w = vec![0.0f64; N_FEATURES];
    let mut b = 0.0f64;
    let mut history = Vec::with_capacity(max_iters + 1);
    history.push(mse(x, y, &w, b));
    for _epoch in 0..max_iters {
        for (xi, &yi) in x.iter().zip(y) {
            let err = yi - predict(xi, &w, b);
            for (wj, &xij) in w.iter_mut().zip(xi) {
                *wj += lr * err * xij;
            }
            b += lr * err;
        }
        history.push(mse(x, y, &w, b));
    }
    (w, b, history)
}

fn plot_convergence(gd_history: &[f64], lms_history: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/convergence.png", output_dir());
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Compute a shared y-range across both series so the 2x MSE gap is visible
    let global_max = gd_history.iter().chain(lms_history.iter())
        .cloned().fold(0.0f64, f64::max) * 1.05;
    let global_min_raw = gd_history.iter().chain(lms_history.iter())
        .cloned().fold(f64::MAX, f64::min);
    // Subtract absolute padding (5% of visible range) so the lower bound
    // works correctly even when global_min_raw is near zero.
    // Floor at 0.0 because MSE is always non-negative.
    let global_min = (global_min_raw - 0.05 * (global_max - global_min_raw)).max(0.0);
    let n_epochs = gd_history.len();

    let gd_final = gd_history.last().unwrap();
    let lms_final = lms_history.last().unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("GD vs LMS — Convergencia (eixo compartilhado)", ("sans-serif", 32))
        .margin(25)
        .x_label_area_size(55)
        .y_label_area_size(90)
        .build_cartesian_2d(0usize..n_epochs, global_min..global_max)?;

    chart.configure_mesh()
        .x_desc("Epoca (0 = inicial; 200 = final)")
        .y_desc("MSE Loss")
        .label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .draw()?;

    // GD curve (blue)
    chart.draw_series(LineSeries::new(
        gd_history.iter().enumerate().map(|(i, &v)| (i, v)),
        BLUE.stroke_width(3),
    ))?
    .label(format!("GD  (MSE final: {:.4})", gd_final))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));

    // LMS curve (red)
    chart.draw_series(LineSeries::new(
        lms_history.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.stroke_width(3),
    ))?
    .label(format!("LMS (MSE final: {:.4})", lms_final))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));

    chart.configure_series_labels()
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Plot salvo: {}", path);
    Ok(())
}

fn plot_scatter(
    x: &[Vec<f64>],
    y: &[f64],
    gd_w: &[f64],
    gd_b: f64,
    lms_w: &[f64],
    lms_b: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/scatter.png", output_dir());
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let x1: Vec<f64> = x.iter().map(|xi| xi[0]).collect();
    let x_min = x1.iter().cloned().fold(f64::MAX, f64::min);
    let x_max = x1.iter().cloned().fold(f64::MIN, f64::max);
    // Symmetric 5% absolute padding so negative x_min values are not clipped
    // (multiplying by 0.95 would move a negative x_min toward zero, cutting data)
    let x_pad = (x_max - x_min) * 0.05;
    let y_min = y.iter().cloned().fold(f64::MAX, f64::min) - SCATTER_Y_PAD;
    let y_max = y.iter().cloned().fold(f64::MIN, f64::max) + SCATTER_Y_PAD;

    let mut chart = ChartBuilder::on(&root)
        .caption("x1 vs y — Regressao Marginal", ("sans-serif", 32))
        .margin(25)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d((x_min - x_pad)..(x_max + x_pad), y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("x1")
        .y_desc("y")
        .label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .draw()?;

    chart.draw_series(x1.iter().zip(y).map(|(&xi1, &yi)| {
        Circle::new((xi1, yi), 3, BLACK.mix(0.3).filled())
    }))?
    .label("Dados")
    .legend(|(x, y)| Circle::new((x + 10, y), 4, BLACK.filled()));

    let step_size = (x_max - x_min) / N_LINE_STEPS as f64;

    let true_line: Vec<(f64, f64)> = (0..=N_LINE_STEPS)
        .map(|i| { let xi1 = x_min + i as f64 * step_size; (xi1, TRUE_W[0] * xi1 + TRUE_BIAS) })
        .collect();
    chart.draw_series(LineSeries::new(true_line, GREEN.stroke_width(3)))?
        .label(format!("Verdadeiro (w0={:.1}, w1={:.1}, b={:.1}) [marginal]", TRUE_W[0], TRUE_W[1], TRUE_BIAS))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(3)));

    let gd_line: Vec<(f64, f64)> = (0..=N_LINE_STEPS)
        .map(|i| { let xi1 = x_min + i as f64 * step_size; (xi1, gd_w[0] * xi1 + gd_b) })
        .collect();
    chart.draw_series(LineSeries::new(gd_line, BLUE.stroke_width(3)))?
        .label(format!("GD  (w0={:.3}, b={:.3}) [regressao marginal em x1; w1 nao entra: E[x2]=0]", gd_w[0], gd_b))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));

    let lms_line: Vec<(f64, f64)> = (0..=N_LINE_STEPS)
        .map(|i| { let xi1 = x_min + i as f64 * step_size; (xi1, lms_w[0] * xi1 + lms_b) })
        .collect();
    chart.draw_series(LineSeries::new(lms_line, RED.stroke_width(3)))?
        .label(format!("LMS (w0={:.3}, b={:.3}) [regressao marginal em x1; w1 nao entra: E[x2]=0]", lms_w[0], lms_b))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));

    chart.configure_series_labels()
        .label_font(("sans-serif", 24))
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Plot salvo: {}", path);
    Ok(())
}


fn plot_pred_vs_actual(
    x: &[Vec<f64>],
    y: &[f64],
    gd_w: &[f64],
    gd_b: f64,
    lms_w: &[f64],
    lms_b: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/pred_vs_actual.png", output_dir());
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area(); // same canvas as other plots; wider format suits legend layout
    root.fill(&WHITE)?;

    // Compute predictions for each model using predict()
    let gd_pred: Vec<f64> = x.iter().map(|xi| predict(xi, gd_w, gd_b)).collect();
    let lms_pred: Vec<f64> = x.iter().map(|xi| predict(xi, lms_w, lms_b)).collect();

    let all_vals: Vec<f64> = y.iter().chain(gd_pred.iter()).chain(lms_pred.iter()).cloned().collect();
    let v_min = all_vals.iter().cloned().fold(f64::MAX, f64::min) - PRED_PAD;
    let v_max = all_vals.iter().cloned().fold(f64::MIN, f64::max) + PRED_PAD;

    // Reuse the canonical mse() function instead of duplicating the formula inline
    let gd_mse = mse(x, y, gd_w, gd_b);
    let lms_mse = mse(x, y, lms_w, lms_b);

    let mut chart = ChartBuilder::on(&root)
        .caption("Predito vs Real: GD e LMS", ("sans-serif", 32))
        .margin(30)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(v_min..v_max, v_min..v_max)?;

    chart.configure_mesh()
        .x_desc("y real")
        .y_desc("y predito")
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 24))
        .draw()?;

    // Ideal diagonal y = x
    chart.draw_series(LineSeries::new(
        vec![(v_min, v_min), (v_max, v_max)],
        GREEN.stroke_width(2),
    ))?
    .label("Ideal (y=x)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));

    // GD predictions
    chart.draw_series(y.iter().zip(gd_pred.iter()).map(|(&yr, &yp)| {
        Circle::new((yr, yp), 4, BLUE.mix(0.5).filled())
    }))?
    .label(format!("GD  (MSE={:.4})", gd_mse))
    .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.mix(0.8).filled()));

    // LMS predictions
    chart.draw_series(y.iter().zip(lms_pred.iter()).map(|(&yr, &yp)| {
        Circle::new((yr, yp), 4, RED.mix(0.5).filled())
    }))?
    .label(format!("LMS (MSE={:.4})", lms_mse))
    .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.mix(0.8).filled()));

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 22))
        .draw()?;

    root.present()?;
    println!("Plot salvo: {}", path);
    Ok(())
}
fn save_report(
    gd_w: &[f64], gd_b: f64, gd_history: &[f64],
    lms_w: &[f64], lms_b: f64, lms_history: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/report.txt", output_dir());
    let mut f = fs::File::create(&path)?;
    writeln!(f, "================================================================")?;
    writeln!(f, "  MINI-PROJETO 3 -- Regressao Linear: GD vs LMS")?;
    writeln!(f, "================================================================")?;
    writeln!(f)?;
    writeln!(f, "Dataset")?;
    writeln!(f, "  Amostras   : {}", N_SAMPLES)?;
    writeln!(f, "  Features   : {}", N_FEATURES)?;
    writeln!(f, "  Pesos reais: [{:.1}, {:.1}]", TRUE_W[0], TRUE_W[1])?;
    writeln!(f, "  Bias real  : {}", TRUE_BIAS)?;
    writeln!(f, "  Ruido std  : {}", NOISE_STD)?;
    writeln!(f)?;
    writeln!(f, "Configuracao")?;
    writeln!(f, "  GD  learning rate : {}", GD_LR)?;
    writeln!(f, "  LMS learning rate : {}", LMS_LR)?;
    writeln!(f, "  Epocas            : {}", MAX_ITERS)?;
    writeln!(f)?;
    writeln!(f, "Resultados")?;
    writeln!(f, "  {:30} {:>12} {:>12} {:>14}", "Metrica", "GD", "LMS", "Ref/Min.Teorico")?;
    writeln!(f, "  {}", "-".repeat(70))?;
    writeln!(f, "  {:30} {:>12.6} {:>12.6} {:>14}", "MSE final",
        gd_history.last().unwrap(), lms_history.last().unwrap(), format!("{:.6} (*)", NOISE_STD * NOISE_STD))?;
    writeln!(f, "  {:30} {:>12.6} {:>12.6} {:>12.6}", "w[0] (true=3.0)", gd_w[0], lms_w[0], TRUE_W[0])?;
    writeln!(f, "  {:30} {:>12.6} {:>12.6} {:>12.6}", "w[1] (true=-2.0)", gd_w[1], lms_w[1], TRUE_W[1])?;
    writeln!(f, "  {:30} {:>12.6} {:>12.6} {:>12.6}", "bias (true=1.0)", gd_b, lms_b, TRUE_BIAS)?;
    writeln!(f, "  {:30} {:>12.6} {:>12.6}", "MSE inicial", gd_history[0], lms_history[0])?;
    let gd_reduction = (1.0 - gd_history.last().unwrap() / gd_history[0]) * 100.0;
    let lms_reduction = (1.0 - lms_history.last().unwrap() / lms_history[0]) * 100.0;
    writeln!(f, "  {:30} {:>11.2}% {:>11.2}%", "Reducao MSE (%)", gd_reduction, lms_reduction)?;
    writeln!(f)?;
    writeln!(f, "Plots gerados")?;
    writeln!(f, "  {}/convergence.png", output_dir())?;
    writeln!(f, "  {}/scatter.png", output_dir())?;
    writeln!(f, "  {}/pred_vs_actual.png", output_dir())?;
    writeln!(f)?;
    writeln!(f, "(*) sigma^2 = limite inferior teorico do MSE (ruido irredutivel), nao um valor de parametro verdadeiro")?;
    writeln!(f, "================================================================")?;
    println!("Relatorio salvo: {}", path);
    Ok(())
}

// =============================================================================
// Multi-seed -- robustez estatistica (media +/- desvio amostral sobre N seeds)
// -----------------------------------------------------------------------------
// Espelha run_multiseed() do projeto-5 (Python), adaptado ao Rust. Cada seed
// gera uma nova realizacao do dataset sintetico y=3x1-2x2+1+ruido; coletamos,
// por seed, o MSE final e os pesos (w0, w1, bias) de GD e de LMS, e agregamos
// media +/- desvio amostral (ddof=1). Este e o RESULTADO OFICIAL. O GD segue
// deliberadamente nao-convergido (lr baixo, didatico); so reportamos mean+/-std.
// =============================================================================

/// Metricas finais de UMA seed (uma realizacao do dataset).
struct SeedMetrics {
    gd_mse: f64, gd_w0: f64, gd_w1: f64, gd_b: f64,
    lms_mse: f64, lms_w0: f64, lms_w1: f64, lms_b: f64,
}

/// media + desvio amostral de uma metrica agregada sobre as seeds.
struct Stat { mean: f64, std: f64 }

/// Resultado multi-seed agregado (o que vai pro report e pro plot).
struct MultiseedStats {
    n: usize,
    seeds: Vec<u64>,
    gd_mse: Stat, gd_w0: Stat, gd_w1: Stat, gd_b: Stat,
    lms_mse: Stat, lms_w0: Stat, lms_w1: Stat, lms_b: Stat,
}

/// Media e desvio amostral (ddof=1, divide por N-1). Com 1 valor, std=0.
fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = if v.len() > 1 {
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    (mean, std)
}

/// Roda o experimento completo para uma seed e devolve as metricas finais.
fn run_seed(seed: u64) -> SeedMetrics {
    let mut rng = StdRng::seed_from_u64(seed);
    // dataset GRANDE (MS_N_SAMPLES): reduz o erro amostral da estimativa
    let (x, y) = generate_dataset(&mut rng, MS_N_SAMPLES);
    let (gd_w, gd_b, gd_history) = batch_gradient_descent(&x, &y, GD_LR, MAX_ITERS);
    let (lms_w, lms_b, lms_history) = lms_widrow_hoff(&x, &y, LMS_LR, MAX_ITERS);
    SeedMetrics {
        gd_mse: *gd_history.last().unwrap(), gd_w0: gd_w[0], gd_w1: gd_w[1], gd_b,
        lms_mse: *lms_history.last().unwrap(), lms_w0: lms_w[0], lms_w1: lms_w[1], lms_b,
    }
}

/// Roda todas as seeds e agrega media +/- desvio por metrica.
fn run_multiseed(seeds: &[u64]) -> MultiseedStats {
    let results: Vec<SeedMetrics> = seeds.iter().map(|&s| run_seed(s)).collect();
    // extrai uma coluna (uma metrica sobre todas as seeds) e agrega
    let col = |f: fn(&SeedMetrics) -> f64| -> Stat {
        let v: Vec<f64> = results.iter().map(f).collect();
        let (mean, std) = mean_std(&v);
        Stat { mean, std }
    };
    MultiseedStats {
        n: seeds.len(),
        seeds: seeds.to_vec(),
        gd_mse: col(|r| r.gd_mse), gd_w0: col(|r| r.gd_w0),
        gd_w1: col(|r| r.gd_w1), gd_b: col(|r| r.gd_b),
        lms_mse: col(|r| r.lms_mse), lms_w0: col(|r| r.lms_w0),
        lms_w1: col(|r| r.lms_w1), lms_b: col(|r| r.lms_b),
    }
}

/// Monta as linhas do relatorio multi-seed (usadas em stdout e no .txt).
fn multiseed_report_lines(stats: &MultiseedStats) -> Vec<String> {
    let mut l: Vec<String> = Vec::new();
    l.push("================================================================".into());
    l.push("  MINI-PROJETO 3 -- Multi-seed (RESULTADO OFICIAL)".into());
    l.push("================================================================".into());
    l.push(format!("Seeds ({}): {:?}", stats.n, stats.seeds));
    l.push(format!("(cada seed = nova realizacao do dataset sintetico y=3x1-2x2+1+ruido, n={} amostras)", MS_N_SAMPLES));
    l.push("Media +/- desvio amostral (ddof=1, divide por N-1)".into());
    l.push("NOTA: GD e DELIBERADAMENTE nao-convergido (lr baixo, didatico).".into());
    l.push(String::new());
    let hdr = format!("  {:<18} {:>22} {:>22}", "Metrica", "GD", "LMS");
    let bar = format!("  {}", "-".repeat(hdr.len().saturating_sub(2)));
    l.push(hdr);
    l.push(bar.clone());
    let row = |name: &str, g: &Stat, m: &Stat| {
        format!("  {:<18} {:>13.4} +/-{:>6.4} {:>13.4} +/-{:>6.4}",
                name, g.mean, g.std, m.mean, m.std)
    };
    l.push(row("MSE final", &stats.gd_mse, &stats.lms_mse));
    l.push(row("w0 (true=3.0)", &stats.gd_w0, &stats.lms_w0));
    l.push(row("w1 (true=-2.0)", &stats.gd_w1, &stats.lms_w1));
    l.push(row("bias (true=1.0)", &stats.gd_b, &stats.lms_b));
    l.push(bar);
    l.push(format!("Referencia teorica: MSE minimo (ruido irredutivel sigma^2) = {:.4}",
                   NOISE_STD * NOISE_STD));
    l
}

/// Escreve o relatorio multi-seed em output/report_multiseed.txt.
fn save_report_multiseed(lines: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/report_multiseed.txt", output_dir());
    let mut f = fs::File::create(&path)?;
    for line in lines {
        writeln!(f, "{}", line)?;
    }
    println!("Relatorio salvo: {}", path);
    Ok(())
}

/// Barras GD vs LMS do MSE final com barra de erro = desvio; piso do eixo y
/// adaptativo (nao fixo em 0). output/metrics_multiseed.png.
fn plot_metrics_multiseed(stats: &MultiseedStats) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/metrics_multiseed.png", output_dir());
    let root = BitMapBackend::new(&path, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let gd_m = stats.gd_mse.mean; let gd_s = stats.gd_mse.std;
    let lms_m = stats.lms_mse.mean; let lms_s = stats.lms_mse.std;

    // Piso adaptativo: baseado no min/max de (mean +/- std), NAO ancorado em 0.
    let data_min = (gd_m - gd_s).min(lms_m - lms_s);
    let data_max = (gd_m + gd_s).max(lms_m + lms_s);
    let range = (data_max - data_min).max(1e-6);
    // desce 15% do range abaixo do menor (mean-std); so trava em 0 se ficar negativo
    let y_lo = (data_min - 0.15 * range).max(0.0);
    let y_hi = data_max + 0.28 * range; // espaco p/ rotulo de valor acima da barra

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("MSE final: GD vs LMS -- media +/- desvio ({} seeds)", stats.n),
                 ("sans-serif", 28))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(95)
        .build_cartesian_2d(0f64..2f64, y_lo..y_hi)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .x_labels(0) // categorias desenhadas manualmente abaixo
        .y_desc("MSE final (media sobre seeds)")
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 24))
        .draw()?;

    // (x0, x1, mean, std, rotulo, cor)
    let bars = [
        (0.25f64, 0.75f64, gd_m, gd_s, "GD", RGBColor(31, 119, 180)),
        (1.25f64, 1.75f64, lms_m, lms_s, "LMS", RGBColor(214, 39, 40)),
    ];
    for (x0, x1, m, s, name, color) in bars {
        let xc = (x0 + x1) / 2.0;
        // barra: do piso adaptativo ate a media
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y_lo), (x1, m)], color.mix(0.65).filled(),
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y_lo), (x1, m)], color.stroke_width(2),
        )))?;
        // barra de erro vertical (mean-std .. mean+std) + caps
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(xc, m - s), (xc, m + s)], BLACK.stroke_width(2))))?;
        for cap_y in [m - s, m + s] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(xc - 0.09, cap_y), (xc + 0.09, cap_y)], BLACK.stroke_width(2))))?;
        }
        // rotulo: categoria + valor mean +/- std, acima da barra de erro
        chart.draw_series(std::iter::once(Text::new(
            format!("{}: {:.4} +/- {:.4}", name, m, s),
            (x0 + 0.02, m + s + 0.06 * range),
            ("sans-serif", 22).into_font(),
        )))?;
    }

    root.present()?;
    println!("Plot salvo: {}", path);
    Ok(())
}

/// Parse do flag CLI `-n <N>` / `--n-seeds <N>` (default 3, minimo 1).
fn parse_n_seeds() -> usize {
    let mut args = std::env::args().skip(1);
    let mut n: usize = 3;
    while let Some(a) = args.next() {
        if a == "-n" || a == "--n-seeds" {
            if let Some(v) = args.next() {
                if let Ok(p) = v.parse::<usize>() {
                    if p >= 1 { n = p; }
                }
            }
        } else if let Some(v) = a.strip_prefix("-n") {
            // forma colada: -n5
            if let Ok(p) = v.parse::<usize>() {
                if p >= 1 { n = p; }
            }
        }
    }
    n
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir())?;

    // Register fonts before any plotting
    register_fonts();

    let n_seeds = parse_n_seeds();
    let seeds: Vec<u64> = (0..n_seeds as u64).map(|i| SEED + i).collect();

    println!("=== Mini-Projeto 3: Regressao Linear GD vs LMS ===");

    // -- 0. Multi-seed: RESULTADO OFICIAL (media +/- desvio) ------------------
    println!("[Multi-seed] rodando {} seeds {:?} ({} amostras, {} features cada)...",
             n_seeds, seeds, MS_N_SAMPLES, N_FEATURES);
    let stats = run_multiseed(&seeds);
    let ms_lines = multiseed_report_lines(&stats);
    println!();
    for line in &ms_lines {
        println!("{}", line);
    }
    save_report_multiseed(&ms_lines)?;
    plot_metrics_multiseed(&stats)?;

    // -- 1. Ilustracao single-seed (seed=42, 1 realizacao) --------------------
    // Gera as figuras single-seed existentes (convergence/scatter/pred_vs_actual
    // + report.txt). NAO e o numero oficial -- e so uma realizacao para ilustrar.
    println!();
    println!("--- Ilustracao single-seed (seed={}, 1 realizacao, {} amostras) ---", SEED, N_SAMPLES);
    let mut rng = StdRng::seed_from_u64(SEED);
    let (x, y) = generate_dataset(&mut rng, N_SAMPLES);
    let (gd_w, gd_b, gd_history) = batch_gradient_descent(&x, &y, GD_LR, MAX_ITERS);
    let (lms_w, lms_b, lms_history) = lms_widrow_hoff(&x, &y, LMS_LR, MAX_ITERS);
    println!("Pesos verdadeiros : w=[{}, {}], b={}", TRUE_W[0], TRUE_W[1], TRUE_BIAS);
    println!("GD  -> w=[{:.6}, {:.6}], b={:.6}, MSE={:.6}", gd_w[0], gd_w[1], gd_b, gd_history.last().unwrap());
    println!("LMS -> w=[{:.6}, {:.6}], b={:.6}, MSE={:.6}", lms_w[0], lms_w[1], lms_b, lms_history.last().unwrap());

    plot_convergence(&gd_history, &lms_history)?;
    plot_scatter(&x, &y, &gd_w, gd_b, &lms_w, lms_b)?;
    plot_pred_vs_actual(&x, &y, &gd_w, gd_b, &lms_w, lms_b)?;
    save_report(&gd_w, gd_b, &gd_history, &lms_w, lms_b, &lms_history)?;

    println!();
    println!("Concluido. OFICIAL: output/report_multiseed.txt + output/metrics_multiseed.png.");
    println!("Ilustracao (seed=42): convergence.png, scatter.png, pred_vs_actual.png, report.txt.");
    Ok(())
}
