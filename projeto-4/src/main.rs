use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

fn generate_dataset(n: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let mut x: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let label = if i < n / 2 { 0.0 } else { 1.0 };
        let cx = if label == 0.0 { -1.5 } else { 1.5 };
        let cy = if label == 0.0 { -1.0 } else { 1.0 };
        let xi = cx + normal.sample(&mut rng);
        let yi = cy + normal.sample(&mut rng);
        x.push([xi, yi]);
        y.push(label);
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let xs: Vec<[f64; 2]> = indices.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    (xs, ys)
}

fn train_test_split<'a>(
    x: &'a [[f64; 2]],
    y: &'a [f64],
    split_idx: usize,
) -> (&'a [[f64; 2]], &'a [f64], &'a [[f64; 2]], &'a [f64]) {
    assert!(split_idx > 0 && split_idx < x.len(),
        "train_test_split: split_idx={} must be in 1..{} (exclusive)", split_idx, x.len());
    (&x[..split_idx], &y[..split_idx], &x[split_idx..], &y[split_idx..])
}

/// Numerically stable sigmoid.
/// For z >= 0: standard form 1/(1+exp(-z)), exp(-z) is small, no overflow.
/// For z < 0:  rewrite as exp(z)/(1+exp(z)), avoids exp(-z) overflow to inf.
#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn predict_proba(x: &[[f64; 2]], w: &[f64; 2], b: f64) -> Vec<f64> {
    x.iter()
        .map(|xi| sigmoid(xi[0] * w[0] + xi[1] * w[1] + b))
        .collect()
}

fn binary_cross_entropy(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len() as f64;
    let eps = 1e-12_f64;
    y.iter()
        .zip(p.iter())
        .map(|(&yi, &pi)| {
            let pi = pi.clamp(eps, 1.0 - eps);
            -(yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln())
        })
        .sum::<f64>()
        / n
}

fn accuracy(y: &[f64], p: &[f64]) -> f64 {
    let correct = y
        .iter()
        .zip(p.iter())
        .filter(|(&yi, &pi)| {
            let pred = if pi >= 0.5 { 1.0 } else { 0.0 };
            (pred - yi).abs() < 1e-9
        })
        .count();
    correct as f64 / y.len() as f64
}

fn confusion_matrix(y: &[f64], p: &[f64]) -> (usize, usize, usize, usize) {
    let (mut tp, mut fp, mut tn, mut fn_) = (0, 0, 0, 0);
    for (&yi, &pi) in y.iter().zip(p.iter()) {
        let pred   = if pi >= 0.5 { 1u8 } else { 0u8 };
        let actual = yi as u8;
        match (pred, actual) {
            (1, 1) => tp  += 1,
            (1, 0) => fp  += 1,
            (0, 0) => tn  += 1,
            (0, 1) => fn_ += 1,
            _      => {}
        }
    }
    (tp, fp, tn, fn_)
}

fn precision_recall_f1(tp: usize, fp: usize, fn_: usize) -> (f64, f64, f64) {
    let precision = if tp + fp  > 0 { tp as f64 / (tp + fp)  as f64 } else { 0.0 };
    let recall    = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

fn compute_gradients(x: &[[f64; 2]], y: &[f64], p: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let (mut dw0, mut dw1, mut db) = (0.0_f64, 0.0_f64, 0.0_f64);
    for ((xi, &yi), &pi) in x.iter().zip(y.iter()).zip(p.iter()) {
        let diff = pi - yi;
        dw0 += diff * xi[0];
        dw1 += diff * xi[1];
        db  += diff;
    }
    (dw0 / n, dw1 / n, db / n)
}

fn gradient_step(
    x: &[[f64; 2]],
    y: &[f64],
    w: &mut [f64; 2],
    b: &mut f64,
    lr: f64,
) -> (f64, f64) {
    let p    = predict_proba(x, w, *b);
    let loss = binary_cross_entropy(y, &p);
    let acc  = accuracy(y, &p);
    let (dw0, dw1, db_val) = compute_gradients(x, y, &p);
    w[0] -= lr * dw0;
    w[1] -= lr * dw1;
    *b   -= lr * db_val;
    (loss, acc)
}

fn train(
    x: &[[f64; 2]],
    y: &[f64],
    epochs: usize,
    lr: f64,
    acc_threshold: f64,
) -> ([f64; 2], f64, Vec<f64>, Option<usize>, Vec<(usize, f64, f64)>) {
    let mut w = [0.0_f64; 2];
    let mut b = 0.0_f64;
    let mut loss_history = Vec::with_capacity(epochs);
    let mut first_thresh: Option<usize> = None;
    let mut csv_log: Vec<(usize, f64, f64)> = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let (loss, acc) = gradient_step(x, y, &mut w, &mut b, lr);
        loss_history.push(loss);
        csv_log.push((epoch + 1, loss, acc));

        // NOTE: acc here is pre-update for this epoch (computed before weight update).
        // first_thresh records the epoch whose *pre-update* accuracy first meets the threshold,
        // which reflects the weights produced by the previous epoch's gradient step.
        if first_thresh.is_none() && acc >= acc_threshold {
            first_thresh = Some(epoch + 1);
        }
        if (epoch + 1) % 100 == 0 || epoch == 0 {
            println!("Epoch {:>4} | Loss: {:.6} | Accuracy: {:.4}", epoch + 1, loss, acc);
        }
    }
    (w, b, loss_history, first_thresh, csv_log)
}

fn plot_decision_boundary(
    x: &[[f64; 2]],
    y: &[f64],
    x_test: &[[f64; 2]],
    y_test: &[f64],
    w: &[f64; 2],
    b: f64,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_x: Vec<&[f64; 2]> = x.iter().chain(x_test.iter()).collect();
    let x0_min = all_x.iter().map(|xi| xi[0]).fold(f64::INFINITY,     f64::min) - 0.5;
    let x0_max = all_x.iter().map(|xi| xi[0]).fold(f64::NEG_INFINITY, f64::max) + 0.5;
    let x1_min = all_x.iter().map(|xi| xi[1]).fold(f64::INFINITY,     f64::min) - 0.5;
    let x1_max = all_x.iter().map(|xi| xi[1]).fold(f64::NEG_INFINITY, f64::max) + 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption("Fronteira de Decisao (treino: solido, teste: oco)", ("sans-serif", 16))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x0_min..x0_max, x1_min..x1_max)?;

    chart.configure_mesh().x_desc("Feature 1").y_desc("Feature 2").draw()?;

    let grid = 100_usize; // heatmap resolution: grid x grid rectangles
    let dx = (x0_max - x0_min) / grid as f64;
    let dy = (x1_max - x1_min) / grid as f64;
    for i in 0..grid {
        for j in 0..grid {
            let gx = x0_min + i as f64 * dx;
            let gy = x1_min + j as f64 * dy;
            let p  = sigmoid(gx * w[0] + gy * w[1] + b);
            // scale p in [0,1] -> channel in [30, 230] to avoid pure black/white background cells
            let r  = (p * 200.0 + 30.0) as u8;
            let bl = ((1.0 - p) * 200.0 + 30.0) as u8;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(gx, gy), (gx + dx, gy + dy)],
                RGBAColor(r, 0, bl, 0.30).filled(), // alpha=0.30: background visible through heatmap
            )))?;
        }
    }

    if w[1].abs() > 1e-10 {
        let pts: Vec<(f64, f64)> = (0..=300) // 300 interpolated points for smooth boundary line
            .map(|i| {
                let gx = x0_min + (x0_max - x0_min) * i as f64 / 300.0;
                let gy = -(w[0] * gx + b) / w[1];
                (gx, gy)
            })
            .filter(|(_, gy)| *gy >= x1_min && *gy <= x1_max)
            .collect();
        chart
            .draw_series(LineSeries::new(pts, BLACK.stroke_width(3)))?
            .label("Fronteira de decisao")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(3)));
    }

    let class0: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter(|(_, &yi)| yi < 0.5).map(|(xi, _)| (xi[0], xi[1])).collect();
    let class1: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter(|(_, &yi)| yi >= 0.5).map(|(xi, _)| (xi[0], xi[1])).collect();

    chart.draw_series(class0.iter().map(|&(cx, cy)| Circle::new((cx, cy), 5, BLUE.mix(0.8).filled())))?
        .label("Classe 0")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.mix(0.8).filled()));
    chart.draw_series(class1.iter().map(|&(cx, cy)| Circle::new((cx, cy), 5, RED.mix(0.8).filled())))?
        .label("Classe 1")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.mix(0.8).filled()));

    // Test set: hollow circles to distinguish from filled train points
    let test0: Vec<(f64, f64)> = x_test.iter().zip(y_test.iter())
        .filter(|(_, &yi)| yi < 0.5).map(|(xi, _)| (xi[0], xi[1])).collect();
    let test1: Vec<(f64, f64)> = x_test.iter().zip(y_test.iter())
        .filter(|(_, &yi)| yi >= 0.5).map(|(xi, _)| (xi[0], xi[1])).collect();

    chart.draw_series(test0.iter().map(|&(cx, cy)| {
        EmptyElement::at((cx, cy))
            + Circle::new((0, 0), 7, BLUE.stroke_width(2))
    }))?.label("Classe 0 (teste)")
      .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.stroke_width(2)));
    chart.draw_series(test1.iter().map(|&(cx, cy)| {
        EmptyElement::at((cx, cy))
            + Circle::new((0, 0), 7, RED.stroke_width(2))
    }))?.label("Classe 1 (teste)")
      .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.stroke_width(2)));

    chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    root.present()?;
    println!("Saved: {path}");
    Ok(())
}

fn plot_loss_curve(
    loss_history: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = loss_history.iter().cloned().fold(0.0_f64, f64::max);
    let min_loss = loss_history.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = max_loss.max(std::f64::consts::LN_2) * 1.05;
    let y_min = min_loss * 0.95;

    let mut chart = ChartBuilder::on(&root)
        .caption("Regressao Logistica - Curva de Loss", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..(loss_history.len() + 1), y_min..y_max)?;

    chart.configure_mesh().x_desc("Epoca").y_desc("Binary Cross-Entropy Loss").draw()?;

    let baseline = std::f64::consts::LN_2;
    chart.draw_series(LineSeries::new(
        vec![(1, baseline), (loss_history.len() + 1, baseline)],
        RED.mix(0.5).stroke_width(1),
    ))?.label("Baseline (ln 2 = 0.6931)")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5).stroke_width(1)));

    chart.draw_series(LineSeries::new(
        loss_history.iter().enumerate().map(|(i, &l)| (i + 1, l)).collect::<Vec<_>>(),
        BLUE.stroke_width(2),
    ))?.label("Loss de Treino")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    root.present()?;
    println!("Saved: {path}");
    Ok(())
}


fn plot_accuracy_curve(
    acc_history: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = 0.0_f64; // start at 0 so the axis represents the full [0,1] accuracy scale
    let y_max = (acc_history.iter().cloned().fold(0.0_f64, f64::max) * 1.02).min(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Regressao Logistica - Curva de Acuracia", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..(acc_history.len() + 1), y_min..y_max)?;

    chart.configure_mesh().x_desc("Epoca").y_desc("Acuracia de Treino").draw()?;

    // Highlight threshold line at 0.95
    chart.draw_series(LineSeries::new(
        vec![(1, 0.95), (acc_history.len() + 1, 0.95)],
        RED.mix(0.5).stroke_width(1),
    ))?.label("Threshold 95%")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5).stroke_width(1)));

    chart.draw_series(LineSeries::new(
        acc_history.iter().enumerate().map(|(i, &a)| (i + 1, a)).collect::<Vec<_>>(),
        BLUE.stroke_width(2),
    ))?.label("Acuracia de Treino")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    root.present()?;
    println!("Saved: {path}");
    Ok(())
}

fn plot_confusion_matrix(
    tp: usize, fp: usize, tn: usize, fn_: usize,
    label: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (480, 520)).into_drawing_area();
    root.fill(&WHITE)?;

    root.draw(&Text::new(
        format!("Matriz de Confusao ({})", label),
        (40, 18),
        ("sans-serif", 20).into_font(),
    ))?;

    let cells: [(usize, usize, usize, &str, RGBColor); 4] = [
        (0, 0, tn,   "TN", RGBColor(173, 216, 230)),
        (0, 1, fp,   "FP", RGBColor(255, 182, 193)),
        (1, 0, fn_,  "FN", RGBColor(255, 182, 193)),
        (1, 1, tp,   "TP", RGBColor(144, 238, 144)),
    ];

    let ox = 80i32;
    let oy = 60i32;
    let cw = 170i32;
    let ch = 170i32;

    for &(row, col, count, lbl, color) in &cells {
        let x0 = ox + col as i32 * cw;
        let y0 = oy + row as i32 * ch;
        root.draw(&Rectangle::new([(x0, y0), (x0 + cw, y0 + ch)], color.filled()))?;
        root.draw(&Rectangle::new([(x0, y0), (x0 + cw, y0 + ch)], BLACK.stroke_width(2)))?;
        root.draw(&Text::new(
            format!("{}: {}", lbl, count),
            (x0 + 30, y0 + ch / 2 - 12),
            ("sans-serif", 20).into_font(),
        ))?;
    }

    root.draw(&Text::new("Pred 0", (ox + 35,      oy - 18), ("sans-serif", 14).into_font()))?;
    root.draw(&Text::new("Pred 1", (ox + cw + 35, oy - 18), ("sans-serif", 14).into_font()))?;
    root.draw(&Text::new("Real 0", (10, oy + ch / 2 - 8),      ("sans-serif", 14).into_font()))?;
    root.draw(&Text::new("Real 1", (10, oy + ch + ch / 2 - 8), ("sans-serif", 14).into_font()))?;
    // Axis header labels
    root.draw(&Text::new("Predito ->", (ox + cw / 2 - 20, oy + 2 * ch + 38), ("sans-serif", 13).into_font()))?;
    root.draw(&Text::new("Real", (2, oy - 18), ("sans-serif", 13).into_font()))?;

    root.present()?;
    println!("Saved: {path}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mini-Projeto 4: Logistic Regression from Scratch ===\n");

    let n_samples   = 300_usize;
    let epochs      = 1000_usize;
    let lr          = 0.1_f64;
    let seed        = 42_u64;
    let train_ratio = 0.8_f64;
    let acc_threshold = 0.95_f64;

    // output_dir is resolved relative to the process working directory (CWD).
    // Run from the project root (`cargo run`) to write to the project output/ directory.
    let output_dir = std::env::current_dir()
        .expect("cannot read current working directory")
        .join("output");
    // Sanity check: warn if not running from the project root (Cargo.toml absent).
    if !output_dir.parent().unwrap().join("Cargo.toml").exists() {
        eprintln!("Warning: Cargo.toml not found in CWD -- output files may land in the wrong place. Run with `cargo run` from the project root.");
    }

    let (x, y) = generate_dataset(n_samples, seed);
    let split_idx = (n_samples as f64 * train_ratio).round() as usize;

    let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, split_idx);

    let n_pos_train = y_train.iter().filter(|&&yi| yi > 0.5).count();
    let n_neg_train = split_idx - n_pos_train;
    let n_pos_test  = y_test.iter().filter(|&&yi| yi > 0.5).count();
    let n_neg_test  = (n_samples - split_idx) - n_pos_test;

    println!("Dataset: {n_samples} amostras (seed={seed})");
    println!("  Treino : {} amostras (Classe 0: {}, Classe 1: {})", split_idx, n_neg_train, n_pos_train);
    println!("  Teste  : {} amostras (Classe 0: {}, Classe 1: {})\n", n_samples - split_idx, n_neg_test, n_pos_test);

    println!("Treinando regressao logistica (lr={lr}, epochs={epochs}):");
    println!("Nota: loss e acuracia por epoca sao pre-atualizacao dos pesos daquela epoca.");
    let (w, b, loss_history, first_thresh_epoch, csv_log) =
        train(x_train, y_train, epochs, lr, acc_threshold);

    let p_train_final    = predict_proba(x_train, &w, b);
    let train_final_loss = binary_cross_entropy(y_train, &p_train_final);
    let train_final_acc  = accuracy(y_train, &p_train_final);
    let (tr_tp, tr_fp, tr_tn, tr_fn) = confusion_matrix(y_train, &p_train_final);
    let (tr_prec, tr_rec, tr_f1)     = precision_recall_f1(tr_tp, tr_fp, tr_fn);

    let p_test    = predict_proba(x_test, &w, b);
    let test_loss = binary_cross_entropy(y_test, &p_test);
    let test_acc  = accuracy(y_test, &p_test);
    let (te_tp, te_fp, te_tn, te_fn) = confusion_matrix(y_test, &p_test);
    let (te_prec, te_rec, te_f1)     = precision_recall_f1(te_tp, te_fp, te_fn);

    println!("\n--- Resultados Finais ---");
    println!("Pesos: w0={:.6}, w1={:.6}, bias={:.6}", w[0], w[1], b);
    println!("Loss Treino Final : {train_final_loss:.6}");
    println!("Loss Teste        : {test_loss:.6}");
    println!("Acuracia Treino   : {:.2}%", train_final_acc * 100.0);
    println!("Acuracia Teste    : {:.2}%", test_acc * 100.0);
    println!("Precisao/Recall/F1 Treino: {:.4}/{:.4}/{:.4}", tr_prec, tr_rec, tr_f1);
    println!("Precisao/Recall/F1 Teste : {:.4}/{:.4}/{:.4}", te_prec, te_rec, te_f1);

    match first_thresh_epoch {
        Some(ep) => println!("Primeira epoca com acc >= {:.2}% (threshold): epoca {ep} (pesos produzidos pela epoca {}; acc medida pre-atualizacao da epoca {ep})", acc_threshold * 100.0, ep - 1),
        None     => println!("Acuracia >= {:.2}% nao atingida em {epochs} epocas", acc_threshold * 100.0),
    }

    if w[1].abs() > 1e-10 {
        let db_slope     = -w[0] / w[1];
        let db_intercept = -b / w[1];
        if db_intercept >= 0.0 {
            println!("Fronteira de decisao: x2 = {:.6} * x1 + {:.6}", db_slope, db_intercept);
        } else {
            println!("Fronteira de decisao: x2 = {:.6} * x1 - {:.6}", db_slope, db_intercept.abs());
        }
    }

    std::fs::create_dir_all(&output_dir)?;
    println!("Diretorio de saida: {}", output_dir.display());

    plot_decision_boundary(x_train, y_train, x_test, y_test, &w, b,
        &output_dir.join("decision_boundary.png").to_str().unwrap())?;
    plot_loss_curve(&loss_history,
        &output_dir.join("loss_curve.png").to_str().unwrap())?;
    let acc_history: Vec<f64> = csv_log.iter().map(|&(_, _, a)| a).collect();
    plot_accuracy_curve(&acc_history,
        &output_dir.join("accuracy_curve.png").to_str().unwrap())?;
    plot_confusion_matrix(tr_tp, tr_fp, tr_tn, tr_fn, "Treino",
        &output_dir.join("confusion_matrix_train.png").to_str().unwrap())?;
    plot_confusion_matrix(te_tp, te_fp, te_tn, te_fn, "Teste",
        &output_dir.join("confusion_matrix_test.png").to_str().unwrap())?;

    let mut csv_content = String::from("epoch,loss,accuracy\n");
    for &(ep, l, a) in &csv_log {
        csv_content.push_str(&format!("{},{:.6},{:.4}\n", ep, l, a));
    }
    std::fs::write(output_dir.join("training_log.csv"), &csv_content)?;
    println!("Saved: {}", output_dir.join("training_log.csv").display());

    let initial_loss_ln2 = std::f64::consts::LN_2; // BCE with zero weights = ln(2)
    let loss_reduction = (1.0 - train_final_loss / initial_loss_ln2) * 100.0;
    // Guard: w[1] may be zero for degenerate datasets; avoids NaN/Inf in report.txt.
    let (slope_str, intercept_str) = if w[1].abs() > 1e-10 {
        (format!("{:.6}", -w[0] / w[1]), format!("{:.6}", -b / w[1]))
    } else {
        ("N/A (w1~0)".to_string(), "N/A".to_string())
    };
    let first_thresh_str = match first_thresh_epoch {
        Some(ep) => format!("epoca {ep}"),
        None     => format!("nao atingida em {epochs} epocas"),
    };

    let acc_pct = acc_threshold * 100.0;
    let report = format!(
        "Mini-Projeto 4 - Logistic Regression Results\n\
         =============================================\n\
         Dataset           : {n_samples} amostras (gaussianas 2D sinteticas, seed={seed})\n\
         Split             : {:.0}% treino / {:.0}% teste\n\
         Treino (neg/pos)  : {n_neg_train} / {n_pos_train}\n\
         Teste  (neg/pos)  : {n_neg_test} / {n_pos_test}\n\
         \n\
         Hiperparametros\n\
         ---------------\n\
         Learning Rate     : {lr}\n\
         Epocas            : {epochs}\n\
         Inicializacao     : zeros\n\
         \n\
         Parametros Aprendidos\n\
         ---------------------\n\
         w0                : {:.6}\n\
         w1                : {:.6}\n\
         bias              : {:.6}\n\
         \n\
         Desempenho (Treino)\n\
         -------------------\n\
         Loss Inicial(BCE) : {:.6}\n\
         Loss Final (BCE)  : {:.6}\n\
         Reducao de Loss   : {:.2}%\n\
         Acuracia          : {:.2}%\n\
         Precisao          : {:.4}\n\
         Recall            : {:.4}\n\
         F1-Score          : {:.4}\n\
         Conf. Matrix:\n\
           TN={tr_tn}  FP={tr_fp}\n\
           FN={tr_fn}  TP={tr_tp}\n\
         Primeira epoca acc>={acc_pct:.2}% (threshold {acc_threshold}): {first_thresh_str}\n\
         \n\
         Desempenho (Teste)\n\
         ------------------\n\
         Loss Final (BCE)  : {:.6}\n\
         Acuracia          : {:.2}%\n\
         Precisao          : {:.4}\n\
         Recall            : {:.4}\n\
         F1-Score          : {:.4}\n\
         Conf. Matrix:\n\
           TN={te_tn}  FP={te_fp}\n\
           FN={te_fn}  TP={te_tp}\n\
         \n\
         Fronteira de Decisao (x2 = m*x1 + c)\n\
         --------------------------------------\n\
         inclinacao m      : {}\n\
         intercepto c      : {}\n\
         \n\
         Arquivos de Saida\n\
         -----------------\n\
         output/decision_boundary.png  (treino: solido, teste: circulo oco)\n\
         output/loss_curve.png\n\
         output/accuracy_curve.png\n\
         output/confusion_matrix_train.png\n\
         output/confusion_matrix_test.png\n\
         output/training_log.csv\n\
         output/report.txt\n",
        train_ratio * 100.0,
        (1.0 - train_ratio) * 100.0,
        w[0], w[1], b,
        initial_loss_ln2, train_final_loss, loss_reduction,
        train_final_acc * 100.0,
        tr_prec, tr_rec, tr_f1,
        test_loss,
        test_acc * 100.0,
        te_prec, te_rec, te_f1,
        slope_str, intercept_str,
    );

    std::fs::write(output_dir.join("report.txt"), &report)?;
    println!("\nRelatorio salvo em {}", output_dir.join("report.txt").display());
    println!("\n=== Concluido ===");
    Ok(())
}