# Guia de Estudo — Mini-Projeto 4: Regressao Logistica

## Pipeline

```
DADOS SINTETICOS (300 amostras, seed=42)
     |
     v
[generate_dataset]  -- 300 pontos 2D, 2 classes gaussianas
     |
     v
[train_test_split]  -- 80/20: 240 treino / 60 teste
     |
     v
[Inicializacao]     -- w0=0, w1=0, b=0
     |
     v
[Loop de Treinamento] -- 1000 epocas (sobre conjunto de treino)
     |
     +----> [predict_proba]       z = w0*x1 + w1*x2 + b
     |             |              p = sigmoid(z)  -- implementacao numericamente estavel (2 ramos, ver main.rs)
     |             v
     +----> [binary_cross_entropy]  L = -(1/n)*sum(y*log(p) + (1-y)*log(1-p))
     |             |
     |             v
     +----> [accuracy + log]      acc = sum(round(p)==y)/n
     |             |              training_log.csv (epoch, loss, acc)
     |             |              first_thresh check (pre-update acc >= 0.95)
     |             v
     +----> [compute_gradients]   dw0 = (1/n)*sum((p-y)*x1)
     |             |              dw1 = (1/n)*sum((p-y)*x2)
     |             |              db  = (1/n)*sum(p-y)
     |             v
     +----> [atualiza pesos]      w0 -= lr*dw0
                                  w1 -= lr*dw1
                                  b  -= lr*db
     |
     v
[Avaliacao]  -- metricas no treino E no teste (separados)
     |
     +----> decision_boundary.png
     +----> loss_curve.png
     +----> accuracy_curve.png
     +----> confusion_matrix_train.png
     +----> confusion_matrix_test.png
     +----> training_log.csv       (epoch, loss, accuracy por epoca)
     +----> report.txt
```

## Matematica

### Modelo

```
z_i = w0*x1_i + w1*x2_i + b           (score linear)

sigma(z) = 1 / (1 + e^{-z})           (funcao sigmoid / logistica)

p_i = sigma(z_i)  in (0, 1)           (probabilidade P(y=1 | x_i))
```

### Funcao de Custo — Binary Cross-Entropy (BCE)

```
L = -(1/n) * SUM_{i=1}^{n} [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

Interpretacao:
- Se `y=1`: minimiza `-log(p)` => empurra `p` para 1
- Se `y=0`: minimiza `-log(1-p)` => empurra `p` para 0
- Loss maxima (aleatoria): `log(2) = 0.6931`

### Gradientes

```
dL/dw_j = (1/n) * SUM_i [ (p_i - y_i) * x_{i,j} ]

dL/db   = (1/n) * SUM_i [ p_i - y_i ]
```

Derivacao completa (chain rule com passos intermediarios):

Nota: nos Passos 1-3 derivamos para um exemplo i (loss por amostra L_i); o fator 1/n da soma entra no Passo 5.

```
Passo 1 — derivada da BCE por amostra (L_i) em relacao a p_i:
dL_i/dp_i = -(y_i/p_i - (1 - y_i)/(1 - p_i))

Passo 2 — derivada da sigmoid em relacao a z_i:
d(sigma)/dz_i = sigma(z_i) * (1 - sigma(z_i)) = p_i * (1 - p_i)

Passo 3 — produto (chain rule), cancelamento algebrico:
dL_i/dz_i = dL_i/dp_i * dp_i/dz_i
         = -(y_i/p_i - (1-y_i)/(1-p_i)) * p_i*(1-p_i)
           [cancela denominadores: (y_i/p_i)*p_i*(1-p_i) = y_i*(1-p_i) e ((1-y_i)/(1-p_i))*p_i*(1-p_i) = (1-y_i)*p_i]
         = -(y_i*(1-p_i) - (1-y_i)*p_i)
         = -(y_i - y_i*p_i - p_i + y_i*p_i)
         = -(y_i - p_i)
         = p_i - y_i

Passo 4 — derivada de z_i em relacao a w_j:
dz_i/dw_j = x_{i,j}

Passo 5 — resultado final:
dL/dw_j = (1/n) * SUM_i [ dL/dz_i * dz_i/dw_j ]
         = (1/n) * SUM_i [ (p_i - y_i) * x_{i,j} ]
```

### Regra de Atualizacao

```
w_j <- w_j - lr * dL/dw_j    (gradient descent)
b   <- b   - lr * dL/db
```

### Fronteira de Decisao

```
p = 0.5  <=>  z = 0  <=>  w0*x1 + w1*x2 + b = 0

x2 = -(w0/w1) * x1 - b/w1
```

Com os pesos aprendidos:

```
x2 = -1.524580 * x1 - 0.075489
```

## Codigo

### Sigmoid e predict_proba

```rust
/// Numerically stable sigmoid.
/// Para z >= 0: forma padrao 1/(1+exp(-z)); exp(-z) e pequeno, sem overflow.
/// Para z < 0:  reescreve como exp(z)/(1+exp(z)), evitando overflow de exp(-z) para inf.
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
```

**Por que sigmoid?** Mapeia qualquer real para (0,1), interpretavel como probabilidade. Gradiente suave em todo dominio.

**Por que a implementacao usa dois ramos?** A forma simples `1/(1+exp(-z))` e numericamente instavel para z muito negativo: `exp(-z)` cresce sem limite (overflow para `inf`), resultando em `1/inf = 0` em vez de um valor proximo a zero mas positivo. O ramo alternativo `exp(z)/(1+exp(z))` para z < 0 e matematicamente equivalente mas usa `exp(z)` com z < 0 (valor pequeno, sem overflow), garantindo precisao em ambas as regioes do dominio.

### Binary Cross-Entropy

```rust
fn binary_cross_entropy(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len() as f64;
    let eps = 1e-12_f64;
    y.iter().zip(p.iter()).map(|(&yi, &pi)| {
        let pi = pi.clamp(eps, 1.0 - eps);   // evita log(0)
        -(yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln())
    }).sum::<f64>() / n
}
```

**Clamping**: necessario para evitar `log(0) = -inf` quando o modelo e muito confiante e errado.

### Compute Gradients + Gradient Step (coracao do algoritmo)

```rust
// Calculo puro de gradientes — sem mutacao, sem side effects.
fn compute_gradients(x: &[[f64; 2]], y: &[f64], p: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let (mut dw0, mut dw1, mut db) = (0.0, 0.0, 0.0);
    for ((xi, &yi), &pi) in x.iter().zip(y.iter()).zip(p.iter()) {
        let diff = pi - yi;
        dw0 += diff * xi[0];
        dw1 += diff * xi[1];
        db  += diff;
    }
    (dw0 / n, dw1 / n, db / n)
}

// Um passo de GD: forward -> loss -> gradientes -> update. Retorna (loss, acc).
// trecho de src/main.rs
fn gradient_step(x: &[[f64; 2]], y: &[f64], w: &mut [f64; 2], b: &mut f64, lr: f64) -> (f64, f64) {
    let p    = predict_proba(x, w, *b);
    let loss = binary_cross_entropy(y, &p);
    let acc  = accuracy(y, &p);
    let (dw0, dw1, db_val) = compute_gradients(x, y, &p);
    w[0] -= lr * dw0;
    w[1] -= lr * dw1;
    *b   -= lr * db_val;
    (loss, acc)
}
```

**Separacao de responsabilidades**: `compute_gradients` e uma funcao matematica pura (calcula gradientes sem efeitos colaterais). `gradient_step` orquestra o passo completo de treinamento. Isso torna o codigo mais testavel e legivel.

**Insight chave**: o erro e simplesmente `(p - y)`, resultado elegante do cancelamento algebraico entre a derivada da BCE e a sigmoid.

### Geracao do Dataset e Split

```rust
fn generate_dataset(n: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let mut x: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let label = if i < n / 2 { 0.0 } else { 1.0 };
        let cx = if label == 0.0 { -1.5 } else { 1.5 };  // Classe 0: centro (-1.5, -1.0)
        let cy = if label == 0.0 { -1.0 } else { 1.0 };  // Classe 1: centro (+1.5, +1.0)
        let xi = cx + normal.sample(&mut rng);
        let yi = cy + normal.sample(&mut rng);
        x.push([xi, yi]);
        y.push(label);
    }
    // Shuffle deterministico com seed=42
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let xs: Vec<[f64; 2]> = indices.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    (xs, ys)
}

fn train_test_split<'a>(x: &'a [[f64; 2]], y: &'a [f64], split_idx: usize)
    -> (&'a [[f64; 2]], &'a [f64], &'a [[f64; 2]], &'a [f64]) {
    (&x[..split_idx], &y[..split_idx], &x[split_idx..], &y[split_idx..])
}
```

O split 80/20 e aplicado sobre o array ja embaralhado, garantindo reproducibilidade com seed=42.

## Resultados

Saida real do programa (seed=42, deterministico):

```
=== Mini-Projeto 4: Logistic Regression from Scratch ===

Dataset: 300 amostras (seed=42)
  Treino : 240 amostras (Classe 0: 116, Classe 1: 124)
  Teste  : 60 amostras (Classe 0: 34, Classe 1: 26)

Treinando regressao logistica (lr=0.1, epochs=1000):
Nota: loss e acuracia por epoca sao pre-atualizacao dos pesos daquela epoca.
Epoch    1 | Loss: 0.693147 | Accuracy: 0.5167
Epoch  100 | Loss: 0.134649 | Accuracy: 0.9542
Epoch  200 | Loss: 0.117320 | Accuracy: 0.9542
Epoch  300 | Loss: 0.111753 | Accuracy: 0.9542
Epoch  400 | Loss: 0.109240 | Accuracy: 0.9542
Epoch  500 | Loss: 0.107919 | Accuracy: 0.9542
Epoch  600 | Loss: 0.107160 | Accuracy: 0.9542
Epoch  700 | Loss: 0.106699 | Accuracy: 0.9542
Epoch  800 | Loss: 0.106408 | Accuracy: 0.9542
Epoch  900 | Loss: 0.106219 | Accuracy: 0.9542
Epoch 1000 | Loss: 0.106094 | Accuracy: 0.9542

--- Resultados Finais ---
Pesos: w0=2.460226, w1=1.613707, bias=0.121817
Loss Treino Final : 0.106093
Loss Teste        : 0.064900
Acuracia Treino   : 95.42%
Acuracia Teste    : 96.67%
Precisao/Recall/F1 Treino: 0.9520/0.9597/0.9558
Precisao/Recall/F1 Teste : 0.9615/0.9615/0.9615
Primeira epoca com acc >= 95.00% (threshold): epoca 2
Fronteira de decisao: x2 = -1.524580 * x1 - 0.075489
```

> accuracy_curve.png visualiza o salto abrupto de 51.67% para 95.42% entre as epocas 1 e 2 — mais evidente na curva de acuracia do que na curva de loss.

> **Nota sobre pre/pos-atualizacao:** Os logs de epoca mostram a loss *antes* da atualizacao dos pesos (pre-update), enquanto "Loss Final" e recomputada *apos* a ultima atualizacao — por isso os valores diferem ligeiramente (0.106094 no log da epoca 1000 vs. 0.106093 final).

### Tabela de Metricas Chave

| Metrica                      | Treino    | Teste     |
|------------------------------|-----------|-----------|
| Loss Inicial (BCE)           | 0.693147  | —         |
| Loss Epoch 100               | 0.134649  | —         |
| Loss Final (pos-treinamento) | 0.106093  | 0.064900  |
| Reducao Total de Loss        | 84.69%    | —         |
| Acuracia de Treino           | 95.42%    | 96.67%    |
| Precisao                     | 0.9520    | 0.9615    |
| Recall                       | 0.9597    | 0.9615    |
| F1-Score                     | 0.9558    | 0.9615    |
| w0 (peso feature 1)          | 2.460226  | —         |
| w1 (peso feature 2)          | 1.613707  | —         |
| bias                         | 0.121817  | —         |
| Inclinacao fronteira         | -1.524580 | —         |
| Intercepto fronteira         | -0.075489 | —         |

**Observacoes:**
- A Loss inicial e `ln(2) = 0.6931` — exatamente o esperado para inicializacao em zeros (probabilidade 0.5 para todos)
- Acuracia de treino de ~51.67% na epoca 1 com pesos zero: sigmoid(0)=0.5 para todos => prediz classe 1 para todos (threshold >= 0.5) => acerta 124/240 (classe 1) e erra 116 (classe 0). A acuracia de 95.42% (229/240) foi atingida ja na **epoca 2** (confirmado pelo training_log.csv).
- Convergencia da loss e mais lenta que a acuracia — o modelo continua ajustando confianca mesmo com acuracia estavel
- O gap treino/teste (95.42% vs 96.67%) e pequeno e favoravel ao teste, indicando boa generalizacao

## Perguntas Provaveis

**P1: Por que usar sigmoid e nao uma funcao linear direta para classificacao?**

R: A funcao linear nao e bounded e pode produzir valores fora de [0,1], impossibilitando interpretacao probabilistica. A sigmoid mapeia qualquer real para (0,1), fornece probabilidades calibradas e tem gradiente suave (sem descontinuidades) que facilita otimizacao via gradiente descendente.

---

**P2: Como e derivado o gradiente da BCE em relacao aos pesos?**

R: Pela chain rule: `dL/dw_j = dL/dp_i * dp_i/dz_i * dz_i/dw_j`. Os passos intermediarios explicitamente:

1. `dL_i/dp_i = -(y_i/p_i - (1-y_i)/(1-p_i))` — derivada da BCE por amostra em relacao a p_i
2. `dp_i/dz_i = p_i*(1-p_i)` — derivada da sigmoid (propriedade conhecida)
3. Produto dos passos 1 e 2: `-(y_i/p_i - (1-y_i)/(1-p_i)) * p_i*(1-p_i)` se simplifica para `p_i - y_i` via cancelamento algebrico (expandir, coletar termos, verificar)
4. `dz_i/dw_j = x_{i,j}`, logo `dL/dw_j = (1/n)*sum((p_i - y_i)*x_ij)`

O resultado `(p - y)` e o erro de predicao multiplicado pela entrada — forma compacta que emerge do cancelamento entre BCE e sigmoid.

---

**P3: O que significa a fronteira de decisao geometricamente?**

R: E o hiperplano (no caso 2D, uma reta) onde `p = 0.5`, ou seja, `z = w^T x + b = 0`. Todos os pontos acima da linha tem `p > 0.5` (classe 1), abaixo tem `p < 0.5` (classe 0). A inclinacao e `-w0/w1` e o intercepto e `-b/w1`.

---

**P4: Por que a loss inicial e exatamente 0.6931 (ln 2)?**

R: Com inicializacao em zeros, `z=0` para todos os pontos, logo `p = sigmoid(0) = 0.5` para todos. A BCE com `p=0.5` e `y` uniforme e `-log(0.5) = log(2) ≈ 0.6931`. Isso e o baseline esperado para um classificador aleatório equilibrado.

---

**P5: Qual a diferenca entre Batch Gradient Descent (usado aqui) e SGD?**

R: Batch GD usa todos os `n` exemplos para calcular o gradiente em cada passo — preciso mas lento para datasets grandes. SGD usa 1 exemplo por vez — ruidoso mas rapido e pode escapar de minimos locais. Mini-batch SGD (k exemplos, tipicamente 32-256) e o compromisso pratico usado em deep learning. Neste projeto, n=240 e cada epoca computa o gradiente sobre todos os 240 exemplos (240 000 avaliacoes no total de 1000 epocas) — viavel para este dataset pequeno.

---

**P6: Como o modelo se comportaria com dados nao-linearmente separaveis?**

R: A regressao logistica encontraria a melhor reta possivel, mas nao conseguiria separar perfeitamente. Para capturar fronteiras curvas, seria necessario: (a) adicionar features polinomiais (`x1^2, x1*x2, x2^2`) como pre-processamento, ou (b) usar modelos nao-lineares como SVM com kernel RBF ou redes neurais.

---

**P7: O que indica a convergencia lenta da loss apos a epoca 2?**

R: A acuracia de treino atinge seu maximo (95.42%) ja na epoca 2 e permanece estavel ate a epoca 1000. Ja a loss continua caindo de 0.6931 (epoca 1, = ln(2)) ate 0.1061 (epoca 1000). Isso ocorre porque a loss mede confianca probabilistica, nao apenas acertos binarios: o modelo continua aumentando as margens de separacao (empurrando `p` para 0 e 1 nos exemplos corretos) mesmo com acuracia ja estabilizada.

Adicionalmente, a acuracia de treino fica limitada a 95.42% porque os 11 pontos mal-classificados (6 FP + 5 FN) estao na regiao de sobreposicao das gaussianas e nenhuma fronteira linear os separa — e um limite do modelo (capacidade), nao da otimizacao. A loss continua caindo porque o modelo aumenta a confianca nas predicoes corretas, mas os 11 erros persistem independentemente de quantas epocas sejam treinadas.

---

**P8: Por que a acuracia inicial e ~51.67% e nao 50%?**

R: Com pesos zero, `sigmoid(0) = 0.5` para todos. O limiar de decisao e `>= 0.5` (inclusivo), entao empates vao para classe 1 — o modelo prediz classe 1 para todos os pontos. (Com limiar `> 0.5`, empates iriam para classe 0 e a acuracia inicial seria 116/240 = 48.33%, nao 51.67%.) No conjunto de treino de 240 amostras temos 124 da classe 1 e 116 da classe 0 (split nao perfeitamente balanceado apos o shuffle). Acertando apenas os 124 exemplos da classe 1: `124/240 = 51.67%`. Com o dataset completo (150/150), seria exatamente 50%.