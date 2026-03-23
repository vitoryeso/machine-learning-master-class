# Guia de Estudo — Projeto 1: K-Means em Embeddings de Imagens
> **Nível:** já sei ~82% → foco em código real, outputs reais, e perguntas do professor

---

## 0. Visão Geral do Pipeline (fluxo completo)

```
22.258 imagens no disco
        ↓  shuffle + seed=42
8.000 amostradas (63 corrompidas dropadas automaticamente)
        ↓  CLIP ViT-B/32 (fastembed-rs, ONNX, batch=64)
embeddings: cada imagem → vetor ℝ⁵¹²
        ↓  L2-normalize
||x̂|| = 1  →  distância euclidiana ≈ similaridade cosseno
        ↓  K-Means++ para K=2..15, 3 runs/K, best por WCSS
labels_k2.txt … labels_k15.txt
        ↓  Elbow + Silhouette → K=4 vence
        ↓  PCA 512D → 2D (power iteration, 50 iters)
6 scatter plots PNG  +  cluster_report.txt
```

---

## 1. CLIP ViT-B/32 — O que é e por que usar

### O que faz
CLIP (Contrastive Language–Image Pre-training, Radford et al., ICML 2021)
é treinado pra alinhar imagens e textos num espaço vetorial comum.
O encoder de imagem é um **Vision Transformer B/32** — patch size 32×32,
produz vetor de **512 dimensões** por imagem.

### Por que é bom pra clustering de coleção pessoal
- Captura **semântica visual**: "foto de pessoa" vs "screenshot" vs "paisagem"
- Não usa metadados (pasta, extensão, tamanho) — só pixels
- Pré-treinado em escala gigante → features ricas sem fine-tuning

### Como é chamado no código
```rust
// main.rs — carrega o modelo ONNX localmente
let mut model = ImageEmbedding::try_new(
    ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
        .with_show_download_progress(true),
)?;

// Batch de 64 imagens por vez
let str_paths: Vec<String> = chunk.iter()
    .map(|p| p.to_string_lossy().into_owned())
    .collect();
match model.embed(&str_paths, Some(BATCH_SIZE)) {
    Ok(embs) => { /* embs: Vec<Vec<f32>> de shape [64][512] */ }
    Err(_)   => { /* fallback: tentar uma por uma para pular corrompidas */ }
}
```

**Output real:**  `8.000 embedded, 63 failed/skipped`

---

## 2. L2-Normalização — Por que normalizar?

### A fórmula (slide 2)
```
x̂ᵢ = xᵢ / ||xᵢ||₂
```

### Intuição geométrica
Sem normalização, pontos com norma maior têm distância euclidiana maior
**independente do ângulo**. Depois da normalização todos os vetores ficam
na hiperesfera unitária ℝ⁵¹² com ||x̂|| = 1, e:

```
||x̂ᵢ - x̂ⱼ||² = 2 - 2·cos(θᵢⱼ)
```

Então **distância euclidiana = função monotônica da distância cosseno**.
K-Means com Euclidiana num espaço L2-normalizado está implicitamente
usando similaridade cosseno — ideal pra embeddings de linguagem/visão.

### Código real
```rust
fn l2_normalize(data: &mut [Vec<f32>]) {
    for v in data.iter_mut() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }
}
// Chamado em main():
l2_normalize(&mut embeddings);
```

**Pergunta que pode vir:** "Por que não usar cosseno direto no K-Means?"
→ K-Means clássico minimiza distância euclidiana. Com L2-norm você
converte o problema pra euclidiana sem mudar o algoritmo.

---

## 3. K-Means++ — Init inteligente

### Problema do K-Means vanilla
Inicialização aleatória dos K centroides pode cair em mínimos locais ruins
(clusters muito desequilibrados). K-Means++ (Arthur & Vassilvitskii, SODA 2007)
resolve com init probabilístico.

### Algoritmo K-Means++ passo a passo
```
1. Escolhe centroide c₁ aleatoriamente
2. Para cada ponto xᵢ, calcula D(xᵢ) = min dist² ao centroide mais próximo
3. Sorteia próximo centroide com prob ∝ D(xᵢ)  ← pontos distantes têm mais chance
4. Repete 2-3 até ter K centroides
5. Roda K-Means padrão a partir daí
```

### Código real (init)
```rust
fn kmeans(data: &[Vec<f32>], k: usize, seed: u64) -> (Vec<usize>, Vec<Vec<f32>>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // 1. Primeiro centroide: aleatório
    centroids.push(data[rng.random_range(0..n)].clone());

    for _ in 1..k {
        // 2. Distância de cada ponto ao centroide mais próximo
        let dists: Vec<f32> = data.iter().map(|p| {
            centroids.iter()
                .map(|c| euclidean_dist_sq(p, c))
                .fold(f32::MAX, f32::min)
        }).collect();

        // 3. Sorteio ponderado por D²
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
    // ... loop EM padrão
}
```

### Loop principal (Expectation-Maximization)
```
Repeat até convergir (ou 100 iters):
  E-step: assign_clusters → cada ponto vai pro centroide mais próximo
  M-step: compute_centroids → recomputa centroide como média dos membros
  Converge se labels não mudam
```

```rust
// Convergência verificada por comparação de labels
let converged = new_labels == labels;
if converged { break; }
```

**Output real (console):**
```
K= 2 | inertia=  3753.7 | silhouette=0.0559
K= 3 | inertia=  3627.7 | silhouette=0.0511
K= 4 | inertia=  3520.9 | silhouette=0.0576   ← best
K= 5 | inertia=  3454.1 | silhouette=0.0558
...
K=15 | inertia=  3087.1 | silhouette=0.0554
```

---

## 4. Elbow Method (WCSS)

### Fórmula
```
WCSS = Σₖ Σ_{xᵢ ∈ Cₖ} ||xᵢ - μₖ||²
```
- `μₖ` = centroide do cluster k
- Mede quão compactos são os clusters internamente
- Sempre diminui com K → busca o "cotovelo" onde a queda fica lenta

### Por que o cotovelo em K=4 é sutil?
Embeddings CLIP em 512D de dados visuais **heterogêneos** (fotos, memes,
wallpapers) não formam clusters perfeitamente separados — o espaço tem
variação contínua. Por isso a curva é suave, não tem cotovelo abrupto.

### Output real
```
K=2 → 3753.7   (queda de -125.0 para K=3)
K=3 → 3627.7   (queda de -107.0 para K=4)
K=4 → 3520.9   ← cotovelo: a próxima queda é menor
K=5 → 3454.1   (queda de -66.8)
K=6 → 3398.0   (queda de -56.1)
```

---

## 5. Silhouette Score

### Fórmulas (slide 3)
```
a(i) = distância média de i aos outros pontos do SEU cluster (coesão)
b(i) = distância média de i ao cluster vizinho mais próximo (separação)

s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

### Interpretação (importante para a banca)
```
s(i) = +1  →  ponto bem no centro do cluster, longe dos vizinhos ✓
s(i) =  0  →  ponto na fronteira entre dois clusters
s(i) = -1  →  ponto provavelmente no cluster errado ✗
```

### Por que 0.0576 é aceitável?
Silhouette baixo (~0.05) em embeddings de alta dimensão é **esperado** porque:
1. Em 512D, distâncias tendem a concentrar (concentration of measure)
2. Dados visuais pessoais são semanticamente contínuos, não há classes discretas
3. Mesmo clusters "reais" se sobrepõem no espaço de features

**Referência para citar:** Rousseeuw, P. J. Comput. Appl. Math., 1987.

### Detalhe de implementação (pode ser perguntado)
Silhouette é O(n²) — caro pra 8000 pontos. O código usa subsample:
```rust
let sample_size = n.min(3000);  // máximo 3000 pontos para calcular
```
Sem isso: 8000² = 64M distâncias por K → inviável.

---

## 6. PCA 2D via Power Iteration

### Por que PCA aqui?
Os embeddings têm 512 dimensões — impossível visualizar. PCA projeta
em 2D preservando a maior variância.

### Power Iteration — como funciona
Em vez de eigendecomposição completa (cara), calcula só os 2 primeiros
componentes principais por iteração de potência:

```
Inicializa vetor v aleatório
Repete 50x:
    scores = X · v          (projeta os dados)
    v_new = Xᵀ · scores    (multiplica transposta)
    v = v_new / ||v_new||   (normaliza)
→ converge pro eigenvector dominante (PC1)
```

### Código real
```rust
// PC1
let mut pc1 = vec![0.0f32; dim];  // 512-dim
pc1[0] = 1.0;                      // init
for _ in 0..50 {
    let scores = project(&pc1, &centered);  // X·v
    let mut new_pc = vec![0.0f32; dim];
    for (v, &s) in centered.iter().zip(&scores) {
        for (np, &x) in new_pc.iter_mut().zip(v) { *np += x * s; }  // Xᵀ·scores
    }
    let norm = new_pc.iter().map(|x| x*x).sum::<f32>().sqrt();
    new_pc.iter_mut().for_each(|x| *x /= norm);
    pc1 = new_pc;
}

// Deflação: remove a variância de PC1 para calcular PC2
let deflated: Vec<Vec<f32>> = centered.iter().zip(&scores1).map(|(v, &s)| {
    v.iter().zip(&pc1).map(|(&x, &p)| x - s * p).collect()  // X - s·pc1ᵀ
}).collect();
// Repete power iteration no deflated para PC2
```

---

## 7. Resultados — Os 4 Clusters

### Tabela real (cluster_report.txt)

| Cluster | N    | %    | Nome dado          | Característcas chave                          |
|---------|------|------|--------------------|-----------------------------------------------|
| 0       | 2125 | 26.6 | Arte digital       | wallpapers 35%, ai_images 32%, landscape 46%  |
| 1       | 1972 | 24.6 | Cenas / ambientes  | iPhone 81%, people 8%, portrait 70%, 8.5MB    |
| 2       | 2301 | 28.8 | Retratos / pessoas | people 50%, iPhone 33%, portrait 75%, 5.9MB   |
| 3       | 1602 | 20.0 | Screenshots/texto  | iPhone 57%, backup 28%, 241KB median          |

### O insight principal (slide mais importante pra apresentar)
**CLIP agrupou por semântica visual, não por metadados.**

Prova:
- Fotos do iPhone aparecem em **todos os 4 clusters** → origem ≠ cluster
- `.png` e `.jpg` distribuídos uniformemente → formato ≠ cluster
- Mas aspect ratio correlaciona com cluster → CLIP capturou orientação
- E tamanho de arquivo correlaciona → CLIP capturou complexidade visual

**Como articular isso:** "O modelo separou o conteúdo semântico do artefato
de origem. Isso valida que os features CLIP carregam informação visual
de alto nível, não baixo nível como metadados de arquivo."

---

## 8. Limitações (saiba defender cada uma)

| Limitação | Por que é real | O que poderia resolver |
|-----------|---------------|------------------------|
| Silhouette baixo (~0.05) | Alta dimensionalidade + dados heterogêneos | Não é necessariamente problema — dados visuais são contínuos |
| PCA 2D perde informação | 512D → 2D comprime muito | UMAP ou t-SNE preservam estrutura local melhor |
| K-Means assume clusters esféricos | Clusters reais podem ser não-convexos | DBSCAN, GMM, clustering hierárquico |
| K fixo a priori | Pode não refletir estrutura real | DBSCAN (K automático), BIC com GMM |

---

## 9. Perguntas Prováveis do Professor + Respostas

**P: "Por que L2-normalizar antes do K-Means?"**
> R: "Pra fazer a distância euclidiana equivaler à distância cosseno.
> Em embeddings de transformers, a direção do vetor carrega semântica —
> não a magnitude. Sem normalização, vetores de maior norma dominariam
> o cálculo de WCSS injustamente."

**P: "Por que usar K-Means++ em vez de K-Means vanilla?"**
> R: "K-Means++ garante inicialização com centroides espalhados —
> cada novo centroide é sorteado com probabilidade proporcional a D²
> ao centroide mais próximo já escolhido. Isso reduz o risco de
> convergência para mínimos locais ruins. Arthur & Vassilvitskii (2007)
> provaram que k-means++ tem bound de custo esperado O(log K) do ótimo."

**P: "Silhouette 0.057 — isso indica clustering ruim?"**
> R: "Não necessariamente. Scores baixos são esperados em espaços de
> alta dimensão com dados heterogêneos. A validação cruzada com metadados
> confirma coerência semântica dos clusters — clusters 0 e 3 têm perfis
> estatisticamente distintos em tamanho, aspect ratio e conteúdo de pasta.
> O silhouette baixo reflete a natureza contínua do espaço de features,
> não ausência de estrutura."

**P: "Por que usar CLIP ao invés de features handcrafted (HOG, SIFT)?"**
> R: "CLIP captura semântica de alto nível treinada em 400M pares
> imagem-texto. Features handcrafted são invariantes a textura/borda local
> mas não entendem 'screenshot' vs 'retrato'. Para clustering de coleção
> pessoal, semântica é mais útil do que textura."

**P: "Como você sabe que K=4 é o certo e não K=3 ou K=5?"**
> R: "Duas métricas concordaram: o cotovelo na curva WCSS em K=4 e o
> pico de silhouette também em K=4 (s=0.0576). Além disso, a inspeção
> qualitativa dos clusters confirma que os 4 grupos têm interpretação
> semântica clara e distinta."

**P: "Por que 3 runs por K? Por que não apenas 1?"**
> R: "K-Means pode convergir para mínimos locais dependendo da inicialização.
> Com 3 runs independentes (seeds diferentes), selecionamos o resultado com
> menor WCSS — garante maior probabilidade de achar um bom mínimo local.
> Para o modelo final com K=4, usamos 5 runs para maior robustez."

**P: "O que é power iteration e por que não SVD completo?"**
> R: "Power iteration é um algoritmo iterativo que converge para o
> eigenvector dominante de XᵀX sem computar a matriz explicitamente.
> SVD completo em 512×8000 seria O(min(n,d)²·max(n,d)) — caro e
> desnecessário quando só precisamos de 2 componentes. Power iteration
> com deflação é O(n·d·iter) — linear no número de pontos."

---

## 10. Números que você precisa saber de cabeça

```
Dataset total:     22.258 imagens
Amostra:            8.000 (seed=42)
Corrompidas:           63 dropadas
Dim embedding:        512 (CLIP ViT-B/32)
K testados:        2..15
Runs por K:            3 (final: 5)
Max iterações:       100

Melhor K:              4
WCSS (K=4):       3520.9
Silhouette (K=4): 0.0576   ← valor mais importante

Cluster 0:  2125 imgs, 26.6%, Arte digital
Cluster 1:  1972 imgs, 24.6%, Cenas/ambientes
Cluster 2:  2301 imgs, 28.8%, Retratos/pessoas
Cluster 3:  1602 imgs, 20.0%, Screenshots/texto
```

---

## 11. Stack Técnico (se perguntarem)

```toml
fastembed  = "5.13"   # CLIP ViT-B/32 via ONNX Runtime — zero Python
plotters   = "0.3"    # geração de PNGs
image      = "0.25"   # leitura de dimensões (width, height)
walkdir    = "2"      # travessia recursiva de diretórios
rand       = "0.9"    # seed=42 para reprodutibilidade
indicatif  = "*"      # progress bar no terminal
```

**Por que 100% Rust?**
- Sem overhead de ambiente Python/conda
- ONNX Runtime via bindings nativos
- Binário único: `cargo build --release` → roda em qualquer máquina

---

*Gerado em 2026-03-23 para preparação da apresentação de Aprendizado de Máquina 2026.1*
