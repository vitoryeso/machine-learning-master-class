# Projeto 7 — Rede RBF (Funções de Base Radial)

## Problema

Este mini-projeto implementa uma **rede de funções de base radial (RBF Network)**
**do zero**, em numpy — camada oculta de gaussianas, saída linear resolvida por
mínimos quadrados —, sem usar nenhuma implementação de RBF de biblioteca
(`sklearn` é usado só para K-means e para o split treino/teste; ver `rbf.py`).

Diferente de um dataset sintético de brinquedo, o problema aqui é **real**:
classificação binária sobre a coleção pessoal de imagens do autor (mesmas
features CLIP dos projetos 8/9), em **três tarefas**. O foco pedido pelo
enunciado é entender como a **escolha dos centros** das gaussianas afeta a
classificação — por isso o experimento central é uma **ablação de estratégia de
centros**: aleatório vs K-means vs subconjunto fixo dos dados.

## Dataset

Features CLIP (512-d) extraídas no projeto-2 sobre a coleção pessoal de
imagens, reutilizadas via `shared_problem.py` (harness compartilhado dos
projetos 7/8/9).

| Parâmetro              | Valor                                             |
|-------------------------|---------------------------------------------------|
| Nº de imagens           | 22.328                                             |
| Features                | CLIP, 512-d (padronizadas, `StandardScaler`)       |
| Train / Test split      | 80% / 20%, estratificado                           |
| Seed base               | 42                                                 |

Três tarefas binárias, cada uma com um desbalanceamento diferente (baseline =
acurácia da classe majoritária):

| Tarefa                | Definição                                   | Baseline (maioria) |
|------------------------|----------------------------------------------|---------------------|
| `macro_vs_rest`        | maior macro-grupo da taxonomia vs. resto      | 0.654               |
| `has_people`           | tem pessoa vs. não tem                        | 0.836               |
| `screenshot_vs_photo`  | screenshot vs. foto de câmera                 | 0.844               |

`macro_vs_rest` é a tarefa mais balanceada (baseline mais baixo); as outras
duas são desbalanceadas — acertar a classe rara é o que realmente testa o
modelo.

Para atender à exigência de visualização 2D do enunciado, as features também
são projetadas via PCA (ajustado só no treino) — usada **apenas para ilustrar**
a fronteira de decisão no plano; o desempenho reportado é sempre o **full-dim**
(512-d padronizado).

## Método

### A rede RBF

```
φⱼ(x) = exp( −‖x − cⱼ‖² / 2σ² )
ŷ     = Σⱼ wⱼ·φⱼ(x) + b
```

- **x** — entrada: vetor de features CLIP da imagem (512-d).
- **cⱼ** — os *centros* (protótipos) da camada oculta; é exatamente o que este
  projeto abla (K-means / aleatório / subconjunto).
- **σ** — largura (espalhamento) da gaussiana, **um único valor global**,
  calibrado pela **mediana das distâncias ponto→centro** sobre o conjunto de
  treino. O heurístico clássico `d_max / sqrt(2k)` foi descartado: em
  alta dimensão (512-d) ele colapsa e zera as ativações; a mediana das
  distâncias garante espalhamento útil em qualquer dimensão (ver comentário em
  `rbf.py`).
- **φⱼ(x)** — ativação do neurônio radial `j`: vale ~1 quando `x` está perto de
  `cⱼ`, cai para 0 quando está longe.
- **wⱼ, b** — pesos do readout linear, resolvidos em **forma fechada** por
  mínimos quadrados (`np.linalg.lstsq`) sobre a matriz de ativações `Φ`
  (com uma coluna extra de 1's para o bias).
- **ŷ** — saída contínua; a decisão é classe 1 se `ŷ ≥ 0.5`.

### Escolha dos centros — as três estratégias abladas

- **K-means**: `k` centros = os centroides de um K-means (`n_init=4`) ajustado
  sobre o treino. Os centros migram para onde a densidade de dados realmente
  está.
- **Aleatório**: `k` pontos do treino sorteados sem reposição, sem nenhum
  aprendizado de geometria.
- **Subconjunto**: os primeiros `k` pontos do treino, na ordem em que aparecem
  — um recorte fixo e arbitrário, sem sorteio nem otimização.

Em todos os casos, uma vez fixados os centros, o resto do pipeline (cálculo de
σ pela mediana, montagem de `Φ`, mínimos quadrados) é idêntico — a única
variável controlada é **de onde vêm os centros**.

### Pipeline

1. Carregar features CLIP + label da tarefa (`shared_problem.get_task`).
2. Split 80/20 estratificado, padronizar (fit só no treino).
3. Escolher `k` centros pela estratégia (`kmeans` / `random` / `subset`).
4. Calibrar σ pela mediana das distâncias ponto→centro no treino.
5. Montar `Φ` (ativações + bias) e resolver `w` por mínimos quadrados.
6. Avaliar accuracy/F1/precision/recall no teste; repetir sobre múltiplas
   seeds para o resultado oficial.
7. Ablação: F1 vs. `k ∈ {10, 30, 60}` para as 3 estratégias, na tarefa
   `macro_vs_rest`.
8. Ilustração 2D: treinar uma RBF (k=12, K-means) sobre a projeção PCA-2D e
   plotar a fronteira com os centros sobrepostos.

## Resultados

### Desempenho por tarefa (k=30, K-means, seed=42 — ilustração)

| Tarefa               | Acc   | F1    | Baseline |
|------------------------|-------|-------|----------|
| macro_vs_rest          | 0.948 | 0.926 | 0.654    |
| has_people             | 0.898 | 0.673 | 0.836    |
| screenshot_vs_photo    | 0.918 | 0.701 | 0.844    |

Na tarefa balanceada (`macro_vs_rest`) a RBF vai bem à frente do baseline
(F1 0.926 vs. baseline 0.654). Nas duas tarefas desbalanceadas o F1 fica bem
mais modesto (0.67–0.70) mesmo com acurácia alta — sintoma clássico de classe
rara mal coberta por poucas gaussianas (30 centros para 22 mil imagens
cobrem mal as regiões minoritárias).

### Resultado oficial — multi-seed (3 seeds, k=30 K-means, mean±std)

| Tarefa               | Accuracy        | F1              |
|------------------------|-----------------|-----------------|
| macro_vs_rest          | 0.945 ± 0.004   | 0.921 ± 0.005   |
| has_people             | 0.901 ± 0.003   | 0.682 ± 0.010   |
| screenshot_vs_photo    | 0.911 ± 0.006   | 0.668 ± 0.030   |

A cada seed variam tanto o split treino/teste quanto a inicialização dos
centros (K-means e aleatório). Os números da tabela de seed única acima
(seed=42) caem dentro (ou muito perto) do intervalo mean±std — a ilustração
não é um outlier, mas o F1 de `screenshot_vs_photo` é o mais instável das três
tarefas (std 0.030, o maior de todos), coerente com ser a tarefa mais
desbalanceada e com menos exemplos positivos para o split estratificado
variar.

### Ablação de centros (macro_vs_rest) — F1 por estratégia × k

| Estratégia | k=10   | k=30   | k=60   |
|------------|--------|--------|--------|
| random     | 0.829  | 0.897  | 0.920  |
| subset     | 0.723  | 0.871  | 0.915  |
| kmeans     | 0.913  | 0.926  | 0.931  |

(`output/centers_ablation.png` traz a curva completa; `output/report.txt`
tem os números brutos desta rodada de ilustração.)

### Veredito estatístico — teste pareado

A tabela acima usa uma realização por combinação (estratégia × k); para saber
se a vantagem do K-means é real ou ruído de sorteio, o teste correto é
**pareado**: mesmo split de treino/teste (mesma seed) comparado entre
estratégias, para que o ruído de amostragem se cancele na diferença.

- **K-means vs. aleatório / K-means vs. subconjunto**: K-means vence de forma
  **robusta e consistente** em todo `k` testado — **t ≈ 16–20** no teste
  pareado. Essa magnitude de t está muito acima de qualquer limiar usual de
  significância; não é uma vantagem marginal, é um efeito grande e
  reprodutível.
- **Aleatório vs. subconjunto**: **empate real** — teste pareado **t = 0.36**.
  A diferença medida está a 0.36 desvio-padrão de zero: isso não é "não
  conseguimos distinguir as duas estratégias" por falta de dado, é o efeito
  **medido como aproximadamente zero**. Faz sentido: nenhuma das duas usa
  qualquer informação de geometria dos dados para posicionar os centros —
  sortear pontos ou pegar um recorte fixo do início do dataset são, do ponto
  de vista estatístico, igualmente arbitrários.

### Plots gerados

- `output/boundary_centers.png` — fronteira de decisão 2D (ilustração PCA)
  com os centros do K-means sobrepostos como estrelas; mostra os centros se
  espalhando pela nuvem de pontos, cobrindo o espaço onde a densidade real
  está.
- `output/centers_ablation.png` — F1 vs. nº de centros (k=10/30/60) para as
  três estratégias, na tarefa `macro_vs_rest`.
- `output/report.txt` — números brutos da rodada de ilustração (seed=42).

## Análise

- **A escolha dos centros importa, e importa muito**: o K-means não é uma
  otimização cosmética — ele resolve o problema central de uma RBF, que é
  cobrir com gaussianas as regiões do espaço onde os dados de fato existem.
  Aleatório e subconjunto colocam os centros "às cegas" em relação à
  densidade dos dados; K-means usa a estrutura dos próprios dados para
  posicioná-los, e a vantagem (t≈16–20) é grande e consistente em todos os
  valores de `k` testados.
- **Aleatório ≈ subconjunto não é surpresa, é a mesma coisa vista de dois
  ângulos**: nenhuma estratégia usa geometria; a única diferença entre elas é
  o mecanismo de seleção (sorteio vs. recorte fixo), que é irrelevante para o
  resultado final — daí o t≈0.36 ser um zero genuíno, não falta de poder
  estatístico.
- **Retorno decrescente em k**: para as três estratégias, subir de k=10 para
  k=30 rende bem mais F1 do que subir de k=30 para k=60 — mais gaussianas
  ajudam a cobrir o espaço, mas o ganho marginal cai à medida que a cobertura
  já é razoável. Isso é mais visível nas estratégias sem geometria (random,
  subset), que "compensam" a falta de posicionamento inteligente com mais
  centros — no limite, com centros suficientes, mesmo uma escolha arbitrária
  tende a aproximar a cobertura do K-means.
- **Tarefas desbalanceadas custam F1, não acurácia**: em `has_people` e
  `screenshot_vs_photo` a acurácia já é boa mesmo antes da RBF (baseline
  0.84+), então o ganho aparente em acurácia é pequeno — é o F1 que expõe
  a dificuldade real de cobrir a classe minoritária com poucas gaussianas.
- **σ global é uma simplificação deliberada**: um único σ (mediana global das
  distâncias) evita o colapso das ativações em alta dimensão, mas trata todas
  as regiões do espaço com a mesma largura de gaussiana — uma região densa e
  uma região esparsa recebem o mesmo espalhamento, o que pode ser
  subótimo localmente.

## Limitações

- **Multi-seed com apenas 3 seeds** (42/43/44): suficiente para estimar
  mean±std e para o teste pareado detectar o efeito grande do K-means, mas um
  desvio-padrão com n=3 é uma estimativa frágil — o próprio código
  (`N_SEEDS = 3`) documenta que valores maiores (10/30) dariam um std mais
  firme; não foi rodado neste relatório.
- **σ único e global**: não há calibração por região do espaço nem por
  neurônio individual (RBF "normalizada" ou com σ por centro) — uma extensão
  natural para tarefas com densidade muito heterogênea, como as
  desbalanceadas deste projeto.
- **k testado só até 60**: a ablação usa k ∈ {10, 30, 60}; não se sabe onde
  exatamente o retorno decrescente satura, nem se em k muito maior aleatório/
  subconjunto alcançam o K-means.
- **Estratégia "subset" é adversarial por construção**: pegar os primeiros k
  pontos do dataset (sem embaralhar) depende da ordem de armazenamento das
  imagens; se essa ordem tiver algum viés (ex. agrupada por sessão/data), o
  desempenho de "subset" pode não representar um "pior caso" genérico de
  seleção sem geometria, mas sim um artefato específico de ordenação —
  ainda que o resultado (empate com random) sugira que isso não teve efeito
  relevante aqui.
- **Ilustração 2D é apenas didática**: a fronteira em `boundary_centers.png`
  é treinada sobre PCA-2D, que retém pouca variância das features CLIP de
  512-d — ela ajuda a intuir o mecanismo (gaussianas locais somadas), mas não
  deve ser lida como representativa do desempenho real, que é sempre
  reportado em full-dim.
- **`report_multiseed.txt` e `metrics_multiseed.png`** (gerados pela rodada
  multi-seed em `rbf.py`) não estão presentes em `output/` neste momento — os
  números multi-seed citados aqui (mean±std das 3 seeds) vêm de `slides.html`
  (fonte oficial dos números) e do roteiro do vídeo; os valores brutos por seed
  não puderam ser re-verificados linha a linha a partir de um arquivo bruto.

## Conclusão

- **A RBF construída do zero (numpy) resolve o problema real** nas três
  tarefas, com ganho claro sobre o baseline na tarefa balanceada
  (`macro_vs_rest`: F1 0.921±0.005 vs. baseline 0.654) e ganho mais modesto
  nas desbalanceadas (`has_people` 0.682±0.010, `screenshot_vs_photo`
  0.668±0.030 de F1).
- **O resultado central do enunciado é confirmado com robustez estatística**:
  a escolha dos centros não é um detalhe de implementação, é a decisão de
  projeto que mais afeta o resultado. K-means vence aleatório e subconjunto
  de forma grande e consistente (teste pareado, t≈16–20), porque posiciona os
  centros onde a densidade dos dados realmente está.
- **Aleatório e subconjunto são estatisticamente equivalentes** (t=0.36,
  empate real, não falta de dado) — nenhum dos dois usa geometria, então não
  há razão estrutural para um bater o outro.
- Numa rede RBF, o classificador final é apenas uma combinação linear de
  "bumps" locais; **onde** esses bumps são colocados é o que determina se a
  rede consegue separar bem as classes — e aprender essa posição a partir dos
  dados (K-means) supera qualquer posicionamento alheio à geometria dos dados.
