# Projeto 2 — Classificação Multi-Task de Imagens Pessoais com Feature Extractors Pré-Treinados

## Problema

Classificação de imagens pessoais (~8000 amostras de D:\media) em duas tarefas simultâneas:
- **image_type** (6 classes): camera_photo, desktop_wallpaper, mobile_screenshot, ai_generated, screenshot_desktop, thumbnail
- **has_people** (binário): presença de pessoas na imagem

As classes foram definidas por regras determinísticas sobre metadados (resolução, aspect ratio, pasta de origem), formando o ground truth. A pergunta central: um modelo consegue aprender essas categorias a partir apenas das representações visuais, sem acesso aos metadados?

## Dataset

- **Origem**: coleção pessoal em D:\media (~8000 imagens, amostra do projeto-1)
- **Embeddings CLIP**: 512-dim, extraídos com fastembed-rs (Qdrant/clip-ViT-B-32-vision, ONNX) no projeto-1
- **Features ConvNeXt**: 768-dim, extraídos com ConvNeXt-Tiny (torchvision, IMAGENET1K_V1, frozen)
- **Labeling**: pipeline de 3 tiers — Tier 1 (resolução exata), Tier 2 (pasta + aspect ratio), Tier 3 (fallback)
- **Split**: 70% treino / 15% validação / 15% teste, estratificado por image_type

### Distribuição de Classes

| Classe | N | % |
|---|---|---|
| camera_photo | 4385 | 54.8% |
| desktop_wallpaper | 1191 | 14.9% |
| ai_generated | 1032 | 12.9% |
| thumbnail | 580 | 7.2% |
| mobile_screenshot | 515 | 6.4% |
| screenshot_desktop | 297 | 3.7% |

has_people: 1323 positivos (16.5%)

## Arquitetura

Modelo multi-task com backbone compartilhado e duas heads de saída:

```
input features (D-dim)
    |
  Linear(D, 256) -> ReLU -> Dropout(0.3)
  Linear(256, 128) -> ReLU -> Dropout(0.3)
    |
  +---------+---------+
  |                   |
Linear(128, 1)    Linear(128, 6)
  |                   |
people_logit      type_logits
```

Loss = BCEWithLogitsLoss(people) + CrossEntropyLoss(type)

Sem sigmoid/softmax explícito no forward — as funções de custo aplicam internamente.

## Feature Extractors Comparados

| | CLIP ViT-B/32 | ConvNeXt-Tiny |
|---|---|---|
| Params (vision) | 87.5M | 28.6M |
| Dataset treino | WIT, 400M pares img+texto | ImageNet-1K, 1.28M imgs |
| Epochs pretraining | 32 | 600 |
| Objetivo | Contrastivo (InfoNCE) | Cross-entropy supervisionado |
| Embedding dim | 512 | 768 |
| Input | 224x224 | 224x224 |

Refs: Radford et al. (2021, arXiv:2103.00020), Liu et al. (2022, arXiv:2201.03545)

## Experimentos

Três configurações, todas com Adam (lr=1e-3, weight_decay=1e-4), cosine annealing, 200 epochs, best checkpoint por val type_f1:

### 1. Linear Probe (CLIP)
- Entrada: CLIP 512-dim
- Modelo: Linear(512, 7) — 3,591 parâmetros
- Sem camadas ocultas

### 2. MLP Probe (CLIP)
- Entrada: CLIP 512-dim
- Modelo: MLP 512->256->128 + 2 heads — 165,127 parâmetros

### 3. MLP Probe (ConvNeXt)
- Entrada: ConvNeXt 768-dim
- Modelo: MLP 768->256->128 + 2 heads — 230,663 parâmetros

## Resultados (Test Set)

| Métrica | Linear (CLIP) | MLP (CLIP) | MLP (ConvNeXt) |
|---|---|---|---|
| type_acc | 0.823 | **0.872** | 0.828 |
| type_f1 (weighted) | 0.817 | **0.869** | 0.826 |
| people_acc | 0.898 | 0.941 | **0.943** |
| people_f1 | 0.665 | 0.827 | **0.828** |
| people_auc | 0.952 | **0.980** | 0.978 |
| Params treináveis | 3,591 | 165,127 | 230,663 |

### F1 por classe (MLP CLIP, melhor modelo):

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ai_generated | 0.86 | 0.85 | 0.85 | 155 |
| camera_photo | 0.89 | 0.92 | 0.90 | 658 |
| desktop_wallpaper | 0.86 | 0.87 | 0.86 | 179 |
| mobile_screenshot | 0.82 | 0.87 | 0.84 | 77 |
| screenshot_desktop | 0.73 | 0.43 | 0.54 | 44 |
| thumbnail | 0.84 | 0.80 | 0.82 | 87 |

## Análise

1. **CLIP > ConvNeXt para image_type**: apesar de ter 3x mais parâmetros no encoder e embedding menor (512 vs 768), CLIP produz features mais discriminativas. A supervisão contrastiva em 400M pares image-text gera representações semânticas mais ricas que a classificação supervisionada em 1.28M imagens.

2. **People empatado**: ambos os encoders capturam igualmente bem a presença humana (~83% F1). É uma feature visual saliente que qualquer rede profunda aprende.

3. **Linear probe CLIP quase empata ConvNeXt MLP**: 3,591 params vs 230,663 params com performance comparável. As features CLIP já são quase linearmente separáveis para essas classes.

4. **screenshot_desktop é a classe mais fraca** (F1=0.54): menor suporte (44 amostras no teste), alta confusão com camera_photo. Screenshots de desktop são visualmente heterogêneos.

5. **ConvNeXt overfita mais**: val_loss subiu de 1.0 para 1.66 durante o treino, enquanto CLIP manteve val_loss relativamente estável (~0.5 a 0.7).

## Limitações

- Dataset é uma amostra de 8000 imagens (de um acervo maior), com classes desbalanceadas
- Labels derivados de metadados (resolução, pasta), não de anotação humana — pode haver ruído
- Features são frozen (sem fine-tuning dos encoders) — limita o potencial de adaptação
- has_people derivado apenas da pasta "people/" — não cobre todas as imagens com pessoas
- Classe screenshot_desktop tem suporte muito baixo (44 no teste)

## Estrutura do Projeto

```
projeto-2/
  extract_metadata.py   — extrai resolução, aspect, folder de cada imagem
  build_dataset.py      — aplica regras determinísticas, gera labels
  linear_probe.py       — experimento linear probe (CLIP)
  mlp_probe.py          — experimento MLP (CLIP)
  convnext_probe.py     — feature extraction ConvNeXt + MLP
  train_all.py          — script unificado: 3 experimentos + curvas + plots
  metadata.json         — metadados das 8000 imagens
  dataset/
    X_embeddings.npy    — features CLIP (8000, 512)
    X_convnext.npy      — features ConvNeXt (8000, 768)
    y_type.npy          — labels de tipo (8000,)
    y_people.npy        — labels de pessoas (8000,)
    class_names.json    — nomes das 6 classes
  output/
    training_curves.png — curvas de loss e val metrics por epoch
    confusion_matrices.png — 3 matrizes de confusão lado a lado
    comparison_bar.png  — barras comparativas de métricas
    all_metrics.json    — métricas de todos os experimentos
```

---

# Parte II — Classificação da taxonomia descoberta por clustering (2 níveis)

## Motivação

A Parte I classifica rótulos **determinísticos** (resolução/pasta). Aqui seguimos a
direção complementar e que fecha o ciclo com o Projeto 1: em vez de rótulos
fabricados por regras, usamos a **taxonomia descoberta pelo K-means hierárquico**
(`hierarchy.json`) como alvo de classificação. O clustering rodou sobre as 22.328
imagens (CLIP 512-d) e produziu **2 níveis**: 5 macro-grupos (G00–G04) e 25
subclasses-folha.

## Pergunta metodológica — circularidade

Treinar o classificador sobre os **mesmos** embeddings CLIP que geraram os clusters
infla a acurácia: os clusters são aproximadamente células de Voronoi naquele
espaço, então classificá-los de volta tende a alta acurácia — porém não perfeita,
pois a padronização das features não preserva a geometria (esfera unitária ≈
cosseno) em que o k-means rodou. Para um teste honesto, classificamos
sobre features **ConvNeXt** (modelo independente). Se a taxonomia for "real", um
espaço de features diferente também deve recuperá-la. Rodamos os dois e comparamos
(`classify_clusters.py`).

## Método

- Alvo: `y_macro` (0–4) e `y_leaf` (0–24) reconstruídos dos `image_indices` do `hierarchy.json`.
- Features: ConvNeXt 768-d (teste honesto) e CLIP 512-d (controle circular), padronizadas.
- Classificador: linear probe (LogisticRegression multinomial), split 80/20 estratificado por folha (seed=42, treino 17.862 / teste 4.466).
- Leitura de 2 níveis: o probe prevê a folha; o macro é derivado pelo mapa folha→macro.

## Resultados

### Single-seed (seed=42, ilustração)

| Features / nível        | Acurácia | Precisão | Recall | F1 (macro) |
|-------------------------|----------|----------|--------|------------|
| CLIP / macro (circular) | 0.9503   | 0.9501   | 0.9486 | 0.9493     |
| CLIP / folha (circular) | 0.8930   | 0.8842   | 0.8834 | 0.8833     |
| **ConvNeXt / macro**    | **0.8269** | 0.8190 | 0.8143 | 0.8164     |
| **ConvNeXt / folha**    | **0.6597** | 0.6260 | 0.6267 | 0.6245     |

Plots em `output/classification/`: `accuracy_comparison.png`, `confusion_macro_convnext.png`, `confusion_leaf_convnext.png`; números completos em `report_classification.txt`.

### Multi-seed (5 seeds, mean±std, ddof=1) — resultado oficial

O experimento acima foi repetido com 5 seeds (split estratificado + `random_state`
do `LogisticRegression` variando; dados reais fixos), para checar se o gap
CLIP→ConvNeXt é estável ou um artefato de uma amostragem particular:

| Features / nível | Acurácia | F1 (macro) |
|-------------------|----------|------------|
| CLIP / macro       | 0.954 ± 0.004 | 0.953 ± 0.004 |
| CLIP / folha        | 0.896 ± 0.005 | 0.884 ± 0.003 |
| ConvNeXt / macro    | 0.833 ± 0.006 | 0.817 ± 0.008 |
| ConvNeXt / folha     | 0.661 ± 0.004 | 0.624 ± 0.004 |

Os valores single-seed batem dentro de ~1 std da média multi-seed em todas as
linhas — a tabela seed=42 acima é consistente com o resultado oficial, apenas
uma amostra pontual dele.

## Conclusão

- **O nível macro é real:** o ConvNeXt (independente) recupera os 5 grupos com ~83% (multi-seed) — a estrutura grossa não é artefato do CLIP.
- **O nível fino é CLIP-específico:** as 25 subclasses caem para ~66% no ConvNeXt — distinções sutis que vivem sobretudo na semântica do CLIP.
- **A circularidade é mensurável (descontando o baseline):** o gap CLIP→ConvNeXt (95.0%→82.7% macro; 89.3%→66.0% folha, seed=42) mistura dois efeitos. Parte dele é apenas o ConvNeXt ser um extrator pior para esta distribuição — na Parte I, com rótulos **externos e determinísticos** (sem circularidade alguma), o ConvNeXt já fica ~4 F1-pts atrás do CLIP (type_f1 0.826 vs 0.869). Só o **excesso** sobre esse baseline (12 pts macro, 23 pts folha) é atribuível à auto-confirmação — e ainda assim excede o baseline com folga.
- **O gap é robusto, não ruído amostral:** no multi-seed, o gap CLIP→ConvNeXt (acc: 0.954→0.833 macro, 0.896→0.661 folha) vale **16–47× o desvio-padrão** observado entre seeds (std de 0.004 a 0.008). Ou seja, a variação seed-a-seed é irrisória frente à diferença entre extratores — a circularidade discutida acima é um efeito real e robustamente mensurável, não um acaso do split seed=42.
- **Limitações:** rótulos vêm de clustering não-supervisionado (sem verdade externa); a rotulagem é **transdutiva** — a partição-alvo foi ajustada pelo K-means global sobre as 22.328 imagens (incl. as de teste), logo os rótulos do teste não são independentes dele (um protocolo plenamente honesto clusterizaria só no treino e atribuiria o teste por centróide mais próximo); linear probe é linear; classes desbalanceadas (folhas de 276 a 4.218 imagens) — por isso reportamos F1 macro-averaged.
