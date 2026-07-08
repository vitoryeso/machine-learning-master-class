# Projeto 5 — Árvore de Decisão: Gini vs Entropia

## Problema

Este mini-projeto implementa e compara dois critérios de divisão de árvores de decisão: **Gini** e **Entropia (Information Gain)**. Atendendo ao enunciado ("construir manualmente"), a árvore CART é implementada **do zero** — critério de impureza, busca gulosa de split, recursão, predição e importância de features —, **sem usar `sklearn.tree`**. O `scikit-learn` é usado apenas para gerar o dataset sintético (`make_classification`) e o split treino/teste; o modelo é todo manual (ver `main.py`).

O objetivo é entender as diferenças práticas dos dois critérios em estrutura da árvore (profundidade, nós, folhas) e desempenho preditivo, além de visualizar as fronteiras de decisão no plano 2D.

## Dataset

Dataset sintético gerado com `sklearn.datasets.make_classification`:

| Parâmetro            | Valor       |
|----------------------|-------------|
| n_samples            | 500         |
| n_features           | 2           |
| n_informative        | 2           |
| n_redundant          | 0           |
| n_clusters_per_class | 1           |
| class_sep            | 0.9         |
| random_state         | 42          |
| Train / Test split   | 400 / 100   |

Dataset 2D puro para que as fronteiras de decisão sejam diretamente visualizáveis. A separação de classes de 0.9 cria um problema de dificuldade moderada.

## Metodologia

### Critério de Impureza — Gini

```
G(t) = 1 - Σ p_k²
```

onde `p_k` é a proporção da classe `k` no nó `t`. Gini varia de 0 (nó puro) a `1 - 1/K` (uniforme); para binário, máximo em 0.5.

### Critério de Impureza — Entropia (Information Gain)

```
H(t) = - Σ p_k · log2(p_k)
```

Por convenção `0·log2(0) = 0`. O ganho de informação de um split é:

```
IG = H(pai) - (|L|/|P|)·H(L) - (|R|/|P|)·H(R)
```

### Construção da árvore (CART manual)

A cada nó, uma **busca gulosa** percorre cada feature e cada ponto-médio entre valores consecutivos como limiar candidato, escolhendo o split que **maximiza a redução de impureza** (`impureza(pai) − média ponderada da impureza dos filhos`). A recursão continua até folhas puras (impureza 0), respeitando `max_depth` e `min_samples_split` quando definidos. A importância de cada feature é a soma das reduções de impureza ponderadas pela fração de amostras do nó, normalizada para somar 1 — a mesma definição do scikit-learn.

### Pipeline

1. Gerar dataset 2D com `make_classification`
2. Split 80/20 estratificado (`stratify=y`, seed=42)
3. Treinar a árvore manual com `criterion='gini'` e `criterion='entropy'`
4. Calcular métricas: accuracy, precision, recall, F1, profundidade, nós, folhas
5. Plotar fronteiras de decisão (predição manual sobre um meshgrid 300×300)
6. Comparar importância das features e varredura de `max_depth`

## Resultados

### Métricas de Classificação (seed=42, ilustração)

| Métrica          | Gini   | Entropy |
|------------------|--------|---------|
| Train Accuracy   | 1.0000 | 1.0000  |
| Test Accuracy    | 0.9600 | 0.9400  |
| Precision        | 0.9423 | 0.9231  |
| Recall           | 0.9800 | 0.9600  |
| F1-Score         | 0.9608 | 0.9412  |

Nesta seed isolada, o **Gini aparenta vantagem em todas as métricas** — 2 pontos percentuais de acurácia de teste — e as predições dos dois critérios divergem em **4 das 100** amostras de teste. **Atenção:** a validação multi-seed abaixo mostra que essa diferença é ruído de amostragem, não um efeito real do critério — ver seção "Validação Multi-Seed".

### Validação Multi-Seed (10 seeds, N=2000) — resultado oficial

A tabela de métricas acima usa uma única seed (42) e um dataset pequeno (N=500) — insuficiente para separar efeito real de ruído de amostragem. Para validar o claim, repetimos o experimento com **10 seeds** em um dataset maior (**N=2000**), reportando média ± desvio padrão (ddof=1):

| Métrica          | Gini            | Entropy         |
|------------------|-----------------|-----------------|
| Test Accuracy    | 0.943 ± 0.033   | 0.945 ± 0.033   |
| F1-Score         | (idem, mesma ordem de grandeza e sobreposição) | |

**Gini ≈ Entropia — indistinguíveis no multi-seed.** Os intervalos se sobrepõem quase totalmente e o ranking até **inverte** em relação à seed única (Entropy nominalmente acima, não Gini). A diferença de "2 p.p." observada com seed=42 era **ruído de amostragem**, não um efeito do critério. O que se mantém direcionalmente real entre os dois critérios é a **estrutura**: a árvore de Entropia tende a ficar mais profunda que a de Gini.

**Teste pareado — mais forte que "indistinguíveis":** o multi-seed acima é *não pareado* — cada seed sorteia uma amostra diferente para avaliar Gini e Entropy, então parte do desvio-padrão observado (±0.033) é ruído de amostragem compartilhado entre os dois critérios, não diferença real. Repetindo com **teste pareado** (mesmo split de treino/teste a cada seed, comparando Gini e Entropy nos mesmíssimos dados, **n=30 seeds**, teste N=400): a diferença de acurácia Gini−Entropy = **−0.0002 ± 0.008** (**t=0.11**) — ou seja, **0.1 desvio-padrão de zero**. Isso não é "não conseguimos distinguir os dois critérios"; é o efeito **medido como zero**. O truque do pareamento é usar os mesmos dados para os dois critérios em cada seed: como o ruído de amostragem afeta Gini e Entropy igualmente, ele se cancela na diferença — por isso o desvio-padrão pareado (±0.008) sai **7× menor** que o desvio-padrão não-pareado (±0.055). O único efeito real e consistente entre os critérios continua sendo estrutural: **Entropy produz árvore mais profunda** (15.5 vs 17.8 níveis, sem poda). Podando com `max_depth=5`, o empate se mantém (**t=0.57**) e a acurácia até melhora um pouco (0.947 vs 0.936) — mais uma evidência de que a árvore irrestrita estava overfitando.

**Lição sobre variância:** aumentar o dataset de teste (N=500 → 2000) **não reduziu** o desvio-padrão entre seeds — pelo contrário, foi de ±0.021 para ±0.033. Isso é o oposto do que aconteceu nos modelos lineares dos projetos 3/4, onde mais dados tende a estabilizar a métrica. A razão é que a **árvore de decisão não-podada é um modelo de alta variância**: ela cresce até folhas puras (ver overfitting abaixo), então pequenas mudanças na amostra de treino/teste mudam a estrutura da árvore e, com ela, o resultado — o desvio entre seeds reflete a **instabilidade do próprio modelo**, não o erro amostral da estimativa de teste. Mais dados por si só não resolve isso; apenas **restringir `max_depth` (poda)** reduziria essa variância. Isso reforça a nota já presente neste relatório sobre validação cruzada k-fold como trabalho futuro — o problema não é só a margem de erro da estimativa (100 amostras de teste), mas a variância estrutural do modelo em si, que k-fold ajudaria a quantificar melhor do que uma única seed.

### Matriz de Confusão (teste, n=100, seed=42, ilustração)

**Gini:**

|              | Pred 0 | Pred 1 |
|--------------|--------|--------|
| **Real 0**   | TN=47  | FP=3   |
| **Real 1**   | FN=1   | TP=49  |

**Entropy:**

|              | Pred 0 | Pred 1 |
|--------------|--------|--------|
| **Real 0**   | TN=46  | FP=4   |
| **Real 1**   | FN=2   | TP=48  |

O Gini errou 4 amostras (3 FP + 1 FN); a Entropia errou 6 (4 FP + 2 FN).

### Estrutura da Árvore (seed=42, ilustração)

| Propriedade   | Gini | Entropy |
|---------------|------|---------|
| Profundidade  | 9    | 13      |
| N. de folhas  | 26   | 27      |
| N. de nós     | 51   | 53      |

A árvore da Entropia é **mais profunda** (13 vs 9). A explicação é geométrica, não de critério de parada: ambas crescem até folhas puras (train accuracy = 1.0), mas a Entropia tem maior curvatura em torno de p=0.5, produzindo reduções de impureza de escala diferente. Isso altera a ordenação gulosa dos splits, e os caminhos de partição resultantes exigem mais níveis para isolar os mesmos pontos.

### Importância das Features

| Feature   | Gini   | Entropy |
|-----------|--------|---------|
| Feature 0 | 0.3580 | 0.3928  |
| Feature 1 | 0.6420 | 0.6072  |

Em ambos os critérios, a **Feature 1 concentra mais poder preditivo** (~61–64%).

### Plots gerados

- `output/boundary_comparison.png` — fronteiras de decisão 2D lado a lado (Gini vs Entropy), pontos de treino (x) e teste (o) sobrepostos
- `output/importance.png` — barras comparando importância das features por critério
- `output/depth_vs_accuracy.png` — varredura de `max_depth` vs acurácia (treino e teste) para os dois critérios, com linhas verticais nas profundidades irrestritas (Gini=9, Entropy=13)
- `output/confusion_matrix.png` — heatmaps das matrizes de confusão no teste
- `output/report.txt` — relatório numérico completo, análise de splits redundantes, contagem de amostras por folha e estrutura textual das árvores

## Conclusão

- **Gini ≈ Entropia — indistinguíveis (resultado oficial, multi-seed):** na validação com 10 seeds e N=2000, Gini (0.943±0.033) e Entropy (0.945±0.033) ficam dentro do mesmo intervalo de variação, com o ranking até invertido em relação à seed única. A "vantagem de 2 p.p." vista com seed=42 (96% vs 94%) era ruído de amostragem, não um efeito real do critério. O que se sustenta direcionalmente é a **estrutura**: a árvore de Entropia tende a crescer mais profunda que a de Gini.
- **Lição de alta variância:** ao contrário dos modelos lineares (P3/P4), aqui **aumentar os dados de teste não encolheu o desvio entre seeds** (foi de ±0.021 para ±0.033) — porque a árvore não-podada é um modelo de **alta variância** por construção (cresce até folhas puras), então a dispersão entre seeds vem da instabilidade do próprio modelo, e não do erro amostral da estimativa de teste. Mais dados não firma o número; só **podar (`max_depth`)** reduziria essa variância.
- **Estrutura difere por geometria, não por parada:** ambas crescem até folhas puras; a diferença de profundidade vem da escala de redução de impureza da entropia reordenar os splits.
- **Overfitting confirmado:** train accuracy = 100% para os dois, test 94–96% (seed=42). As árvores têm folhas com 1 só amostra (10 no Gini, 8 na Entropy), evidência de memorização de exemplos isolados.
- **Limitações:** sem `max_depth`, ambas overfitam (dados sintéticos permitem splits até folhas puras). O `depth_vs_accuracy.png` mostra que profundidades menores já atingem o platô de acurácia de teste — os níveis extras são supérfluos. Validação cruzada k-fold é recomendada como trabalho futuro: além da margem de ±~4 p.p. de uma estimativa com 100 amostras de teste, o resultado multi-seed mostra que a própria variância estrutural do modelo (não só o erro amostral) precisa ser quantificada — k-fold ajudaria a capturar essa instabilidade de forma mais sistemática que seeds isoladas.
- **Nota de implementação:** árvore construída manualmente; a varredura de profundidade re-treina a árvore manual para cada `max_depth`, confirmando o comportamento de generalização.
