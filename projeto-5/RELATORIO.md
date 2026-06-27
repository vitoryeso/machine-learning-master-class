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

### Métricas de Classificação

| Métrica          | Gini   | Entropy |
|------------------|--------|---------|
| Train Accuracy   | 1.0000 | 1.0000  |
| Test Accuracy    | 0.9600 | 0.9400  |
| Precision        | 0.9423 | 0.9231  |
| Recall           | 0.9800 | 0.9600  |
| F1-Score         | 0.9608 | 0.9412  |

Diferente do que se vê em muitos datasets (onde Gini e Entropia tendem a empatar), aqui o **Gini levou vantagem em todas as métricas** — 2 pontos percentuais de acurácia de teste. As predições dos dois critérios divergem em **4 das 100** amostras de teste, confirmando que a escolha do critério produziu modelos distintos.

### Matriz de Confusão (teste, n=100)

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

### Estrutura da Árvore

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

- **O critério teve efeito pequeno, mas real neste dataset:** o Gini ficou 2 p.p. acima em acurácia (96% vs 94%) e gerou uma árvore mais rasa (9 vs 13). Não é o empate "clássico" — a geometria específica deste dataset favoreceu o Gini.
- **Estrutura difere por geometria, não por parada:** ambas crescem até folhas puras; a diferença de profundidade vem da escala de redução de impureza da entropia reordenar os splits.
- **Overfitting confirmado:** train accuracy = 100% para os dois, test 94–96%. As árvores têm folhas com 1 só amostra (10 no Gini, 8 na Entropy), evidência de memorização de exemplos isolados.
- **Limitações:** sem `max_depth`, ambas overfitam (dados sintéticos permitem splits até folhas puras). O `depth_vs_accuracy.png` mostra que profundidades menores já atingem o platô de acurácia de teste — os níveis extras são supérfluos. Validação cruzada k-fold é recomendada como trabalho futuro, já que com 100 amostras de teste a estimativa de acurácia tem margem de ±~4 p.p.
- **Nota de implementação:** árvore construída manualmente; a varredura de profundidade re-treina a árvore manual para cada `max_depth`, confirmando o comportamento de generalização.
