# Guia de Estudo — Projeto 5: Arvore de Decisao (Gini vs Entropia)

## Pipeline

```
Dataset sintetico 2D (500 amostras, 2 classes)
        |
        v
  Train/Test Split  (80% / 20%, stratify=y, seed=42)
        |
   +----+----+
   |         |
   v         v
DecisionTree  DecisionTree
criterion=    criterion=
'gini'        'entropy'
   |         |
   v         v
  Predict   Predict
  (test)    (test)
   |         |
   +----+----+
        |
        v
  Metricas: accuracy, precision, recall, F1
  Estrutura: depth, n_leaves, n_nodes
  Feature importance
        |
        v
  Plots: fronteiras de decisao | importancia features
        |
        v
  report.txt  |  boundary_comparison.png  |  confusion_matrix.png  |  importance.png  |  depth_vs_accuracy.png
```

## Matematica

### Impureza de Gini

```
G(t) = 1 - sum(p_k^2)
```

Para binario (p = proporcao da classe positiva):

```
G = 1 - p^2 - (1-p)^2 = 2*p*(1-p)
```

- Maximo: G = 0.5 quando p = 0.5 (no mais impuro)
- Minimo: G = 0.0 quando p = 0 ou p = 1 (no puro)

### Entropia de Shannon

```
H(t) = -sum(p_k * log2(p_k))
```

- Maximo: H = 1.0 quando p = 0.5 (binario)
- Minimo: H = 0.0 quando no e puro
- Convencao: `0 * log2(0) = 0` (limite quando `p_k -> 0`), garantindo `H = 0` para nos puros.

### Ganho de Informacao

```
IG = H(pai) - (|L|/|P|)*H(L) - (|R|/|P|)*H(R)
```

### Reducao de Impureza — Gini (mesma logica)

```
delta_G = G(pai) - (|L|/|P|)*G(L) - (|R|/|P|)*G(R)
```

### Busca exaustiva do split (CART)

Para cada feature j e cada threshold theta, o CART calcula o score e escolhe o par (j*, theta*) que maximiza a reducao de impureza. Complexidade media de treinamento: O(n * d * log n), onde n=n_samples e d=n_features; pior caso: O(n^2 * d) quando os splits nao reduzem o subconjunto de forma balanceada.

### Importancia da Feature (sklearn)

```
importance(j) = sum over all nodes t that split on j: (n_t / N) * delta_impureza(t)
```

Normalizado para soma = 1.

## Codigo

### Geracao do dataset e split

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42

X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1,
    class_sep=0.9, random_state=SEED,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
```

### Treinamento e comparacao

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

criteria = ["gini", "entropy"]
trees = {}
metrics = {}

for crit in criteria:
    clf = DecisionTreeClassifier(criterion=crit, random_state=SEED)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    trees[crit] = clf
    metrics[crit] = {
        "train_accuracy": train_acc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "depth": clf.get_depth(),
        "n_leaves": clf.get_n_leaves(),
        "n_nodes": clf.tree_.node_count,
        "feature_importances": clf.feature_importances_.tolist(),
        "confusion_matrix": cm.tolist(),
    }
```

### Fronteira de decisao

```python
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors as mcolors

colors = ["#4e79a7", "#f28e2b"]
cmap = mcolors.ListedColormap(colors)

# main.py uses X (full dataset) to ensure the boundary grid covers all points
# including test; passing X_train instead would risk clipping test points
# outside the grid extent.
DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict",
    alpha=0.3, ax=ax, cmap=cmap,
)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
           cmap=cmap, marker="x", s=40, alpha=0.6, linewidths=1.0)  # train points (x markers)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
           cmap=cmap, edgecolors="k", s=40)  # test points (circle markers)
```

### Texto da arvore para inspecao

```python
from sklearn.tree import export_text
# max_depth=clf.get_depth() ensures the full tree is printed;
# the default max_depth=10 silently truncates trees deeper than 10 levels
# (e.g., the Entropy tree is depth 14 and would be truncated without this).
print(export_text(clf, feature_names=["f0", "f1"], max_depth=clf.get_depth()))
```

## Resultados

Saida real da execucao (Python 3.14.4, sklearn 1.9.0, matplotlib 3.11.0):

```
Dataset: 500 samples, 2 features, 2 classes
Train: 400  Test: 100

--------------------------------------------
Metric                       Gini    Entropy
--------------------------------------------
train_accuracy             1.0000     1.0000
accuracy                   0.9500     0.9500
precision                  0.9245     0.9245
recall                     0.9800     0.9800
f1                         0.9515     0.9515
depth                           9         14
n_leaves                       26         28
n_nodes                        51         55
--------------------------------------------

Feature importance (gini): f0=0.3443, f1=0.6557
Feature importance (entropy): f0=0.3842, f1=0.6158

Confusion matrix (gini): TN=46 FP=4 FN=1 TP=49
Confusion matrix (entropy): TN=46 FP=4 FN=1 TP=49

Predictions identical (gini vs entropy): False
Samples where predictions differ: 6
```

**Observacoes chave:**
- Accuracy identica: 95.0% em ambos os criterios
- Entropia produz arvore ~56% mais profunda (14 vs 9 niveis)
- Entropia gera ~8% mais nos (7.8%; 55 vs 51)
- Feature 1 domina a importancia em ambos (~62-66%)
- Ambos os plots mostram fronteiras em forma de grade (eixos-paralelos)
- `depth_vs_accuracy.png`: sweep max_depth=1-17 (range dinamico) para ambos os criterios; tabela numerica completa em report.txt (secao "Depth-vs-accuracy sweep"); acuracia Gini atinge 95% pela primeira vez em depth=4, sobe para 96% em depth=5 (pico), e volta a 95% a partir de depth=6 onde se mantem; Entropy estabiliza definitivamente em 95% a partir de depth=8. **Atencao:** esse pico pontual de Gini (96% em depth=5) e ruido desta seed unica — o teste pareado oficial (30 seeds) mostra Gini e Entropia empatados na pratica (diferenca de acuracia praticamente zero); o unico efeito real e estrutural: Entropy produz arvores consistentemente mais profundas
- Ambos erram 5 amostras no total, mas apenas 2 em comum (indices 49, 59); os outros 3+3 erros sao exclusivos de cada criterio (Gini-exclusivos: indices 34, 81, 82; Entropy-exclusivos: indices 51, 76, 92)
- `confusion_matrix.png`: matrizes identicas (TN=46 FP=4 FN=1 TP=49) apesar de 6 predicoes diferentes — coincidencia numerica, nao indicador de arvores equivalentes

## Perguntas Provaveis

**P1: Qual a diferenca matematica entre Gini e Entropia como criterios de divisao?**

R: Gini mede impureza como `1 - sum(p_k^2)` (sem logaritmo), enquanto a Entropia usa `-sum(p_k * log2(p_k))`. Para binario, Gini varia de 0 a 0.5 e Entropia de 0 a 1.0. Ambas sao maximizadas quando as classes estao em proporcao igual (50/50) e zeradas em nos puros. A diferenca e que a Entropia penaliza mais fortemente impurezas intermediarias, tornando-a mais sensivel a distribuicoes balanceadas.

**P2: Por que neste experimento ambos os criterios produziram a mesma accuracy?**

R: Gini e Entropia geram fronteiras de decisao muito similares na pratica. A diferenca nao esta na qualidade preditiva mas na estrutura da arvore — quantos splits sao feitos e em que profundidade. No test set de 100 amostras, ambas erraram 5 exemplos cada -- mas NAO os mesmos: np.array_equal(y_pred_gini, y_pred_entropy) retornou False, com 6 exemplos de discordancia (6% do test set). As metricas agregadas coincidem porque o numero total de erros e o mesmo. Isso e empiricamente conhecido na literatura: a escolha do criterio raramente muda mais de 1-2% de accuracy.

**P3: Por que a arvore com Entropia foi mais profunda (14 vs 9 niveis)?**

R: A Entropia tem maior curvatura em torno de p=0.5: uma divisao que vai de 50/50 para 60/40 gera uma reducao absoluta de impureza maior em Entropia do que em Gini. Exemplo numerico (split simetrico: pai a p=0.5, dois filhos iguais a p=0.4 e p=0.6): Gini reduction = Gini(0.5) - avg(Gini(0.6),Gini(0.4)) = 0.50 - 0.48 = 0.02; Entropy reduction = Entropy(0.5) - avg(Entropy(0.6),Entropy(0.4)) = 1.0000 - 0.9710 ≈ 0.0290. Esta diferenca de escala altera a ordenacao gulosa de splits em cada no: as escolhas divergem entre os dois criterios, produzindo geometrias de particao distintas. Ambas as arvores crescem com as mesmas condicoes de parada (min_impurity_decrease=0.0, padrao) e ambas atingem treino perfeito (train_accuracy=1.0 para ambos) -- portanto a diferenca nao e que Entropia "aceita mais splits antes de parar". O que acontece e que os caminhos alternativos de particao gerados pelas diferentes escolhas gulosas requerem mais niveis para segmentar os mesmos pontos de treino em folhas puras sob Entropia. Verificado em 10 seeds (0-9): Entropy mais profunda em 6, igual em 4, nunca mais rasa — resultado consistente com esta explicacao geometrica *(verificacao manual, fora do script main.py)*.

**P4: O que e overfitting em arvores de decisao e como controlar?**

R: Sem restricoes, o CART cresce ate que cada folha contenha um unico exemplo (ou exemplos de uma so classe), memorizando o dataset de treino. Para controlar: (a) `max_depth` limita a profundidade; (b) `min_samples_split` exige minimo de amostras para dividir um no; (c) `min_samples_leaf` exige minimo em cada folha; (d) `ccp_alpha` (cost-complexity pruning) poda a arvore apos o treinamento removendo nos que contribuem pouco. Neste projeto, as arvores cresceram sem restricao (depth=9 e 14), o que causa overfitting confirmado (train_accuracy=1.0 para ambos).

**P5: O que representa a importancia de features em arvores de decisao?**

R: E a soma ponderada da reducao de impureza causada por todos os splits que usam aquela feature, normalizada pelo total. Feature 1 teve importancia 0.6557 (Gini) — significa que os splits em Feature 1 causaram 65.6% da reducao total de impureza no dataset de treino. Nao indica causalidade, apenas poder preditivo dentro do modelo.

**P6: Quando voce escolheria Entropia sobre Gini na pratica?**

R: Entropia e preferida quando a maior sensibilidade em torno de p=0.5 e desejavel -- especificamente: (a) quando se busca uma interpretacao baseada em teoria da informacao (ganho de informacao); (b) quando splits proximos a 50/50 precisam ser penalizados com maior granularidade, o que pode gerar fronteiras de decisao mais finas em regioes de sobreposicao de classes. Neste projeto, a maior curvatura da Entropia em torno de p=0.5 produziu uma arvore significativamente mais profunda (14 vs 9 niveis) -- uma manifestacao concreta de como a escolha do criterio afeta a estrutura da arvore mesmo quando a acuracia de teste e identica. Gini e preferivel quando velocidade e prioridade (sem operacao de logaritmo) ou quando a diferenca de sensibilidade ao redor de p=0.5 nao e relevante para a tarefa. Na pratica, tunar `max_depth` ou `ccp_alpha` tem muito mais impacto na acuracia final do que a escolha entre Gini e Entropia.

**P7: O que e o algoritmo CART e como ele escolhe os splits?**

R: CART (Classification and Regression Trees) e o algoritmo guloso que o sklearn usa internamente. Para cada no: (1) para cada feature j, ordena os valores unicos; (2) testa todos os possiveis thresholds theta (pontos medios entre valores consecutivos); (3) calcula a reducao de impureza para cada par (j, theta); (4) escolhe o par que maximiza a reducao. O processo se repete recursivamente em cada filho ate atingir condicao de parada. Complexidade media de treinamento: O(n * d * log n), onde n=n_samples e d=n_features; pior caso: O(n^2 * d) quando os splits nao reduzem o subconjunto de forma balanceada.

**P8: Como interpretar a fronteira de decisao de uma arvore de decisao?**

R: Arvores de decisao produzem fronteiras ortogonais (paralelas aos eixos das features), pois cada split e univariado — divide apenas uma feature por vez. No plot 2D, isso aparece como uma grade de retangulos, cada um correspondendo a uma folha da arvore. Quanto maior a profundidade, mais fragmentada e a grade. Isso contrasta com modelos como SVM (fronteiras curvas) ou regressao logistica (fronteira linear). A interpretabilidade e a vantagem: cada regiao pode ser explicada como "se Feature 0 <= x E Feature 1 <= y, entao Classe 0".
