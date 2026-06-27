# Projeto 5 — Arvore de Decisao: Gini vs Entropia

## Problema

Este mini-projeto implementa e compara dois criterios de divisao de arvores de decisao (Decision Trees): **Gini** e **Entropia (Information Gain)**. O objetivo e entender as diferencas praticas em termos de estrutura da arvore (profundidade, numero de nos) e desempenho preditivo em um dataset sintetico 2D, alem de visualizar as fronteiras de decisao resultantes.

## Dataset

Dataset sintetico gerado com `sklearn.datasets.make_classification`:

| Parametro            | Valor       |
|----------------------|-------------|
| n_samples            | 500         |
| n_features           | 2           |
| n_informative        | 2           |
| n_redundant          | 0           |
| n_clusters_per_class | 1           |
| class_sep            | 0.9         |
| random_state         | 42          |
| Train / Test split   | 400 / 100   |

Dataset 2D puro (sem features redundantes) para que as fronteiras de decisao sejam diretamente visualizaveis. A separacao de classes de 0.9 cria um problema de dificuldade moderada — nao trivialmente linear, mas com regioes bem definidas.

## Metodologia

### Criterio de Impureza — Gini

```
G(t) = 1 - sum(p_k^2)
```

onde `p_k` e a proporcao da classe `k` no no `t`. Gini varia de 0 (no puro) a `1 - 1/K` (distribuicao uniforme). Para binario: maximo em 0.5. O CART escolhe o split que maximiza a reducao de G(pai) - media ponderada de G dos filhos.

### Criterio de Impureza — Entropia (Information Gain)

```
H(t) = -sum(p_k * log2(p_k))
```

Por convencao, `0 * log2(0) = 0` (limite quando `p_k -> 0`), garantindo `H = 0` para nos puros.

O ganho de informacao de um split e:

```
IG = H(pai) - (|L|/|P|)*H(L) - (|R|/|P|)*H(R)
```

onde `L`, `R` sao os filhos esquerdo e direito, `P` o no pai.

### Pipeline

1. Gerar dataset 2D com `make_classification`
2. Dividir em train (80%) / test (20%) com `stratify=y`
3. Treinar `DecisionTreeClassifier(criterion='gini')` e `criterion='entropy'` com `random_state=42`
4. Calcular metricas: accuracy, precision, recall, F1, profundidade, nos, folhas
5. Plotar fronteiras de decisao com `DecisionBoundaryDisplay`
6. Comparar importancia das features

## Resultados

### Metricas de Classificacao

| Metrica          | Gini   | Entropy |
|------------------|--------|---------|
| Train Accuracy   | 1.0000 | 1.0000  |
| Test Accuracy    | 0.9500 | 0.9500  |
| Precision        | 0.9245 | 0.9245  |
| Recall           | 0.9800 | 0.9800  |
| F1-Score         | 0.9515 | 0.9515  |

### Matriz de Confusao

Ambos os criterios produzem a mesma matriz de confusao:

|                  | Pred. Negativo | Pred. Positivo |
|------------------|---------------|----------------|
| **Real Negativo**| TN = 46       | FP = 4         |
| **Real Positivo**| FN = 1        | TP = 49        |

O modelo tende a classificar amostras negativas como positivas (4 FP vs 1 FN), o que explica a precisao inferior ao recall: precision = 49/(49+4) = 92.45%, recall = 49/(49+1) = 98.0%. Este desequilibrio FP/FN nao e causado por desbalanceamento de classes — o test set e perfeitamente balanceado (50 amostras de cada classe). A origem e geometrica: com class_sep=0.9, algumas amostras da classe 0 estao na regiao de sobreposicao proxima a fronteira da classe 1, tornando-as mais suscetiveis a classificacao erronea como positivos do que o inverso.

### Estrutura da Arvore

| Propriedade   | Gini | Entropy |
|---------------|------|---------|
| Profundidade  | 9    | 14      |
| N. de folhas  | 26   | 28      |
| N. de nos     | 51   | 55      |

### Importancia das Features

| Feature   | Gini   | Entropy |
|-----------|--------|---------|
| Feature 0 | 0.3443 | 0.3842  |
| Feature 1 | 0.6557 | 0.6158  |

### Plots gerados

- `output/boundary_comparison.png` — fronteiras de decisao 2D lado a lado (Gini vs Entropy), com pontos de teste sobrepostos
- `output/importance.png` — grafico de barras comparando importancia das features para cada criterio
- `output/depth_vs_accuracy.png` — varredura de max_depth (1-17) vs acuracia no test set para ambos os criterios, com linhas verticais pontilhadas nas profundidades irrestritas (Gini=9, Entropy=14)
- `output/confusion_matrix.png` — heatmaps lado a lado das matrizes de confusao (Gini e Entropy) no test set, com contagens anotadas; confirma visualmente que os dois criterios produzem exatamente a mesma distribuicao de erros
- `output/report.txt` — relatorio numerico completo com tabela de metricas, matrizes de confusao, analise de splits redundantes e estrutura completa das arvores em texto

## Conclusao

**Desempenho identico (metricas agregadas):** Ambos os criterios atingiram exatamente a mesma accuracy (95%), precision (92.45%), recall (98%) e F1 (95.15%) no test set. Apesar das metricas agregadas identicas, as predicoes nao sao identicas: os criterios discordam em 6 exemplos (6% do test set). Em 3 dessas amostras (indices 34, 81, 82), o Gini erra e Entropy acerta; nas outras 3 (indices 51, 76, 92), o inverso. Ambos compartilham 2 erros comuns (indices 49, 59) nas 94 amostras em que concordam. A tabela abaixo resume todos os erros individuais (8 amostras implicadas no total — inclui os 6 exemplos de discordancia entre os criterios e os 2 erros compartilhados, indices 49 e 59, nos quais ambos os criterios erram da mesma forma):

| Indice | Classe Real | Gini Pred | Entropy Pred | Gini Errou? | Entropy Errou? |
|--------|-------------|-----------|--------------|-------------|----------------|
| 34     | 0           | 1         | 0            | Sim         | Nao            |
| 49     | 0           | 1         | 1            | Sim         | Sim            |
| 51     | 0           | 0         | 1            | Nao         | Sim            |
| 59     | 0           | 1         | 1            | Sim         | Sim            |
| 76     | 1           | 1         | 0            | Nao         | Sim            |
| 81     | 1           | 0         | 1            | Sim         | Nao            |
| 82     | 0           | 1         | 0            | Sim         | Nao            |
| 92     | 0           | 0         | 1            | Nao         | Sim            |

Isso e comum: Gini e Entropia tendem a produzir desempenho preditivo similar na maioria dos datasets. O ponto central e que ambas as arvores atingiram memorization completa do treino (train_acc=1.0), portanto a acuracia no teste e determinada exclusivamente pela geometria das fronteiras nos pontos de teste — e neste caso as duas geometrias distintas cometem exatamente 5 erros cada, ainda que em 3 exemplos diferentes (os 6 de discordancia compensam-se: 3 Gini-exclusivos vs 3 Entropy-exclusivos).

**A acuracia identica NAO e suspeita** — e uma coincidencia numerica: ambas erraram 5 amostras mas em locais distintos, produzindo o mesmo TN=46/FP=4/FN=1/TP=49 por acaso (mesmo numero de FP e FN em cada criterio). Arvores com estruturas diferentes podem atingir metricas agregadas identicas quando os erros individuais se compensam desta forma.

**Estrutura diferente:** A arvore Entropy e 56% mais profunda (14 vs 9) e tem 8% mais nos (55 vs 51). A explicacao e geometrica, nao de criterio de parada: ambas as arvores crescem com as mesmas condicoes de parada (min_impurity_decrease=0.0, padrao do sklearn), e ambas atingem treino perfeito (train_accuracy=1.0) -- ou seja, ambas crescem ate folhas puras. A diferenca nao e que a Entropia "aceita mais splits antes de parar": ela aceita exatamente os mesmos tipos de splits (qualquer reducao positiva de impureza). O que difere e a escala numerica da reducao: a Entropia tem maior curvatura em torno de p=0.5, produzindo reducoes absolutas maiores para os mesmos splits candidatos. Isso altera a ordenacao gulosa de splits em cada no -- as escolhas divergem entre os dois criterios -- e os caminhos alternativos de particao resultantes requerem mais niveis para segmentar os mesmos pontos de treino em folhas puras. Exemplo numerico (split simetrico: pai a p=0.5, dois filhos a p=0.4 e p=0.6): Gini reduction = Gini(0.5) - avg(Gini(0.6), Gini(0.4)) = 0.50 - 0.48 = 0.02; Entropy reduction = Entropy(0.5) - avg(Entropy(0.6), Entropy(0.4)) = 1.0000 - 0.9710 ≈ 0.0290. Esta diferenca de escala altera quais splits sao escolhidos em cada no, gerando geometrias de particao distintas, onde alguns caminhos exigem mais divisoes sob Entropia do que sob Gini para atingir pureza. Nota: esta ordering e dataset-dependente -- para make_classification 2D com n_clusters_per_class=1, Entropy tende a produzir arvores com profundidade >= Gini (verificado em 10 seeds: Entropy mais profunda em 6, igual em 4, nunca mais rasa). O inverso pode ocorrer em datasets com outras geometrias (ex: multiplos clusters por classe, alta dimensionalidade), mas a mudanca de seed sozinha nao reverte o resultado neste tipo de dataset.

> **Nota de reproducibilidade:** A verificacao de 10 seeds (0-9) foi realizada manualmente fora do script `main.py`. O script principal roda apenas com `SEED=42`. Para reproduzir os resultados das 10 seeds, execute um loop externo alternando `random_state` em `make_classification` e `train_test_split` de 0 a 9. Os resultados obtidos foram: Entropy mais profunda em 6 seeds, igual em 4, nunca mais rasa.

O fato de que a arvore Entropy tem 56% mais profundidade sem penalidade de test accuracy sugere que os splits adicionais sao benigno-superfluos neste dataset (nao confundir com splits redundantes tecnicamente, que compartilham feature e threshold identicos). O plot `output/depth_vs_accuracy.png` (gerado por este codigo) apresenta essa varredura de max_depth=1..17 (range dinamico: max_natural_depth + 3) para ambos os criterios; a tabela numerica completa por profundidade esta em `output/report.txt` (secao "Depth-vs-accuracy sweep"). O sweep mostra que a acuracia de teste Gini atinge 95% pela primeira vez em depth=4, sobe para 96% em depth=5 (pico isolado), e volta a 95% a partir de depth=6 onde se mantem; a acuracia Entropy estabiliza definitivamente em 95% a partir de profundidade 8. Isso confirma que os niveis irrestritos (9 e 14) sao superfluous alem de profundidades 5-8: os splits adicionais da arvore Entropy nao adicionam valor preditivo neste dataset.

**Feature 1 mais importante:** Em ambos os criterios, a Feature 1 concentra mais poder preditivo (~65% Gini, ~62% Entropy), sugerindo que ela separa as classes com maior eficiencia.

**Overfitting confirmado:** Train accuracy = 100% para ambos os criterios, test accuracy = 95%. A diferenca de 5 p.p. entre treino e teste e evidencia de overfitting (confirmado por train_accuracy=1.0 para ambos). Ressalva: esta analise e baseada em um unico split 80/20; sem validacao cruzada k-fold, nao e possivel distinguir entre overfitting real e variancia da estimativa de acuracia no conjunto de teste de 100 amostras. Com 26 folhas e 400 amostras de treino, a media e ~15.4 amostras/folha, mas o minimo e 1 amostra por folha (10 folhas com exatamente 1 amostra — calculado via tree_.n_node_samples e impresso em  secao Leaf sample counts) — memoriza sim exemplos isolados, e os splits extras capturam ruido local sem agregar generalizacao. Para Entropy, com 28 folhas: 400/28 ≈ 14.3 amostras/folha, com minimo igualmente de 1 amostra por folha (10 folhas com 1 amostra, idem verificado e impresso em report.txt) — igualmente fragmentado e com memorization de exemplos isolados, consistente com a maior profundidade.

**Splits aparentemente redundantes na arvore Gini:** O texto da arvore Gini (gerado por `export_text`) exibe dois nos consecutivos com `f0 <= 0.13`, que a primeira vista parecem identicos. Inspecao dos thresholds internos revela que sao distintos a partir da terceira casa decimal (0.13318 vs 0.12690) — diferem em ~0.0063, NAO visivel a olho nu no texto exportado (`export_text` arredonda para 2 casas decimais, exibindo ambos como `f0 <= 0.13`); o artefato ocorre por razoes opostas: 0.13318 arredonda para baixo no terceiro decimal (3<5), resultando em 0.13; e 0.12690 arredonda para cima no terceiro decimal (6>=5), resultando igualmente em 0.13. Assim, dois thresholds distintos (0.12690 e 0.13318) exibem o mesmo valor 0.13 no export_text por caminhos opostos de arredondamento; a diferenca so e visivel inspecionando `tree_.threshold` diretamente. Sao splits reais, nao degeneram. Este e um artefato de exibicao do arredondamento de `export_text`, nao um split redundante genuino.

**Limitacoes:** Sem `max_depth`, ambas as arvores overfitam o treino (os dados sinteticos permitem splits ate folhas puras). Em producao, hiperparametros como `max_depth`, `min_samples_split` e `ccp_alpha` (pruning) seriam essenciais para controlar a profundidade. O plot `depth_vs_accuracy.png` mostra que max_depth=4 ja atinge 95% de acuracia para Gini com estrutura mais simples (depth=4 vs 9), e max_depth=8 para Entropy (depth=8 vs 14), reduzindo o gap de complexidade sem penalidade de acuracia neste dataset. Adicionalmente, com apenas 100 amostras no conjunto de teste e um unico split 80/20, a estimativa de acuracia de 95% tem margem de erro de aproximadamente +/-4 p.p. a 95% de confianca (intervalo aproximado: [90.7%, 99.3%], pelo metodo normal: SE = sqrt(0.95*0.05/100) = 0.0218). Sem validacao cruzada k-fold, nao e possivel distinguir com confianca estatistica entre overfitting real e variancia da estimativa de acuracia. Validacao cruzada (ex: k=5 ou k=10) e recomendada como trabalho futuro.
