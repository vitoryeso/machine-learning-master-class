# Projeto 6 — Random Forest: Ensemble e Variabilidade

## Problema

Este mini-projeto implementa um **Random Forest** para classificação binária 2D, construído **manualmente** (sem `sklearn.ensemble`): bootstrap (bagging), subamostragem aleatória de features por split e voto de maioria — reaproveitando a árvore CART do Projeto 5 como estimador-base. O `scikit-learn` é usado apenas para gerar o dataset sintético (`make_classification`) e o split treino/teste; o ensemble é todo manual (ver `random_forest.py`).

**A tese que fecha o arco dos dois projetos:** o Projeto 5 diagnosticou, via teste pareado multi-seed, que a árvore de decisão não-podada é um estimador de **alta variância** — o desvio entre seeds não caía com mais dados, porque a instabilidade era do *modelo*, não do erro amostral da estimativa. O Random Forest é a **cura clássica** para esse problema: o bagging faz a média de muitas árvores diversas (bootstrap + features aleatórias) e **reduz a variância sem inflar o viés**. Este projeto mede exatamente isso, com o mesmo ferramental multi-seed/pareado do Projeto 5: árvore única vs floresta, na mesma realização de dados por seed, comparando não só a acurácia média mas sobretudo o **desvio entre seeds**.

## Dataset

Mesmo gerador do Projeto 5, `sklearn.datasets.make_classification`:

| Parâmetro            | Ilustração 2D | Multi-seed (oficial) |
|-----------------------|---------------|-----------------------|
| n_samples             | 500           | 2000                  |
| n_features            | 2             | 2                     |
| n_informative         | 2             | 2                     |
| n_redundant           | 0             | 0                     |
| n_clusters_per_class  | 1             | 1                     |
| class_sep             | 0.9           | 0.9                   |
| Train / Test split    | 80% / 20%     | 80% / 20% (teste ~400)|
| Seeds                 | 42            | 42..51 (10 seeds)     |

Dataset 2D para que as partições de cada árvore e a fronteira combinada sejam diretamente visualizáveis. O dataset maior (N=2000) no multi-seed dá estimativas de teste mais precisas por seed (~400 amostras de teste vs ~100 no P5-ilustração).

## Metodologia

### Árvore-base (CART reaproveitada do Projeto 5, com uma adição)

A árvore é a mesma implementação manual do Projeto 5 (impureza Gini, busca gulosa de split, recursão até folha pura), com um parâmetro novo: `max_features`. Quando `None`, a árvore considera todas as features em cada split (comportamento do P5). Quando definido (`max_features=1` de 2 features), a cada nó sorteia-se um subconjunto aleatório de features candidatas e o split ótimo é buscado só dentro dele — é o **"random" do Random Forest**, que força diversidade entre as árvores mesmo quando o bootstrap sorteia amostras parecidas.

### Random Forest manual — bagging + voto

```
Para cada árvore da floresta (N_TREES = 51):
  1. Bootstrap: sorteia n amostras COM reposição do treino (bagging)
  2. Treina uma árvore CART nesse bootstrap, com max_features=1 (de 2)
Predição: cada árvore vota (0 ou 1); decide por MAIORIA (média dos votos >= 0.5)
```

Duas fontes de aleatoriedade tornam as árvores diversas: (1) o **bootstrap** — cada árvore vê uma reamostragem diferente do treino, com repetições e omissões; (2) a **subamostragem de features** — cada split considera um subconjunto aleatório de candidatas. Árvores individuais treinadas assim são instáveis e enviesadas localmente, mas a **combinação por voto** cancela boa parte desse ruído porque os erros de cada árvore não são perfeitamente correlacionados.

Número de árvores: `N_TREES = 51` (ímpar, evita empate de voto no caso binário).

### Pipeline

1. Gerar dataset 2D (`make_classification`), split 80/20 estratificado
2. Treinar árvore única (todas as features, sem poda) — o baseline do P5
3. Treinar Random Forest (51 árvores, bootstrap, `max_features=1`)
4. Multi-seed **pareado**: 10 seeds, mesma realização de dados para árvore e floresta em cada seed, calculando accuracy, F1 e a **diferença pareada** (floresta − árvore) por seed
5. Visualizar: partições individuais de árvores da floresta vs partição combinada (voto) vs árvore única
6. Varredura de número de árvores (1 a 101) vs desvio da acurácia entre seeds — para mostrar a redução de variância em função do tamanho do ensemble

## Resultados

### Multi-seed pareado (10 seeds, N=2000, 51 árvores) — resultado oficial

| Métrica    | Árvore única      | Random Forest     |
|------------|-------------------|--------------------|
| Accuracy   | 0,943 ± 0,034     | **0,960 ± 0,024**  |

A floresta vence em **duas frentes ao mesmo tempo**: a **média** sobe (0,943 → 0,960) e o **desvio entre seeds** cai (0,034 → 0,024) — uma redução de variância de **~1,4×**. Isso é exatamente o comportamento esperado do bagging: reduzir a variância do estimador sem pagar em viés (a média não piora, melhora).

**Teste pareado (floresta − árvore, mesma realização por seed):**

```
diferença de acurácia = +0,017   |   t = 4,00
```

Um `|t|` de 4,00 é um sinal forte — muito acima do limiar informal de ~2 para "efeito real e não ruído de uma seed isolada". A floresta é **consistentemente melhor** que a árvore única em praticamente todas as seeds testadas, não só na média.

### Variância vs número de árvores

Variando o tamanho do ensemble (1, 3, 5, 11, 21, 51, 101 árvores) e medindo o desvio da acurácia entre as mesmas 10 seeds: a variabilidade **cai monotonicamente** conforme o número de árvores cresce, consistente com a intuição estatística de que a variância de uma média de estimadores (aproximadamente independentes) escala com `1/n_árvores`. O ganho marginal é maior nas primeiras árvores (1 → 11) e depois satura — dobrar de 51 para 101 árvores já traz pouco ganho adicional de estabilidade, sugerindo que 51 é um ponto de operação razoável custo/benefício.

### Plots gerados

- `output/forest_partitions.png` — 4 árvores individuais da floresta (partições diversas, bagging) na linha de cima; árvore única do P5 vs Random Forest combinado vs mapa de fração de votos (incerteza na fronteira) na linha de baixo
- `output/metrics_multiseed.png` — barras árvore vs floresta (accuracy, F1) com barra de erro = desvio entre seeds; a barra da floresta é visivelmente menor
- `output/variance_vs_ntrees.png` — curva do desvio da acurácia entre seeds em função do número de árvores (escala log), mostrando a queda da variabilidade
- `output/report_multiseed.txt` — relatório numérico completo do multi-seed pareado

## Conclusão

- **A tese se confirma: bagging reduz variância sem custo de viés.** Árvore única 0,943±0,034 → Random Forest 0,960±0,024 — a média **sobe** e o desvio entre seeds **cai ~1,4×**. Não é um trade-off; é uma dupla melhora, exatamente o que a teoria de ensemble prevê para estimadores de alta variância e baixo viés (como árvores não-podadas).
- **O teste pareado confirma que não é sorte de seed:** diferença de acurácia floresta−árvore = +0,017 com **t=4,00** — um efeito grande e consistente, não ruído de amostragem isolado.
- **A variância cai com mais árvores, como esperado de uma média de estimadores:** a varredura de 1 a 101 árvores mostra queda monotônica do desvio entre seeds, com retornos decrescentes — a maior parte do ganho já aparece com poucas dezenas de árvores.
- **O mecanismo é visível nas partições 2D:** árvores individuais da floresta têm fronteiras recortadas e diferentes entre si (bootstrap + features aleatórias); o voto de maioria produz uma fronteira mais suave e estável que qualquer árvore individual — inclusive mais estável que a árvore única do P5, treinada com todas as features e sem regularização.
- **Custo do ensemble:** treinar e avaliar 51 árvores é ~51× mais caro computacionalmente que uma árvore única; para este dataset pequeno (N=2000, 2 features) o custo é irrelevante, mas escalaria em datasets maiores ou de alta dimensão — nesse caso, paralelizar o treino das árvores (embaraçosamente paralelo, cada árvore é independente) é a otimização natural.
- **Limitações:** `max_features=1` (de 2 features) é o mínimo possível de subamostragem em 2D; em datasets com mais features, esse hiperparâmetro (tipicamente `sqrt(n_features)`) merece uma varredura própria. Também não exploramos poda das árvores-base dentro da floresta — combinar bagging com árvores levemente podadas é uma direção comum na prática (ex.: `max_depth` moderado) e ficaria como trabalho futuro, assim como comparar contra o `RandomForestClassifier` do scikit-learn como sanity-check da implementação manual.
