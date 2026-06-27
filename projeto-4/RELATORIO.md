# Mini-Projeto 4 — Regressao Logistica: Modelo, Gradiente e Fronteira de Decisao

## Problema

O objetivo deste mini-projeto e implementar a Regressao Logistica **do zero** — sem bibliotecas de ML — em Rust. Isso inclui:

- Derivar e implementar a funcao sigmoid
- Calcular o custo via Binary Cross-Entropy (BCE)
- Atualizar os pesos via gradiente descendente analitico
- Visualizar a fronteira de decisao aprendida e a curva de loss

## Dataset

**Dados sinteticos 2D gerados via distribuicoes gaussianas** (seed=42, deterministico).

| Atributo         | Valor                         |
|------------------|-------------------------------|
| Total de amostras| 300                           |
| Dimensionalidade | 2 features (x1, x2)           |
| Classes          | Binaria (Classe 0 / Classe 1) |
| Classe 0 (neg)   | 150 amostras, centro (-1.5, -1.0) |
| Classe 1 (pos)   | 150 amostras, centro (+1.5, +1.0) |
| Ruido            | Normal(0, 1) em cada feature  |
| Balanceamento    | 50% / 50%                     |

Cada classe e amostrada de uma gaussiana isotrópica 2D com desvio padrao 1.0. As classes sao linearmente separaveis com alguma sobreposicao (distancia entre centros de sqrt(3.0^2 + 2.0^2) = sqrt(13) ~3.61 unidades no espaco de features, gerando sobreposicao pequena mas nao nula). A taxa de erro de Bayes estimada para este problema e de aproximadamente 3.5-4%, consistente com os 11/240 erros de treino observados (4.58% de erro, proximo ao limite teorico dado a sobreposicao gaussiana com std=1.0 e separacao de ~3.61 unidades). O valor exato e Phi(-sqrt(13)/2) = Phi(-1.803) ≈ 3.57%. O erro de teste observado (2/60 = 3.33%) e ligeiramente abaixo da estimativa do erro de Bayes (3.57%), o que e consistente com a variancia amostral elevada para n=60 (IC 95% Wilson: [0.9%, 11.4%]); nao indica violacao do limite teorico.

## Metodologia

### 1. Modelo

O modelo logístico e um classificador linear com saída em probabilidade:

```
z = w0*x1 + w1*x2 + b         (combinacao linear)
p = sigmoid(z) = 1 / (1 + e^(-z))    (probabilidade da classe 1)
yhat = 1  se p >= 0.5 else 0   (predicao binaria)
```

A implementacao usa a forma de dois ramos para estabilidade numerica: para z >= 0 usa 1/(1+e^{-z}); para z < 0 usa e^z/(1+e^z), evitando overflow de e^{-z} para valores muito negativos de z.

### 2. Funcao de Custo — Binary Cross-Entropy

```
L(w, b) = -(1/n) * sum[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

Onde `p_i = sigmoid(w^T x_i + b)`. Para `y=1` penaliza `p` pequeno; para `y=0` penaliza `p` grande.

### 3. Gradientes (resultado analitico)

```
dL/dw_j = (1/n) * sum[ (p_i - y_i) * x_ij ]
dL/db   = (1/n) * sum[ p_i - y_i ]
```

Esses gradientes decorrem da composicao chain-rule entre BCE e sigmoid — a sigmoid cancela de forma elegante, resultando em `(p - y)`.
A derivacao passo a passo (5 etapas, com cancelamento algebrico explicito) esta em **GUIA_ESTUDO.md**, secao *Gradientes / Derivacao completa*.

### 4. Atualizacao dos Pesos (Gradient Descent)

```
w_j <- w_j - lr * dL/dw_j
b   <- b   - lr * dL/db
```

Inicializacao: `w0 = w1 = b = 0`. Taxa de aprendizado: `lr = 0.1`. Epocas: `1000`.

### 5. Fronteira de Decisao

A fronteira e o conjunto de pontos onde `p = 0.5`, i.e., `z = 0`:

```
w0*x1 + w1*x2 + b = 0
=> x2 = -(w0/w1)*x1 - b/w1
```

### 6. Split Treino/Teste

O dataset de 300 amostras e dividido 80/20 apos o shuffle deterministico (seed=42):

- **Treino**: 240 amostras (Classe 0: 116, Classe 1: 124)
- **Teste**: 60 amostras (Classe 0: 34, Classe 1: 26)

O modelo e treinado exclusivamente nas 240 amostras de treino; a avaliacao no conjunto de teste estima a generalizacao.

## Resultados

### Parametros Aprendidos

| Parametro | Valor     |
|-----------|-----------|
| w0        | 2.460226  |
| w1        | 1.613707  |
| bias (b)  | 0.121817  |

### Desempenho

| Metrica                  | Treino    | Teste     |
|--------------------------|-----------|-----------|
| Loss Inicial (BCE)       | 0.693147  | —         |
| Loss Final (BCE)         | 0.106093  | 0.064900  |
| Reducao de Loss          | 84.69%    | —         |
| Acuracia                 | 95.42%    | 96.67%    |
| Precisao                 | 0.9520    | 0.9615    |
| Recall                   | 0.9597    | 0.9615    |
| F1-Score                 | 0.9558    | 0.9615    |

O gap treino/teste e pequeno (95.42% vs 96.67%), indicando boa generalizacao sem overfitting visivel.
A loss de teste ser substancialmente menor (0.065 vs 0.106) e consistente com a variancia esperada
para um subconjunto aleatorio de 60 amostras: com n=60, o erro padrao da BCE e elevado e o gap observado
esta dentro da variabilidade amostral esperada. Nao ha evidencia de data leakage — o modelo foi treinado
exclusivamente nas 240 amostras de treino. A diferenca nao indica problema sistematico.

### Matriz de Confusao

**Treino (240 amostras):**

|              | Pred 0 | Pred 1 |
|--------------|--------|--------|
| **Real 0**   | TN=110 | FP=6   |
| **Real 1**   | FN=5   | TP=119 |

**Teste (60 amostras):**

|              | Pred 0 | Pred 1 |
|--------------|--------|--------|
| **Real 0**   | TN=33  | FP=1   |
| **Real 1**   | FN=1   | TP=25  |

### Fronteira de Decisao

```
x2 = -1.524580 * x1 - 0.075489
```

A inclinacao negativa reflete que classe 1 ocupa a regiao superior-direita e classe 0 a inferior-esquerda — a fronteira separa as duas gaussianas diagonalmente.

### Arquivos de Saida

| Arquivo                                  | Conteudo                                         |
|------------------------------------------|--------------------------------------------------|
| `output/decision_boundary.png`           | Heatmap de probabilidade + linha de fronteira + scatter (treino: solido, teste: oco) |
| `output/loss_curve.png`                  | Curva BCE vs. epoca, mostrando convergencia suave de 0.6931 (baseline ln 2) ate ~0.1061 na epoca 1000 (loss de treino; valores pre-atualizacao de cada epoca) |
| `output/accuracy_curve.png`              | Curva de acuracia vs. epoca (pre-atualizacao), evidenciando o salto de 51.67% para 95.42% entre as epocas 1 e 2 |
| `output/confusion_matrix_train.png`      | Matriz de confusao visual (conjunto de treino)   |
| `output/confusion_matrix_test.png`       | Matriz de confusao visual (conjunto de teste)    |
| `output/training_log.csv`               | Log por epoca: epoch, loss, accuracy             |
| `output/report.txt`                      | Metricas numericas completas                     |

## Conclusao

- A regressao logistica implementada do zero atingiu 95.42% de acuracia de treino e **96.67% de acuracia de teste** em 1000 epocas, confirmando a implementacao correta do gradiente e boa generalizacao.
- A Loss caiu de 0.693 (baseline aleatoria ln(2)) para 0.106 no treino — reducao de 84.69%, indicando boa separacao aprendida.
- O modelo e linear: a fronteira e uma reta. Para dados nao-linearmente separaveis seria necessario features polinomiais ou modelos nao-lineares.
- A acuracia de treino de 95.42% (229/240) foi atingida apos apenas 1 passo de gradiente descendente (registrado como epoca 2 no log, pois o log exibe a acuracia pre-atualizacao de cada epoca); a loss continuou decrescendo suavemente ate a epoca 1000 sem um ponto de convergencia discreto. Isso ocorre porque a loss mede confianca probabilistica continua enquanto a acuracia e uma metrica binaria com limiar em 0.5.
- Limitacoes identificadas: (1) batch gradient descent completo — inviavel para datasets grandes; (2) ausencia de regularizacao L2 (para dados linearmente separaveis sem sobreposicao, os pesos divergem sem regularizacao; neste dataset com sobreposicao, os pesos convergem a valores finitos — 2.46, 1.61 — mas regularizacao L2 ainda melhoraria a margem de generalizacao); (3) sem criterio formal de convergencia (treinamento para em `epochs` fixas independente da variacao da loss); (4) a curva `loss_curve.png` exibe apenas a loss de treino por epoca — a loss de teste (0.0649) e avaliada somente ao final do treinamento, nao por epoca, portanto nao aparece como overlay no grafico. Neste experimento especifico, o plateau de acuracia de 95.42% a partir da epoca 2 corresponde ao limite de Bayes do problema (~3.57% de erro), tornando um criterio de parada irrelevante na pratica; para outros datasets, a ausencia de criterio de convergencia pode resultar em treinamento desnecessariamente longo ou em overfitting em epocas tardias.
- Direcoes de melhoria: adicionar regularizacao L2 (`loss += lambda * (w0^2 + w1^2)`) reduziria a magnitude dos pesos aprendidos e melhoraria generalizacao; incluir features polinomiais (`x1^2, x1*x2, x2^2`) permitiria fronteiras curvas para dados nao-linearmente separaveis.
- O erro de teste observado (2/60 = 3.33%) e consistente com o limite teorico de Bayes estimado para este problema (~3.57%, dado por Phi(-sqrt(13)/2) = Phi(-1.803)), confirmando que o modelo se aproxima do melhor classificador linear possivel dado o ruido intrinseco de sobreposicao gaussiana.