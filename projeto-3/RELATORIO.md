# Mini-Projeto 3 — Regressao Linear: GD vs LMS

## Problema

Este mini-projeto compara dois algoritmos de otimizacao para treinar um modelo de regressao linear simples:

- **Batch Gradient Descent (GD)**: calcula o gradiente sobre todos os N exemplos do dataset a cada iteracao e atualiza os pesos uma vez por epoca.
- **LMS / Widrow-Hoff (SGD online)**: atualiza os pesos a cada exemplo individual, percorrendo o dataset sequencialmente em cada epoca.

O objetivo e entender as diferencas praticas entre os dois metodos em termos de velocidade de convergencia, qualidade final dos pesos aprendidos, e sensibilidade ao learning rate.

## Dataset

Dataset sintetico gerado com a funcao linear:

```
y = 3*x1 - 2*x2 + 1 + ruido
```

| Parametro       | Valor                        |
|-----------------|------------------------------|
| N amostras      | 300                          |
| N features      | 2 (x1, x2 ~ Normal(0,1))    |
| Pesos verdadeiros | w = [3.0, -2.0]            |
| Bias verdadeiro | b = 1.0                      |
| Ruido           | Normal(0, sigma=0.5)         |
| Seed            | 42 (reproducivel)            |

O ruido gaussiano com desvio padrao 0.5 implica um MSE minimo teorico de sigma^2 = 0.25.

## Metodologia

### Modelo

Regressao linear: `y_hat = w^T * x + b`

Funcao de custo (MSE):

```
L(w, b) = (1/N) * sum_i (y_hat_i - y_i)^2
```

### Batch Gradient Descent

Gradiente calculado sobre todo o dataset:

```
grad_w = sum_i (y_hat_i - y_i) * x_i   # acumula soma bruta
grad_b = sum_i (y_hat_i - y_i)

w <- w - alpha * (1/N) * grad_w         # divisao por N acontece aqui, nao antes
b <- b - alpha * (1/N) * grad_b
```

*Nota: o gradiente exato de L = (1/N)*sum(...) inclui um fator 2 (derivada da funcao quadratica). O fator 2 e omitido na implementacao e absorvido pelo learning rate alpha (convencao comum -- equivale a usar alpha_efetivo = 2*alpha da derivada completa). Ver tambem GUIA_ESTUDO.md.*

Parametros: learning rate alpha = 0.01, 200 epocas.

> **Nota de design:** o lr=0.01 foi escolhido intencionalmente para demonstrar a sensibilidade do GD a escolha de hiperparametros. Com essa configuracao, o GD *nao* converge completamente em 200 epocas -- o que e esperado e desejado: trata-se de um experimento controlado para evidenciar o trade-off entre lr e velocidade de convergencia, nao uma comparacao desequilibrada entre algoritmos. O trade-off geral e: lr maior leva a convergencia mais rapida mas aumenta o risco de instabilidade; lr menor e mais seguro mas requer mais epocas.

### LMS / Widrow-Hoff

Regra de atualizacao por exemplo individual (online):

```
erro_i = y_i - y_hat_i      (sinal positivo: correcao)

w <- w + eta * erro_i * x_i
b <- b + eta * erro_i
```

Parametros: learning rate eta = 0.001 (menor que o GD para estabilidade online), 200 epocas.

A diferenca de sinal (+ vs -) reflete a convencao: LMS minimiza o erro quadratico pelo lado do exemplo atual em vez de pelo gradiente global.

### Implementacao

Rust puro, sem bibliotecas de ML. Geracao de dados com `rand_distr::Normal`, plots com `plotters` (backend bitmap, fonte DejaVu via ab_glyph).

### Visualizacoes

Tres graficos sao gerados para avaliar os resultados sob angulos complementares:

1. **convergence.png** -- curvas de MSE por epoca para GD e LMS, permitindo comparar velocidade de convergencia ao longo das epocas.
2. **scatter.png** -- projecao 2D (x1 vs y) com as retas de regressao marginal E[y|x1] aprendidas por cada algoritmo e a reta verdadeira. A dispersao vertical ampla e estrutural (dominada por Var[-2*x2]=4), nao indica erro do modelo.
3. **pred_vs_actual.png** -- diagrama de dispersao no plano (y_real, y_predito) usando *todas* as 2 features simultaneamente. Pontos proximos da diagonal ideal (y=x) indicam bom ajuste. Este grafico complementa o scatter.png ao capturar o erro global do modelo sem omitir nenhuma feature.

## Resultados

| Metrica           | GD         | LMS        | Ref/Min.Teorico |
|-------------------|------------|------------|-----------------|
| MSE final         | 0.479627 (**) | 0.235422   | 0.250000 (*)    |
| w[0] (true=3.0)   | 2.636558   | 2.966010   | 3.000000   |
| w[1] (true=-2.0)  | -1.668652  | -2.007297  | -2.000000  |
| bias (true=1.0)   | 0.825018   | 0.985971   | 1.000000   |
| MSE inicial       | 14.303548  | 14.303548  | —          |
| Reducao MSE (%)   | 96.65%     | 98.35%     | —          |

(**) GD nao convergiu completamente em 200 epocas com lr=0.01 — ver Nota de design em Metodologia.

**Plots gerados:**
- `output/convergence.png` — curvas de perda MSE por epoca para cada algoritmo
- `output/scatter.png` — dados originais com as retas aprendidas (GD azul, LMS vermelho, verdadeiro verde)
- `output/pred_vs_actual.png` — predito vs real para GD e LMS (diagrama de dispersao no plano y_pred x y_real com diagonal ideal)

(*) sigma^2 = limite inferior teorico do MSE (ruido irredutivel), nao um valor de parametro verdadeiro.

**Nota sobre o scatter plot:** O grafico exibe x1 vs y (projecao 2D de um modelo com 2 features). A dispersao vertical ampla nao e causada principalmente pelo ruido gaussiano (Var[ruido] = sigma^2 = 0.25), mas pela contribuicao da feature omitida x2: Var[-2*x2] = 4 * Var[x2] = 4, dezesseis vezes maior que o ruido. As retas representam a regressao marginal E[y|x1] = w[0]*x1 + b, que e a projecao correta em 2D; a dispersao restante e estrutural, nao erro do modelo.

O MSE reportado e o MSE de treinamento (sem split treino/teste). Para regressao linear com ruido gaussiano e N >> d, o otimismo do MSE de treinamento e da ordem de (d/N)*sigma^2 = (2/300)*0.25 ≈ 0.0017, negligivel. A comparacao com sigma^2 e valida.

O LMS convergiu para pesos muito proximos dos verdadeiros (erro < 0.04 em todos os parametros) e atingiu MSE = 0.235, ligeiramente abaixo do limite teorico de ruido (0.25) — variacao estatistica normal com 300 amostras. O GD com lr=0.01 e 200 epocas ainda nao convergiu completamente: pesos com erro de ate 0.363 em w[0] e MSE = 0.48, quase o dobro do minimo.

## Conclusao

- **LMS (online SGD)** com lr pequeno convergiu melhor neste cenario: as atualizacoes frequentes (N=300 por epoca) permitiram 200 epocas x 300 amostras = 60.000 passos de gradiente vs apenas 200 do GD batch.
- **GD batch** e mais estavel (sem oscilacoes) mas requer mais epocas ou learning rate maior para atingir o mesmo MSE final. Para GD sobre MSE com features N(0,1), o limite de estabilidade lr < 2/lambda_max((1/N)*X^T*X) e satisfeito com larga margem (lr=0.01 << ~2 ~ limite teorico para features N(0,1)); a escolha lr=0.01 e conservadora e garante convergencia monotonica ao custo de mais epocas (ver Nota de design em Metodologia). Nota: o convergence.png registra o MSE no limite de cada epoca; oscilacoes intra-epoca do LMS (N=300 atualizacoes individuais por epoca) nao sao visiveis nessa resolucao temporal.
- O **trade-off central**: GD usa gradiente exato (menor variancia, mas uma atualizacao por epoca); LMS usa gradiente ruidoso (maior variancia, mas N atualizacoes por epoca).
- **Limitacao**: o LMS com lr muito grande diverge. O GD com lr muito grande oscila. A escolha de hiperparametros e critica e independente para cada algoritmo.
- Para datasets grandes, o LMS (SGD) e o preferido pois permite N passos de gradiente por epoca ao custo total de O(Nd), identico ao GD, mas com cada passo custando O(d) — possibilitando atualizacoes incrementais uteis quando N e muito grande ou quando o dataset chega em streaming.
- O diagrama predito vs real (`pred_vs_actual.png`) confirma visualmente: os pontos do LMS estao mais concentrados em torno da diagonal ideal (y=x), enquanto os do GD mostram dispersao maior, consistente com o MSE quase dobrado.
