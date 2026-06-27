# Guia de Estudo — Mini-Projeto 3: Regressao Linear GD vs LMS

## Pipeline

```
1. GERACAO DO DATASET
   seed=42 -> rand_distr::Normal(0,1) -> x1, x2
   y = 3*x1 - 2*x2 + 1 + Normal(0, 0.5)
   N = 300 amostras

2. INICIALIZACAO DOS MODELOS
   w = [0.0, 0.0]
   b = 0.0
   (mesma inicializacao para GD e LMS)

3a. BATCH GRADIENT DESCENT (200 epocas)
    Para cada epoca:
      err_i = (w.x_i + b) - y_i       (para todo i em 1..N)
      grad_w_sum += err_i * x_i        # acumula soma bruta (nao media)
      grad_b_sum += err_i
      w -= lr * grad_w_sum / N         (lr = 0.01; divisao por N acontece aqui, nao antes)
      b -= lr * grad_b_sum / N
    Registra MSE inicial (antes do treino) + MSE ao fim de cada epoca -> 201 pontos no historico (indice 0 = estado inicial)

3b. LMS / WIDROW-HOFF (200 epocas)
    Para cada epoca:
      Para cada (x_i, y_i):
        err_i = y_i - (w.x_i + b)     (sinal invertido!)
        w += lr * err_i * x_i         (lr = 0.001)
        b += lr * err_i
    Registra MSE inicial (antes do treino) + MSE ao fim de cada epoca -> 201 pontos no historico (indice 0 = estado inicial)

4. PLOTS
   convergence.png    : MSE por epoca (GD azul | LMS vermelho)
   scatter.png        : dados + retas aprendidas (x1 vs y) + reta verdadeira (verde, referencia)
                         Nota: a dispersao vertical e dominada por Var[-2*x2]=4 (feature omitida), nao pelo ruido (Var[ruido]=0.25). As retas mostram a regressao marginal E[y|x1]=w0*x1+b, integrando x2 via E[x2]=0 — projecao 2D correta.
   pred_vs_actual.png : predito vs real para GD e LMS (ambos os modelos usam todas as 2 features; diagonal verde = modelo perfeito)

5. REPORT
   output/report.txt com tabela comparativa de metricas
```

## Matematica

### Modelo de regressao linear

```
y_hat = w^T * x + b
      = w[0]*x[0] + w[1]*x[1] + b
```

### Funcao de custo MSE

```
L(w, b) = (1/N) * sum_{i=1}^{N} (y_hat_i - y_i)^2
```

### Gradiente do MSE (derivadas parciais)

**Derivacao matematica (fator 2/N -- formula exata, use em provas e derivacoes):**

```
dL/dw_j = (2/N) * sum_i (y_hat_i - y_i) * x_ij
dL/db   = (2/N) * sum_i (y_hat_i - y_i)
```

**Implementacao (fator 2 absorvido em alpha -- use ao descrever o codigo):**

O fator 2 e omitido e absorvido pelo learning rate alpha (equivalente a usar alpha_efetivo = 2*alpha da derivada completa). Isso nao altera o caminho de otimizacao -- apenas escala o lr efetivo. Em provas e derivacoes formais, cite a formula com fator 2/N; ao descrever a implementacao, cite a formula com fator 1/N.

### Regra de atualizacao GD (Batch)

Na implementacao, o fator 2 e omitido e absorvido pelo learning rate alpha
(equivalente a usar alpha_efetivo = 2 * alpha da derivada completa):

```
w <- w - alpha * (1/N) * sum_i (y_hat_i - y_i) * x_i    # fator 2 absorvido em alpha
b <- b - alpha * (1/N) * sum_i (y_hat_i - y_i)
```

### Regra de atualizacao LMS (Widrow-Hoff)

```
delta_i = y_i - y_hat_i          (erro: sinal positivo)
w <- w + eta * delta_i * x_i
b <- b + eta * delta_i
```

Equivalente a GD com N=1 e sinal corrigido. Converge para o mesmo minimo em expectativa, mas com passos ruidosos.

### Condicao de convergencia do LMS (populacao / assimptotica)

Para garantir convergencia, o learning rate eta deve satisfazer:

```
0 < eta < 2 / lambda_max
```
*(condicao populacional/assimptotica; lambda_max = autovalor maximo de R = E[x x^T] -- ver estimativa empirica abaixo)*

**Condicao teorica (populacao):** lambda_max e o maior autovalor de R = E[x x^T]. Para features normalizadas, lambda_max ~= 1 e eta < 2.

**Importante — media vs MSE:** a condicao `0 < eta < 2/lambda_max` garante convergencia da *media* dos pesos E[w(k)] para o otimo (convergencia de primeira ordem). Para convergencia do *MSE* (momento de segunda ordem / estabilidade da variancia dos pesos), a condicao mais restrita e necessaria e:

```
0 < eta < 1 / lambda_max(R)
```

*(Widrow & Stearns 1985; Haykin, Adaptive Filter Theory). Para lambda_max ~ 1 (features N(0,1)), isso significa eta < 1 para MSE vs eta < 2 para a media. O LMS com eta=0.001 satisfaz ambas com larga margem.*

**Na pratica (estimativa empirica):** o codigo opera sobre a matriz empirica (1/N)*X^T*X. A condicao efetiva e sobre o autovalor maximo dessa matriz amostral. Para este dataset com N=300 e features ~ N(0,1), ambas convergem para ~1, portanto a condicao numerica e a mesma. Em geral, populacao e amostra coincidem em expectativa mas diferem por realizacao — para exames que perguntam sobre o dataset especifico, citar a versao empirica e mais preciso.

## Codigo

### Geracao do dataset

```rust
fn generate_dataset(rng: &mut StdRng) -> (Vec<Vec<f64>>, Vec<f64>) {
    let feat_dist = Normal::new(0.0, 1.0).unwrap();
    let noise_dist = Normal::new(0.0, NOISE_STD).unwrap();
    let mut x = Vec::with_capacity(N_SAMPLES);  // acumula feature vectors
    let mut y = Vec::with_capacity(N_SAMPLES);  // acumula targets
    for _ in 0..N_SAMPLES {
        let xi: Vec<f64> = (0..N_FEATURES).map(|_| feat_dist.sample(rng)).collect();
        let yi = dot(&xi, &TRUE_W) + TRUE_BIAS + noise_dist.sample(rng);
        x.push(xi);
        y.push(yi);
    }
    (x, y)  // retorna tupla (features, targets)
}
```

Cada amostra xi tem N_FEATURES=2 valores independentes ~ Normal(0,1). O target yi e a combinacao linear verdadeira mais ruido gaussiano.

### Batch Gradient Descent (nucleo)

```rust
for _iter in 0..max_iters {
    let mut grad_w = vec![0.0f64; N_FEATURES];
    let mut grad_b = 0.0f64;
    // Acumula gradiente sobre todos os N exemplos
    for (xi, &yi) in x.iter().zip(y) {
        let err = predict(xi, &w, b) - yi;   // y_hat - y
        for (gw, &xij) in grad_w.iter_mut().zip(xi) {
            *gw += err * xij;
        }
        grad_b += err;
    }
    // Atualiza com media do gradiente
    for (wj, gw) in w.iter_mut().zip(&grad_w) {
        *wj -= lr * gw / n;
    }
    b -= lr * grad_b / n;
}
```

### LMS / Widrow-Hoff (nucleo)

```rust
for _epoch in 0..max_iters {
    for (xi, &yi) in x.iter().zip(y) {
        let err = yi - predict(xi, &w, b);   // y - y_hat (sinal +)
        for (wj, &xij) in w.iter_mut().zip(xi) {
            *wj += lr * err * xij;           // correcao proporcional ao erro
        }
        b += lr * err;
    }
}
```

A diferenca chave: LMS atualiza os pesos N=300 vezes por epoca, GD atualiza 1 vez. Por isso o LMS pode usar lr menor e ainda convergir mais rapido em termos de epocas.

## Resultados

Saida real da execucao (seed=42, N=300, 200 epocas):

```
Pesos verdadeiros : w=[3, -2], b=1
GD  -> w=[2.636558, -1.668652], b=0.825018, MSE=0.479627
LMS -> w=[2.966010, -2.007297], b=0.985971, MSE=0.235422
```

| Metrica           | GD         | LMS        | Ref/Min.Teorico |
|-------------------|------------|------------|-----------------|
| MSE final         | 0.479627   | 0.235422   | 0.250 (*)  |
| w[0]              | 2.636558   | 2.966010   | 3.000000        |
| w[1]              | -1.668652  | -2.007297  | -2.000000       |
| bias              | 0.825018   | 0.985971   | 1.000000        |
| MSE inicial       | 14.303548  | 14.303548  | —               |
| Reducao MSE       | 96.65%     | 98.35%     | —               |

(*) sigma^2 = limite inferior do MSE (ruido irredutivel). O LMS atingiu 0.235 < 0.250 por variacao estatistica normal com N=300 — ver P4.

- LMS atingiu MSE = 0.235, proximo do limite teorico de ruido (sigma^2 = 0.25)
- GD com lr=0.01 nao convergiu completamente em 200 epocas: MSE = 0.48
- Erro absoluto nos pesos: GD: |w0|=0.36, |w1|=0.33, |b|=0.17 (media ~0.29); LMS: |w0|=0.034, |w1|=0.007, |b|=0.014

## Perguntas Provaveis

**P1: Qual e a diferenca matematica entre GD e LMS?**

R: GD calcula o gradiente exato sobre todo o dataset antes de cada atualizacao. LMS (Widrow-Hoff) usa apenas um exemplo por atualizacao, aplicando a regra `w += eta * (y - y_hat) * x`. GD tem variancia zero no gradiente (dado o dataset), mas apenas 1 atualizacao por epoca. LMS tem gradiente ruidoso (estimativa estochastica), mas N atualizacoes por epoca.

**P2: Por que o LMS usou lr=0.001 e o GD usou lr=0.01?**

R: No LMS, como ha N=300 atualizacoes por epoca e cada uma pode superestimar o gradiente verdadeiro, o lr precisa ser menor para estabilidade. Com lr grande, o LMS pode divergir (pesos oscilam em torno do otimo ou explodem). O GD com gradiente exato tolera lr maior pois a direcao e garantidamente de descida. Nota: o limite teorico de estabilidade para este dataset e eta < 2/lambda_max(R) ≈ 2 (features ~ N(0,1)). O valor eta=0.001 e deliberadamente conservador — 2000x abaixo do limite — para enfatizar a vantagem do numero de atualizacoes por epoca em vez da magnitude do lr.

**P3: Por que o LMS convergiu para pesos mais proximos dos verdadeiros?**

R: Efetivamente o LMS fez 200 epocas * 300 amostras = 60.000 atualizacoes de gradiente, enquanto o GD fez 200. Mesmo com passos menores, a quantidade de atualizacoes foi muito maior. Nota: costuma-se citar tambem o efeito de regularizacao implicita do SGD online, mas esse efeito nao foi verificado neste projeto — a principal explicacao e a quantidade de atualizacoes (60.000 vs 200).

**P4: O MSE final do LMS (0.235) ficou abaixo do limite teorico (0.25). Isso e possivel?**

R: Sim. sigma^2 = 0.25 e o limite assimptotico esperado, mas com uma realizacao especifica de 300 amostras o MSE empirico pode ficar ligeiramente acima ou abaixo. E variacao estatistica normal — nao e overfitting porque os pesos aprendidos estao muito proximos dos verdadeiros.

**P5: O que aconteceria se aumentassemos o learning rate do GD?**

R: Com lr maior (ex: 0.05), o GD convergiria mais rapido nas primeiras epocas mas poderia oscilar em torno do minimo ou ate divergir.

A implementacao usa `gradient = (1/N)*sum(err*x)` (sem fator 2), portanto a Hessiana efetiva e `H = (1/N)*X^T*X` e a condicao de estabilidade que se aplica ao codigo e:

```
lr < 2 / lambda_max((1/N)*X^T*X)
```

Se voce derivar formalmente a partir de `L = (1/N)*sum(...)`, a derivada exata inclui fator 2, resultando em `H = (2/N)*X^T*X` e na condicao `lr < 1/lambda_max((1/N)*X^T*X)`. Essa formula e correta para a derivacao matematica completa, mas **nao se aplica ao codigo como escrito** — o fator 2 foi absorvido pelo learning rate na implementacao. Usar a formula derivada formalmente para prever o comportamento do codigo daria um limite duas vezes mais conservador do que o necessario.

Para este dataset com features ~ N(0,1), `(1/N)*X^T*X ~ I` e lambda_max ~ 1 (aproximacao populacional; a matriz empirica com N=300 amostras pode desviar ~10-15% da identidade por realizacao, mas a condicao lr < 2 ainda e satisfeita com larga margem), entao a condicao efetiva do codigo e `lr < 2`, e o valor lr=0.01 e muito conservador (200x abaixo do limite), garantindo convergencia monotonica — ao custo de mais epocas para atingir o mesmo MSE que o LMS. A formula lr < 2/lambda_max(R) do LMS (onde R = E[x x^T]) coincide numericamente aqui, mas o caminho de derivacao e diferente — nao e uma analogia direta.

**P6: Qual algoritmo usar em producao e por que?**

R: Para datasets grandes, SGD/LMS e preferido pois o custo por atualizacao e O(d) (dimensao das features) e nao O(N*d) como no GD. O GD batch tem custo proibitivo quando N e na casa de milhoes. Mini-batch SGD (usado em deep learning) e o meio-termo: agrega B exemplos por passo, reduzindo a variancia sem o custo total do batch.

**P7: Como a regra LMS se relaciona com o algoritmo Perceptron?**

R: Ambos sao regras de atualizacao por exemplo com a forma `w += lr * erro * x`. A diferenca e que o Perceptron usa erro discreto: (y - y_hat) e 0 quando a predicao esta correta e nao-nulo (ex: +1 ou -1) quando errada — atualiza so quando (y - y_hat) != 0 — e aplica funcao de ativacao de limiar. O LMS usa erro continuo y - y_hat real, atualizando sempre, sem funcao de ativacao. Historicamente, Widrow e Hoff publicaram o LMS em 1960, contemporaneo ao Perceptron de Rosenblatt (1958).

**P8: O que o pred_vs_actual.png mostra que o scatter.png nao mostra?**

R: O scatter.png e uma projecao 2D (x1 vs y) que omite x2, entao as retas mostradas sao regressoes marginais E[y|x1] — corretas, mas incompletas. O pred_vs_actual.png usa *ambas* as features simultaneamente: o eixo x e o valor real y e o eixo y e a predicao y_hat = w[0]*x1 + w[1]*x2 + b. A diagonal verde y=x e o preditor ideal (erro zero). A dispersao ao redor dessa diagonal visualiza diretamente o MSE: quanto mais concentrados em torno da diagonal, menor o erro. No grafico, LMS (vermelho, MSE=0.2354) esta mais concentrado na diagonal que GD (azul, MSE=0.4796), o que corrobora os numeros do report. O scatter.png mostra onde as retas estao no espaco x1, mas nao captura o ajuste completo com as 2 features; o pred_vs_actual.png captura o erro global do modelo sem omitir nenhuma variavel.

**P9: Por que as oscilacoes do LMS nao aparecem no convergence.png?**

R: O convergence.png registra o MSE calculado *ao fim de cada epoca* — ou seja, depois que todos os N=300 exemplos da epoca ja foram processados e os pesos ja foram atualizados 300 vezes. As oscilacoes intra-epoca do LMS (flutuacoes dos pesos a cada um dos 300 exemplos individuais dentro da epoca) acontecem, mas nao sao visiveis nessa resolucao temporal porque o grafico so amostra o estado dos pesos no ponto de checkpoint (final de epoca). Para observar as oscilacoes seria necessario registrar o MSE apos cada atualizacao individual — resultando em 60.000 pontos em vez de 200.
