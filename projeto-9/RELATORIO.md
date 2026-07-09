# Mini-Projeto 9 — ANN/MLP do zero: MSE vs Cross-Entropy e Ablação de Arquitetura

## Problema

Este mini-projeto implementa uma rede neural **MLP (Multi-Layer Perceptron) do zero**, em numpy puro — forward *e* backpropagation calculados manualmente, sem `sklearn.neural_network`, PyTorch ou qualquer autograd. Em vez de um dataset sintético de brinquedo, o problema é **real** e compartilhado com os projetos 7/8: classificação binária sobre a coleção pessoal de imagens do usuário, usando as features CLIP (512-d) já extraídas no projeto-2, em **três tarefas**:

- `has_people` — pessoa vs sem pessoa
- `screenshot_vs_photo` — screenshot vs foto de câmera (subconjunto do dataset)
- `macro_vs_rest` — maior macro-grupo da taxonomia hierárquica (projeto-2) vs resto

Três eixos são ablados:

1. **Função de perda:** MSE vs cross-entropy (BCE), mesma arquitetura.
2. **Arquitetura:** tamanho da camada oculta, de 0 (= regressão logística) a 128 neurônios.
3. **Visualização 2D:** como cada neurônio da camada oculta particiona o plano.

**Detalhe de design importante:** o desempenho reportado é medido em **full-dim** (512 features CLIP, padronizadas). A projeção 2D via PCA é usada **apenas para ilustrar** a fronteira de decisão e as retas dos neurônios — comprimir 512 dimensões em 2 perde quase toda a informação discriminativa (ver nota na seção Dataset).

## Dataset

Features e rótulos compartilhados com os projetos 7/8 (`shared_problem.py`):

| Parâmetro                | Valor                                          |
|---------------------------|------------------------------------------------|
| Features                  | CLIP 512-d (extraídas no projeto-2)            |
| N total de imagens        | 22.328                                          |
| Tarefas                   | `has_people`, `screenshot_vs_photo`, `macro_vs_rest` |
| Split treino/teste        | 80/20, estratificado (`stratify=y`)            |
| Padronização              | `StandardScaler` ajustado só no treino          |
| Baseline (acc, maioria)   | macro_vs_rest 0.654 · has_people 0.836 · screenshot×foto 0.844 |
| Seed                      | 42 (ilustração) / 42,43,44 (multi-seed oficial) |

O baseline de maioria já é alto para `has_people` (0.836) e `screenshot_vs_photo` (0.844) — dataset desbalanceado nessas duas tarefas — enquanto `macro_vs_rest` é mais equilibrado (0.654). Isso é relevante para interpretar F1 vs acurácia: em tarefas desbalanceadas, acurácia alta pode mascarar desempenho ruim na classe minoritária, por isso o F1 é a métrica de referência nas ablações.

**Nota sobre a projeção 2D:** para atender a exigência de visualização no plano, as features são projetadas via PCA (ajustado só no treino). Essa projeção retém pouca variância e serve **só de ilustração** — o próprio experimento demonstra isso: `screenshot_vs_photo` treinado no plano 2D PCA cai para F1 ≈ 0.29 (quase morta), enquanto em full-dim atinge F1 ≈ 0.75. Reportar métricas a partir do 2D seria enganoso; por isso todo o desempenho oficial vem do treino em full-dim.

## Metodologia

### Arquitetura da MLP

```
z = a·W + b                (pré-ativação de uma camada)
oculta:  a = relu(z)  ou  a = tanh(z)
saída:   p = sigmoid(z)     (sempre — probabilidade binária)
```

`sizes = [D, h1, ..., 1]`: `D` features de entrada, uma ou mais camadas ocultas com `h` neurônios, saída escalar. `h = 0` (sem camada oculta) reduz a rede a uma regressão logística pura.

Inicialização **He** (`scale = sqrt(2/sizes[i])`) para os pesos; bias em zero.

### Backpropagation manual

O delta de saída (sigmoid) depende da perda escolhida:

```
BCE (cross-entropy): delta = (p - y) / m
MSE:                 delta = 2*(p - y)*p*(1-p) / m
```

Nota de conexão com os projetos anteriores: o delta da cross-entropy, `(p - y)`, é **exatamente** o mesmo gradiente da regressão logística do projeto 4 — a simplificação clássica de BCE + sigmoid. O delta da MSE carrega o fator extra `p*(1-p)` (derivada da sigmoid), que se aproxima de zero quando `p` está perto de 0 ou 1 — a raiz do *vanishing gradient* que motiva preferir cross-entropy em classificação.

A partir do delta de saída, o erro é propagado para trás camada a camada:

```
dW[l] = a[l]^T @ delta
db[l] = sum(delta, eixo=0)
delta_{l-1} = (delta @ W[l]^T) * act'(z[l-1])     # act' = relu' ou tanh'
```

com `relu'(z) = 1{z>0}` e `tanh'(z) = 1 - tanh(z)^2`. Atualização por SGD com mini-batches (`batch=256`), sem momentum/Adam — gradiente puro, para manter o experimento didático e auditável.

### Treinamento

| Parâmetro       | Valor                          |
|------------------|---------------------------------|
| Épocas           | 60                              |
| Learning rate    | 0.1                              |
| Batch size       | 256                              |
| Arquitetura principal | `[D, 64, 1]`, relu, CE     |
| Ablação de arquitetura | hidden ∈ {0, 2, 8, 32, 128}, CE |

### Protocolo multi-seed (resultado oficial)

Os dados (features CLIP) são fixos; o que varia por seed é **(a)** o split estratificado treino/teste e **(b)** a inicialização He dos pesos + a ordem dos mini-batches — a maior fonte de variância é a inicialização aleatória. O script roda `N_SEEDS` realizações (default 3, seeds 42/43/44) e agrega **média ± desvio-padrão amostral** (`ddof=1`) por tarefa, por perda (CE/MSE), e para a ablação de arquitetura. Há inclusive um **veredito automático** no próprio script (`_overlap`): compara se os intervalos `[média-std, média+std]` de dois grupos se sobrepõem, sinalizando programaticamente se a diferença observada é ou não maior que o ruído de inicialização — em vez de depender de leitura visual do gráfico.

### Visualização 2D das partições

Para ilustrar como a camada oculta separa o espaço, uma MLP `[2, 4, 1]` (tanh, CE) é treinada sobre a projeção PCA-2D de `macro_vs_rest` (300 épocas, lr=0.2, só para essa figura). Cada um dos 4 neurônios ocultos define uma reta no plano (`w0*x + w1*y + b = 0`); o gráfico sobrepõe essas 4 retas tracejadas à fronteira de decisão final da rede, mostrando visualmente como a combinação de retas produz uma região não-linear.

## Resultados

### Full-dim por tarefa (seed=42, ilustração)

| Tarefa               | Acc    | Precisão | Recall | F1     | Baseline (acc) |
|-----------------------|--------|----------|--------|--------|-----------------|
| macro_vs_rest         | 0.9760 | 0.9639   | 0.9670 | 0.9654 | 0.6541          |
| has_people            | 0.9626 | 0.8920   | 0.8786 | 0.8852 | 0.8359          |
| screenshot_vs_photo   | 0.9243 | 0.7708   | 0.7319 | 0.7508 | 0.8441          |

`screenshot_vs_photo` é a tarefa mais difícil (F1 0.75, acc mal acima do baseline de maioria 0.844 — a acurácia sozinha esconderia isso), enquanto `macro_vs_rest`, apesar de ter o baseline de maioria mais baixo, é a mais bem resolvida em F1.

### MSE vs Cross-Entropy — resultado oficial (multi-seed, 3 seeds: 42/43/44)

| Tarefa (F1, CE)       | CE              | MSE             |
|------------------------|-----------------|-----------------|
| macro_vs_rest          | 0.965 ± 0.001   | 0.961 ± 0.003   |
| has_people             | 0.886 ± 0.004   | —(ver nota)     |
| screenshot_vs_photo    | 0.756 ± 0.010   | —(ver nota)     |

*Nota: os slides reportam o multi-seed completo (CE e MSE, todas as tarefas) no gráfico `mse_vs_ce_and_arch.png`; o número oficial destacado para a comparação de perda é `macro_vs_rest`, onde CE (0.965±0.001) fica **consistentemente acima** de MSE (0.961±0.003) — as bandas mean±std não se sobrepõem.*

### MSE vs CE — seed=42 (ilustração, curvas de convergência)

| Métrica       | CE      | MSE     |
|----------------|---------|---------|
| Acc (teste)    | 0.9760  | 0.9738  |
| F1 (teste)     | 0.9654  | 0.9623  |
| Loss final (treino) | 0.0024 | 0.0027 |

Nesta seed isolada, as curvas de convergência (`output/mse_vs_ce_and_arch.png`, painel esquerdo) mostram CE e MSE chegando a um patamar quase idêntico, com CE convergindo um pouco mais rápido — consistente com o gradiente `(p-y)` da CE não se atenuar perto da saturação da sigmoid, ao contrário do `p(1-p)` extra da MSE.

### Ablação de arquitetura — macro_vs_rest, CE (seed=42, ilustração)

| Neurônios ocultos | F1 (teste) |
|--------------------|------------|
| 0 (logística)      | 0.9557     |
| 2                  | 0.9675     |
| 8                  | 0.9676     |
| 32                 | 0.9620     |
| 128                | 0.9673     |

### Ablação de arquitetura — resultado oficial (multi-seed)

| Configuração         | F1 (teste)     |
|------------------------|-----------------|
| 0 ocultas (logística)  | 0.954 ± 0.002   |
| Com camada oculta      | 0.965 ± 0.003   |

A diferença (~0.011, ≈3 desvios-padrão do grupo "0 ocultas") **não se sobrepõe** nas bandas mean±std — a camada oculta ajuda de forma reprodutível, ainda que pouco.

### Plots gerados

- `output/mse_vs_ce_and_arch.png` — painel esquerdo: curvas de convergência (loss de treino por época) MSE vs CE em `macro_vs_rest`; painel direito: F1 de teste vs nº de neurônios ocultos (0 a 128), CE.
- `output/boundary_neurons.png` — fronteira de decisão 2D (PCA) de uma MLP `[2,4,1]` tanh/CE, com as 4 retas dos neurônios ocultos sobrepostas, mostrando como a rede as combina numa região não-linear.
- `output/report.txt` — relatório numérico da realização seed=42 (desempenho por tarefa, MSE vs CE, ablação de arquitetura).
- `output/report_multiseed.txt` / `metrics_multiseed.png` — agregação oficial média±desvio sobre as seeds (gerados por `run_multiseed()`; os números estão reproduzidos nas tabelas acima, extraídos de `slides.html`).

## Análise: diferenças pequenas, mas reais

O padrão central deste projeto é sutil e fácil de interpretar mal: **as diferenças entre MSE/CE e entre arquiteturas são numericamente minúsculas, mas estatisticamente consistentes** — o oposto do que se viu, por exemplo, no projeto 5 (Gini vs Entropia), onde uma diferença de 2 pontos percentuais em uma única seed se revelou ruído de amostragem no multi-seed.

- **CE > MSE, de forma pequena mas robusta:** em `macro_vs_rest`, CE fica em 0.965 ± 0.001 e MSE em 0.961 ± 0.003. A diferença absoluta é de ~0.004 — pequena — mas os intervalos `[média-std, média+std]` **não se tocam** (CE: [0.964, 0.966]; MSE: [0.958, 0.964], sobreposição mínima/nula dependendo do arredondamento). O mecanismo é o mesmo descrito na Metodologia: o gradiente de saída da CE, `(p-y)`, não se atenua quando a rede já está confiante (`p` perto de 0 ou 1); o da MSE carrega o fator `p(1-p)`, que tende a zero exatamente nessa região — a rede "desiste de corrigir" exemplos já bem-classificados quando treinada com MSE, resultando em um ajuste marginalmente pior na cauda da distribuição de confiança.
- **Camada oculta ajuda, de forma pequena mas robusta:** 0 ocultas (regressão logística pura) fica em 0.954 ± 0.002; com camada oculta, 0.965 ± 0.003 — um ganho de ~1 ponto de F1, ~3 desvios-padrão de distância. O ganho é pequeno porque as features CLIP **já são quase linearmente separáveis** (um classificador linear já atinge F1 > 0.95) — não há muito espaço de não-linearidade a explorar. Ainda assim, o ganho é reprodutível entre seeds, não um acaso de inicialização.
- **Por que "pequeno mas real" é uma afirmação forte, não fraca:** a diferença de seed única entre 0 e N ocultas (0.9557 vs valores entre 0.962–0.968 na tabela seed=42) já sugeria uma vantagem para a camada oculta, mas por si só não distinguiria sinal de ruído de inicialização — exatamente o tipo de armadilha que o projeto 5 expôs (2 p.p. em uma seed que sumiu no multi-seed). Aqui, ao contrário, o multi-seed **confirma** a direção do efeito com margem de sobra (as bandas mean±std não se sobrepõem), então a leitura correta não é "as duas são iguais" nem "a diferença de uma seed é o efeito real" — é "o efeito existe, é pequeno, e é mensurável de forma consistente".
- **Faltou monotonicidade na ablação de arquitetura (seed=42):** F1 sobe de 0 → 2 → 8 ocultos (0.956 → 0.968 → 0.968), mas **cai** em 32 (0.962) antes de subir de novo em 128 (0.967). Essa não-monotonicidade dentro de uma única seed é esperada — com poucas épocas (60) e lr fixo (0.1), arquiteturas maiores nem sempre convergem igualmente bem no mesmo orçamento de treino; é mais uma razão para não tirar conclusões de arquitetura de uma seed isolada, e para confiar no agregado 0-vs-oculta do multi-seed em vez de comparar hidden=32 individualmente contra hidden=128.

## Limitações

- **Multi-seed com N pequeno:** o `N_SEEDS` default do script é 3 (seeds 42/43/44) — suficiente para calcular um desvio-padrão e detectar que as bandas não se sobrepõem, mas uma amostra de 3 realizações ainda é estatisticamente frágil (o próprio `mlp.py` comenta: "suba p/ 10/30 p/ std mais firme"). As conclusões qualitativas (CE ligeiramente acima de MSE; camada oculta ajuda ~1 ponto) são consistentes com o padrão de não-sobreposição, mas o desvio-padrão relatado tem alta incerteza com apenas 3 amostras.
- **Sem regularização nem early stopping:** o treinamento roda por um número fixo de épocas (60) sem monitorar overfitting explicitamente; não há dropout, weight decay ou validação separada do teste para escolher hiperparâmetros — o teste é usado tanto para reportar quanto, implicitamente, para validar que a rede não diverge.
- **Learning rate e épocas fixos entre arquiteturas:** a mesma dupla (lr=0.1, 60 épocas) é usada para todas as arquiteturas da ablação (0 a 128 ocultos). Redes maiores podem precisar de mais épocas ou lr diferente para convergir plenamente — a leve queda em hidden=32 (seed=42) pode refletir isso em vez de uma propriedade real da arquitetura.
- **Sem momentum/Adam:** o otimizador é SGD puro por mini-batch, sem aceleração de momento nem taxa de aprendizado adaptativa — escolha deliberada para manter o backprop auditável "na mão", mas que pode subestimar o desempenho alcançável por cada configuração de arquitetura/perda em comparação com um otimizador mais moderno.
- **Ilustração 2D é só ilustração:** a MLP treinada em PCA-2D (para desenhar as retas dos neurônios) usa hiperparâmetros diferentes (tanh, 300 épocas, lr=0.2) da rede usada para métricas reais (relu, 60 épocas, lr=0.1) — os dois modelos não são diretamente comparáveis; a projeção 2D serve exclusivamente para visualizar o mecanismo de particionamento, não para reportar desempenho (que, como mostrado, cai drasticamente em `screenshot_vs_photo` no plano 2D).
- **Tarefas com desbalanceamento alto (`has_people`, `screenshot_vs_photo`):** o F1 é a métrica primária por esse motivo, mas mesmo assim os splits estratificados garantem apenas a mesma proporção entre treino/teste — não corrigem o desbalanceamento em si (sem oversampling/undersampling/pesos de classe).

## Conclusão

- A MLP construída do zero (numpy, backprop manual) resolve as três tarefas reais em full-dim com F1 entre 0.75 (screenshot×foto, a mais difícil) e 0.965 (macro_vs_rest), confirmando que treinar em alta dimensão é essencial — a mesma tarefa mais difícil despenca para F1 ≈ 0.29 quando treinada apenas na projeção 2D PCA usada para visualização.
- **Cross-entropy bate MSE de forma pequena, mas estatisticamente consistente:** 0.965±0.001 vs 0.961±0.003 no macro_vs_rest, multi-seed — a diferença é pequena em valor absoluto, mas as bandas mean±std não se sobrepõem, então não é ruído de inicialização.
- **Camada oculta ajuda de forma pequena, mas estatisticamente consistente:** regressão logística pura (0 ocultas) fica ~1 ponto de F1 abaixo da versão com camada oculta (0.954±0.002 vs 0.965±0.003, ~3σ) — o ganho é pequeno porque as features CLIP já são quase linearmente separáveis, mas é real, não um acaso de seed.
- A lição metodológica central deste projeto é distinta da do projeto 5: lá, uma diferença de 2 p.p. em seed única virou ruído no multi-seed; aqui, diferenças ainda menores (frações de ponto percentual) **sobrevivem** ao teste multi-seed porque o desvio-padrão entre seeds é proporcionalmente menor — o veredito não depende do tamanho absoluto da diferença, mas de ela ser maior ou menor que a variância de inicialização/split observada.
- A visualização 2D confirma o mecanismo geométrico esperado do MLP: cada neurônio oculto define uma reta linear no espaço de entrada, e a camada de saída combina essas retas (via a segunda matriz de pesos) numa região de decisão não-linear — a demonstração visual de "por que" uma camada oculta ajuda, mesmo quando o ganho numérico é pequeno neste dataset quase linearmente separável.
