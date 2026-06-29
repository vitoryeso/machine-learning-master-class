# Guia de Estudo — Aprendizado de Máquina 2026.1

Companion didático dos mini-projetos da disciplina **Aprendizado de Máquina** — PPgEEC / UFRN.
Explica cada método do zero: **intuição → matemática (com cada termo explicado) → como apareceu no seu projeto → pegadinhas → exercícios de fixação (com gabarito)**.

Aluno: **Vitor Y. F. Freitas** · `vitoryeso@outlook.com`

> Como usar: leia a seção "Fundamentos" primeiro (o vocabulário comum). Depois cada método pode ser lido sozinho. Os exercícios têm gabarito em `<details>` — tente antes de abrir.

---

## Sumário

- [Fundamentos (o vocabulário comum)](#fundamentos)
- [1. K-Means — descobrir grupos sem rótulos](#1-k-means)
- [2. Classificar uma taxonomia descoberta](#2-classificar-taxonomia)
- [3. Regressão Linear: Gradient Descent vs LMS](#3-regressao-linear)
- [4. Regressão Logística](#4-regressao-logistica)
- [5. Árvore de Decisão: Gini vs Entropia](#5-arvore)
- [7. Redes RBF](#7-rbf)
- [8. SVM com kernel RBF](#8-svm)
- [9. Redes Neurais (MLP)](#9-mlp)
- [10. Redes Convolucionais (CNN)](#10-cnn)
- [Visão de conjunto — a progressão](#visao-de-conjunto)

---

<a name="fundamentos"></a>
## Fundamentos (o vocabulário comum)

Antes dos métodos, os conceitos que reaparecem em todos.

### O que é "aprender"?
Temos exemplos `(x, y)`: **x** é a *entrada* (um vetor de números que descreve algo — chamado **features**), e **y** é o *alvo* (o que queremos prever). Aprender = achar uma função `f` tal que `f(x) ≈ y` em exemplos **que o modelo nunca viu, de outra distribuição** — formalmente, minimizando a **função de perda** (o erro médio nos exemplos): `L = (1/N) Σᵢ ℓ(f(xᵢ), yᵢ)`, onde `ℓ` mede o erro de um exemplo (previsto vs real). Se `y` é categoria → **classificação**; se `y` é número contínuo → **regressão**; se não há `y` → **não-supervisionado** (ex.: clustering).

### Treino, validação e teste
Dividimos os dados:
- **Treino** — o modelo ajusta seus parâmetros aqui.
- **Teste** — medimos o desempenho aqui, em dados *nunca vistos*. É a única medida honesta de **generalização**.
- (**Validação** — opcional, pra escolher hiperparâmetros sem "sujar" o teste.)

> **Pegadinha — data leakage:** qualquer informação do teste que vaza pro treino infla o resultado. Ex.: ajustar a normalização (`StandardScaler`) usando o dataset inteiro. O certo é ajustar **só no treino** e aplicar no teste.

### Métricas de classificação
Da **matriz de confusão** (VN, FP, FN, VP = verdadeiros/falsos negativos/positivos):

```
Acurácia  = (VP + VN) / total              → fração de acertos
Precisão  = VP / (VP + FP)                  → dos que chamei de positivo, quantos eram
Recall    = VP / (VP + FN)                  → dos positivos reais, quantos achei
F1        = 2·(Precisão·Recall)/(Prec+Rec)  → média harmônica de precisão e recall
```

> **Por que não só acurácia?** Com classes **desbalanceadas** (ex.: 84% de uma classe), prever sempre a maioria já dá 84% de acurácia — parece bom, mas o modelo é inútil. F1 (ou F1 macro-averaged) expõe isso porque exige acertar a classe rara. Sempre compare contra o **baseline da classe majoritária**.

### Gradiente descendente (GD)
A receita universal pra "ajustar parâmetros". Definimos a **função de perda** explícita `L(θ) = (1/N) Σᵢ ℓ(f(xᵢ), yᵢ)` (o erro médio do modelo, em função dos parâmetros `θ`). O **gradiente** `∇L` aponta na direção de maior *aumento* da perda; então andamos no sentido oposto:

```
θ ← θ − α · ∇L(θ)
```

- **θ** — os parâmetros (pesos) do modelo.
- **∇L** — o gradiente: a derivada da perda em relação a cada parâmetro.
- **α** (learning rate) — o tamanho do passo. Pequeno = lento mas estável; grande = rápido mas pode oscilar/divergir.

Repetimos até a perda parar de cair. Quase todo modelo "treinável" deste guia usa alguma variante disso.

---

<a name="1-k-means"></a>
## 1. K-Means — descobrir grupos sem rótulos

### Intuição
Você tem pontos e quer agrupá-los em **K** grupos, sem rótulos. K-Means alterna duas perguntas até estabilizar: *"a que grupo cada ponto pertence?"* e *"onde fica o centro de cada grupo?"*. É como organizar uma festa em rodas: cada pessoa vai pra roda mais próxima, e cada roda se reposiciona no meio de quem chegou.

### Matemática
Minimiza a soma das distâncias quadradas de cada ponto ao centro do seu grupo (a **inércia** / WCSS):

```
WCSS = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²
```

- **C_k** — o conjunto de pontos do grupo k.
- **μ_k** — o **centróide**: a média dos pontos do grupo k.
- **‖x − μ_k‖²** — distância euclidiana ao quadrado entre o ponto e o centro.

O algoritmo (Lloyd) repete:
1. **Atribuição:** cada `x` vai pro centróide mais próximo.
2. **Atualização:** cada `μ_k` vira a média dos pontos atribuídos a ele.

Como escolher **K**? Duas ferramentas:
- **Elbow (cotovelo):** plote WCSS vs K; o "cotovelo" (onde a queda desacelera) sugere o K.
- **Silhouette:** para cada ponto, `s = (b − a)/max(a,b)`, onde **a** = distância média aos do próprio grupo (coesão) e **b** = distância média ao grupo vizinho mais próximo (separação). `s≈1` ótimo, `s≈0` na fronteira, `s<0` provavelmente no grupo errado.

### No seu projeto (Projeto 1)
Você rodou K-Means sobre **embeddings CLIP** (vetores de 512-d) de 8.000 imagens pessoais. Elbow e silhouette concordaram em **K=4**. O silhouette foi baixo (~0.058) — **e isso é esperado** em alta dimensão com dados visuais contínuos (não há classes discretas perfeitas). O insight: o CLIP agrupou por **semântica visual**, não por metadados (fotos de iPhone caíram em todos os clusters).

### Pegadinhas
- K-Means assume grupos **esféricos** e de tamanho parecido — falha em grupos alongados/curvos (aí: DBSCAN, GMM).
- É sensível à inicialização → use **K-Means++** (espalha os centros iniciais).
- Em alta dimensão, **L2-normalize** os vetores: aí distância euclidiana ≈ distância de cosseno (o que importa em embeddings).

### Exercícios
1. Por que a WCSS *sempre* diminui quando aumentamos K? O que isso implica pra usar WCSS sozinha na escolha de K?
2. Um ponto tem silhouette `s = −0.3`. O que isso diz sobre ele?
3. Você tem 2 grupos em formato de duas luas entrelaçadas. K-Means vai separá-los bem? Por quê?

<details><summary>Gabarito</summary>

1. Com mais centros, cada ponto fica em média mais perto de *algum* centro, então a soma das distâncias só cai (no limite K=N, WCSS=0). Por isso WCSS sozinha sempre "prefere" K maior — precisamos do *cotovelo* (ganho marginal) ou de outra métrica (silhouette) pra escolher.
2. `s<0` significa que ele está, em média, **mais perto do grupo vizinho** do que do próprio grupo — provavelmente foi atribuído ao cluster errado (está na região de outro grupo).
3. **Não.** K-Means cria fronteiras lineares (células de Voronoi, convexas) entre centros; duas luas entrelaçadas não são linearmente separáveis por centros, então pedaços de uma lua serão roubados pela outra. DBSCAN (densidade) resolveria.
</details>

---

<a name="2-classificar-taxonomia"></a>
## 2. Classificar uma taxonomia descoberta (probes + circularidade)

### Intuição
E se você **não tem rótulos**, mas quer um classificador? Uma ideia: usar o clustering (Projeto 1) pra *descobrir* classes, e então treinar um classificador pra atribuí-las a imagens novas. Mas há uma armadilha sutil de honestidade científica.

### A pegadinha central: circularidade
Se você gera os rótulos com K-Means **sobre os embeddings CLIP**, e depois treina o classificador **sobre os mesmos CLIP**, a acurácia fica artificialmente alta — os clusters são, por construção, regiões coerentes naquele espaço (quase células de Voronoi). É quase "prever a si mesmo". O teste honesto: classificar sobre um espaço de features **independente** (ex.: ConvNeXt). Se a taxonomia for "real", outro modelo também deve recuperá-la.

### Linear probe
Um **linear probe** é o classificador mais simples possível em cima de features prontas: uma regressão logística (ver §4) sobre os vetores. Se features já são boas, um modelo linear basta — é uma sonda da *qualidade da representação*.

### No seu projeto (Projeto 2)
Você reconstruiu 2 níveis de rótulos (5 macro-grupos, 25 folhas) do clustering hierárquico e treinou linear probes:

| Features | macro (acc) | folha (acc) |
|---|---|---|
| CLIP (circular — gerou os clusters) | 0.95 | 0.89 |
| **ConvNeXt (honesto — independente)** | **0.83** | **0.66** |

Leitura: o nível **macro é "real"** (um espaço independente recupera 83%), mas as 25 subclasses são mais **específicas do CLIP** (caem pra 66%). O *gap* CLIP→ConvNeXt **quantifica** a circularidade.

### Exercícios
1. Por que classificar os clusters no próprio espaço que os gerou infla a acurácia?
2. Seu colega reporta 99% classificando clusters K-Means com os mesmos embeddings. Que pergunta você faz?
3. O que um linear probe com alta acurácia te diz sobre as *features* (não sobre o classificador)?

<details><summary>Gabarito</summary>

1. Porque os clusters são definidos por proximidade nesse espaço — então as fronteiras entre eles já são (aproximadamente) lineares ali. O classificador só precisa redescobrir as fronteiras de Voronoi, tarefa quase trivial → acurácia inflada e pouco informativa.
2. "Você classificou no **mesmo** espaço que gerou os clusters? Tente um espaço de features independente." (a métrica é circular).
3. Que as features já são (quase) **linearmente separáveis** para aquela tarefa — ou seja, a representação carrega a informação relevante de forma acessível. O mérito é da representação, não do modelo linear.
</details>

---

<a name="3-regressao-linear"></a>
## 3. Regressão Linear: Gradient Descent vs LMS

### Intuição
Ajustar uma reta (ou hiperplano) que melhor prevê um número. A questão do projeto não é *o modelo*, e sim **como otimizá-lo**: olhar todos os dados antes de cada passo (Batch GD) ou corrigir a cada exemplo (LMS/online)?

### Matemática
Modelo: `ŷ = wᵀx + b`. Perda (erro quadrático médio):

```
L(w,b) = (1/N) Σ_i (ŷ_i − y_i)²
```

- **x** — features de entrada; **ŷ** — previsão; **y** — alvo real.
- **w, b** — pesos e viés (o que treinamos).
- **N** — número de exemplos.

O gradiente (derivada de L) dá a direção de correção:

```
∂L/∂w = (2/N) Σ_i (ŷ_i − y_i)·x_i
∂L/∂b = (2/N) Σ_i (ŷ_i − y_i)
```

Duas estratégias de atualização:
- **Batch GD:** usa o gradiente sobre **todos** os N exemplos → **1 passo por época**. Gradiente exato, estável.
- **LMS / Widrow-Hoff:** atualiza a cada exemplo: `w ← w + η·(y−ŷ)·x` → **N passos por época**. Gradiente ruidoso, mas muitos passos.

### No seu projeto (Projeto 3)
Dados sintéticos (`y = 3x₁ − 2x₂ + 1 + ruído`), então você *conhece* os pesos verdadeiros. Em 200 épocas: o GD deu 200 passos; o **LMS deu 200×300 = 60.000**. Resultado: LMS chegou a MSE 0.235 (≈ o mínimo teórico do ruído, 0.25), GD ficou em 0.48 — **mas o GD não convergiu de propósito** (lr=0.01 escolhido pra mostrar a sensibilidade ao hiperparâmetro).

### Pegadinhas
- O **fator 2** do gradiente costuma ser absorvido no learning rate (convenção).
- LMS com lr grande **diverge**; GD com lr grande **oscila**. A escolha de lr é crítica e independente.
- Para dados gigantes/streaming, LMS (SGD) vence: muitos passos baratos.

### Exercícios
1. Por que o LMS pode convergir mais rápido que o GD mesmo "vendo" os mesmos dados?
2. O que acontece com o LMS se o learning rate for grande demais?
3. O MSE final do LMS deu *abaixo* do mínimo teórico do ruído (σ²=0.25). Isso é um bug?

<details><summary>Gabarito</summary>

1. Porque dá **N atualizações por época** (uma por exemplo) contra **1** do batch GD. Cada passo é ruidoso, mas o número muito maior de passos leva os pesos pra perto do ótimo mais rápido em termos de épocas.
2. As atualizações por exemplo ficam grandes demais e "passam do ponto" repetidamente → os pesos **divergem** (explodem) em vez de convergir.
3. Não necessariamente — é **variação estatística amostral** (com N=300, o MSE de treino flutua em torno de σ²). É o MSE de *treino*; ficar um pouco abaixo de σ² é esperado, não viola o limite teórico (que vale em expectativa / no teste).
</details>

---

<a name="4-regressao-logistica"></a>
## 4. Regressão Logística

### Intuição
Classificação binária: em vez de prever um número qualquer, prevemos uma **probabilidade** entre 0 e 1. Pegamos a combinação linear e a "esmagamos" no intervalo (0,1) com a função sigmoid.

### Matemática
```
z = wᵀx + b
p = sigmoid(z) = 1 / (1 + e^(−z))     → probabilidade da classe 1
```
- **x** — features; **w, b** — pesos e viés; **z** — o "score" linear.
- **p** — probabilidade prevista; decisão: classe 1 se `p ≥ 0.5`.

Perda: **Binary Cross-Entropy (BCE)** — penaliza confiança errada:
```
L = −(1/n) Σ_i [ y_i·log(p_i) + (1−y_i)·log(1−p_i) ]
```
- Se `y=1`, o termo `log(p)` empurra `p` pra cima; se `y=0`, `log(1−p)` empurra pra baixo.

**A beleza:** a derivada da BCE composta com a sigmoid se simplifica elegantemente:
```
∂L/∂w = (1/n) Σ_i (p_i − y_i)·x_i
```
O gradiente é só **(previsto − real) × entrada** — o mesmo formato da regressão linear! (Não é coincidência: ambas são *modelos lineares generalizados*.)

### No seu projeto (Projeto 4)
Implementada **do zero** em Rust. Atingiu **96,7% de acurácia no teste**. O ponto sofisticado: o erro de teste (3,33%) praticamente **encostou no limite de Bayes** (~3,57%, o erro irredutível dado o overlap das gaussianas) — ou seja, o gargalo é o **ruído do problema**, não o modelo.

### Pegadinhas
- A sigmoid satura: para `z` muito grande/negativo, `e^(−z)` dá overflow → implemente em **dois ramos** (z≥0 e z<0).
- O modelo é **linear**: a fronteira é uma reta. Dados não-lineares exigem features polinomiais ou MLP (§9).
- Sem regularização e com dados perfeitamente separáveis, os pesos **divergem** (vão pro infinito). L2 resolve.

### Exercícios
1. Por que usamos BCE e não MSE como perda na logística?
2. O que significa, geometricamente, `p = 0.5`?
3. A acurácia saltou pra 95% já na 1ª época mas a loss continuou caindo por 1000 épocas. Como ambas as coisas são verdade ao mesmo tempo?

<details><summary>Gabarito</summary>

1. Com a sigmoid, a MSE vira **não-convexa** (cheia de mínimos locais e regiões de gradiente quase nulo onde a sigmoid satura). A BCE é **convexa** para a logística e seu gradiente não satura quando o modelo está muito errado → treino mais estável e rápido. (E a BCE é a perda de máxima verossimilhança do modelo.)
2. É a **fronteira de decisão**: o conjunto de pontos onde `z = wᵀx+b = 0`. De um lado `p>0.5` (classe 1), do outro `p<0.5` (classe 0). É uma reta/hiperplano.
3. A **acurácia** é binária (limiar em 0.5): assim que `p` cruza 0.5 para o lado certo, já conta como acerto. A **loss** mede *confiança contínua*: continua melhorando enquanto `p` se aproxima de 0 ou 1, muito depois de a classificação já estar correta.
</details>

---

<a name="5-arvore"></a>
## 5. Árvore de Decisão: Gini vs Entropia

### Intuição
Uma sequência de perguntas sim/não ("a feature 1 é maior que 0.3?") que vão partindo o espaço em retângulos, até cada região ser "pura" (de uma classe só). A pergunta do projeto: qual critério usar pra escolher a melhor pergunta em cada nó?

### Matemática
A cada nó, escolhemos o **split** (feature + limiar) que mais reduz a **impureza**. Dois critérios:

```
Gini:     G = 1 − Σ_k p_k²
Entropia: H = − Σ_k p_k · log₂(p_k)
```
- **p_k** — proporção da classe k no nó.
- Nó **puro** (uma classe só) → G=0 e H=0. Nó 50/50 (binário) → G=0.5, H=1.0 (máxima impureza).

O **ganho** de um split é a impureza do pai menos a média ponderada da impureza dos filhos:
```
Ganho = Impureza(pai) − (|E|/|P|)·Imp(E) − (|D|/|P|)·Imp(D)
```
(E, D = filhos esquerdo/direito; P = pai). Escolhe-se gulosamente o split de maior ganho.

### No seu projeto (Projeto 5)
Árvore **construída manualmente** (sem `sklearn.tree`). Com Gini vs Entropia no mesmo dataset: **acurácias quase iguais** (Gini 96% / Entropy 94%), mas **estruturas diferentes** — a árvore da Entropia ficou mais profunda (13 vs 9 níveis). A explicação é geométrica: a entropia tem maior curvatura perto de p=0.5, o que muda a *ordem* dos splits gulosos. E confirmou **overfitting**: 100% no treino, ~95% no teste, com folhas de 1 amostra (memorização).

### Pegadinhas
- Sem `max_depth`, a árvore cresce até folhas puras → **decora** o treino (overfitting). Controle com `max_depth`, `min_samples_split`, poda (`ccp_alpha`).
- Gini vs Entropia quase nunca muda muito o desempenho — é uma decisão secundária.
- Métricas agregadas iguais **não** significam predições iguais: erros podem se compensar.

### Exercícios
1. Calcule o Gini de um nó com 8 exemplos: 6 da classe A e 2 da classe B.
2. Por que uma árvore sem profundidade máxima quase sempre overfita?
3. Duas árvores têm a mesma acurácia (95%) mas profundidades 9 e 14. Qual você prefere e por quê?

<details><summary>Gabarito</summary>

1. p_A = 6/8 = 0.75, p_B = 0.25. `G = 1 − (0.75² + 0.25²) = 1 − (0.5625 + 0.0625) = 1 − 0.625 = `**`0.375`**.
2. Porque ela continua criando splits até cada folha ser pura — inclusive folhas com 1 único exemplo. Isso **memoriza** o ruído específico do treino (pontos isolados), que não se repete no teste → gap treino/teste grande.
3. A de **profundidade 9** (mesma acurácia, menos complexa). Pela navalha de Occam, o modelo mais simples com igual desempenho generaliza melhor e é mais interpretável; os níveis extras da outra são supérfluos.
</details>

---

<a name="7-rbf"></a>
## 7. Redes RBF (Funções de Base Radial)

### Intuição
Espalhe alguns "morros" (gaussianas) pelo espaço, centrados em pontos-protótipo. Cada morro "acende" quando a entrada está perto do seu centro. A decisão final é uma **combinação ponderada** desses morros. É um classificador feito de bumps locais.

### Matemática
Camada oculta de gaussianas + saída linear:
```
φ_j(x) = exp( −‖x − c_j‖² / 2σ² )      (ativação do neurônio radial j)
ŷ = Σ_j w_j · φ_j(x) + b
```
- **x** — entrada; **c_j** — o **centro** (protótipo) do neurônio j; **σ** — a largura do morro.
- **φ_j(x)** — vale ≈1 se x está perto de c_j, →0 se longe.
- **w_j, b** — pesos da saída, resolvidos por **mínimos quadrados** (forma fechada, sem GD).

A decisão de projeto crítica é **onde colocar os centros**: aleatório, por **K-Means** (§1), ou subconjunto dos dados.

### No seu projeto (Projeto 7)
RBF **manual** sobre features CLIP. Ablação de centros: **K-Means vence** em todo k (F1 macro 0.93), amostragem aleatória fica perto, subconjunto fixo atrás. Detalhe importante: o **σ** precisa ser calibrado pela *escala dos dados* — você usou a **mediana das distâncias ponto→centro**, porque a heurística clássica `d_max/√(2k)` **colapsa em alta dimensão** (zera todas as ativações). Mais centros → melhor, com retorno decrescente.

### Pegadinhas
- **σ é tudo:** pequeno demais → morros viram picos isolados (tudo vira 0, modelo não aprende); grande demais → morros se fundem (perde resolução).
- Em alta dimensão, distâncias se concentram → escolha de σ adaptativa é essencial.
- Centros por K-Means conectam diretamente com o Projeto 1 (clustering ajuda a posicionar protótipos).

### Exercícios
1. O que acontece com `φ_j(x)` se σ → 0? E se σ → ∞?
2. Por que K-Means dá centros melhores que aleatório?
3. A saída da RBF é resolvida por mínimos quadrados, não por gradiente descendente. Por que isso é possível aqui (e não na logística)?

<details><summary>Gabarito</summary>

1. σ→0: cada morro vira um pico infinitamente estreito → `φ_j(x)≈0` para qualquer x que não seja exatamente o centro → ativações nulas, modelo inútil. σ→∞: `φ_j(x)≈1` para todo x → todos os morros "acesos" igualmente → perde poder de discriminar (tudo parece igual).
2. K-Means posiciona os centros onde os dados **realmente se concentram** (regiões densas), então os morros cobrem bem o espaço de entrada. Centros aleatórios podem cair em regiões vazias, desperdiçando neurônios.
3. Porque, **com os centros e σ fixos**, as ativações `φ_j(x)` são fixas e a saída `ŷ = Σ w_j φ_j + b` é **linear nos pesos w**. Minimizar erro quadrático de um modelo linear nos parâmetros tem solução fechada (pseudo-inversa). Na logística, a sigmoid torna a saída não-linear nos pesos → sem forma fechada, precisa de GD.
</details>

---

<a name="8-svm"></a>
## 8. SVM com kernel RBF

### Intuição
A SVM procura a fronteira que separa as classes com a **maior folga possível** (margem máxima) — a fronteira "mais segura". E o truque do **kernel** permite fronteiras curvas sem calcular coordenadas em alta dimensão: basta medir *similaridades* entre pares de pontos.

### Matemática
A decisão usa só alguns pontos de treino — os **vetores de suporte**:
```
K(x, xᵢ) = exp( −γ·‖x − xᵢ‖² )           (kernel RBF: similaridade)
f(x) = Σ_i α_i · y_i · K(x, xᵢ) + b
ŷ = sinal( f(x) )
```
- **x** — entrada; **xᵢ** — os **vetores de suporte** (pontos críticos, na margem); **y_i** — rótulo deles (±1).
- **K** — kernel: quão parecido x é de cada vetor de suporte.
- **γ** (gamma) — largura do kernel: **alto** = influência muito local; **baixo** = suave/global.
- **α_i** — peso de cada vetor de suporte; **C** — regularização: troca margem larga ↔ erros no treino.

### No seu projeto (Projeto 8)
SVM-RBF (sklearn) sobre CLIP, com varredura **C × γ**. F1 macro 0.95 — o **melhor** entre os modelos baseados em similaridade. Lição da varredura: **γ é o hiperparâmetro crítico** — γ alto (0.1) faz o kernel só "ver" o vizinho imediato → o modelo decora e o F1 **colapsa pra 0.09**; γ baixo/`scale` generaliza. O C, numa faixa ampla, mudou pouco. Os vetores de suporte se concentram na **zona de sobreposição** (a fronteira incerta).

### Pegadinhas
- SVM escala **O(N²)** com o número de amostras → inviável pra datasets enormes (você subamostrou pra 6.000).
- γ e σ da RBF são o "mesmo" parâmetro de largura, em formas recíprocas (γ ∝ 1/σ²).
- Só os vetores de suporte importam — remover os outros pontos não muda a fronteira.

### Exercícios
1. O que é um "vetor de suporte" e por que os outros pontos são irrelevantes?
2. γ muito alto leva a quê? E γ muito baixo?
3. RBF Network (§7) e SVM-RBF usam a mesma gaussiana. Qual a diferença conceitual de *onde* ficam os centros?

<details><summary>Gabarito</summary>

1. São os pontos de treino **na margem ou que a violam** — os mais difíceis/fronteiriços. A fronteira de margem máxima é determinada só por eles; pontos longe da fronteira não a influenciam, então podem ser removidos sem mudar `f(x)`.
2. γ alto → kernel estreito, cada ponto só influencia sua vizinhança imediata → fronteira super-flexível que **decora** (overfit), colapsando a generalização. γ baixo → kernel largo → fronteira quase linear, podendo **underfittar**.
3. Na **RBF Network** *você* escolhe os centros (aleatório/K-Means/subset). Na **SVM-RBF** os centros são **selecionados automaticamente** pelo treino — são exatamente os vetores de suporte (os pontos críticos). A SVM "descobre" quais protótipos importam.
</details>

---

<a name="9-mlp"></a>
## 9. Redes Neurais (MLP)

### Intuição
Uma regressão logística é **um** neurônio (uma fronteira reta). Empilhe vários neurônios em camadas e a rede passa a **dobrar e combinar** retas em fronteiras curvas arbitrárias. Cada neurônio oculto aprende uma reta; a camada de saída combina essas retas numa região não-linear.

### Matemática
Camadas de transformação linear + ativação não-linear:
```
camada: a = ativação(Wᵀ·entrada + b)     (ReLU/tanh nas ocultas, sigmoid na saída)
```
- **W, b** — pesos e viés de cada camada (aprendidos).
- **ativação** — a não-linearidade; sem ela, empilhar camadas colapsaria numa única transformação linear.

Treino por **backpropagation** = regra da cadeia aplicada camada a camada. O sinal de erro na saída é:
```
δ_saída = (p − y)        com cross-entropy
δ_saída = 2(p − y)·p(1−p)  com MSE
```
e propaga pra trás: `δ_camada = (δ_próxima · W) · ativação'`. O `(p−y)` é o mesmo da logística (§4) — o MLP é a logística "empilhada".

### No seu projeto (Projeto 9)
MLP **do zero** (numpy, backprop manual) sobre CLIP, comparando MSE vs cross-entropy e ablando arquitetura. Achados honestos:
- **MSE ≈ cross-entropy** neste problema (F1 0.962 vs 0.965; CE converge um pouco mais rápido).
- **0 camadas ocultas (= logística) já dá F1 0.956** → as features CLIP são *quase linearmente separáveis*, então a não-linearidade do MLP agrega pouco em alta dimensão. Ela importa mais no plano 2D comprimido (onde as classes se sobrepõem). Foi o **melhor** dos 3 classificadores na tarefa fácil (F1 macro 0.965).

### Pegadinhas
- Sem ativação não-linear, N camadas = 1 camada (o produto de matrizes é uma matriz).
- Mais neurônios ≠ sempre melhor: pode overfittar e o ganho satura.
- A inicialização dos pesos importa (He/Xavier) pra o gradiente não explodir/sumir.

### Exercícios
1. Por que uma rede sem ativações não-lineares é equivalente a um único neurônio linear?
2. No seu projeto, por que adicionar camadas ocultas quase não melhorou?
3. O `δ_saída = (p − y)` aparece tanto na logística quanto no MLP com cross-entropy. Por quê?

<details><summary>Gabarito</summary>

1. Compor transformações lineares dá outra transformação linear: `W₂(W₁x) = (W₂W₁)x = W'x`. Sem a não-linearidade entre as camadas, toda a rede colapsa numa única matriz `W'` — ou seja, um modelo linear, incapaz de fronteiras curvas.
2. Porque as features **CLIP já são quase linearmente separáveis** em 512 dimensões — a fronteira boa já é (quase) uma reta nesse espaço. A capacidade não-linear extra do MLP tem pouco o que fazer; ela só ajudaria se a relação fosse genuinamente não-linear naquele espaço.
3. Porque a saída usa sigmoid + cross-entropy nos dois casos, e a álgebra da regra da cadeia entre essas duas funções **cancela** os termos, sobrando exatamente `(p−y)`. O MLP só propaga esse mesmo sinal de erro pelas camadas anteriores.
</details>

---

<a name="10-cnn"></a>
## 10. Redes Convolucionais (CNN)

### Intuição
Um MLP trataria cada pixel como uma feature independente, ignorando que pixels *vizinhos* formam bordas, texturas, objetos. A CNN **desliza filtros pequenos** pela imagem, detectando padrões locais (uma borda, um canto) em qualquer posição. Camadas profundas combinam padrões simples em conceitos complexos.

### Matemática
A operação de **convolução** (um filtro varrendo a entrada):
```
S(i,j) = Σ_m Σ_n I(i+m, j+n) · K(m,n) + b
A = ReLU(S)  →  MaxPool  →  ... →  camada densa
```
- **I** — a imagem de entrada (altura × largura × **canais**; RGB=3, cinza=1).
- **K** — o **filtro/kernel** (K×K), pesos *aprendidos* que detectam um padrão local.
- **b** — viés; **S** — o **feature map** (resposta do filtro em cada posição i,j).
- **ReLU** — não-linearidade; **MaxPool** — reduz a resolução resumindo cada região (dá invariância a pequenos deslocamentos).

Os mesmos pesos do filtro são usados em toda a imagem (**weight sharing**) — por isso a CNN tem poucos parâmetros e detecta o padrão em qualquer lugar. Para sinais **1D** (séries temporais), o filtro desliza em 1 dimensão.

### No seu projeto (Projeto 10)
CNN (PyTorch) na tarefa real **foto vs screenshot** (imagens da sua coleção, baixadas do Drive). Resultados:
- **2D RGB (F1 0.92) > 2D cinza (0.89) > 1D (0.87)** — a estrutura espacial (2D) e o canal de cor agregam.
- Ablação: número de filtros e tamanho do kernel têm ponto ótimo (k=3 melhor; k=7 cai — filtro grande demais pra 64×64).
- **Feature maps:** os filtros do conv1 viram **detectores de borda** nas caixas de UI dos screenshots, e respondem a **texturas/gradientes** nas fotos. É essa diferença que a rede usa pra classificar.

### Pegadinhas
- Convolução explora **estrutura espacial** — colapsar a imagem num vetor (1D/MLP) joga isso fora.
- Filtro grande demais numa imagem pequena perde localidade (overfit/pior).
- Precisa de mais dados que modelos rasos; data augmentation e/ou transfer learning ajudam.

### Exercícios
1. Por que a CNN 2D superou a versão 1D (perfil de linha) na mesma tarefa?
2. O que é "weight sharing" e por que reduz o número de parâmetros?
3. Olhando os feature maps, como você *explicaria* que a rede aprendeu algo sensato?

<details><summary>Gabarito</summary>

1. A 2D **preserva a estrutura espacial** (vizinhança de pixels em duas dimensões), que é onde vivem bordas, formas e a distinção entre a UI retangular de um screenshot e a cena natural de uma foto. Colapsar pra um perfil 1D (média por linha) **descarta** a maior parte dessa informação espacial.
2. É usar o **mesmo filtro** (mesmos pesos) em todas as posições da imagem, em vez de pesos diferentes por pixel. Assim um filtro de borda com K×K pesos detecta bordas em qualquer lugar — drasticamente menos parâmetros que um MLP totalmente conectado, e ganha invariância a translação.
3. Mostrando que os filtros respondem a **padrões interpretáveis e coerentes com a classe**: nos screenshots eles acendem nas bordas retangulares da UI; nas fotos, nas texturas naturais. Se os mapas fossem ruído sem relação com a imagem, a rede teria "decorado" sem aprender estrutura.
</details>

---

<a name="visao-de-conjunto"></a>
## Visão de conjunto — a progressão

Os métodos não são isolados; eles formam uma escada:

```
K-Means (1)        descobre grupos sem rótulos  ─┐
                                                  ├─→ usa clusters como classes (2)
Regressão Linear (3)  ajustar + otimizar (GD/LMS)
        │ + sigmoid + BCE
Reg. Logística (4)    1 neurônio → fronteira reta
        │ empilha neurônios + backprop
MLP (9)               fronteiras não-lineares
        │ troca neurônios por convoluções
CNN (10)              explora estrutura espacial

Árvore (5)   particiona o espaço com perguntas (não-paramétrico)
RBF (7)      bumps gaussianos; centros que VOCÊ escolhe
SVM (8)      bumps gaussianos; centros que o treino ESCOLHE (margem máxima)
```

### Meta-ablação (mesmo problema, 3 classificadores)
Como você usou o **mesmo dataset e split** (features CLIP) em 7/8/9, dá pra comparar maçã-com-maçã (F1):

| Tarefa | RBF (7) | SVM (8) | MLP (9) |
|---|---|---|---|
| macro_vs_rest | 0.93 | 0.95 | **0.97** |
| has_people | 0.67 | **0.90** | 0.89 |
| screenshot×foto | 0.70 | **0.76** | 0.75 |

**Leitura:** o **MLP** lidera no problema fácil/separável; o **SVM** ganha nas tarefas desbalanceadas (a margem máxima lida melhor com classe rara); a **RBF manual** (poucos centros) fica atrás quando a classe positiva é pequena. Não existe "melhor modelo" universal — depende do problema (*No Free Lunch*).

---

### Conceitos que reaparecem (cole na geladeira)
- **Gradiente descendente** (3,4,9,10) — a receita de otimização.
- **`(p − y)`** — o sinal de erro que une logística e MLP.
- **Largura da gaussiana** σ/γ (7,8) — pequeno demais decora, grande demais borra.
- **Overfitting** (5,8,9,10) — treino ≫ teste; controle com regularização/profundidade/poda.
- **Baseline da classe majoritária** — toda acurácia precisa ser comparada a ele.
- **Data leakage** — ajuste normalização/seleção só no treino.

> Bons estudos. Cada seção tem o `slides.html` e o `ROTEIRO.md` do projeto correspondente como material complementar — este guia é a "teoria por trás".
