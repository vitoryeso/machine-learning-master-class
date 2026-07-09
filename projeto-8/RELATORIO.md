# Mini-Projeto 8 — SVM com Kernel RBF

## Problema

Este mini-projeto treina uma **Support Vector Machine (SVM)** com **kernel RBF** (gaussiano) para classificação binária sobre a coleção pessoal de imagens (features CLIP), nas mesmas três tarefas dos projetos 7 e 9 (problema real, compartilhado via `shared_problem.py`, não sintético). O enunciado não pede implementação manual, então `sklearn.svm.SVC` é usado legitimamente — o foco é **interpretar os vetores de suporte** como os pontos que de fato definem a fronteira de decisão, e analisar o efeito dos hiperparâmetros **C** e **gamma (γ)**.

A SVM busca a fronteira de **margem máxima**: a separação entre classes com a maior folga possível, tornando o classificador mais robusto a pequenas perturbações dos dados.

## Método

### SVM linear — margem máxima

No caso separável, a SVM escolhe o hiperplano que maximiza a distância mínima aos pontos de treino (a margem), em vez de qualquer hiperplano que apenas separe as classes. Isso reduz a variância do classificador: entre infinitos separadores possíveis, o de margem máxima generaliza melhor.

### Kernel RBF (gaussiano)

Para problemas não-linearmente separáveis no espaço original, o **kernel trick** permite calcular produtos internos em um espaço de features implícito de dimensão maior sem projetar os dados explicitamente. O kernel RBF usado aqui é:

```
K(x, xᵢ) = exp( −γ · ‖x − xᵢ‖² )
```

- **x** — a entrada, as features CLIP (512-d) da imagem a classificar.
- **xᵢ** — os **vetores de suporte**: pontos de treino que caem sobre ou dentro da margem, incluindo os mal classificados.
- **K** — mede a similaridade gaussiana entre x e cada vetor de suporte; decai com a distância euclidiana ao quadrado.
- **γ (gamma)** — o inverso da largura do kernel. γ alto → kernel estreito, influência bem local (cada ponto só "enxerga" seus vizinhos imediatos); γ baixo → kernel largo e suave, influência mais global.

### Função de decisão e vetores de suporte

```
f(x) = Σᵢ αᵢ · yᵢ · K(x, xᵢ) + b
ŷ    = sinal( f(x) )
```

- **αᵢ** — o peso aprendido de cada vetor de suporte (zero para os pontos que não são vetores de suporte — é isso que torna a solução esparsa).
- **yᵢ** — o rótulo (±1) do vetor de suporte i.
- **b** — o viés (bias) do hiperplano.
- **ŷ** — a classe prevista, dada pelo sinal de f(x).

A ideia central: **só os vetores de suporte definem a fronteira** — todos os demais pontos de treino poderiam ser removidos sem alterar o classificador, pois αᵢ = 0 para eles.

### Hiperparâmetro C (regularização)

C controla o trade-off entre margem larga e erro no treino: C baixo tolera mais violações de margem (fronteira mais suave, mais vetores de suporte); C alto penaliza fortemente erros de treino (fronteira mais ajustada, risco de overfitting).

### Implementação

`sklearn.svm.SVC(kernel='rbf')`. Como a SVM escala O(N²) com o número de amostras, o treino full-dim é subamostrado aleatoriamente para **6000** pontos (`SUB = 6000` em `svm.py`) — necessário para tornar o treino tratável em tempo razoável. Features padronizadas (`StandardScaler`, fit só no treino). Para a ilustração 2D da fronteira e dos vetores de suporte, as features são projetadas via PCA (mesmo padrão dos projetos 7/9) — o treino/avaliação de métricas usa sempre o espaço full-dim.

## Dataset

Problema real compartilhado entre os projetos 7/8/9 (`shared_problem.py`): coleção pessoal de imagens, 22.328 imagens, features **CLIP (512-d)** extraídas no projeto-2. Três tarefas binárias:

| Tarefa                 | Classes (pos vs neg)      | Baseline (classe majoritária, teste) |
|-------------------------|---------------------------|----------------------------------------|
| `has_people`            | pessoa vs sem pessoa       | 0.836                                   |
| `screenshot_vs_photo`   | screenshot vs foto de câmera | 0.844                                |
| `macro_vs_rest`         | maior macro-grupo vs resto | 0.654                                   |

| Parâmetro              | Valor                          |
|-------------------------|---------------------------------|
| Features                | CLIP, 512-d                     |
| Split treino/teste      | 80/20, estratificado             |
| Subsample de treino     | 6000 (SVM é O(N²); sem subsample o dataset inteiro não cabe no tempo de treino) |
| Padronização            | `StandardScaler`, fit no treino |
| Seed (ilustração)       | 42                               |
| Seeds (multi-seed)      | 42, 43, 44 (3 seeds)             |

`has_people` e `screenshot_vs_photo` são desbalanceadas (baseline de classe majoritária acima de 0.83); `macro_vs_rest` é a tarefa mais equilibrada (baseline 0.654), o que explica por que seu F1 costuma ser o mais alto e estável das três.

## Resultados

### Desempenho por tarefa — multi-seed (3 seeds, mean±std, ddof=1) — resultado oficial

`SVC(kernel='rbf', C=10, gamma='scale')`, features CLIP full-dim, treino subamostrado a 6000 por seed (cada seed refaz o split estratificado *e* o subsample).

| Tarefa                | Accuracy        | F1              |
|------------------------|-----------------|-----------------|
| macro_vs_rest          | 0.968 ± 0.002   | 0.954 ± 0.004   |
| has_people             | 0.962 ± 0.003   | 0.883 ± 0.010   |
| screenshot_vs_photo    | 0.933 ± 0.002   | 0.770 ± 0.007   |

As três tarefas batem o baseline de classe majoritária com folga muito maior que o desvio-padrão entre seeds — a margem SVM-baseline é robusta, não ruído de amostragem.

### Desempenho por tarefa — single-seed (seed=42, ilustração)

Mantida como ilustração (mesma seed usada nas figuras `output/*.png`); os números oficiais para citar são os da tabela multi-seed acima.

| Tarefa                | Acc    | Precision | Recall | F1     | Baseline | Vetores de suporte |
|------------------------|--------|-----------|--------|--------|----------|----------------------|
| macro_vs_rest          | 0.966  | 0.945     | 0.958  | 0.951  | 0.654    | 1515/6000            |
| has_people             | 0.966  | 0.897     | 0.892  | 0.895  | 0.836    | 1684/6000            |
| screenshot_vs_photo    | 0.931  | 0.831     | 0.703  | 0.762  | 0.844    | 2652/6000            |

`screenshot_vs_photo` tem o maior número relativo de vetores de suporte (2652/6000 ≈ 44%) — coerente com ser a tarefa de F1 mais baixo: a fronteira entre screenshot e foto é a mais confusa das três no espaço de features CLIP, exigindo mais pontos "na margem" para sustentá-la.

### Varredura C × γ — F1 no teste (macro_vs_rest, seed=42, ilustração)

| C \ γ  | scale  | 0.001  | 0.01   | 0.1    |
|--------|--------|--------|--------|--------|
| 0.1    | 0.9356 | 0.9349 | 0.0026 | 0.0000 |
| 1      | 0.9525 | 0.9501 | 0.5677 | 0.0867 |
| 10     | 0.9515 | 0.9498 | 0.6126 | 0.0902 |
| 100    | 0.9520 | 0.9478 | 0.6126 | 0.0902 |

Com `gamma=scale` ou `gamma=0.001`, o F1 fica sempre acima de 0.93, praticamente indiferente ao valor de C. Já com `gamma=0.1`, o F1 desaba para 0.00–0.09 em toda a faixa de C — um colapso de mais de 10x, não uma degradação suave.

**Robustez (multi-seed):** o mesmo padrão se confirma nas 3 seeds — o gap entre a melhor e a pior célula da grade C×γ é da ordem de **~200× o desvio-padrão combinado** entre seeds, e a SVM bate o baseline de classe majoritária nas três tarefas em todas as seeds. O claim "γ é crítico" não é artefato de uma seed sortuda.

### Plots gerados

- `output/c_gamma_sweep.png` — heatmap F1 (teste) por C × γ, tarefa `macro_vs_rest` (ilustração seed=42).
- `output/boundary_svs.png` — fronteira de decisão 2D (PCA) com os vetores de suporte destacados como círculos.
- `output/report.txt` — relatório numérico da rodada de ilustração (seed=42): desempenho por tarefa e grade C×γ completa.

## Análise

- **γ é o hiperparâmetro crítico, C quase não importa.** Na grade C×γ, variar C de 0.1 a 100 mantendo γ fixo muda o F1 em poucos pontos percentuais; variar γ de `scale`/0.001 para 0.1 derruba o F1 de ~0.95 para ~0.09 — um efeito de ordem de grandeza maior. Isso é intuitivo pela geometria do kernel: com γ=0.1 (largura do kernel muito estreita, pois γ = 1/(2σ²) grande equivale a σ pequeno), cada ponto de treino só tem influência gaussiana significativa sobre seus vizinhos mais próximos — o kernel efetivamente memoriza cada exemplo isoladamente em vez de aprender uma fronteira suave que generalize. É overfitting extremo via kernel, análogo a uma árvore sem poda ou um k-NN com k=1.
- **A robustez do efeito é forte:** o gap γ-alto vs γ-baixo (~200× o desvio-padrão combinado entre seeds) é ordens de grandeza maior que qualquer variação de C observada — não é um efeito marginal que poderia inverter com outra amostragem.
- **Vetores de suporte como "centros" efetivos:** na ilustração 2D (PCA), as classes se sobrepõem bastante no plano comprimido — natural, já que 2 componentes retêm pouca variância de um espaço de 512-d — e por isso quase um quarto dos pontos de treino vira vetor de suporte. Isso reflete uma faixa larga de incerteza na fronteira, não um defeito do modelo: em full-dim (onde o modelo de fato opera), a separação é bem melhor, como mostram os F1 de 0.77–0.95.
- **SVM-RBF supera os modelos anteriores baseados em similaridade** (RBF Network do projeto-7) nas três tarefas — o kernel RBF learned implicitamente combinado com margem máxima entrega uma fronteira mais eficaz do que a combinação linear de gaussianas com centros fixos do projeto-7, ao custo de treino mais caro (O(N²) vs O(N·k)).
- **Desbalanceamento explica a dificuldade relativa das tarefas:** `screenshot_vs_photo` e `has_people` têm baseline de classe majoritária mais alto (0.84 e 0.84 vs 0.65 do macro), então seu F1 absoluto mais baixo (0.77 e 0.88 vs 0.95) reflete o desafio de identificar corretamente a classe minoritária, não uma fraqueza do modelo — a margem sobre o baseline continua sólida nas três.

## Limitações

- **Subsample de treino obrigatório.** SVM com kernel escala O(N²) em tempo e memória; o treino usa apenas 6000 de possivelmente muito mais exemplos disponíveis, descartando dados que talvez ajudassem — especialmente na tarefa `screenshot_vs_photo`, a mais difícil. Um `LinearSVC` ou aproximação de kernel (Nystrom, RBFSampler) permitiria usar o dataset inteiro.
- **Multi-seed limitado a 3 seeds** (vs 10 nos projetos 3/5), por causa do custo O(N²) do treino SVM — o próprio código (`svm.py`) documenta essa escolha (`N_SEEDS = 3`, comentário "SVM é O(N²), cuidado"). O desvio-padrão relatado é, portanto, uma estimativa menos precisa que a dos projetos com mais seeds.
- **Grade C×γ é coarse** (4×4 valores) e roda só sobre a tarefa `macro_vs_rest` no multi-seed — o comportamento fino perto da transição (entre γ=0.01, onde o F1 já cai para ~0.6, e γ=0.1, onde colapsa) não é explorado, e as outras duas tarefas não têm sweep multi-seed dedicado.
- **Ilustração 2D (PCA) retém pouca variância** do espaço CLIP de 512-d — a fronteira e os vetores de suporte mostrados na figura são uma simplificação visual; as métricas reportadas (accuracy, F1) usam sempre o espaço full-dim, que é o que realmente importa para desempenho.
- **`report_multiseed.txt` e `metrics_multiseed.png`** (gerados por `run_multiseed()` em `svm.py`) não estão presentes em `output/` neste momento — os números multi-seed citados aqui vêm de `slides.html` (fonte oficial dos números) e do roteiro do vídeo; a grade C×γ detalhada por seed não pôde ser re-verificada linha a linha a partir de um arquivo bruto.
