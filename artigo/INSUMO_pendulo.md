# INSUMO — Projeto Pêndulo-N com PPO (PufferLib)

> **Documento de insumo bruto para artigo científico.** Mineração exaustiva do banco de
> prompts do Vitor (`prompt_research.messages`, Postgres yalien) — histórico de todas as
> sessões de agente (Claude Code / Codex) que tocaram o projeto `cartpole_pendulum_n` /
> N-pêndulo em PufferLib.
>
> Convenção: **[FATO]** = resultado/dado observado no treino ou benchmark. **[HIPÓTESE/DEBATE]**
> = raciocínio, especulação, decisão de design, plano não confirmado ou beco-sem-saída.
> Datas em **BRT (UTC-3)**. Fonte = grupo (claude/codex) + sessão (source_path).
>
> Gerado 2026-07-08. As seções estão organizadas pelos 5 temas pedidos. Priorizado o que é
> ÚNICO do banco de prompts: os DEBATES e o RACIOCÍNIO (o "porquê/como pensamos"), já que o
> lado código/resultado foi coberto em paralelo.

---

## Índice de sessões-fonte (mapa cronológico)

| # | Data (BRT) | Grupo | Sessão (id curto) | Assunto |
|---|---|---|---|---|
| S0 | 2026-03-13 | codex | `019ce827…` (yalien) | Gênese: primeiro contato com PufferLib, "hello world" de env |
| S1 | 2026-06-10→12 | claude | `6a5b7364…` (work) | Double-pendulum swing-up: CPU→CUDA, MinGRU, reward shaping inicial |
| S2 | 2026-06-12 | claude | `bdd760a4…` | SPS NumPy vs C; decisão de portar env pra C |
| S3 | 2026-06-12→13 | claude | `53a9f012…` (work) | N=2, bugs RNN stale, redução do modelo, NF4 QAT |
| S4 | 2026-06-13 | claude | `c1c4538a…` (work) | Continuação treino N=2 |
| S5 | 2026-06-17→18 | claude | `cbe0bce6…` (work) | (batch B — pendente) |
| S6 | 2026-06-18→26 | claude | `c35637cf…` (work) | Gravity-transfer, curriculum, quad-pendulum, CUDA loop kernelizer |
| S7 | 2026-06-20 | claude | `84600e77…` | Bug: training morre junto com subprocess `claude` (yeclaudebot) |
| S8 | 2026-06-22→30 | claude | `b256985f…` (dl_model_compression) | Debate de enquadramento do paper + achado gravity-transfer |
| S9 | 2026-06-26→28 | claude | `5832f132…` | Saga bf16 / otimização CUDA / n=5 from-scratch |
| S10 | 2026-06-28→07-05 | claude | `96cf0ef9…` | POMDP "olhos fechados" LSTM + N=2 blind + n=5→1T/2T |

*(Batches A e B ainda em processamento; suas seções serão anexadas ao fim.)*

---

# TEMA 1 — O AMBIENTE / O PROBLEMA

## 1.1 Física do pêndulo-N (integrador Lagrangiano)

- **[FATO]** (S3, 2026-06-12) O env é um **carrinho (cart) com N pêndulos acoplados em série**.
  Estado raw exposto pelo C via `env_get_states` para N=2 é `[n, 6]`: `x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂`.
  A física (≈228 linhas de C) inclui um **integrador de Lagrangiana 3×3** para 2 pêndulos
  (função `solve3`), `xorshift32` para random, `wrap_π`, tudo estático/inline.
- **[FATO]** (S3) A shared library C expõe: `env_create(n, seed)`, `env_reset_all(obs)`,
  `env_step(actions, obs, rewards, dones, truncs)` (step vetorizado), `env_get_states(out)`,
  `env_set_state(i, ...)` (debug/vídeo), `env_destroy()`. **O C é só o env** — o training loop
  (PPO, RNN, NF4 QAT) é 100% Python. "Não tem training em C — faz sentido, seria pesado demais
  reimplementar backprop."
- **[FATO]** (S9, 2026-06-27 12:26–12:31) **A matriz de massa do pêndulo-N tem estrutura regular
  e generalizável em N**:
  - `A[0][0] = CART + n·m`
  - `A[0][k] = (n−k+1)·m·l·cos(θ_k)`
  - `A[j][j] = (n−j+1)·m·l²`
  - `A[j][k] = (n−max(j,k)+1)·m·l²·cos(θ_j − θ_k)`
  - com massa efetiva `M_k = (N−k+1)·m`.
  O env quad original (`quadruple_pendulum_cuda.cu`) era N=4 "desenrolado à mão" (eliminação
  Gaussiana manual 4×4, `OBS_SIZE=14` fixo). Foi generalizado para **N arbitrário** em
  `npendulum_cuda_wrapper.py` — kernel CuPy `RawKernel` com loops na montagem da matriz +
  eliminação Gaussiana n×n.
- **[FATO]** (S9) Validação de paridade do kernel genérico contra o quad hardcoded:
  diff máxima **8.8e-7** (ruído fp32 puro) no 1º passo, ~1e-6 mantido em 6 passos → "física correta".
  (Em S8 há menção a "parity vs NumPy a 5e-7".)

## 1.2 Crescimento do espaço de estado com N (o eixo de dificuldade)

- **[FATO]** (S10, 2026-06-28 22:57) Regra explícita: **DOF = n+1** (carro + n ângulos) →
  **estado = 2(n+1) = 2n+2**; **observação = 2 + 3n** (cada ângulo entra como `sinθ, cosθ, θ̇`
  em vez de θ cru):

  | N | estado | observação |
  |---|---|---|
  | n=1 | 4D `[x,ẋ,θ₁,θ̇₁]` | 5D |
  | n=2 | 6D | 8D |
  | n=3 | 8D | 11D |
  | n=4 | 10D | 14D |
  | n=5 | 12D | 17D |
  | n=6 | 14D | 20D |

- **[HIPÓTESE/pedagógico]** (S10) O uso de `sin/cos` no lugar de θ cru **evita a descontinuidade
  circular** (θ=0 e θ=2π são vizinhos fisicamente mas distantes numericamente) — "jeito padrão
  de dar ângulo pra rede neural".
- **[HIPÓTESE/design de hardware]** (S10, 2026-06-28 22:55) Mapeamento pra um sistema real:
  cada junta passiva teria **encoder rotativo** (mede θ; θ̇ por diferenciação temporal), o carro
  teria encoder linear + motor. O sistema é **sub-atuado (underactuated)**: **1 motor (o carro)
  para n+1 DOF** — as juntas dos elos são todas passivas/livres, giram só por gravidade+inércia.
  "É por isso que swing-up de pêndulo múltiplo é benchmark de controle: domar muita coisa
  caótica mexendo em UMA só [variável de controle]."

## 1.3 Observação e ação

- **[FATO]** Observação normalizada: `obs = [x/X, ẋ/V, (sinθ, cosθ, θ̇)×n]` = 2+3n dims
  (S8, prompt POMDP; S10). Ação **discreta** (esquerda/direita/nada — S1 usa ação discreta e
  discute a dificuldade dela para swing-up).
- **[FATO]** (S9) SPS do env **sozinho** (sem policy), kernel CuPy genérico: n=4: 333M SPS ·
  **n=5: 260M SPS** (obs=17, state=12) · n=6: 212M SPS (obs=20, state=14) — todos sem NaN.
  "n=5 só ~22% mais lento que n=4 apesar do solver 6×6 vs 5×5."

## 1.4 Design da RECOMPENSA — swingup vs balance, e o "center-farming"

Este é um dos fios centrais do projeto e o mais rico em debate.

- **[FATO]** (S1, 2026-06-10→12, double pendulum) Primeiras tentativas de reward para swing-up
  com **ação discreta** foram muito difíceis:
  - Reward multiplicativo `up1*up2` — "quando ambos estão em baixo, reward≈0, sem gradiente pra
    progresso parcial" (local ótimo de reward ~5.3).
  - Fórmula que aparece no código: `(0.4*up1 + 0.4*up2 + 0.2*up1*up2) * centered − 0.001*(θ̇1²+θ̇2²)`.
  - **[HIPÓTESE]** "swing-up discreto é muito hard"; o agente "ganha reward balançando mas não
    estabiliza no topo" — reward subiu a 231.8 mas métrica "upright" (|θ|<0.3 rad = 17°) ficou 0%.
- **[FATO]** (S10, 2026-06-28 23:16) **Descoberta do "center-farming" (reward hacking flagrado
  em ato).** Reward original do env: `(0.5·height + hold)·center`, onde `center = 1−(x/X)²`.
  Um agente **cego** (só via `x, ẋ`) parou o carro **dead-center** e bancou `center≈1` toda vez
  enquanto o pêndulo ficava pendurado — gerando **episode-return alto (212)**, *maior* que o
  controle full-obs (66), **sem resolver o swing-up**.
- **[FATO]** (S10) Introdução de **métrica honesta**: `cosθ` e `%up` (cos θ > 0.6) lidos do
  **estado físico real** do env, não do reward proxy:

  | config | obs | cosθ | %up | \|x\| | veredito |
  |---|---|---|---|---|---|
  | full_mlp | 5D | −0.086 | 26.9% | 2.22 | genuinamente sobe o pêndulo |
  | partial_mlp | (x,ẋ) | −0.392 | 12.3% | **1.07** (menor) | "center-farm" |

- **[HIPÓTESE→FATO]** (S10, 2026-06-28 23:17) "Reward-hacking caught in the act: o agente
  otimizou o retorno, achou o termo `center` como caminho barato, e abandonou a tarefa difícil."
- **[FATO]** (S10) **Correção v1**: reward **swing-up de-confundido** — calculado no wrapper
  `SlicedEnv` a partir de `final_obs` (pré-reset, todo step), **puramente height, sem termo
  center**, sem tocar o kernel CUDA n=5 compartilhado. Após a correção: `partial_mlp` return
  caiu de 212 → ~80 ("centering-farm removido, retornos agora rastreiam altura real").
- **[FATO]** (S10, 2026-07-01 10:09/10:11) **Segundo problema de reward**, descoberto quando o
  Vitor **assistiu ao vídeo** do checkpoint 245B: pole upright ~92% em ambos os agentes, mas o
  cego tinha **|ẋ| do carro 4× maior** e **|θ̇| (wobble) 2.5× maior** que o full-obs.
  "A métrica %up>0.6 escondeu isso (0.6 = 53° off vertical — uma barra baixa)."
  - **[HIPÓTESE]** Duas causas: (1) reward só paga por altura, nunca por quietude → zero
    gradiente pra calma; (2) parte é **inerente à cegueira** — active sensing / dual control:
    o agente precisa manter o carro se mexendo pra "sentir" o pêndulo invisível; ficar parado
    degrada a estimativa de θ.
  - **[FATO]** **Reward v2** (2026-07-01 12:43): `reward = height + up_weight·½·(stillness +
    centering)`, com `up_weight ≈ cosθ` (**zero quando o pêndulo está pendurado** → gated, não
    farmável — pêndulo pendurado parado no centro ganha zero). `stillness = 1 − 2·(|ẋ|/V +
    mean|θ̇|/W)`; `centering = 1 − |x|/X_threshold`.
  - **[HIPÓTESE]** "Um pouco de jitter residual é inevitável para um agente cego — sensoriamento
    ativo. Reward v2 mata a agitação gratuita, mas eyes-closed pode nunca ficar tão parado
    quanto full-obs. **Propriedade genuína do POMDP, não bug.**"

## 1.5 Gravidade como eixo de dificuldade (design do ambiente)

- **[FATO]** (S8, 2026-06-22) Env com **envs próprios N=1..4** (depois estendido até N=8, ver
  1.6), **curriculum por escala de gravidade**, transfer zero-shot, 50B+ steps, checkpoints.
- **[HIPÓTESE/DEBATE]** (S8, 2026-06-22 21:00) **"Gravidade como knob de projeto"** + o truque
  técnico: reescalar **`dt ∝ 1/√g`** (e força/max_steps junto) pra manter o tempo de episódio
  constante entre regimes de gravidade. "Nenhum pêndulo padrão expõe isso." (Framing pro paper.)

## 1.6 Cobertura de N

- **[FATO/relatado]** (S8, 2026-06-22 21:01) Vitor: "eu tava desafiado em fazer o n pendulo até
  uns 8x kkkk" — sugere que N chegou a ser testado **até N=8** (não só N=1..4). Reação:
  "N=8 fortalece o ângulo (eixo N de 1 a 8 = dificuldade escalável de verdade)."

---

# TEMA 2 — PPO E TREINO

## 2.1 Arquitetura da rede

- **[FATO]** (S3, 2026-06-12) Arquitetura **PufferNet**: `Linear → RNN → Linear` (value fundido
  no decoder), **H=48, 1 layer (~5.1k params)**. Era H=128/2L antes; foi **reduzido pra H=48/1L**
  nessa sessão. Init **ortogonal em todas as matrizes**.
- **[FATO]** (S1) Estilo MinGRU bias-free: `Linear encoder (no bias) → MinGRU (1-2 layers, cada
  um só um `Linear(H→3H, no bias)`) → Linear decoder (actions+value fundidos, no bias)`.
  Exemplo concreto: **6.496 params, todos bias-free**, hidden=32. "pura matrix, zero bias".
- **[FATO]** (S1) Modelo minúsculo (hidden=4, GRU) = **108 params, 0.4 KB** (4 matrizes) — **não
  aprendeu** ("4 floats de memória não é suficiente pro double pendulum caótico").

## 2.2 Hiperparâmetros

- **[FATO]** (S3, 2026-06-12) Config default: `TOTAL_STEPS=50M`, `NUM_ENVS=1024`, `LR=5e-4`,
  `GAMMA=0.99`.
- **[FATO]** (S8, prompt POMDP) `MAX_STEPS=1500`, `HORIZON=128` (janela de BPTT no update).
- **[FATO]** (S1) Descoberta empírica sobre `NUM_ENVS`: com **64 envs × 2048 steps = 131k
  transitions/rollout**, completavam ~1000 eps em só **2 updates de PPO** → aprendizado pobre.
  Reduzir o rollout (mais updates por step) melhorou muito o aprendizado, ao custo de SPS.

## 2.3 Curvas e taxa de sucesso por N

- **[FATO]** (S1, double pendulum 0.367kg) Sequência de reward: −0.6 → 231.8 (mas 0% upright) →
  com correções, **255 (avg50), 76-77% upright** nos melhores eps. Depois, com segmented scan +
  10M steps: **reward 132, todos os 500 steps sobrevivem, eval até 195**.
- **[FATO]** (S9, quad N=4) Curva 0→500B completa: 0.1B ~42 (quase aleatório) → 10B ~440 (salto
  from-scratch) → **platô longo 10–230B em ~450–600** → **bf16 ON em 230B** → subida final até
  **rew@eval ~1045** (max 1838) em 500B. "O platô foi o que te fez pedir LR schedule e episódios
  mais longos" — estender **MAX_STEPS pra 1500** foi o que "quebrou" o platô.
- **[FATO]** (S9, n=5 from-scratch, obs=17) 100M: rew@eval 61 → 200M: **347** → trajetória
  200B: 100M:61 → 20B:435 → 80B:454 → 137B:463 → **200B: 468 (platô firme, "CONCLUÍDO")**.
- **[HIPÓTESE]** (S9) "**n=5 é genuinamente mais difícil que n=4**: n=4 chegou a ~1045 em 500B;
  n=5 platôou em ~468 em 200B. 5 elos caóticos acoplados = espaço de controle muito mais duro."
- **[HIPÓTESE]** (S9) Interpretação do platô n=5: "Reward por step clamped [0,1.5], teto de
  episódio ~2250. n=5 em ~468 = ~21% do teto (n=4 chegou a 46%). Episódios duram os 1500 steps
  inteiros (não caem) → a política **sobrevive mas não otimiza** — platô de otimização/capacidade,
  não teto de reward." Recomendações: rede maior (512 hidden), reward shaping, ou transfer do n=4.

## 2.4 Warm-restart de LR (SGDR) para destravar platô — n=5 → 1T → 2T

- **[FATO]** (S9, 2026-06-28 21:24) LR bump simples (200B→235B, LR 2e-4) **NÃO funcionou** —
  35B estável em ~466, "sem breakthrough nem instabilidade".
- **[FATO]** (S10, 2026-07-02/03) n=5 estendido a **1T e 2T via warm-restart de LR** (a cada
  extensão LR volta a 1.5e-4 e re-anela):

  | marco | rew@eval |
  |---|---|
  | 500B (pré-extensão) | 530 |
  | ~505B (dip pós warm-restart) | ~514 |
  | 861B | 606 |
  | **1T (final)** | **612.6** (max 614.1 @990B) — **+15%** sobre o platô 530 |
  | 1T→2T dip | 612 → ~550 |
  | 2T (após reboot que perdeu ~27h, chegou a 1405B) | ~575, ainda < 612 |

- **[HIPÓTESE→confirmada]** (S10, 2026-07-03 20:48) "Warm-restart é textbook cyclic-LR/SGDR:
  cada bump escapa a bacia atual pra uma melhor, depois re-converge e achata — com **retornos
  decrescentes por ciclo** (530→610 foi +80; o próximo bump dá menos)." A 2ª extensão confirmou
  (~575 vs 612 esperado) — "diminishing returns confirmado".

## 2.5 Bugs de treino encontrados e corrigidos

- **[FATO]** (S3, 2026-06-12) **Hidden states da RNN estavam stale entre rollouts** — o hidden
  não era carregado corretamente, gerando **ratio de importância errado** no PPO. Corrigido.
- **[FATO]** (S3) **Truncation vs termination**: bootstrap correto no time-limit (não tratar
  truncação como término real). Corrigido.
- **[FATO]** (S1) **Parallel scan ignorando episode boundaries**: o Heinsen scan do MinGRU
  tratava o rollout como sequência contínua, **corrompendo gradientes** quando episódios resetam
  no meio do rollout. Fix 1: loop `forward_step` sequencial (correto mas 4× mais lento: 4.5K vs
  18K SPS). Fix 2: **segmented scan** que reseta o estado nos `done` boundaries (8.9K SPS).
- **[FATO]** (S10) **CUDA graph sobre RNG**: teste decisivo confirmou "RNG advances across graph
  replays: True" — o replay **não congela** o ruído estocástico do Gumbel (exploração intacta).
- **[FATO]** (S10) `USE_BF16` no MinGRU/LSTM **quebra**: hidden persistente é fp32, ativações
  autocast são bf16, `torch.lerp` rejeita dtypes mistos → **recorrentes rodam em fp32 sempre**
  (só o MLP fica em bf16).
- **[FATO]** (S10) Bugs de infra: watchdog se autoexterminava por regex casando string obsoleta
  ("500B CONCLUÍDO"); string-sort de checkpoints (`n5_99900M.pt` ordenava depois de
  `n5_500000M.pt`); parser de scatter cortava dígito do nome do checkpoint (cobria 0-50B rotulado
  como 0-500B). Todos corrigidos.
- **[FATO]** (S7, 2026-06-20) Bug de orquestração (yeclaudebot): **o training era filho do
  subprocess `claude`; quando `claude` terminava, o Windows matava o treino junto** (Job Object
  herdado). Fase [0→100M] funcionava (60s < 145s do claude), [100→200M] morria. Fix: lançar via
  `Start-Process` do PowerShell (desacopla de verdade no Windows).

---

# TEMA 3 — MODO "BLIND" / POMDP (n=1, observação parcial)

O experimento "vassoura de olhos fechados": treinar um cart-pendulum **n=1 que observa SÓ
`(x, ẋ)`** — sem ver o ângulo do pêndulo — com política **recorrente** que infere o estado
oculto do histórico. Rodou em paralelo ao n=5. (Prompt de partida: S8, 2026-06-28 23:05.)

## 3.1 Design do experimento e controles

- **[FATO/plano]** (S8, prompt) Estrutura incremental: (1) baseline n=1 obs completa (5D) + MLP
  → deve aprender swing-up trivial; (2) POMDP: reduzir obs a `(x,ẋ)` (fatiar `obs[:, :2]`) + MLP
  → **deve falhar** (controle negativo, sem memória `(x,ẋ)` instantâneo é ambíguo); (3) trocar
  MLP por `pm.MinGRU`/LSTM (política recorrente) → ver se aprende a **inferir o pêndulo invisível
  do histórico**. Critério de sucesso: RNN-em-(x,ẋ) equilibra e MLP-em-(x,ẋ) não.
- **[FATO/detalhe técnico]** (S8, prompt) **Reset do hidden state**: quando um env termina
  (done/trunc), o hidden da RNN PRECISA zerar pra aquele env (MLP tem `state=()`, nada a resetar).
  Com MAX_STEPS=1500 e HORIZON=128, terminação dentro do horizonte é rara — "pode começar
  ignorando reset intra-horizonte e refinar depois".

## 3.2 Resultado inesperado: o controle negativo NÃO falhou (por center-farming)

- **[FATO]** (S10, 2026-06-28) Stage 1 (full-obs 5D + MLP): rew@eval sobe a **66**.
  Stage 2 (obs parcial + MLP): **NÃO falhou como esperado** — subiu a **212 de return**, porque o
  episode-return mentiu (center-farming, ver §1.4).
- **[FATO]** (S10) MinGRU vs MLP com reward original (com center): o RNN **NÃO bateu o memoryless**:

  | config | vê θ? | memória | cosθ | %up |
  |---|---|---|---|---|
  | full_mlp | ✅ | ❌ | −0.085 | 26.9% |
  | partial_mlp | ❌ | ❌ | −0.397 | 12.1% |
  | partial_mingru | ❌ | MinGRU | −0.452 | **10.1% (pior que MLP)** |

- **[HIPÓTESE→confirmada]** (S10) "O RNN não bateu o memoryless — **ambos os cegos convergiram
  pro mesmo escape hatch (center-farming)**. A RNN gastou capacidade de memória aperfeiçoando a
  centralização, não inferindo θ, porque esse gradiente é mais fácil de subir. **O experimento
  como desenhado não conseguia testar a hipótese.**"

## 3.3 Após de-confundir o reward: o platô do RNN (o "~11%")

- **[FATO]** (S10, 2026-06-29 00:04) Com reward swing-up puro, 200M steps, 5 seeds:

  | config | vê θ? | memória | cosθ | %up |
  |---|---|---|---|---|
  | full_mlp | ✅ | none | **−0.06** | **26.7%** |
  | partial_mlp | ❌ | none | −0.41 | 12.0% |
  | partial_mingru | ❌ | MinGRU | −0.41 | **11.6% (= baseline)** |
  | partial_mingru_mlp (sandwich) | ❌ | MinGRU+MLP | −0.45 | 10.3% (= baseline) |

  → Este é o **"~11% up vs 26.7% full-obs"** citado na memória do Vitor. O RNN estava colado no
  baseline memoryless — não estava usando a memória.

- **[HIPÓTESE, suspeito nº1] BPTT-from-zero.** (S10) `Policy.forward` no update roda
  `forward_train` da RNN **sempre a partir de h=0** para cada segmento de 128 steps; mas durante
  o rollout o hidden é carregado ao longo de todo o episódio (até 1500 steps). "A política é
  treinada sobre uma **trajetória de crença fria (cold-start)** que difere da **trajetória quente
  (warm)** que gerou as ações — **corrompendo o credit assignment**." Este é o mecanismo que
  explica o platô ~11%.

## 3.4 O fix: TBPTT stateful + LSTM + long cook → recuperação total

- **[FATO]** (S10) Implementado: hidden `(h,c)` capturado no início de cada janela de 128 steps
  e alimentado como estado inicial do `nn.LSTM` no update — o gradiente vê a mesma crença "quente"
  que gerou as ações. Rollout mudado de `inference_mode` para `no_grad` (pra reusar o hidden no
  grafo autograd).
- **[FATO]** (S10) **Curva completa do "eyes-closed broom" (n=1 cego, só x,ẋ, LSTM stateful):**

  | steps | %up | nota |
  |---|---|---|
  | 3B | ~10.8% | platô memoryless |
  | 9B | 25.3% | **iguala o full-obs MLP** |
  | 18–21B | 26.4–38.7% | passa o baseline |
  | 27B | **7.3%** | colapso |
  | 37.0B | 47.9–50% | high-water (5-seed: +0.28 cosθ) |
  | 37.2B | 6.6% | colapso de novo |
  | 40B | 52.6% | |
  | 60B | 59.6–60% | |
  | 80B | 64.3% | |
  | 100B | 75.5% | |
  | 120B | 87.2% | |
  | 130B | 89.3% | |
  | **134B** | **94.6%** | pico single-seed (robusto 5-seed ~83%) |
  | **245B** | **96.8% up** | pico oficial (vídeo final) — ≈ full-obs MLP (~97%) |

- **[FATO/conclusão N=1]** (S10) **"Memória recupera totalmente a observabilidade."** Full-obs
  resolve em ~6B (97%); eyes-closed resolve em ~135–245B (~90–97%). **O custo da cegueira é
  eficiência amostral (~20–40×), não teto de performance.**
- **[FATO — nuance importante]** (S10) A comparação inicial "full-obs 27% vs eyes-closed" tinha
  sido feita contra um **controle full-obs fraco** (1 layer / metade dos envs). O full-obs bem
  dimensionado (2 layers, 16k envs) atinge **97% em <1B steps**. Ou seja, o número honesto é
  full-obs 97% (rápido) vs eyes-closed 96.8% (245B) — **teto igual, custo amostral diferente**.
- **[FATO/qualidade]** (S10) Mesmo com cosθ quase idêntico (0.89 vs 0.95, up 92% vs 97%), o cego
  tem **|ẋ| 4× maior** e **|θ̇| 2.5× maior** — capturado só ao **assistir ao vídeo**, não pela
  métrica escalar.

## 3.5 N=2 blinded (double pendulum, olhos fechados)

- **[FATO]** (S10, lançado 2026-07-01, cozinhando até ≥07-05) LSTM hidden **256** (vs 128 do
  n=1), **528K params**, reward v2 (swing-up + stillness + centering gated). Trajetória
  (tip-up% médio dos 2 links):

  | steps | tip-up% | cosθ |
  |---|---|---|
  | 10–30B | ~28% (platô inicial) | +0.01–0.03 |
  | 40B | 33.8% | +0.10 |
  | 60B | 34.1% | +0.13 |
  | **90B** | **39.2% (pico)** | +0.19 |
  | 141B | ~37% | |
  | 142B (pausado pra liberar GPU pro n=5) | ~38–40% | |

- **[HIPÓTESE]** (S10) "N=2 blind é difícil em duas frentes: precisa inferir **DOIS** pêndulos
  acoplados de UM sinal de carro, E balancear um pêndulo duplo é caoticamente difícil até com
  observação completa. Mais lento que n=1 (que estava ~69% aos 90B) — pode assentar bem abaixo
  de 90%, ou romper muito mais tarde. Dado o histórico de 5 reversões de snapshot, não cravar cedo."
- **[HIPÓTESE/debate de design em aberto]** (S10) hidden=256 escolhido "pra dar espaço a
  representar dois estados de crença" — mas **cuDNN LSTM BPTT escala com hidden²**, então 256 =
  ~4× trabalho de matriz = ~2× mais lento (1.4M vs 2.76M SPS solo). Debate não resolvido: manter
  256 e ter paciência, ou reiniciar em 128 (2× mais rápido, apostando que a capacidade do n=1
  basta) perdendo ~98B já treinados. **Vitor não decidiu nas mensagens lidas.**
- **[FATO]** (S10) N=2 foi pausado/retomado várias vezes por disputa de GPU com o n=5→1T/2T
  (pausas "limpas", sem perda por checkpointar).

---

# TEMA 4 — ENGENHARIA CUDA / PERF

## 4.1 SPS: da física NumPy ao env nativo em C/CuPy

- **[FATO]** (S2, 2026-06-12) Env inicial em NumPy: **~9.860 SPS** na RTX 3060 (`np.linalg.solve`
  da física a cada step + loop Python do rollout). **GPU peak 0.03 GB** → "o gargalo não é a rede,
  é o ambiente em NumPy no CPU". Decisão: portar env pra C.
- **[FATO]** (S2) Após port pra C: benchmark **10K (64 envs) → 146K (1024) → 386K SPS (8192)**;
  no treino real **9.9K → 125K SPS sustentado (12,7×)**.
- **[FATO]** (S1) SPS iniciais no double pendulum (antes do env C maduro): 9.5K (CPU) → 14.6K
  (CUDA, ~1.5×) — "GPU subutilizada, gargalo é o loop Python do env, não a GPU".
- **[FATO]** (S9) Migração C→CuPy `RawKernel` (JIT via NVRTC): env-only ~4.6× mais rápido isolado,
  mas só **~2× end-to-end** por causa de **sync device↔host↔device por step** no boundary
  PyTorch↔CuPy (lei de Amdahl) — isso motivou toda a investigação de CUDA graph/bf16.
- **[FATO]** (S8) Em produção citou-se **5,78M SPS** como throughput do env; e o fator **12,7×**
  como speedup a ancorar.

## 4.2 bf16 — a alavanca real (o "~1.8×")

- **[FATO]** (S9/S10) Benchmarks isolados na RTX 3060:
  - Update-only (idle GPU): TF32 670ms/6.26M SPS → bf16 341ms/12.31M SPS = **1.97×**.
  - Rollout-only: TF32 322ms/13.0M SPS → bf16 188ms/22.3M SPS = **1.71×**.
- **[FATO]** (S9) A/B de **aprendizado** (3 seeds, do checkpoint ~219–226B, +200M steps):
  TF32 rew@eval 627.5 (std 27.3) vs bf16 639.5 — diferença 12.0 **dentro do ruído** → PASS.
  "bf16 não degrada, roda em metade do tempo."
- **[FATO]** (S9) Em produção (quad N=4): **4.0M → 6.0–7.2M SPS sustentado (~1.5–1.8×)**, zero NaN
  em 270B steps extras, rew@eval final **1031** (subiu de ~597 quando bf16 entrou em 230B). Este
  é o **"~1.8×" da memória**, confirmado e detalhado.

## 4.3 Memory-bandwidth-bound — a causa raiz (roofline)

- **[FATO]** (S9/S10) RTX 3060 (GA106): FP32 ~12.7 TFLOP/s, BF16/FP16 tensor core ~51 TFLOP/s,
  TF32 ~25.5 TFLOP/s, DRAM ~360 GB/s.
- **[FATO]** O op dominante (GEMM K=N=256) tem intensidade aritmética **AI≈85 FLOP/byte**; o
  **ridge point** da 3060 é 142. AI < ridge ⇒ **memory-bound**. Teto efetivo = AI×BW =
  85×360 ≈ **30 TFLOP/s** (não os 51 teóricos de compute).
- **[FATO]** Sweep de tamanho de GEMM (M de 16384→1048576): eficiência sobe 15.0→24.4 TFLOP/s e
  **platôa em ~24.4 TFLOP/s = 79% da banda de memória** por volta de M=262144 — exatamente o M do
  update real (seg×T=2048×128). "O update já roda a **78% da banda** — já está no platô."

## 4.4 CUDA graph = paridade (não fusion vencendo)

- **[FATO]** (S10) Primeira medição (contra réplica `bench_loop.py`) deu "1.52×" — mas era
  **espantalho**: a réplica usava política aleatória (~42k terminações/epoch → ~42k syncs DtoH),
  enquanto a produção (política treinada) completa os 1500 steps (syncs desprezíveis). Contra a
  produção real (~3.86M SPS): CUDA graph deu só **paridade (~1.05–1.09×)**.
- **[FATO]** Só quando **bf16 foi adicionado dentro do graph** (rollout+update) ficou 1.87–2.05× —
  e a alavanca ali também era o bf16, não a captura de grafo.

## 4.5 Kernel fusion (Triton) vs cuBLAS — não sobreviveu à integração

- **[FATO]** (S10) Kernel Triton single-layer (`GELU(X@W+b)`, memory-bound, fp16): **2.88× mais
  rápido que cuBLAS+GELU separado** (15.3 vs 5.3 TFLOP/s), porque cuBLAS faz GEMM e GELU como 2
  kernels (2 round-trips HBM) e o Triton funde e escreve uma vez. Composto 2-layer (h1 residente
  em registradores): **3.26× vs cuBLAS**.
- **[FATO — a reversão crucial]** Integrado no `forward_eval` real (fp16, com casts na fronteira
  bf16↔fp16): **regrediu para 0.83× (mais lento)**. Causa: o microbench comparava contra um
  caminho **lento** do cuBLAS (casts fp32 explícitos entre GEMMs) que não reflete produção (bf16
  autocast sem upcast); e o kernel fp16 precisava de 2 casts de fronteira que comiam a economia.
- **[FATO — retentativa correta]** Reescrito **bf16-nativo, sem casts**: rollout eager 133.3ms vs
  fused-bf16 124.2ms = **1.07× real** (parity, argmax-agree 99.8%). Mas **end-to-end apenas +2%**
  (rollout é só 30% do epoch; o gargalo é o update/backward = 65%).
- **[HIPÓTESE→rejeitada]** Backward custom fundido: rejeitado porque o Triton GEMM próprio é 1.6×
  mais lento que cuBLAS puro (0.358 vs 0.225ms) — teto estimado 5-7% com risco de regressão,
  "provável net-negativo".

## 4.6 Staleness (gradient staleness) — throughput real, mas degrada aprendizado

- **[FATO]** (S9/S10) Pular backward a cada K minibatches (K=2, reusando `.grad` velho): update
  343.5→229.5ms = **1.50× no update**, projetado 1.27× full-loop. Aplicado ao vivo no n=5: SPS
  real **~9.7M (1.35× medido)**; nas primeiras 5 fases (~500M steps) rew@eval segurou (~448 vs
  454, ruído).
- **[FATO — a reversão]** Com observação sustentada (2.5B steps): rew@eval caiu de **454 → 426**,
  degradação real e crescente, não ruído. **Revertido** para K=1, rollback pro checkpoint
  pré-staleness. "Gradiente stale = otimização grosseira que afasta a política do ótimo devagar."
- **[HIPÓTESE/lição]** "O trade só aparece com **observação sustentada**, não num snapshot."

## 4.7 INT8 / quantização — morto em todas as variantes

- **[FATO]** (S10) cuBLAS int8 (`_int_mm`): **0.89× (mais lento)** que bf16, 6× mais erro.
  Triton int8 bem-tunado: 0.94×. Melhor caso possível (int8 in E out, full-int8, 3× menos bytes):
  só **1.05×** — prova de que o kernel pequeno é **overhead/occupancy-bound**, não bandwidth nem
  compute-bound; e full-int8 seria proibitivo pra qualidade de treino RL.

## 4.8 NF4 QAT (quantização da policy)

- **[FATO]** (S3, 2026-06-12) **NF4 QAT** (4-bit NormalFloat, Dettmers 2023) integrado com toggle
  `USE_NF4`. Artefatos: `cartpole_pendulum_n_policy.pt` (fp32) e `..._policy_nf4.pt` (NF4).
- **[FATO]** (S8) NF4-QAT na policy RNN = **8× compressão**, policy treina normal sob NF4.
- **[FATO — lacuna metodológica honesta]** (S8, 2026-06-22 20:56) **"não há A/B FP32-vs-NF4 de
  reward"** — o NF4 ficou sempre ligado; FP32 só serviu de baseline de tamanho. "Sem essa
  ablação, o NF4 não fecha como claim — é infra boa, não resultado."

## 4.9 torch.compile / observabilidade

- **[FATO]** (S10) `triton-windows` instala limpo em py3.14 (o agente havia errado dizendo "dead
  end" sem testar). compile no update: default ~1.10–1.11×, max-autotune ~1.14× ("Not enough SMs"
  na 3060 explica por que max-autotune ≈ default). compile no rollout: **piora** (0.97×). Aplicado
  ao n=5: só-no-update deu 6.8M→7.6M SPS (+9% end-to-end).
- **[FATO]** (S10) `ncu`/`nsys` não instaláveis (sem CUDA toolkit no Windows), mas
  `torch.profiler`+kineto+CUPTI funcionam. Contadores de HW ("achieved occupancy", DRAM
  throughput via CUPTI) ficaram **bloqueados** pela permissão de perf counter do Windows/NVIDIA.
  Alternativa que funcionou: **occupancy determinística** calculada de specs do device (28 SMs,
  65536 regs/SM, 100KB shared/SM, 48 warps/SM) + metadata do kernel — sem contador nenhum.

---

# TEMA 5 — INSIGHTS / DEBATES TRANSVERSAIS

## 5.1 Lições-mestras metodológicas (repetidas ~6× no projeto)

- **[HIPÓTESE/lição] "Não conclua de um snapshot — só vale com observação sustentada."**
  Casos: center-farming (o return mentia), "600M sem separação" (era slow-start), staleness
  ("parecia ok" nas primeiras 5 fases mas degradava em 2.5B), "full-obs = eyes-closed ~52%"
  (era platô intermediário; real era 97% vs 60%), n=5 "500B convergido" (tinha +15% escondido via
  warm-restart), N=2 "platô ~28%" (ainda subindo).
- **[HIPÓTESE/lição] "API aceitou ≠ dado saiu."** Erros repetidos: `torch.compile`/Triton "dead
  end" (falso, não tinha testado); "sem CUPTI" (falso, olhou no env Python errado); API de
  métricas CUPTI "funcionando" (aceitava nomes sem erro mas não coletava — falha silenciosa).
- **[HIPÓTESE/lição] "Micro-benchmark vs baseline-espantalho não sobrevive à integração;
  end-to-end é a verdade."** A saga do kernel fusion Triton (wash → 2.88× → 3.26× → 0.83× → 1.07×
  real → +2% e2e) é o caso-escola citado pelo próprio agente como a lição mais dura.
- **[HIPÓTESE/lição] "Medir o valor físico, não o proxy escalar."** episode-return mentiu
  (center-farming); %up>0.6 escondeu o jitter (só o vídeo revelou); "solved" precisa ser
  cosθ/%up honesto, não reward bruto.

## 5.2 Debates de arquitetura

- **[HIPÓTESE/debate] "MLP dentro do MinGRU" — rejeitado.** MinGRU só é paralelizável (Heinsen
  scan) porque a transição de estado é **linear**; inserir uma MLP na recorrência quebra o scan
  (viraria LSTM sequencial lento). Solução correta = **sandwich** MLP→recorrência linear→MLP
  (mesma receita de SSMs tipo Mamba/minGRU paper), mantendo o núcleo recorrente rápido e a
  não-linearidade fora dele.
- **[HIPÓTESE/debate metodológico, cobrado pelo Vitor]** A ablation de throughput inicial variou
  n_envs/hidden/layers **junto** com as levers de execução (precision/compile/graph) — o Vitor
  apontou que isso mistura *mudança de workload* com *mudança de execução*, invalidando
  conclusões. Corrigido: fixar modelo+batch, variar só a execução.
- **[HIPÓTESE/diretriz do Vitor]** (S9, 2026-06-27 14:12) "**Observabilidade extrema antes de
  otimizar**" — motivou a stack de roofline/occupancy determinística que depois reverteu a
  conclusão errada "Triton = wash" para "Triton bate cuBLAS 2.88–3.26× em kernels memory-bound
  fundíveis" (mesmo não sobrevivendo à integração e2e).

## 5.3 O achado-estrela: transfer zero-shot por escala de gravidade

- **[FATO/citação do Vitor]** (S8, 2026-06-22 20:56) A hipótese de transfer por gravidade foi
  **ideia do próprio Vitor**, verbatim: *"pq simplesmente então não treina só o de 10xg e solta
  ele no regime real sem q ele saiba?"* — e funcionou. O agente registrou "the user was right all
  along", "a fascinating finding".
- **[FATO]** (S8) **Treinar em 10×g converge em ~100M steps vs ~3B steps no regime alvo (1×g)** —
  ~30× mais eficiente em amostras — **e transfere zero-shot pra 1×g com reward ~139**, comparável
  a curriculum/annealing tradicional.
- **[FATO]** (S8) Transfer é **monótono na distância ao alvo**; há um "sweet-spot" ~1,5–2×g no
  triple-pêndulo; no quad a vantagem sub-g desaparece ("1×g ≈ π⁻¹×g"). Cobertura N=1..4 (revisado
  até N=8).

## 5.4 Debate: como enquadrar como paper científico

Debate completo em S8 (2026-06-22 20:28→21:01), depois adiado.

- **[FATO/contexto]** (2026-06-22 20:28) Vitor: "são 2 papers. um é umas sessões projetos do
  n-pendulo q fiz com pufferlib. eu descobri várias coisas legais, queria uma revisão pra ver se
  dá pra encaixar um paper rápido pq gerei bastante conteúdo."
- **[HIPÓTESE/DEBATE — 1ª rejeição]** (20:59) "SPS puro" como contribuição foi rejeitado: "esse
  throughput é majoritariamente do PufferLib (vetorização em C é literalmente o ponto da
  biblioteca), não seu. Sozinho é demo/workshop, com risco de 'isso o PufferLib já faz'."
- **[HIPÓTESE/DEBATE — 2ª proposta]** (21:00) "eficiência de compute via escala física": treinar
  no regime de gravidade que converge ~30× mais rápido (amostras E wall-clock) + transfer
  zero-shot dá a mesma policy por ~1/30 do compute. SPS/wall-clock como suporte, não tese.
- **[HIPÓTESE/DEBATE — enquadramento final proposto: "environment engineering"]** (21:00) **O env
  em si é a contribuição**. Tese: "Um benchmark de n-pêndulo rápido e parametrizado para controle
  sub-atuado, com **dois eixos de dificuldade controláveis — número de elos N e gravidade** —
  projetado para estudar currículo, transfer e eficiência de compute."
  - Novidade: eixo N parametrizado (Gym Pendulum/Acrobot e MuJoCo InvertedDoublePendulum têm N
    fixo); gravidade como knob com o truque `dt ∝ 1/√g`; velocidade como propriedade habilitadora.
  - **Gap honesto:** "Papers de benchmark vivem de adoção/reusabilidade… hoje são DLLs ad-hoc e
    scripts — esse é o maior gap." Falta release limpo/instalável, posicionamento vs
    Gym/MuJoCo/dm_control, e escrita do zero.
  - Veredito: cabe em "workshop de RL, NeurIPS Datasets & Benchmarks, ou até ENIAC/BRACIS" — mas
    **não é paper pronto** (sem rascunho escrito).
- **[FATO/fonte]** (20:56) Os achados foram recuperados dos **chats do "yeclaudebot" (Telegram →
  `claude -p`) no xmain** — existe um corpus de transcripts do bot com a "voz" do Vitor sobre os
  achados, potencialmente ainda não totalmente minerado.
- **[FATO — adiamento]** (21:00→S8 2026-06-26 12:12) Por deadline do paper BPMNPlan naquele dia,
  o n-pêndulo foi deixado como backlog: "esse vive em `world/PufferLib`, não aqui; é outro paper."

## 5.5 Insights de infraestrutura (runs multi-dia)

- **[HIPÓTESE/observação]** (S10) Sleep/reboot do Windows é risco real: (a) sleep de 4.4h
  "wedgeou" o contexto CUDA do processo LSTM (ficou "vivo" mas 35× mais devagar) — corrigido com
  sleep off + watchdog que detecta "wedged-but-alive"; (b) reboot completo (2026-07-04→05)
  derrubou tudo sem auto-restart por ~27h (o watchdog não sobrevive a reboot do SO).
- **[HIPÓTESE/pivô de agenda]** (S9/S10) Encerramento da via de otimização pura de SPS na 3060:
  "bf16 (1.8×) + compile (+11% update) = **teto físico, memory-bandwidth-bound a 79% da banda**."
  Daí o leverage do projeto migrou explicitamente de **"velocidade" para "ciência"** (capacidade
  N, depois POMDP).

---

## 5.6 Gênese do projeto (contexto)

- **[FATO]** (S0, 2026-03-13, codex/yalien) Primeiro contato com o **PufferLib** (repo clonado em
  `C:\Users\vitor\world\PufferLib`). Descobriu o caminho mínimo de env
  (`gymnasium.Env` → `pufferlib.emulation.GymnasiumPufferEnv`), notou que `pip install -e .`
  não funciona no Windows (setup.py levanta "Unsupported system: Windows") — rodou via
  `PYTHONPATH=.`. (Boa parte dessa sessão deriva pra outro tema — DiagramEnv/BPMN — não pêndulo.)

---

> **Status:** núcleo científico (batches C, D2 + leituras diretas S0-S2, S7) escrito.
> **Pendente de append:** batch A (S3/S4 detalhado, sessões 06-12/13) e batch B (S5/S6,
> sessões 06-17/18/26). Ver seções "APPEND" abaixo quando disponíveis.
