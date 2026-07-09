# ESBOÇO — Artigo final de Aprendizado de Máquina (PPgEEC/UFRN)

**Título (final, fixado pelo brief):**
*Aprendizado por Reforço em GPU para o Pêndulo-N: um Estudo de Throughput, Precisão Mista e Escalabilidade*

**Autor:** Vitor Y. F. Freitas (vitoryeso@outlook.com), PPgEEC/UFRN
**Formato:** SBC, português, ≤12 páginas. Arquivo: `artigo/artigo.tex` (1º draft).

---

## Tese / ângulo

Swing-up do pêndulo-N (cart + N elos rígidos, subatuado, caótico, ação discreta
bang-off-bang) exige orçamento gigante de amostras (500B–1,6T steps). O insight que
justifica tudo: **capacidade da rede NÃO é o gargalo** (obs = 2+3n dims; n=21 → 65;
MLP pequena basta) — o gargalo é **exploração sob dinâmica instável/subatuada**, que
se compra com amostras. Logo **throughput é a alavanca científica**, e o artigo é um
*estudo* (honesto por design) em 3 pilares que dão nome ao título.

## Espinha estrutural (os 3 pilares — aplicados em Resultados E Análise)

1. **Throughput** — env GPU-nativo (CuPy RawKernel, zero-copy DLPack), 7–8M SPS e2e;
   roofline diagnostica o laço como memory-bound (posicionado AQUI como chave de
   leitura do pilar 2).
2. **Precisão Mista** — bf16 1,87× (A/B 3 seeds, zero NaN) + NF4-QAT 8× (ressalva:
   sem A/B de qualidade). Contrapontos honestos: CUDA graph = paridade (1,52× era
   espantalho), Triton fundido 2,88× micro → +2% e2e. Lição: o ganho veio de
   PRECISÃO, não de kernelização.
3. **Escalabilidade** — duas dimensões explícitas: (a) escalar o SISTEMA (32.768
   envs, trilhões de steps viáveis em GPU de consumo — entregue); (b) escalar o
   PROBLEMA (n=1→5 — fronteira aberta). Throughput *viabiliza tentar*, não garante.

## Outline final do .tex

| Seção | Conteúdo |
|---|---|
| Abstract (EN) + Resumo (PT) | organizados pelos 3 eixos do título |
| 1. Introdução | problema, subatuação, tese capacity-não-é-gargalo (quote destacado), contribuições = 3 pilares |
| 2. Estado da Arte | RL massivo em GPU (Isaac Gym, Brax, EnvPool, Madrona, Sample Factory, PufferLib); PPO/GAE; controle subatuado (Spong, Åström & Furuta, Boubaker survey); precisão mista (Micikevicius); NF4 (Dettmers/QLoRA) |
| 3. Metodologia | 3.1 env (matriz de massa Eq.1, RawKernel genérico-N via #define, obs/ação/reward Eq.3 + aviso center farmável, paridade 8,8e-7); 3.2 escada de n (L=1m/M=1kg constantes, Fibonacci 1..21, escala Quanser, discreto→contínuo); 3.3 PPO (Tabela 1 hparams n=5, warm-restart SGDR); 3.4 variante POMDP (3 braços, reward de-confundido, métricas físicas, TBPTT stateful); 3.5 pilha de sistemas; 3.6 metodologia de medição (3 princípios) |
| 4. Resultados | 4.1 Pilar 1: Tabela SPS (env-only 260–333M; e2e 7,2–7,7M; legado 125K), roofline (Fig 1); 4.2 Pilar 2: Tabela speedups com veredito, bf16 A/B, heatmaps (Fig 2), CUDA graph/Triton/NF4; 4.3 Pilar 3: (a) sistema, (b) problema — Tabela por n, n=1 full+POMDP, n=5 curva (Fig 3) |
| 5. Análise | 5.1 Pilar 1: capacidade não é gargalo (3 evidências); 5.2 Pilar 2: lições de medição (espantalho, staleness 454→426 revertido, valor físico vs proxy / center-farming / vídeo); 5.3 Pilar 3: onde PPO quebra com n; 5.4 Pilar 3: POMDP + mecanismo TBPTT (BPTT-from-zero); 5.5 Limitações (5 itens honestos) |
| 6. Conclusões | 3 pilares recapitulados sem overclaim + futuro (A/B NF4, currículo/gravidade/transfer/energy-shaping p/ n≥5, GNN/1D-conv p/ generalizar n, escada até n=21, concluir n=2 blind) |
| Bibliografia | `thebibliography` embutida, labels autor-ano, 13 entradas, sem DOI |

## Números usados (todos do brief/insumos — nenhum inventado)

| Item | Valor |
|---|---|
| Obs / estado | 2+3n / 2n+2; ação {−10,0,+10} N |
| Reward | clamp((0,5·height+hold)·center, 0, 1,5); MAX_STEPS 1500; teto ~2250 |
| SPS env-only | n=4: 333M · n=5: 260M · n=6: 212M |
| SPS e2e | n=4: 7,2M · n=5: 7,7M; legado CPU/C 125K (~60× — derivado de 7,7M/125K, ver "incertos") |
| PPO n=5 | 32768 envs, H=128, 16 mb/1 epoch, γ0,99 λ0,95 clip0,2 vf0,5 vf_clip±1 gn0,5, LR cos 1,5e-4→1e-5, ent 0,005→0,001, MLP 2×256 |
| bf16 | 1,87× e2e (992→529 ms); A/B: TF32 627,5±27,3 vs bf16 639,5; zero NaN em 270B |
| torch.compile | +9% (6,8→7,6M SPS), update only; rollout piora |
| Triton fundido | 2,88× vs cuBLAS (0,807→0,281 ms); +2% e2e; multi-camada travou em shared-mem |
| CUDA graph | ~1,05× (paridade); 1,52× = espantalho (política aleatória, ~42k syncs) |
| Staleness K=2 | 1,50× update / 1,35× e2e; 454→426 em 2,5B → revertido |
| int8 | 0,89–1,05× → morto (occupancy-bound) |
| Roofline 3060 | 360 GB/s, 51 TFLOP/s fp16-TC, ridge 142; GEMM AI≈85; satura 24,4 TFLOP/s = 79% banda em M≈262144 |
| NF4-QAT | 8× compressão; SEM A/B de qualidade (lacuna declarada) |
| n=1 full | 2188/2250, 97% up — resolvido |
| n=1 POMDP LSTM | 96,8% up @245B; custo da cegueira ~20–40× amostral |
| n=1 POMDP MLP | ~11% (controle negativo) |
| n=2 POMDP | ~38%, em progresso |
| n=4 full | ~1031/2250, parcial/instável |
| n=5 full | platô 610–624/2250 (~27%) @1,6T; curva 100M→61, 500M→372, warm-restart 530→612,6 (1T), 2T→~575 |
| Center-farming | cega return 212 vs full 66; corrigido só na variante POMDP |
| Escada | L=1,0 m, M=1,0 kg, x±2 m, F±10 N, dt=0,005 s; n∈{1,2,3,5,8,13,21} |

## Citações (13, thebibliography, autor-ano)

Makoviychuk 2021 (Isaac Gym) · Freeman 2021 (Brax) · Weng 2022 (EnvPool) ·
Shacklett 2023 (Madrona) · Petrenko 2020 (Sample Factory) · Suárez 2024 (PufferLib)
· Schulman 2017 (PPO) · Schulman 2016 (GAE) · Spong 1995 (Acrobot) · Åström &
Furuta 2000 (energy swing-up) · Boubaker 2013 (survey pêndulo invertido —
usado p/ framing contínuo/benchmark) · Micikevicius 2018 (mixed precision) ·
Dettmers 2023 (QLoRA/NF4).

## Figuras referenciadas (existem no repo — caminhos verificados 2026-07-08)

| Fig | Arquivo | via `\graphicspath` |
|---|---|---|
| 1 roofline | `C:/Users/vitor/world/PufferLib/npendulum5_output/ablation/roofline_sweep.png` | `roofline_sweep.png` |
| 2 heatmaps | `.../ablation/ablation_heatmaps.png` | `ablation_heatmaps.png` |
| 3 curva n5 | `.../npendulum5_output/n5_500B_reward.png` | `n5_500B_reward.png` |

`\graphicspath{{../../PufferLib/npendulum5_output/}{...}}` assume compilar de dentro
de `artigo/` com `PufferLib` irmão de `machine-learning-master-class` — **ajustar se
a árvore for outra** (ou copiar os PNGs pra `artigo/fig/`). Alternativa disponível:
`n5_1T_scatter.png` (scatter até 1T) se quiser figura da extensão 1T.

## Pontos incertos / a revisar antes de entregar

1. **Compilação NÃO testada** — sem LaTeX instalado no xmain (checado: sem pdflatex/
   tectonic/latexmk). Checagem estática passou (envs, labels, refs, cites, braces).
   Precisa de `sbc-template.sty` (há fallback mínimo p/ rascunho embutido no .tex).
2. **"~60×" GPU vs linha legada** — derivado por mim (7,7M / 125K); o brief só dá os
   dois extremos. Conferir se quer manter a razão explícita.
3. **Shacklett 2023 (Madrona)** — lista de autores/título abreviados de memória;
   conferir a entrada exata (TOG 42(4), 2023) antes da versão final.
4. **Retorno POMDP** — a tabela por n usa "---" no retorno das linhas POMDP (a
   variante usa reward de-confundido, escala incomparável). Ok?
5. **%up de n=4/n=5** — declarado "n/d" (não logado) na tabela e nas limitações,
   conforme brief.
6. **Boubaker 2013** — citação real, mas foi minha escolha p/ cobrir "pêndulo
   flexível/continuum" (o brief pedia a área sem dar referência específica).
7. **Páginas** — estimo ~10–12 pág. em sbc-template com 3 figuras; se estourar 12,
   cortar primeiro: quote da tese na intro, parágrafo int8, detalhes do TBPTT.

## Decisões de escrita

- Tom "estudo": nenhuma claim de que n≥4 foi resolvido; n=5 sempre "fronteira aberta".
- Resultados negativos com o mesmo peso dos positivos (tabela de speedups tem coluna
  "veredito").
- Center-farming aparece 3×: metodologia (aviso), POMDP (correção), limitações
  (permanece na linha CUDA).
- A linha "125K SPS" aparece SÓ como contexto histórico da linha legada CPU/C, como
  mandado — o paper centra na linha CUDA genérica-N.
