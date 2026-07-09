# ESBOÇO — Artigo final AM 2026.1 (revisão rápida)

**Arquivo:** `artigo/artigo.tex` (LaTeX SBC, 12pt, babel brazil; requer `sbc-template.sty` do zip do professor nesta pasta).
**Tese:** seed única engana; multi-seed (mean±std, ddof=1) + teste pareado por seed resolvem.
**Bancada:** 8 métodos dos miniprojetos (P1 K-means, P3 GD/LMS, P4 logística, P5 árvore, P7 RBF, P8 SVM, P9 MLP, P10 CNN; P2 usado como estudo de circularidade).

## Outline

1. **Resumo/Abstract** — tese + 2 descobertas + números-âncora (±2,5→±0,3 p.p.; 7×; t=0,11/1,3 vs t≈5/16–20).
2. **Introdução** — 2 exemplos enganosos (P4 96,7%, P5 Gini>Entropy); RQ1 (o que sobrevive), RQ2 (origem da variância), RQ3 (ganho do pareamento).
3. **Estado da Arte** — Dietterich 1998; Salzberg 1997; Nadeau & Bengio 2003; Demšar 2006; Henderson 2018; Reimers & Gurevych 2017; Bouthillier 2021; Pineau 2021. Posicionamento: não propomos teste novo; transportamos o ferramental p/ bancada didática controlada.
4. **Metodologia** — 4.1 os 8 métodos (1–2 frases cada, "do zero" destacado); 4.2 dados (sintéticos c/ verdade conhecida; CLIP 512-d reais; 450 imagens foto×screenshot); 4.3 protocolo multi-seed (eq. mean/std ddof=1); 4.4 teste pareado (d_s por seed, t = mean/SE); 4.5 diagnóstico variância-da-medida vs variância-do-modelo (EP binomial √(p(1-p)/n)).
5. **Resultados** — P4 (tab. multi-seed), P3 (tab. GD×LMS), P5 (tab. single vs não-pareado vs pareado + Fig. paired_viz.png), P7 (tab. ablação centros), P10 (tab. modalidades pareado), 5.6 demais (P8, P9, P2, meta-comparação).
6. **Análise** — 6.1 Descoberta 1 (tabela dos 2 regimes de variância); 6.2 Descoberta 2 (3 desfechos do pareamento; mecânica via covariância; 7× ≈ 50× mais seeds); 6.3 cinco recomendações de reporte; 6.4 limitações (3 seeds impreciso; normalidade do t; dados reais fixos; sem correção múltipla; k-fold futuro).
7. **Conclusões** — 4 contribuições + 4 trabalhos futuros (Nadeau-Bengio, Wilcoxon/correção, decomposição de variância à la Bouthillier, parear a meta-comparação).

## Números usados (fonte no repo)

| Projeto | Número | Fonte |
|---|---|---|
| P4 | Acc 96,29%±0,27%; F1 0,963±0,003; BCE 0,098±0,005; erro 3,71%±0,27%; Bayes 3,57%=Φ(−√13/2); 10 seeds, teste ~2000 | `projeto-4/output/report_multiseed.txt` |
| P4 | seed única 96,67% (teste n=60); EP binomial ≈2,5 p.p.; 1 seed chegou a 100% | `projeto-4/RELATORIO.md` |
| P3 | MSE GD 0,5018±0,0118 vs LMS 0,2517±0,0039 (piso 0,25); gap ≈20× std; pesos LMS 2,996±0,011 / −2,001±0,006 / 0,991±0,016; std encolheu ~4× c/ N=10.000 | `projeto-3/RELATORIO.md` |
| P5 | seed 42: Gini 0,96 vs Entropy 0,94 (depth 9 vs 13; folhas c/ 1 amostra 10/8) | `projeto-5/output/report.txt` |
| P5 | não-pareado 10 seeds N=2000: 0,943±0,033 vs 0,945±0,033 (ranking inverte); std cresceu ±0,021→±0,033 (N=500→2000) | `projeto-5/RELATORIO.md` |
| P5 | pareado 30 seeds (teste N=400): d=−0,0002±0,008, t=0,11; std pareado 7× < não-pareado (±0,055); depth 15,5 vs 17,8; poda max_depth=5: t=0,57, acc 0,947 vs 0,936 | `projeto-5/RELATORIO.md` + `slides.html` |
| P5 | figura não-pareado vs pareado | `projeto-5/output/paired_viz.png` |
| P7 | ablação F1 k=10/30/60: kmeans 0,913/0,926/0,931; random 0,829/0,897/0,920; subset 0,723/0,871/0,915 | `projeto-7/output/report.txt` |
| P7 | pareado: kmeans t≈16–20 (robusto em todo k); random vs subset t=0,36 (empate real); oficial 3 seeds: macro F1 0,921±0,005; has_people 0,682±0,010; screenshot 0,668±0,030 | `projeto-7/slides.html` |
| P10 | seed 42 F1: 2D-RGB 0,9167, 2D-gray 0,8913, 1D 0,8723; ablação k=3 0,9126 / k=7 0,7928 | `projeto-10/output/report.txt` |
| P10 | pareado 30 seeds: 2D>1D t≈5 (real); RGB vs gray t=1,3 (empate) | `projeto-10/slides.html` |
| P8 | 3 seeds: macro F1 0,954±0,004; has_people 0,883±0,010; screenshot 0,770±0,007; γ=0,1 colapsa F1→0,09 (efeito ~200× std) | `projeto-8/slides.html` |
| P9 | 3 seeds: macro F1 CE 0,965±0,001 vs MSE 0,961±0,003; 0 ocultas 0,954±0,002 (~3σ abaixo); has_people 0,886±0,004; screenshot 0,756±0,010 | `projeto-9/slides.html` |
| P2 | 5 seeds: CLIP 0,954/0,896 vs ConvNeXt 0,833/0,661 (acc macro/folha); gap = 16–47× std (0,004–0,008) | `projeto-2/RELATORIO.md` |
| Meta | macro_vs_rest F1: MLP 0,965±0,001 > SVM 0,954±0,004 > RBF 0,921±0,005 | `GUIA_DE_ESTUDO_COMPLETO.md` |

## Citações (todas reais; sem DOI inventado)

- Dietterich (1998), Neural Computation — testes 5x2cv/McNemar
- Salzberg (1997), Data Mining and Knowledge Discovery — armadilhas
- Nadeau & Bengio (2003), Machine Learning — variância do erro de generalização
- Demšar (2006), JMLR — Wilcoxon/Friedman/Nemenyi
- Henderson et al. (2018), AAAI — reprodutibilidade em RL
- Reimers & Gurevych (2017), EMNLP — distribuições de scores em NLP
- Bouthillier et al. (2021), MLSys — decomposição de variância em benchmarks
- Pineau et al. (2021), JMLR — programa de reprodutibilidade NeurIPS
- Métodos: Widrow & Hoff 1960 (LMS); Breiman et al. 1984 (CART); Broomhead & Lowe 1988 (RBF); Cortes & Vapnik 1995 (SVM); Rumelhart et al. 1986 (backprop); LeCun et al. 1998 (CNN); Radford et al. 2021 (CLIP)

## Pontos incertos p/ revisão

1. **"Monte-Carlo confirmou" (P4)** — o repo NÃO tem menção a Monte-Carlo; o argumento no repo é o previsor binomial analítico (√(p(1-p)/n)) + consistência por-seed. O artigo usa só o binomial. Se houve simulação MC em outro lugar, adicionar.
2. **P10 multi-seed mean±std** — o repo só tem a tabela seed=42 + os t pareados (slides/ROTEIRO citam "~0,88 / 0,86 / 0,83" aproximados, 30 seeds). A tabela do artigo usa seed=42 rotulada como ilustração + veredito pareado. Se existir report multi-seed do P10, substituir.
3. **P7 t≈16–20 e t=0,36** — fonte é `slides.html` (claim do projeto), não um report .txt; conferir se há log numérico do teste pareado do P7.
4. **Inconsistência menor no P9** — slides citam MLP oculta como 0,965±0,001 num bullet e 0,965±0,003 noutro; artigo usa ±0,001 (valor da tabela oficial/meta-ablação).
5. **Desvio não-pareado do P5** — RELATORIO cita ±0,033 (10 seeds N=2000) e ±0,055 (não-pareado no protocolo de 30 seeds); o "7×" refere-se a ±0,055 vs ±0,008 — mantido assim no texto; conferir redação.
6. **N seeds heterogêneo (3/5/10/30)** — tratado nas Limitações; o professor pode pedir uniformização.
7. **Compilação** — precisa de `sbc-template.sty` na pasta (zip do prof). Bibliografia embutida via `thebibliography` (labels autor-ano); migrar p/ `.bib` + `\bibliographystyle{sbc}` na versão final se exigido.
8. **Figura** — caminho relativo `../projeto-5/output/paired_viz.png`; copiar a imagem p/ `artigo/` se for compilar fora da árvore do repo.
9. **Tamanho** — estimado ~10–12 págs; se estourar 12, cortar primeiro §5.6 (demais projetos) e a Tabela 2 (P3 pesos).
