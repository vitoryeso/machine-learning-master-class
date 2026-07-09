# Projeto 10 — CNN para Dados 1D e 2D

## Problema

Este mini-projeto implementa CNNs (PyTorch) para uma tarefa de classificação binária **real**, não sintética: distinguir **foto de câmera** vs **screenshot**, usando 450 imagens da coleção pessoal do autor (baixadas do Google Drive). A tarefa foi escolhida por ser visualmente saliente — screenshots têm bordas retangulares de UI, fotos têm texturas/gradientes de cena natural — o que permite interpretar os feature maps aprendidos.

O experimento compara três eixos de design de uma CNN:

- **2D vs 1D**: convolução sobre a imagem inteira (2D) vs sobre um sinal 1D (perfil de intensidade por linha, colapsando altura e canais).
- **RGB vs escala de cinza**: efeito de ter 3 canais de entrada vs 1.
- **Ablação de filtros/kernel**: número de filtros (`nf`) e tamanho do kernel (`k`) na camada convolucional.

Além disso, os **feature maps** da primeira camada convolucional são visualizados para dar intuição sobre o que cada filtro aprendeu a detectar.

## Dataset

Imagens pessoais, organizadas por pasta = rótulo (`data/foto/`, `data/screenshot/`):

| Parâmetro          | Valor                                    |
|--------------------|-------------------------------------------|
| N total            | 450 imagens                                |
| Foto (classe 0)    | 200                                         |
| Screenshot (classe 1) | 250                                      |
| Resolução          | 64×64 (resize + normalização [0,1])        |
| Canais originais   | RGB (3 canais)                              |
| Train / Test split | 360 / 90 (80/20)                            |
| Seed (ilustração)  | 42                                          |
| Device             | CUDA (RTX 3060, xmain)                     |

As imagens são carregadas com PIL, convertidas para RGB, redimensionadas para 64×64 e cacheadas em `.npz` (`data/cache_64.npz`) para permitir reprodução em máquinas sem PIL (ex.: hosts de compute do laboratório usados para o multi-seed).

## Método

### Convolução (2D e 1D)

```
S(i,j) = Σ_c Σ_m Σ_n I(i+m, j+n, c)·K(m,n,c) + b
A = ReLU(S)  →  MaxPool  →  ...  →  fc
```

Termos:

- **I** — entrada: a imagem (H×W×canais); RGB tem 3 canais, cinza tem 1. A soma percorre também os canais `c` — é por isso que, em princípio, RGB carrega mais informação bruta que cinza.
- **K** — o filtro/kernel (K×K), pesos **aprendidos** que detectam um padrão local (ex.: uma borda).
- **b** — viés do filtro.
- **S** — o feature map: a resposta do filtro em cada posição (i,j) da entrada.
- **ReLU + MaxPool** — não-linearidade e redução de resolução (resume cada região a seu valor máximo).
- **1D** — mesma ideia, mas o "sinal" é o perfil de intensidade por linha: cada imagem é reduzida a um vetor de comprimento H (média sobre canais e sobre as colunas), e o kernel 1D desliza sobre esse vetor. Colapsar a imagem em 1D descarta toda a estrutura espacial ao longo do eixo das colunas.

### Arquiteturas

- **CNN2D** (`in_ch`, `nf=16`, `k=3`): `Conv2d(in_ch→nf) → ReLU → MaxPool(2) → Conv2d(nf→nf*2) → ReLU → MaxPool(2) → Linear(→1)`. Usada para RGB (`in_ch=3`) e cinza (`in_ch=1`, obtido por média dos 3 canais).
- **CNN1D** (`nf=16`, `k=5`): mesma topologia (2 conv + pool), mas com `Conv1d` sobre o perfil de intensidade (`sinal = média sobre canais e sobre colunas`, comprimento 64).
- Treino: Adam (lr=1e-3), 15 épocas, batch=64, `BCEWithLogitsLoss` (saída = logit único, classificação binária).

### Feature maps

Após treinar a CNN2D-RGB, os mapas de ativação (`ReLU(conv1(x))`) são extraídos para uma imagem de cada classe do conjunto de teste, permitindo comparar visualmente o que os mesmos filtros "enxergam" em um screenshot vs em uma foto.

## Resultados

### Single-seed (seed=42, ilustração)

| Modalidade   | Acc    | F1     | Precisão | Recall |
|--------------|--------|--------|----------|--------|
| **2D-RGB**   | 0.9111 | **0.9167** | 0.9362 | 0.8980 |
| 2D-gray      | 0.8889 | 0.8913 | 0.9535   | 0.8367 |
| 1D-perfil    | 0.8667 | 0.8723 | 0.9111   | 0.8367 |

Nesta seed isolada, o ranking aparenta ser claro: **2D-RGB > 2D-gray > 1D**. A seção seguinte mostra que esse ranking mistura um efeito real com ruído de amostragem.

### Multi-seed (3 seeds, F1, mean±std) — resultado oficial

| Modalidade | F1 (mean ± std) |
|------------|------------------|
| **2D-RGB** | **0.884 ± 0.024** |
| 2D-gray    | 0.839 ± 0.068     |
| 1D         | 0.814 ± 0.060     |

O ranking das médias (2D-RGB > 2D-gray > 1D) se mantém, mas os desvios-padrão são grandes o suficiente para que os intervalos de 2D-gray e 1D se sobreponham amplamente com 2D-RGB — e entre si. Com apenas 90 imagens de teste e um dataset total de 450, a CNN é um modelo de **alta variância**: trocar a seed muda o split e a inicialização dos pesos, e isso move o F1 mais do que a diferença entre modalidades RGB vs cinza. Esse é exatamente o cenário em que uma comparação **não pareada** (cada seed sorteia treino/teste diferente para cada modalidade) não separa sinal de ruído — é preciso o teste pareado abaixo.

### Teste pareado (n=30 seeds) — veredito

Repetindo o experimento com **o mesmo split de treino/teste e a mesma seed comparando as modalidades nos mesmíssimos dados** (30 seeds), a diferença sai da média ruidosa e vira dois vereditos distintos:

| Comparação      | Efeito                                  | Estatística | Veredito |
|-----------------|------------------------------------------|--------------|----------|
| **2D vs 1D**    | 2D > 1D                                   | t ≈ 5        | **Sinal real** — a estrutura espacial importa |
| **RGB vs gray** | diferença ≈ 0                             | t = 1,3      | **Empate estatístico** — a cor não agrega informação nesta tarefa |

O truque do pareamento é o mesmo já usado nos projetos anteriores (P3/P5): ao usar os mesmos dados de treino/teste para as duas modalidades em cada seed, o ruído de amostragem compartilhado se cancela na diferença, sobrando só o efeito estrutural. O resultado é que o "ranking" RGB > gray > 1D do single-seed **misturava um efeito real (2D supera 1D, t≈5) com ruído (RGB e gray empatam, t=1,3)** — a leitura ingênua da tabela de uma seed só teria atribuído à cor um ganho que na verdade não existe.

### Ablação (F1 teste, seed=42, 2D-RGB)

| Filtros (`nf`) | F1     | Kernel (`k`) | F1     |
|----------------|--------|--------------|--------|
| 4              | 0.8846 | 3            | 0.9126 |
| **16**         | **0.9184** | 5        | 0.8911 |
| 32             | 0.9000 | 7            | 0.7928 |

`nf=16` é o ponto ótimo entre poucos filtros (sub-representação) e muitos (overfitting num dataset de 360 imagens de treino). Para o kernel, `k=3` é o melhor e `k=7` **despenca** (0.79): num mapa de 64×64, um kernel 7×7 cobre uma fração grande demais da imagem logo na primeira camada, perdendo a localidade que a convolução deveria explorar.

### Plots gerados

- `output/modalities_ablation.png` — barras de F1 por modalidade (2D-RGB / 2D-gray / 1D) e por ablação (nº de filtros, tamanho do kernel).
- `output/feature_maps.png` — mapas de ativação do `conv1` (6 filtros) para um screenshot e uma foto lado a lado.
- `output/report.txt` — relatório numérico da rodada single-seed (métricas completas + ablação).

## Análise

**Por que 2D vence 1D de verdade (t≈5):** a convolução 2D preserva a vizinhança espacial nas duas dimensões da imagem; o perfil 1D é obtido tirando a média sobre canais **e** sobre as colunas, o que destrói toda a informação de onde, horizontalmente, cada padrão ocorre. Os feature maps do `conv1` (RGB, `output/feature_maps.png`) tornam esse efeito visível: para um **screenshot**, os filtros acendem nas **bordas retangulares** da caixa de UI — viraram detectores de aresta/linha; para uma **foto**, os mesmos filtros respondem de forma **difusa**, a texturas e gradientes da cena natural, sem bordas nítidas. É exatamente esse contraste espacial (bordas nítidas vs textura difusa) que a rede 2D consegue explorar e que o colapso para 1D apaga.

**Por que RGB e cinza empatam de verdade (t=1,3):** a diferença de 4,5 pontos percentuais de F1 vista no single-seed (0.917 vs 0.891) parecia sugerir que a cor ajuda. O teste pareado mostra que não — o sinal que separa screenshot de foto nesta tarefa (bordas de UI vs textura de cena) já está presente no canal de luminância; a informação cromática adicional dos 3 canais RGB não move a agulha de forma estatisticamente distinguível de zero. Isso é coerente com a intuição do parágrafo anterior: o que a CNN usa para discriminar é a **geometria** da imagem (bordas retas vs gradientes orgânicos), não a paleta de cores.

**Lição metodológica (recorrente nos projetos anteriores):** a tabela de uma seed só (single-seed) é enganosa por dois motivos simultâneos aqui — (1) o dataset é pequeno (90 imagens de teste), então qualquer métrica isolada tem margem de erro grande; e (2) a comparação entre modalidades usando seeds independentes (multi-seed não pareado) não separa quanto da variação observada é efeito real de quanto é ruído de amostragem compartilhado. Só o teste pareado, ao fixar o mesmo split para as modalidades comparadas em cada seed, resolve essa ambiguidade — e o resultado é qualitativamente diferente do que a tabela single-seed sugeria: o ranking "RGB > gray > 1D" continua valendo nas médias, mas apenas **uma** das duas comparações (2D vs 1D) é estatisticamente sustentável.

**Ablação de filtros/kernel:** o padrão observado (pico em `nf=16`, `k=3`, queda acentuada em `k=7`) é consistente com a intuição de que, numa imagem pequena (64×64) e um dataset pequeno (360 treino), tanto a capacidade excessiva (`nf=32`, `k=7`) quanto a insuficiente (`nf=4`) degradam o desempenho — só que a ablação foi rodada **apenas na seed=42**, sem repetição multi-seed, então a posição exata do ótimo (`nf=16` vs `32`, por exemplo) deve ser lida como indicativa, não como um veredito estatístico como o do 2D-vs-1D.

## Limitações

- **Dataset pequeno e pessoal**: 450 imagens (200 foto / 250 screenshot) de uma coleção específica do autor. O desbalanceamento leve (250 vs 200) e a origem pessoal (fotos de pessoas, screenshots do próprio uso) limitam a generalização do resultado para "foto vs screenshot" em geral — outras coleções de foto (paisagens, documentos) ou de screenshot (apps mobile, jogos) podem ter estatísticas de borda/textura diferentes.
- **Alta variância do modelo**: com teste de apenas 90 imagens, o desvio-padrão do F1 entre seeds chega a 0,068 (2D-gray) — maior que muitas das diferenças que se está tentando medir. Isso é o motivo direto de precisar do teste pareado em vez de comparar médias ± desvio de rodadas independentes.
- **Assimetria de seeds entre experimentos**: a tabela multi-seed usa 3 seeds (rodada oficial default do script) enquanto o teste pareado usa 30 seeds — números de repetição diferentes por desenho do experimento (3 seeds já bastam para ilustrar a variância; 30 seeds são necessários para o teste pareado ter poder estatístico). Isso deve ficar explícito para não confundir "resultado oficial" (3 seeds) com "veredito pareado" (30 seeds) como se fossem a mesma rodada.
- **Ablação não pareada/não multi-seed**: os números de `nf` e `k` vêm de uma única seed (42); não se sabe se o ótimo em `nf=16, k=3` é robusto a variação de seed, apenas que a tendência qualitativa (kernel grande demais prejudica; poucos filtros sub-representam) é plausível dado o tamanho da imagem e do dataset.
- **Sem validação cruzada nem data augmentation**: com um dataset de 450 imagens, k-fold e/ou augmentation (flips, crops, jitter de cor) tenderiam a estabilizar as estimativas e são um próximo passo natural — especialmente relevante para testar se a conclusão "cor não ajuda" se sustenta com mais dados de treino.
- **Feature maps são ilustrativos, não quantitativos**: a leitura de "bordas para screenshot, textura difusa para foto" é uma interpretação visual de 6 filtros de uma única execução — não há uma métrica formal (ex.: energia de borda por canal) quantificando esse contraste no relatório.
