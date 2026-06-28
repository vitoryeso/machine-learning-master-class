# Roteiro de vídeo — Projeto 10 (CNN para dados 1D e 2D)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:05)*
"Olá. Neste miniprojeto eu construo redes convolucionais para um problema real: distinguir foto de screenshot, usando 450 imagens da minha coleção pessoal — fotos de pessoas contra capturas de tela, baixadas do meu Google Drive. A ideia é demonstrar como a arquitetura convolucional se adapta a diferentes tipos de dado: comparo uma CNN 2D, que vê a imagem inteira, com uma CNN 1D; comparo entrada RGB contra escala de cinza, pra ver o efeito dos canais; e no final visualizo os feature maps, ou seja, o que cada filtro realmente enxerga. O treino roda na minha GPU, uma RTX 3060."

### Slide 2 — Método *(1:05 – 2:35)*
"O coração da CNN é a convolução. A equação diz: o feature map S na posição i,j é a soma, sobre uma pequena janela, do produto entre a imagem e o filtro, mais um viés. Deixa eu explicar cada termo: **I** é a entrada, a imagem, com altura, largura e canais — RGB tem 3 canais, escala de cinza tem 1; **K** é o filtro, ou kernel, uma matriz pequena de pesos que a rede **aprende** para detectar um padrão local, como uma borda; **b** é o viés desse filtro; e **S** é o feature map resultante, que mede a resposta do filtro em cada posição. Depois aplico uma não-linearidade, a ReLU, e um max-pooling, que reduz a resolução resumindo cada região. Empilho duas dessas camadas e termino numa camada totalmente conectada que dá a decisão. Para a versão 1D, a mesma ideia: o sinal é o perfil de intensidade por linha da imagem — um vetor — e o kernel 1D desliza sobre ele."

### Slide 3 — Resultados *(2:35 – 3:55)*
"Os números mostram três lições. Primeira: a CNN 2D RGB chega a 0,92 de F1; a versão em escala de cinza cai um pouco, pra 0,89 — ou seja, **o canal de cor agrega informação**. Segunda: a CNN 1D, no perfil de linha, fica em 0,87 — pior que a 2D, porque ao colapsar a imagem num vetor eu **jogo fora a estrutura espacial** que a convolução 2D explora. Terceira, na ablação à direita: o número de filtros tem um ponto ótimo em 16, e o tamanho do kernel também — kernel 3 é o melhor, e kernel 7 despenca pra 0,79, porque um filtro grande demais numa imagem de 64 por 64 perde a localidade."

### Slide 4 — Feature maps e conclusão *(3:55 – 5:00)*
"E esta é a parte que dá intuição. Aqui estão os feature maps da primeira camada convolucional para um screenshot, em cima, e uma foto, embaixo. No screenshot, repare que os filtros acendem exatamente nas bordas retangulares da caixa de interface — eles viraram detectores de aresta e linha. Já na foto, os mesmos filtros respondem de forma difusa, às texturas e gradientes da cena natural, sem bordas nítidas. É justamente essa diferença de resposta que a rede aprende a usar pra separar as duas classes. E é por isso que a versão 2D vence a 1D: ela preserva essa estrutura espacial que os filtros exploram. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só "2D RGB 0.92, 1D 0.87, k=3 melhor".*
