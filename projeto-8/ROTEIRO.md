# Roteiro de vídeo — Projeto 8 (SVM com kernel RBF)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:05)*
"Olá. Neste miniprojeto eu treino uma máquina de vetores de suporte — uma SVM — com kernel RBF, para classificação binária da minha coleção pessoal de imagens, nas mesmas três tarefas dos projetos anteriores. A SVM busca a fronteira de **margem máxima** — a que separa as classes com a maior folga possível. O foco do enunciado é interpretar os **vetores de suporte** como os centros efetivos do modelo, e analisar o efeito dos hiperparâmetros C e gamma."

### Slide 2 — Método *(1:05 – 2:35)*
"A SVM com kernel é definida por estas equações. Primeiro o kernel: K de x e x_i igual a exponencial de menos gamma vezes a distância ao quadrado entre eles. Explicando os termos: **x** é a entrada, as features da imagem; **x_i** são os vetores de suporte, que são pontos do treino sobre ou dentro da margem (incluindo os mal classificados); **K** mede a similaridade gaussiana entre x e cada vetor de suporte; e **gamma** controla o inverso da largura desse kernel — gamma alto deixa o kernel estreito, com influência bem local; gamma baixo deixa o kernel largo e suave. A decisão é a função f de x: a soma, sobre os vetores de suporte, de **alpha_i** vezes **y_i** vezes o kernel, mais um viés b — onde alpha_i é o peso aprendido de cada vetor de suporte e y_i é o rótulo dele, mais ou menos um. A classe sai do sinal de f de x. E tem ainda o **C**, a regularização: ele controla o trade-off entre uma margem larga e errar menos no treino."

### Slide 3 — Resultados *(2:35 – 3:55)*
"Os números, em full-dim, com C igual a 10 e gamma scale: macro-grupo 0,95 de F1, pessoa vs não 0,89, e screenshot vs foto 0,76. Uma observação prática: a SVM escala com o quadrado do número de amostras, então eu subamostrei o treino para 6 mil pontos. À direita está a varredura de C contra gamma, com o F1 no teste. E aqui está a lição: **gamma é o hiperparâmetro crítico**. Quando gamma fica alto, tipo 0,1, o kernel só enxerga o vizinho imediato de cada ponto, o modelo decora o treino e o F1 colapsa pra 0,09. Com gamma baixo ou no modo scale, generaliza bem, na casa de 0,95. Já o C, dentro de uma faixa ampla, quase não muda o resultado."

### Slide 4 — Vetores de suporte e conclusão *(3:55 – 5:00)*
"Esta ilustração 2D mostra os vetores de suporte como círculos. A ideia central da SVM é essa: só esses pontos críticos, os que ficam perto da fronteira, é que a definem — todos os outros pontos poderiam ser removidos sem mudar nada. Repare que no plano 2D comprimido as classes se sobrepõem muito, então quase um quarto dos pontos vira vetor de suporte: é uma faixa larga de incerteza sustentando a fronteira. Concluindo: a SVM-RBF entregou o melhor desempenho entre os modelos baseados em similaridade, e a análise mostra que escolher gamma corretamente é o que separa um modelo que generaliza de um que decora. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só "gamma alto colapsa, scale generaliza".*
