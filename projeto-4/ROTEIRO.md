# Roteiro de vídeo — Projeto 4 (Regressão Logística do Zero)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Neste miniprojeto eu construí uma regressão logística **do zero**, em Rust, sem nenhuma biblioteca de machine learning — sigmoid, função de custo e gradiente, tudo na mão. O problema é classificação binária em 2D: duas gaussianas com 300 pontos, dividas 80/20 entre treino e teste. Três números resumem o resultado e eu vou voltar neles: **96,7% de acurácia no teste**, um gradiente que se reduz elegantemente a `(p menos y)`, e um erro de Bayes teórico de **3,57%** — o melhor que qualquer classificador linear conseguiria neste problema."

### Slide 2 — Formulação e gradiente *(1:10 – 2:35)*
"A formulação: faço a combinação linear, passo pela sigmoid pra virar uma probabilidade, e o custo é a Binary Cross-Entropy. Um detalhe de implementação que uma biblioteca esconde: a sigmoid foi escrita em dois ramos, um para z positivo e outro para negativo, pra evitar overflow numérico. E aqui está a parte mais bonita, à direita: quando você aplica a regra da cadeia entre a BCE e a sigmoid, os termos **se cancelam** e o gradiente sobra simplesmente como a média de `(p menos y)` vezes `x` — o erro de previsão vezes a entrada. Treinei com learning rate 0,1, por mil épocas."

### Slide 3 — Fronteira e desempenho *(2:35 – 3:55)*
"À esquerda, a fronteira de decisão aprendida: o fundo é um heatmap da probabilidade prevista, e a reta é onde p é igual a 0,5. Como o modelo é linear, a fronteira é necessariamente uma reta, separando as duas gaussianas na diagonal. À direita, as métricas: 95,4% de acurácia no treino e **96,7% no teste**, com precisão, recall e F1 todos acima de 0,95. O gap pequeno entre treino e teste indica boa generalização, sem overfitting visível."

### Slide 4 — Conclusão *(3:55 – 5:00)*
"Concluindo: a implementação do zero está correta — a acurácia de teste confirma o gradiente analítico. E o detalhe que fecha a história: o erro de teste, de 3,33%, praticamente **encosta no limite de Bayes** de 3,57%, dado por Phi de menos raiz de 13 sobre 2. Isso mostra que o gargalo aqui é o **ruído intrínseco do problema**, não o modelo. Como limitações: por ser linear, a fronteira é uma reta — dados não-lineares exigiriam features polinomiais; e como melhorias eu aponto regularização L2 e um critério formal de parada. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só acurácia e o gap treino/teste.*
