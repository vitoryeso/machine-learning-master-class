# Roteiro de vídeo — Projeto 9 (ANN/MLP: Fronteiras Não-Lineares)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Neste miniprojeto eu construí uma rede neural MLP **do zero**, em numpy — forward e backpropagation na mão, sem framework. E em vez de um dataset de brinquedo, ataco um problema **real**: classificação binária da minha coleção pessoal de imagens, usando as features CLIP do projeto anterior, em três tarefas — pessoa vs não, screenshot vs foto, e um macro-grupo vs o resto. Um detalhe de design importante: o desempenho eu meço nas features completas, em alta dimensão; o plano 2D, via PCA, eu uso só pra **ilustrar** como os neurônios particionam o espaço — porque comprimir 512 dimensões em 2 perde quase toda a informação."

### Slide 2 — Método *(1:10 – 2:25)*
"A rede: cada camada faz uma combinação linear seguida de ativação — relu ou tanh nas ocultas, sigmoid na saída pra dar uma probabilidade. O backpropagation é manual, e aqui está a conexão com os projetos anteriores: o delta da saída é exatamente `(p menos y)` quando uso cross-entropy — o mesmo gradiente da regressão logística do projeto 4 — ou `2(p−y)·p(1−p)` quando uso MSE. A partir daí, propago o erro pelas camadas multiplicando pela derivada da ativação. Três coisas eu ablo: a função de perda (MSE contra cross-entropy), o tamanho da camada oculta (de zero a 128 neurônios), e a visualização 2D de como cada neurônio recorta o plano."

### Slide 3 — Resultados *(2:25 – 3:50)*
"Os números, em full-dim: macro-grupo 97,6% de acurácia, pessoa 96,3%, screenshot vs foto 92,4%. E aqui está a validação da minha escolha de treinar em full-dim: a tarefa screenshot, que no plano 2D PCA tinha F1 de 0,29 — praticamente morta —, em alta dimensão sobe pra 0,75. O gráfico à direita mostra duas ablações: à esquerda, MSE contra cross-entropy convergindo — as duas chegam quase no mesmo lugar, com a cross-entropy um pouco mais rápida; à direita, o F1 contra o número de neurônios ocultos."

### Slide 4 — Partições e interpretação *(3:50 – 5:00)*
"E esta é a parte mais bonita: cada uma das 4 retas tracejadas é um neurônio oculto, e a rede **combina** essas retas numa região não-linear que isola o grupo laranja — é literalmente a camada oculta particionando o plano. Duas conclusões, e aqui eu rodei múltiplas seeds pra não confiar num resultado isolado. Primeira: MSE e cross-entropy praticamente empatam, mas não são idênticas — repetindo o treino em várias seeds, a cross-entropy fica um fio consistentemente acima do MSE onde há sinal, algo como 0,965 contra 0,961 no macro-grupo. É uma vantagem pequena, mas que se repete toda vez, então é real. Segunda: uma rede com **zero** camadas ocultas — ou seja, uma regressão logística — já vai muito bem sozinha, mas ter a camada oculta ajuda sim, um pouquinho: no multi-seed ela fica reprodutivelmente cerca de um ponto de F1 acima da versão sem camada oculta. É um ganho pequeno, porque as features CLIP já são quase linearmente separáveis, mas é ganho de verdade, não ruído de uma seed só. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só os 3 números de F1.*
