# Roteiro de vídeo — Projeto 7 (Redes RBF)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:05)*
"Olá. Neste miniprojeto eu construí uma rede de funções de base radial — uma RBF Network — do zero, em numpy, para um problema real: classificação binária da minha coleção pessoal de imagens, usando as features CLIP, em três tarefas. A RBF tem uma camada oculta de gaussianas e uma saída linear. O foco que o enunciado pede é entender como a **escolha dos centros** dessas gaussianas afeta a classificação — e eu comparo três estratégias: centros aleatórios, por K-means, e por subconjunto dos dados."

### Slide 2 — Método *(1:05 – 2:30)*
"A rede tem duas equações. A primeira é a ativação de cada neurônio radial: φ_j de x igual a exponencial de menos a distância ao quadrado entre x e o centro c_j, dividida por dois sigma ao quadrado. Deixa eu explicar cada termo: **x** é a entrada, o vetor de features CLIP da imagem; **c_j** é o centro, um protótipo no espaço — é justamente o que vou ablar; **sigma** é a largura da gaussiana, que eu calibro pela mediana das distâncias dos pontos aos centros; e **φ_j** é a ativação, que vale perto de 1 quando x está próximo do centro e cai pra zero quando está longe. A segunda equação é a saída: y-chapéu é a soma das ativações φ_j ponderadas pelos pesos **w_j**, mais um viés **b** — e eu resolvo esses pesos em forma fechada, por mínimos quadrados. A decisão final é classe 1 se y-chapéu for maior ou igual a 0,5."

### Slide 3 — Resultados *(2:30 – 3:50)*
"Com 30 centros por K-means, em full-dim, olhando a média de três seeds pra não depender de sorte de inicialização: no macro-grupo, que é a tarefa mais estável, dá 0,92 de F1; screenshot vs foto fica em 0,67; e pessoa vs não, em 0,68 — essas duas, por serem desbalanceadas, variam mais de seed pra seed, com poucas gaussianas a rede cobre mal as regiões da classe rara. O gráfico à direita é a ablação principal: F1 contra o número de centros, para as três estratégias. A lição, que se confirma rodando várias seeds e comparando estatisticamente, é clara: **a escolha dos centros importa**. O K-means vence de forma robusta e consistente em todos os valores de k, porque posiciona os centros onde os dados realmente estão. Já entre aleatório e subconjunto, a diferença é zero de verdade — um empate real, não falta de dado. E aumentar o número de centros melhora, com retorno decrescente."

### Slide 4 — Partições e conclusão *(3:50 – 5:00)*
"Esta ilustração no plano 2D mostra a intuição: cada estrela é um centro c_j; em volta dele a gaussiana acende, e a soma ponderada dessas gaussianas desenha a região laranja que separa a classe. Repare que os centros do K-means se espalham pela nuvem de pontos, cobrindo bem o espaço — é por isso que eles ganham. Concluindo: numa rede RBF, o classificador é só a combinação linear de bumps locais, então **onde** você coloca esses bumps — os centros — é a decisão de projeto mais importante, e posicioná-los com K-means, que aprende a geometria dos dados, é o que dá o melhor resultado. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só "K-means vence em todo k".*
