# Roteiro de vídeo — Projeto 5 (Árvore de Decisão: Gini vs Entropia)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Neste miniprojeto eu construí uma árvore de decisão **manualmente**, do zero, sem usar o `DecisionTreeClassifier` do sklearn — implementei o cálculo de impureza, a busca de split e a recursão na mão. O objetivo é comparar os dois critérios clássicos de divisão, Gini e Entropia, no mesmo dataset 2D de 500 amostras, com split 80/20. Os três números do slide já adiantam a história: rodando multi-seed, com 10 seeds, **Gini e Entropia empatam** em acurácia de teste; a diferença real está na profundidade das árvores, 9 contra 13 níveis; e ambas atingiram 100% no treino — o que, como vou mostrar, é overfitting."

### Slide 2 — Critérios *(1:10 – 2:25)*
"Os dois critérios medem a impureza de um nó. O Gini é 1 menos a soma dos quadrados das proporções das classes, com máximo de 0,5 no caso binário. A Entropia usa o logaritmo, chega a 1,0, e a partir dela definimos o ganho de informação: a impureza do pai menos a média ponderada da impureza dos filhos. A construção é igual nos dois: uma busca gulosa que, a cada nó, escolhe a feature e o limiar que **mais reduzem a impureza**, e cresce até as folhas ficarem puras. A diferença, que vai importar no final, é que a Entropia tem maior curvatura em torno de p igual a 0,5."

### Slide 3 — Fronteiras e desempenho *(2:25 – 3:50)*
"À esquerda está a comparação pareada: de um lado o cruzamento bruto entre os dois critérios, do outro o resultado emparelhado seed a seed — e é aí que aparece o veredito de verdade. À direita, a tabela mostra uma seed específica, a 42, só como ilustração: nela o Gini teve 96% de acurácia contra 94% da Entropia, com recall de 0,98 e F1 de 0,96. Mas isso é sorte de amostra: rodando pareado em 30 seeds, a diferença de acurácia entre os dois critérios é praticamente **zero** — bem menor que a variação natural entre seeds. O que de fato muda é a estrutura da árvore: nessa seed, 9 níveis e 26 folhas no Gini, contra 13 níveis e 27 folhas na Entropia — a Entropia consistentemente gera árvores mais profundas."

### Slide 4 — Conclusão *(3:50 – 5:00)*
"Concluindo: o critério **não teve efeito real na acurácia**. Com 30 seeds e teste pareado, a diferença entre Gini e Entropia é praticamente **zero** — bem menor que o ruído natural entre seeds. Os 2 pontos percentuais que a seed única mostrava eram só sorte de amostra pequena. O único efeito real é estrutural: a Entropia consistentemente gera árvores mais profundas, em média 15,5 contra 17,8 níveis sem poda, porque sua maior curvatura reordena os splits gulosos. E podando as duas em profundidade máxima 5, o empate se mantém e a acurácia até melhora um pouco — confirmando que aquela profundidade extra era overfitting puro, não sinal. Isso bate com o que vi sem poda: 100% no treino contra 94 a 96% no teste, com 8 a 10 folhas de uma única amostra — memorização pura. O sweep de profundidade mostra que níveis menores já atingem o platô, então o excesso é supérfluo. Em produção, eu usaria `max_depth`, poda e validação cruzada. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só a acurácia pareada e a diferença de profundidade.*
