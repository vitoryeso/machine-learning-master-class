# Roteiro de vídeo — Projeto 5 (Árvore de Decisão: Gini vs Entropia)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Neste miniprojeto eu construí uma árvore de decisão **manualmente**, do zero, sem usar o `DecisionTreeClassifier` do sklearn — implementei o cálculo de impureza, a busca de split e a recursão na mão. O objetivo é comparar os dois critérios clássicos de divisão, Gini e Entropia, no mesmo dataset 2D de 500 amostras, com split 80/20. Os três números do slide já adiantam a história: a acurácia de teste foi 96% com Gini contra 94% com Entropia; as árvores têm profundidades diferentes, 9 e 13; e ambas atingiram 100% no treino — o que, como vou mostrar, é overfitting."

### Slide 2 — Critérios *(1:10 – 2:25)*
"Os dois critérios medem a impureza de um nó. O Gini é 1 menos a soma dos quadrados das proporções das classes, com máximo de 0,5 no caso binário. A Entropia usa o logaritmo, chega a 1,0, e a partir dela definimos o ganho de informação: a impureza do pai menos a média ponderada da impureza dos filhos. A construção é igual nos dois: uma busca gulosa que, a cada nó, escolhe a feature e o limiar que **mais reduzem a impureza**, e cresce até as folhas ficarem puras. A diferença, que vai importar no final, é que a Entropia tem maior curvatura em torno de p igual a 0,5."

### Slide 3 — Fronteiras e desempenho *(2:25 – 3:50)*
"À esquerda estão as fronteiras de decisão das duas árvores. Repare que as partições são visivelmente diferentes — os retângulos não coincidem. À direita, as métricas: o Gini levou vantagem em tudo, com 96% de acurácia, recall de 0,98 e F1 de 0,96, contra 94% da Entropia. As árvores também diferem em estrutura: 9 níveis e 26 folhas no Gini, contra 13 níveis e 27 folhas na Entropia. E apesar de treinarem no mesmo dado, as predições das duas divergem em 4 das 100 amostras de teste — ou seja, a escolha do critério realmente muda o modelo."

### Slide 4 — Conclusão *(3:50 – 5:00)*
"Concluindo: o critério teve um efeito pequeno mas real. O Gini ficou 2 pontos percentuais acima e, curiosamente, gerou uma árvore mais rasa — 9 contra 13 níveis. A explicação é geométrica: a maior curvatura da entropia reordena os splits gulosos e acaba abrindo mais níveis pra separar os mesmos pontos. Também confirmei overfitting nas duas: 100% no treino contra 94 a 96% no teste, com cerca de 8 a 10 folhas contendo uma única amostra — memorização pura. O sweep de profundidade mostra que níveis menores já atingem o platô de acurácia, então a profundidade extra é supérflua. Em produção, eu usaria `max_depth`, poda e validação cruzada pra controlar isso. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 3 lendo só acurácia e a divergência de 4 amostras.*
