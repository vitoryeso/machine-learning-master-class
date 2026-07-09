# Roteiro de vídeo — Projeto 6 (Random Forest: Ensemble e Variabilidade)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Este miniprojeto fecha o arco do estudo de variância que comecei no Projeto 5. Lá eu tinha diagnosticado, com teste pareado multi-seed, que a árvore de decisão não-podada é um estimador de **alta variância** — o desvio entre seeds não caía com mais dados, porque a instabilidade era do modelo, não do erro amostral. Aqui eu construí, também manualmente, um **Random Forest**: bagging, ou seja bootstrap das amostras, mais subamostragem aleatória de features por split, mais voto de maioria, reaproveitando a árvore CART do Projeto 5 como base. Os números já contam a história: rodando o mesmo multi-seed pareado, a acurácia da árvore única foi 0,943 mais ou menos 0,034; a floresta foi 0,960 mais ou menos 0,024 — a média sobe **e** o desvio cai. E o teste pareado deu t igual a 4, ou seja, a floresta é consistentemente melhor, não é sorte de seed."

### Slide 2 — Método *(1:10 – 2:25)*
"O Random Forest tem duas fontes de aleatoriedade que geram diversidade entre as árvores. Primeiro, o bootstrap: cada árvore da floresta é treinada numa reamostragem do treino sorteada com reposição — algumas amostras aparecem repetidas, outras ficam de fora. Segundo, a subamostragem de features: a cada split, em vez de considerar as duas features disponíveis, a árvore sorteia um subconjunto — aqui, só uma de duas — e busca o melhor corte só dentro dele. Isso é o 'random' do nome. Cada árvore sozinha fica mais instável e enviesada localmente, mas a predição final não é de uma árvore só: é o **voto de maioria** das 51 árvores da floresta, um número ímpar para não empatar no caso binário. É essa combinação que cancela boa parte do ruído individual."

### Slide 3 — Resultados *(2:25 – 3:50)*
"À esquerda, a figura mostra quatro árvores individuais da floresta — repare como as partições são diferentes entre si, fruto do bootstrap e das features sorteadas — e embaixo, a árvore única do Projeto 5 ao lado da floresta combinada e do mapa de fração de votos, que mostra a incerteza na fronteira. À direita, o gráfico de barras resume o multi-seed pareado: floresta com acurácia média maior e barra de erro visivelmente **menor** que a árvore única — 0,943 mais ou menos 0,034 contra 0,960 mais ou menos 0,024, uma redução de variância de cerca de 1,4 vezes. E o teste pareado, que compara floresta e árvore na mesma realização de dados por seed, deu diferença de mais 0,017 de acurácia com t de 4 — um efeito grande e consistente."

### Slide 4 — Conclusão *(3:50 – 5:00)*
"Concluindo: a tese do Projeto 5 se confirma na prática. O bagging **reduz a variância sem piorar o viés** — na verdade aqui a média até melhorou, então foi uma dupla vitória, não um trade-off. Eu também variei o número de árvores de 1 até 101 e o desvio entre seeds cai de forma monotônica, com retornos decrescentes — a maior parte do ganho de estabilidade já aparece com algumas dezenas de árvores, batendo com a intuição de que a variância de uma média de estimadores cai proporcionalmente ao número de árvores. O custo é computacional: treinar 51 árvores é 51 vezes mais caro que uma só, mas cada árvore é independente, então dá pra paralelizar sem dor. Como trabalho futuro, ficaria testar profundidades diferentes para as árvores-base e comparar contra o `RandomForestClassifier` do scikit-learn como conferência da implementação manual. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 2 lendo só o bootstrap e o voto de maioria, sem detalhar a subamostragem de features.*
