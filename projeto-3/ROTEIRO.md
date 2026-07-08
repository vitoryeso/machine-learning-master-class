# Roteiro de vídeo — Projeto 3 (Regressão Linear: GD vs LMS)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Neste miniprojeto eu comparo dois jeitos de treinar **o mesmo** modelo de regressão linear — mesma loss, o erro quadrático médio — mudando só **como** os pesos são atualizados. De um lado o Batch Gradient Descent, que olha todos os 300 exemplos e dá uma única atualização por época. Do outro, o LMS, ou Widrow-Hoff, que corrige os pesos a cada exemplo individual. A diferença parece pequena, mas olhem os números: na mesma quantidade de épocas, o GD dá 200 passos de gradiente e o LMS dá **60 mil**. Os dados são sintéticos de propósito — como gerei a partir de `y = 3x1 − 2x2 + 1` mais ruído, eu conheço os pesos verdadeiros e posso medir o erro absoluto, não só a loss. Tudo em Rust puro, sem bibliotecas de machine learning."

### Slide 2 — Método e derivação *(1:10 – 2:30)*
"Começo pela derivação, que o enunciado pede. A loss é a média de `(ŷ − y)²`. Derivando em relação a `w`, a regra da cadeia traz o fator 2 e sobra a média de `(ŷ − y)` vezes `x` — esse é o gradiente exato, e o fator 2 eu absorvo no learning rate. À direita estão as duas regras lado a lado. O GD usa esse gradiente médio sobre todo o dataset, com learning rate 0,01. O LMS usa o erro de **um** exemplo por vez, com learning rate menor, 0,001, pra manter a estabilidade online. E é aqui que nasce a diferença de 200 contra 60 mil passos: o LMS dá uma atualização por amostra, 300 por época."

### Slide 3 — Resultado *(2:30 – 3:55)*
"O gráfico à esquerda mostra o MSE por época, pra uma seed. O LMS, em vermelho, despenca em poucas épocas; o GD, em azul, desce devagar. E isso não é sorte de uma rodada só: rodei com dez seeds, num dataset bem maior, e o padrão se confirma — o LMS fica praticamente no piso teórico de 0,25, de forma consistente, com os pesos recuperados sempre muito perto dos verdadeiros. O GD fica em torno do dobro disso, também de forma estável entre seeds. Mas atenção a um ponto metodológico importante: o GD **não convergiu de propósito**. Eu escolhi o learning rate 0,01 justamente pra evidenciar a sensibilidade do GD ao hiperparâmetro — não é uma comparação desequilibrada, é um experimento controlado pra mostrar o trade-off."

### Slide 4 — Conclusão *(3:55 – 5:00)*
"Concluindo: o LMS venceu **neste cenário** porque seus 60 mil passos o levaram perto do ótimo — o que o torna ideal pra datasets grandes ou em streaming. Mas não é de graça: o GD é mais estável, sem oscilação, enquanto o LMS com learning rate grande diverge. O eixo central do projeto é esse trade-off — gradiente exato com poucos passos contra muitos passos ruidosos, mesmo custo por época, estratégias opostas. E a escolha do learning rate é crítica e independente para cada algoritmo. Obrigado."

---
*4 slides · ~5:00. Se estourar, comprima o slide 2 falando só a regra final do gradiente.*
