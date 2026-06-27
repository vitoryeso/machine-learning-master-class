# Roteiro de vídeo — Projeto 2 (Classificação da taxonomia descoberta)

**Alvo:** ~5 min · 4 slides (~75s cada) · abra `slides.html`, tela cheia com `F`.

---

### Slide 1 — Problema *(0:00 – 1:10)*
"Olá. Este miniprojeto fecha o ciclo do projeto anterior. No projeto 1, eu não tinha rótulos para a minha coleção pessoal de imagens, então usei K-means pra descobrir grupos. Agora a pergunta é: e se eu usar esses grupos descobertos como **classes** e treinar um classificador pra atribuí-las a imagens novas? É exatamente isso — classificação sobre uma taxonomia que eu mesmo descobri, sem verdade externa. São 22 mil imagens, organizadas em dois níveis pelo clustering: 5 macro-grupos e 25 subclasses-folha. E o classificador é um linear probe sobre features pré-extraídas."

### Slide 2 — Pipeline e o teste honesto *(1:10 – 2:35)*
"O pipeline é este: embeddings CLIP entram no K-means hierárquico, que produz os dois níveis de classes; esses rótulos viram o alvo do linear probe. Mas tem uma armadilha metodológica que eu quero destacar, porque é o ponto central do projeto. Se eu treinar o classificador sobre os **mesmos** embeddings CLIP que geraram os clusters, a acurácia vai ser inflada — os clusters são, por construção, células de Voronoi naquele espaço, então classificá-los de volta é quase trapaça. A solução: classificar sobre features de um modelo **independente**, o ConvNeXt. Se a taxonomia for real, um outro espaço de features também deve conseguir recuperá-la. Eu rodo os dois e comparo — o CLIP vira o controle, o ConvNeXt vira o teste honesto."

### Slide 3 — Resultados *(2:35 – 3:55)*
"E os números mostram exatamente isso. Com o CLIP, o classificador acerta 95% no nível macro e 89% no nível folha — alto, como esperado pela circularidade. Já com o ConvNeXt, independente, cai para 83% no macro e 66% na folha. O gráfico deixa o contraste claro: as barras cinza, do CLIP, ficam sempre acima das azuis, do ConvNeXt. Esse afastamento não é ruído — é a medida de quanto da 'acurácia' do CLIP seria apenas auto-confirmação. O teste foi feito em 4.466 imagens, com split estratificado 80/20, e as métricas são macro-averaged pra não mascarar as classes pequenas."

### Slide 4 — Interpretação *(3:55 – 5:00)*
"A leitura final tem três pontos. Primeiro: o nível macro é **real** — um espaço independente recupera os 5 grupos com 83%, então a estrutura grossa não é um artefato do CLIP. Segundo: o nível fino é **CLIP-específico** — as 25 subclasses caem pra 66% no ConvNeXt, ou seja, as distinções sutis vivem sobretudo na semântica que o CLIP aprendeu. Terceiro: a circularidade ficou **mensurável** — o gap entre os dois espaços quantifica o viés. Como limitações: os rótulos vêm de clustering não-supervisionado, sem verdade externa; o linear probe não captura fronteiras não-lineares; e as classes são desbalanceadas, de 295 a 4.218 imagens. Obrigado."

---
*4 slides · ~5:00. Se estourar, encurte o slide 1 indo direto pro "classificar a taxonomia descoberta".*
