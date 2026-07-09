# INSUMO — Discussão de design/física do pêndulo-N (fonte: ChatGPT, resgatado pelo Vitor)

Discussão conceitual que embasa a MOTIVAÇÃO e o DESIGN EXPERIMENTAL do paper.
Threads-chave extraídos abaixo; texto integral logo em seguida.

## Threads para o artigo
1. **Framing "discreto → contínuo":** o pêndulo-N é uma escada do rígido (n=1) → cadeia articulada → limite de corda/haste flexível (n→∞). O framing publicável: *"cart-mounted flexible inverted pendulum as the continuum limit of an n-link pendulum"*. Mantém **L_total e M_total constantes** enquanto l_i = L_total/n → 0.
2. **Rationale da escada de n:** n ∈ {1,2,3,5,8,13,21} (Fibonacci) cobre 3 regimes — rígido (1–3), articulado complexo/caótico (5–13), quase-contínuo (21+). L_total=1.0 m, M_total=1.0 kg, x_lim=±2 m, F=±10 N, dt=0.005 s. **Regra crítica:** `l_i = L_total/n` com L_total FIXO (não l_i fixo — senão o pêndulo total cresce e muda o problema).
3. **Insight que amarra o ângulo de sistemas (IMPORTANTE):** *"policy capacity is usually not the bottleneck; the bottleneck is structured exploration under unstable, underactuated dynamics."* O estado é pequeno (2+3n; n=21 → só 65 dims), uma MLP [128,128] segura n≤10. Ou seja: não dá pra resolver com rede maior → **precisa de mais AMOSTRAS** → é por isso que a **aceleração/throughput (CUDA/bf16) é a alavanca real**. (Este é o elo que conecta a física ao ângulo de otimização escolhido.)
4. **Para generalizar entre n:** viés estrutural — 1D-Conv (cadeia ordenada), GNN/message-passing (elos=nós, juntas=arestas), ou shared-per-link-encoder+pooling. (Trabalho futuro.)
5. **Ancoragem física real:** Quanser linear (pêndulos ~33,65 cm e 64,13 cm, curso do carrinho 81,4 cm) → sistemas de lab na escala de dezenas de cm. Refs implícitas: *flexible inverted pendulum on a cart*, *energy shaping control*, *double inverted pendulum* (elos 0.5 m, x∈[-2,2], F∈[-10,10]).

> NOTA DE HONESTIDADE: esta discussão sonhava com n→21 (limite contínuo); o projeto REAL alcançou n≤5 (n=4 parcial, n=5 platô ~27%). O paper usa a escada como MOTIVAÇÃO/design mas reporta resultados só até n=5.

---

## Texto integral (ChatGPT)

[Sobre "equilibrar uma corda" — 3 níveis: pendurada (fácil), ideal invertida (mal-posta, corda só transmite tensão não compressão), haste/cabo flexível invertido (difícil mas possível → controlar modos de vibração, estado = [cart, θ, a₁, a₂...], truncar em poucos modos). Framing: cart-pole clássico → flexible inverted pendulum → PDE control → continuum soft body control. Meio-termo publicável: "cart-mounted flexible inverted pendulum as the continuum limit of an n-link pendulum".]

[Range de tamanho de elo: L_total ∈ [0.5, 1.0] m, l_i = L_total/n. Escada n=1,2,3,5,8,13,21 (l: 1.0→0.048 m). Manter L_total constante (limite contínuo: n→∞, l_i→0, n·l_i=L_total const). Escala física Quanser ~33-64 cm. dt ≤ T_min/50, T_i≈2π√(l_i/g). Curso do carrinho x_max ∈ [1.5,3.0]·L_total pra swing-up. Config: L=1.0, M=1.0, x=±2, F=±10 N, dt=0.005.]

[Tamanho da rede: capacity NÃO é o gargalo. Estado 2+3n pequeno (n=21→65). MLP [128,128] pra n≤10, [256,256] pra n=13-21. O difícil = reward shaping + curriculum (n=1→...→21; balance→swingup→robustez) + normalização + dinâmica confiável. Pra generalizar em n: 1D-Conv / GNN / shared-encoder+pooling. Frase-chave: "policy capacity is usually not the bottleneck; the bottleneck is structured exploration under unstable, underactuated dynamics."]
