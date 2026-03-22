# Prompt para o Cowork (Claude Web)

Cole o texto abaixo no Claude Web/Desktop para gerar os slides via Cowork.
Arraste os graficos da pasta output/ para dentro da conversa.

---

Crie uma apresentacao de 3 slides para disciplina de mestrado sobre a seguinte analise de Machine Learning. Estilo tecnico, limpo, sem emojis. Cada slide deve ter bullet points concisos e espaco para graficos.

=== SLIDE 1: Metodologia ===

Titulo: "Analise Exploratoria com K-Means em Embeddings de Imagens"

- Dataset: 22.258 imagens pessoais (fotos, wallpapers, AI art, screenshots, memes)
- Amostra: 8.000 imagens (amostragem aleatoria simples, 63 corrompidas descartadas)
- Embedding: CLIP ViT-B/32 via fastembed-rs (Rust, ONNX runtime) - vetores de 512 dimensoes
- Pre-processamento: resize 224x224 + ImageNet normalization (interno ao CLIP), L2-normalization pos-embedding
- Clustering: K-Means++ (Arthur & Vassilvitskii, 2007), 3 runs por K, K=2..15
- Metricas: Elbow Method (WCSS) e Silhouette Score (Rousseeuw, 1987)
- Visualizacao: PCA 2D via power iteration
- Pipeline: 100% Rust, binario unico (sem dependencias Python)

=== SLIDE 2: Determinacao do K Otimo ===

GRAFICO: elbow_silhouette.png (arrastar para o slide)

Resultados:
- Cotovelo sutil em K=4 na curva de inercia
- Melhor silhouette score em K=4 (0.0576)
- Scores de silhueta baixos (~0.05) sao esperados para embeddings 512D com dados visuais heterogeneos
- Interpretacao: os clusters existem mas nao sao perfeitamente separados - inerente a natureza continua de features visuais

Tabela resumida:
| K | Inercia | Silhouette |
|---|---------|------------|
| 2 | 3753.7 | 0.0559 |
| 4 | 3520.9 | 0.0576 (best) |
| 8 | 3290.3 | 0.0488 |
| 15 | 3087.1 | 0.0554 |

=== SLIDE 3: Analise dos Clusters e Metadados ===

GRAFICOS: pca_by_cluster.png + pca_by_folder.png (arrastar)

4 clusters identificados:
- Cluster 0 (26.6%): "Arte digital" - wallpapers 35%, AI images 32%, maioria landscape/square
- Cluster 1 (28.8%): "Pessoas/retratos" - 50% people, 75% portrait, 5.9MB mediano
- Cluster 2 (20.0%): "Screenshots" - iPhone 57%, capturas de tela 12%, 241KB mediano
- Cluster 3 (24.6%): "Fotos pessoais" - iPhone 81%, 70% portrait, 8.5MB mediano

Insight chave: CLIP separou por semantica visual, NAO por metadados. Fotos de iPhone aparecem nos 4 clusters. Formato (.png/.jpg) e distribuido uniformemente. Tamanho e aspect ratio correlacionam com conteudo, nao com cluster assignment.

Limitacoes: silhouette baixo, PCA 2D perde informacao, K-Means assume clusters esfericos.
Trabalhos futuros: UMAP/t-SNE, DBSCAN, outros modelos (DINOv2), cross-modal search.
