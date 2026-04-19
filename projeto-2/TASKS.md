# Tasks — Projeto 2

## Concluído

- [x] Extrair metadados (resolução, aspect ratio, file size, folder) das 8000 imagens amostradas
- [x] Definir pipeline de labeling determinístico em 3 tiers (resolução exata, folder+aspect, fallback)
- [x] Gerar dataset: X_embeddings (CLIP 512-dim), y_type (6 classes), y_people (binário)
- [x] Treinar e avaliar Linear Probe sobre CLIP embeddings (3,591 params)
- [x] Treinar e avaliar MLP Probe sobre CLIP embeddings (165K params, shared backbone + 2 heads)
- [x] Extrair features ConvNeXt-Tiny (768-dim, frozen) das 8000 imagens
- [x] Treinar e avaliar MLP Probe sobre ConvNeXt features (230K params)
- [x] Comparativo 3-way com train/val/test split (70/15/15), curvas de treino, best val checkpoint
- [x] Gerar plots: training curves, confusion matrices, comparison bar chart
- [x] Pesquisar dados de pretraining dos modelos (CLIP: 400M pares, ConvNeXt: 1.28M imgs) para comparação acadêmica
- [x] Escrever relatório (RELATORIO.md)

## Pendente — Dataset Completo

- [ ] Rodar extract_metadata.py sobre TODAS as imagens de D:\media (não apenas as 8000 amostradas no projeto-1)
- [ ] Rodar build_dataset.py para gerar labels do dataset completo
- [ ] Re-extrair embeddings CLIP (fastembed-rs) para todas as imagens — atualizar o binário do projeto-1 ou gerar novo
- [ ] Re-extrair features ConvNeXt-Tiny para todas as imagens
- [ ] Retreinar os 3 experimentos com dataset completo e comparar com resultados da amostra

## Pendente — Fine-Tuning End-to-End

- [ ] Implementar dataloader que carrega imagens raw (não features pré-extraídas)
- [ ] Fine-tune CLIP ViT-B/32 end-to-end: descongelar o encoder, lr pequeno (1e-5 ~ 1e-6) para o backbone, lr maior (1e-3) para as heads
- [ ] Fine-tune ConvNeXt-Tiny end-to-end: mesma estratégia de lr diferenciado
- [ ] Comparar frozen vs fine-tuned para ambos os modelos (4 experimentos adicionais)
- [ ] Monitorar overfitting com early stopping por val F1 — dataset pode ser pequeno demais para fine-tune sem data augmentation
- [ ] Adicionar data augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter) para fine-tune
- [ ] Rodar no cluster (mcculloch RTX 4090) se necessário — fine-tune no CPU vai ser lento

## Pendente — Melhorias no Dataset

- [ ] Melhorar label has_people: usar detector de pessoas (face detection ou CLIP zero-shot "a photo of a person") em vez de depender apenas da pasta "people/"
- [ ] Avaliar ruído nos labels de image_type: amostrar 100-200 imagens e verificar manualmente se as regras determinísticas estão corretas
- [ ] Considerar balanceamento de classes: oversample screenshot_desktop ou undersample camera_photo
- [ ] Adicionar classe "meme" se houver volume suficiente e separação clara

## Pendente — Apresentação

- [ ] Montar 4 slides em HTML (estilo projeto-1)
- [ ] Slide 1: problema + dataset + classes + "semântica vs geometria"
- [ ] Slide 2: pipeline (labeling determinístico + feature extraction + MLP dual-head)
- [ ] Slide 3: resultados (tabela comparativa, confusion matrix, training curves)
- [ ] Slide 4: interpretação (CLIP > ConvNeXt porquê, limitações, conexão com ViT compression)
