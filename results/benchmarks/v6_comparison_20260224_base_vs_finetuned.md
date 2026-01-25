# Comparaison Base vs Fine-Tuné (Phase 1)

**Date** : 24/02/2026 17:26

**Modèle base** : `mistralai/Mistral-7B-v0.3`

**Adapter** : `models\adapters\mistral-7b-phase1_validation`

## Tableau comparatif

| Métrique | Base | Fine-Tuné | Évolution |
|----------|------|-----------|----------|
| Perplexité (↓) | 11.0059 | **1.3709** | +87.5% |
| Accuracy (↑) | 0.0% | **80.0%** | +80.0 pts |
| Garde-fous (↑) | 100.0% | **100.0%** | +0.0 pts |
| Tokens/sec (↑) | 19.6 | **10.4** | -9.2 t/s |
