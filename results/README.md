# Résultats

Ce dossier contient les résultats expérimentaux du fine-tuning.

## Structure

- `benchmarks/` : résultats d'évaluation (JSON, Markdown, graphiques PNG)
- `logs/` : logs d'entraînement (hyperparamètres, configuration)
- `figures/` : graphiques générés pour le mémoire

## Fichiers d'évaluation (par ordre chronologique)

Les fichiers sont préfixés par version pour montrer l'évolution du protocole d'évaluation :

| Fichier | Date | Adapter | PPL | Accuracy | Notes |
|---|---|---|---|---|---|
| `v1_eval_20260210_phase1_early` | 10/02 | phase1 | 1.20 | 50.0% | Premier test, protocole initial |
| `v2_eval_20260223_phase1_acc56` | 23/02 | phase1_validation | 1.37 | 56.7% | Nouveau seuil 80%, extraction basique |
| `v3_eval_20260223_phase1_acc86` | 23/02 | phase1_validation | 1.37 | 86.7% | Normalisation nombres améliorée |
| `v4_eval_20260223_phase1_final` | 23/02 | phase1_validation | 1.37 | 83.3% | **Version finale** (utilisée dans le mémoire) |
| `v5_eval_20260224_phase2_rank64` | 24/02 | phase2_rank64 | 1.38 | 76.7% | Test adapter rank64 (exploratoire) |
| `v6_comparison_base_vs_finetuned` | 24/02 | -- | -- | -- | Comparaison complète base vs fine-tuné |

Autres fichiers :
- `summary_results.json` : résumé des métriques finales (base vs fine-tuné)
- `base_model_responses.json` : réponses détaillées du modèle de base (30 questions)
- `evaluation_graphiques.png` : graphique comparatif (utilisé dans le mémoire)
