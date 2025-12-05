# Scripts

## Organisation

### `data_collection/`

Scripts de collecte de données brutes depuis les APIs publiques.

- `01_collect_raw_data.py` — télécharge les données DECP et budgets
- `02_extract_decp.py` — filtre les DECP par département et période

### `data_preparation/`

Pipeline de génération du corpus d'entraînement.

- `run_pipeline.py` — script principal, génère le corpus complet JSONL

### `fine_tuning/`

Scripts d'entraînement et de test du modèle.

- `train_lora.py` — configuration baseline (r=32, context=512)
- `train_optimized.py` — configurations optimisées (validation split, early stopping)
- `test_model.py` — test rapide sur questions prédéfinies

### `evaluation/`

Scripts d'évaluation et de benchmarking.

- `evaluate_model.py` — évaluation complète (perplexité, loss, précision)
- `compare_configs.py` — compare tous les adapters entraînés
- `benchmark_external.py` — comparaison avec modèles externes

## Workflow complet

```powershell
# 1. Préparer les données
python data_preparation/run_pipeline.py

# 2. Entraîner le modèle
python fine_tuning/train_optimized.py

# 3. Tester
python fine_tuning/test_model.py

# 4. Évaluer
python evaluation/evaluate_model.py
```

## Sorties

- Corpus : `data/fine_tuning/training_data_final_12gb.jsonl` (~4 792 paires)
- Adapters : `models/adapters/mistral-7b-*` (adapters LoRA)
- Logs : `results/logs/training_*.json`
- Benchmarks : `results/benchmarks/`

Chaque sous-dossier contient un README avec la documentation détaillée.
