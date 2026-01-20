# Évaluation

Scripts d'évaluation et de benchmarking du modèle fine-tuné.

## Scripts disponibles

### `evaluate_model.py`

Évaluation complète du modèle : perplexité, loss, précision factuelle (seuil 80 %), vitesse d'inférence. Compare automatiquement avec le modèle de base.

```powershell
python evaluate_model.py
```

Sortie : `results/benchmarks/evaluation_report_{timestamp}.json`

Le script supporte la reprise automatique en cas d'interruption (sauvegarde après chaque métrique).

### `compare_configs.py`

Compare tous les adapters entraînés pour identifier la meilleure configuration. Teste chaque adapter sur 5 questions représentatives et génère un rapport comparatif.

```powershell
python compare_configs.py
```

Sorties : `results/benchmarks/comparison_{timestamp}.json` et `.md`

### `eval_phase1.py`

Évaluation spécifique de la phase 1 (validation split + early stopping).

### `benchmark_external.py`

Comparaison avec des modèles externes (GPT, Claude) sur les mêmes questions.

### `eval_comparison.py` / `dashboard_results.py`

Outils de visualisation et de comparaison des résultats.

## Workflow typique

```powershell
# Évaluer un modèle
python evaluate_model.py

# Comparer tous les adapters
python compare_configs.py
```

## Dépendances

```powershell
pip install -r requirements.txt
```

Nécessite `matplotlib`, `seaborn` et `pandas` pour les graphiques.

Durée estimée : 10-15 minutes (GPU), 2-3 h (CPU).
