# Préparation des données

## Script principal

### `run_pipeline.py`

Pipeline complet de génération du corpus d'entraînement. Il exécute les étapes suivantes :

1. Collecte des données réelles (RNE, DECP) depuis data.gouv.fr
2. Génération de questions variées
3. Nettoyage du corpus
4. Fusion et déduplication
5. Optimisation pour 12 GB de VRAM

```powershell
python run_pipeline.py
```

Sortie : `data/fine_tuning/training_data_final_12gb.jsonl` (~4 800 paires Q/R)

## Temps d'exécution

- Collecte des données : 10-15 min
- Génération Q/R : 20-30 min
- Nettoyage et fusion : 5-10 min

Total : environ 45 minutes.

## Sources de données

Les données brutes sont téléchargées automatiquement depuis :
- [DECP](https://www.data.gouv.fr/fr/datasets/donnees-essentielles-de-la-commande-publique/) (marchés publics)
- [RNE](https://api.rne.fr/) (élus municipaux)

## Corpus final

| Métrique | Valeur |
|---|---|
| Paires Q/R | 5 460 |
| Tokens | 542K |
| Compatible 12 GB | oui (< 600K tokens) |
| Source | 100 % données publiques (data.gouv.fr) |
| Duplicatas | 0 (vérification SHA-256) |

## Fichiers générés

```
data/fine_tuning/
├── training_data_final_12gb.jsonl          Corpus final
├── training_data_final_12gb_metadata.json  Métadonnées
└── training_data_final_12gb_BACKUP.jsonl   Backup
```
