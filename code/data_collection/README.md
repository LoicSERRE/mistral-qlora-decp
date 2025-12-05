# Data Collection

Scripts de collecte de données brutes depuis sources publiques.

## Scripts

### `01_collect_raw_data.py`
Télécharge toutes les données brutes nécessaires.

**Sources** :
- PIAF (3.8K paires Q/R)
- DECP (marchés publics XML)
- Budgets municipaux (10 villes)
- Délibérations (4 collectivités)

```powershell
python 01_collect_raw_data.py
```

**Sortie** : `data/raw/` (~620 MB)

### `02_extract_decp.py`
Filtre et extrait les marchés publics pertinents du DECP.

**Filtres** :
- 9 départements Sud France (11, 13, 30, 31, 33, 34, 66, 69, 81)
- Période : 2023-2025
- Format : CSV léger

```powershell
python 02_extract_decp.py
```

**Sortie** : `data/processed/decp_filtered.csv` (~2.9 MB, 10K marchés)

## Note

Ces scripts sont appelés automatiquement par `data_preparation/run_pipeline.py`.

Il n'est généralement **pas nécessaire** de les lancer manuellement.

## Dépendances

- `requests` - Téléchargement HTTP
- `lxml` - Parsing XML (DECP)
- `pandas` - Manipulation CSV
