# Données

## Structure

- `fine_tuning/` : corpus d'entraînement au format JSONL (5 460 paires Q/R, 542K tokens)
- `test_questions.json` : questions de test pour l'évaluation

## Sources

Les données proviennent exclusivement de sources publiques françaises :
- DECP (marchés publics) via data.gouv.fr
- RNE (élus municipaux) via api.rne.fr

Les données brutes sont téléchargées automatiquement par le pipeline `code/data_preparation/run_pipeline.py`.
