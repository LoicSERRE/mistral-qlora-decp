# Feuille de route

## Planning global

- Semestre 7 (oct-déc 2025) : recherche et préparation
- Semestre 8 (jan-mars 2026) : expérimentation et rédaction

## Semestre 7 — Fondations

### Mois 1 : Introduction et état de l'art

- [X] Rédaction du contexte et de la problématique
- [X] État de l'art LLM et fine-tuning
- [X] Tests des LLM généralistes (baseline)
- [X] Analyse des limites, formulation de l'hypothèse

### Mois 2 : Sélection du modèle

- [X] Analyse comparative des modèles candidats
- [X] Tests techniques (capacité fine-tuning sur matériel personnel)
- [X] Choix justifié du modèle (Mistral 7B v0.3)
- [X] Définition de l'architecture technique (QLoRA 4-bit)

### Mois 3 : Préparation des données

- [X] Identification des sources (DECP, RNE, data.gouv.fr)
- [X] Collecte des données (9 départements sud de la France)
- [X] Nettoyage et formatage du corpus
- [X] Validation du dataset (5 460 paires Q/R, 542K tokens)

## Semestre 8 — Expérimentation

### Mois 4 : Fine-tuning

- [X] Configuration de l'environnement (RTX 4070 Ti, CUDA)
- [X] Première expérience de fine-tuning (baseline)
- [X] Optimisation : validation split + early stopping
- [X] Entraînement final (92.1 min, 5.33 GB VRAM)

### Mois 5 : Évaluation et benchmarking

- [X] Création du benchmark de test (30 questions)
- [X] Évaluation du modèle (PPL 1.37, précision 83.3 %)
- [X] Analyse des résultats
- [X] Discussion et interprétation

### Mois 6 : Rédaction et soutenance

- [X] Rédaction du mémoire (méthodologie, résultats)
- [X] Rédaction du mémoire (discussion, conclusion)
- [X] Relecture et corrections
- [ ] Création du poster
- [ ] Préparation de la soutenance orale
- [ ] Soutenance

## Jalons

| Date     | Jalon                            |
| -------- | -------------------------------- |
| oct.     | Sujet validé                    |
| nov.     | Modèle choisi (Mistral 7B v0.3) |
| déc.    | Dataset prêt (5 460 paires)     |
| jan.     | Modèle entraîné               |
| Mi-jan.  | Résultats analysés             |
| Fev.  | Rédaction mémoire              |
| Début-Mars | Finalisation mémoire + création poster |
| Mi-mars | Rendu du mémoire + Soutenance |
