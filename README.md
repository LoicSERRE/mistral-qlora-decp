# Fine-tuning Mistral 7B sur données publiques françaises

Projet de mémoire de fin d'études (CESI, 2025-2026). L'objectif est de fine-tuner Mistral 7B v0.3 avec QLoRA sur un corpus de données publiques territoriales françaises, en utilisant uniquement du matériel grand public (RTX 4070 Ti, 12 GB VRAM, budget 0 €).

Mémoire complet : [`memoire/memoire_ADS.tex`](memoire/memoire_ADS.tex)
Code source : **https://github.com/LoicSERRE/mistral-qlora-decp**

## Résultats obtenus

| Métrique              | Valeur                                           |
|-----------------------|--------------------------------------------------|
| Perplexité            | 1.37 (amélioration de 87.5 % vs base)           |
| Précision factuelle   | 83.3 % (25/30 questions au seuil 80 %)          |
| Loss (cross-entropy)  | 0.3155 (amélioration de 86.8 % vs base)         |
| Vitesse d'inférence   | 10.4 tokens/s (-47 % vs base)                   |
| Garde-fous            | 100 % de refus corrects (7/7)                   |
| Temps d'entraînement  | 72.9 minutes                                    |
| VRAM utilisée         | 6.48 GB (54 % de la carte)                      |
| Corpus d'entraînement | 4 792 paires Q/R, ~488K tokens                  |

## Prérequis

- Python 3.10+
- GPU NVIDIA >= 12 GB VRAM (testé sur RTX 4070 Ti)
- CUDA 11.8+
- ~1 GB d'espace disque
- Compte HuggingFace avec accès à [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) (uniquement pour le fine-tuning)

## Installation

```powershell
git clone https://github.com/LoicSERRE/mistral-qlora-decp.git
cd mistral-qlora-decp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

### 1. Corpus de fine-tuning (déjà inclus)

Le corpus canonique `data/fine_tuning/training_data_final_12gb.jsonl` **(4 792 paires Q/R)** est inclus dans le repo  c'est exactement le corpus utilisé pour entraîner les adapters.

Le pipeline ne le régénère **pas** par défaut :

```powershell
python code/data_preparation/run_pipeline.py --auto
#  Détecte le corpus existant et s'arrête proprement
```

Pour régénérer depuis les APIs (données fraîches, résultat légèrement différent) :

```powershell
python code/data_preparation/run_pipeline.py --auto --force
```

> **Sources de données** (si `--force`) :
> - **RNE** : Répertoire National des Élus  [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/repertoire-national-des-elus-1/)
> - **DECP** : Données Essentielles de la Commande Publique  [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/r/16962018-5c31-4296-9454-5998585496d2) (~900 MB)
> - **Délibérations** : Actes SCDL  [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/)

### 2. Fine-tuning (~73 minutes, GPU requis)

Définir votre token HuggingFace (nécessaire pour télécharger Mistral 7B) :

```powershell
$env:HF_TOKEN = "hf_votre_token_ici"
```

```powershell
# Version recommandée : validation split + early stopping
python code/fine_tuning/train_optimized.py

# Version alternative : LoRA classique
python code/fine_tuning/train_lora.py
```

Les adapters LoRA sont sauvegardés dans `models/adapters/`.

### 3. Évaluation et test

```powershell
# Tester le modèle sur des questions libres
python code/fine_tuning/test_model.py

# Évaluation complète (perplexité, précision factuelle, garde-fous, vitesse)
python code/evaluation/evaluate_model.py
```

### 4. Notebook interactif

Le notebook [`code/notebooks/pipeline_complet.ipynb`](code/notebooks/pipeline_complet.ipynb) reproduit l'ensemble du pipeline de manière interactive.

| Section             | Contenu                                                     | GPU requis           |
|---------------------|-------------------------------------------------------------|----------------------|
| 1. Installation     | Dépendances pip                                             | Non                  |
| 2. Configuration    | Chemins, modèle, system prompt                              | Non                  |
| 3. Collecte données | PIAF (HuggingFace), DECP mensuels, budgets, délibérations  | Non                  |
| 4. Filtrage DECP    | 9 départements Sud France                                   | Non                  |
| 5. Corpus Q/R       | Génération + fusion + déduplication (~5 000 paires)        | Non                  |
| 6. Fine-tuning      | QLoRA Mistral 7B, r=32, 3 epochs                            | **Oui (12 GB VRAM)** |
| 7. Évaluation       | Perplexité, précision factuelle, garde-fous, vitesse        | **Oui**              |

## Structure du projet

```
ADS/
 code/
    data_preparation/
       01_collect_real_data.py          Collecte RNE + DECP + délibérations
       02_generate_varied_questions.py  Génération Q/R (~10 templates)
       03_clean_existing_corpus.py      Nettoyage qualité
       04_merge_and_deduplicate.py      Fusion + déduplication SHA-256
       05_optimize_for_12gb.py          Sélection pour 12 GB VRAM
       run_pipeline.py                  Orchestrateur (point d'entrée)
    fine_tuning/
       train_optimized.py  Entraînement recommandé (+ validation + early stopping)
       train_lora.py       Entraînement LoRA classique
       test_model.py       Test interactif du modèle
    evaluation/
       evaluate_model.py   Évaluation complète
    notebooks/
        pipeline_complet.ipynb  Notebook AZ
 config/
    system_prompt_specialized.txt
 data/
    fine_tuning/
       training_data_final_12gb.jsonl   Corpus canonique (4 792 paires)
    test_questions.json
 memoire/                    Mémoire LaTeX
 models/
    adapters/               Adapters LoRA (non versionnés  trop volumineux)
 results/
     benchmarks/             Résultats d'évaluation
     logs/                   Logs d'entraînement
```

## Reproductibilité

Le corpus `training_data_final_12gb.jsonl` est versionné dans le repo. Il contient exactement les 4 792 paires utilisées pour entraîner les adapters (seed 42, split 80/10/10).

> **Note** : Les adapters `.safetensors` (~800 MB) ne sont pas dans le repo (trop volumineux). Ils peuvent être régénérés avec le corpus fourni :
>
> ```powershell
> python code/fine_tuning/train_lora.py
> ```

## Matériel utilisé

- **GPU** : NVIDIA RTX 4070 Ti (12 GB VRAM)
- **CPU** : AMD Ryzen 9 7900X
- **RAM** : 64 GB DDR5-5600
- **Quantification** : QLoRA 4-bit (NF4)

## Licence

Ce projet est sous licence MIT  voir [LICENSE](LICENSE).

---

Loïc SERRE  CESI, 2025-2026
