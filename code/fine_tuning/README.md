# Fine-tuning

Scripts d'entraînement du modèle Mistral 7B avec LoRA/QLoRA.

## Scripts disponibles

### `train_lora.py` (baseline)

Configuration de base : LoRA r=32, alpha=64, context=512, batch=1x8.

```powershell
python train_lora.py
```

### `train_optimized.py` (recommandé)

Script d'entraînement avec plusieurs configurations et validation split.

Configurations disponibles :
- `baseline` — identique à train_lora.py
- `phase1_validation` — validation 15 % + early stopping
- `phase2_context1024` — context 512 -> 1024
- `phase2_rank64` — context 1024 + rank 64

Pour changer de configuration, éditer `ACTIVE_CONFIG` dans le fichier puis lancer :

```powershell
python train_optimized.py
```

Sortie : `models/adapters/mistral-7b-{config_name}/`

### `test_model.py`

Test rapide du modèle sur 5 questions prédéfinies.

```powershell
python test_model.py
```

## Configuration matérielle requise

- GPU NVIDIA avec 12 GB de VRAM minimum
- Quantification QLoRA 4-bit (NF4)
- Durée : 1 à 2 h par entraînement

## Hyperparamètres LoRA

```python
LORA_CONFIG = {
    'r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    'target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ]
}
```

Paramètres entraînables : environ 12M (0.17 % du modèle).

## Sorties

Après entraînement, le dossier `models/adapters/` contient :

```
adapter_config.json          Config LoRA
adapter_model.bin            Poids (~24 MB)
tokenizer.json               Tokenizer
training_metadata.json       Métadonnées
```

## Dépendances

```powershell
pip install -r requirements.txt
```
