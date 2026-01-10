"""
FINE-TUNING OPTIMISÉ AVEC VALIDATION + EARLY STOPPING
======================================================

VERSION AMÉLIORÉE avec :
 Validation split 15%
 Early stopping (patience=3)
 Best model saved (pas dernier)
 LoRA rank flexible (32 ou 64)
 Context length flexible (512, 1024, 2048)

GAINS ATTENDUS vs version actuelle :
- Perplexity : -2 à -5%
- Accuracy : +3-8 pts
- Overfitting : -30 à -50%
"""

import os
import json
import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import numpy as np

# ==================== CONFIGURATION ====================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fine_tuning"
MODEL_DIR = PROJECT_ROOT / "models"
ADAPTERS_DIR = MODEL_DIR / "adapters"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-v0.3"

# ==================== CONFIGURATIONS OPTIMISÉES ====================

OPTIMIZATION_CONFIGS = {
    "baseline": {
        "name": "Baseline (actuel)",
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_length": 512,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "validation_split": None,
        "early_stopping": False
    },
    
    "phase1_validation": {
        "name": "Phase 1 : Validation + Early Stopping",
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_length": 512,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "validation_split": 0.10,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.01
    },
    
    "phase2_context1024": {
        "name": "Phase 2A : Context 1024",
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_length": 1024,
        "batch_size": 1,              # Réduit pour éviter OOM avec context 1024
        "gradient_accumulation": 8,    # Compense le batch réduit
        "validation_split": 0.15,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.01
    },
    
    "phase2_rank64": {
        "name": "Phase 2B : Context 1024 + Rank 64",
        "lora_rank": 64,
        "lora_alpha": 128,
        "max_length": 1024,
        "batch_size": 1,              # Réduit pour éviter OOM avec context 1024
        "gradient_accumulation": 8,    # Compense le batch réduit
        "validation_split": 0.10,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.01
    },
    
    "phase3_optimal": {
        "name": "Phase 3 : Configuration Optimale",
        "lora_rank": 64,
        "lora_alpha": 128,
        "max_length": 1024,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "validation_split": 0.15,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.01,
        "learning_rate": 3e-4  # Peut être testé
    }
}

# Choisir config (À MODIFIER)
ACTIVE_CONFIG = "phase2_rank64"  # ← CHANGER ICI POUR PHASE 2, 3, etc.

CONFIG = OPTIMIZATION_CONFIGS[ACTIVE_CONFIG]

print(f"\n{'='*80}")
print(f"CONFIGURATION ACTIVE : {CONFIG['name']}")
print(f"{'='*80}\n")

# System prompt
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'accès aux données publiques françaises, notamment :
- DECP (Données Essentielles de la Commande Publique)
- RNE (Répertoire National des Élus)

Tu réponds avec précision en citant tes sources. Si une information n'est pas dans ton corpus, tu le dis clairement."""


# ==================== CHARGEMENT DONNÉES ====================

def load_and_split_data(file_path, validation_split=None, test_split=0.10):
    """
    Charge corpus et split en train/validation/test
    
    Args:
        file_path: Chemin corpus JSONL
        validation_split: % pour validation (ex: 0.15 = 15%)
        test_split: % pour test (ex: 0.10 = 10%)
    
    Returns:
        train_data, val_data, test_data
    """
    
    print(f"\n{'='*80}")
    print("CHARGEMENT ET SPLIT DU CORPUS")
    print(f"{'='*80}")
    print(f"Fichier : {file_path}")
    
    # Charger JSONL
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f" {len(data):,} paires chargées")
    
    # Mélanger avec seed fixe (reproductibilité)
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = [data[i] for i in indices]
    
    # Calculer splits
    n_total = len(data)
    n_test = int(n_total * test_split)
    
    if validation_split:
        n_val = int(n_total * validation_split)
        n_train = n_total - n_test - n_val
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]
        
        print(f"\n Répartition avec validation :")
        print(f"   • Train      : {n_train:,} ({n_train/n_total*100:.1f}%)")
        print(f"   • Validation : {n_val:,} ({n_val/n_total*100:.1f}%)")
        print(f"   • Test       : {n_test:,} ({n_test/n_total*100:.1f}%)")
    else:
        n_train = n_total - n_test
        
        train_data = data[:n_train]
        val_data = None
        test_data = data[n_train:]
        
        print(f"\n Répartition sans validation :")
        print(f"   • Train      : {n_train:,} ({n_train/n_total*100:.1f}%)")
        print(f"   • Test       : {n_test:,} ({n_test/n_total*100:.1f}%)")
    
    return train_data, val_data, test_data


def format_instruction(prompt, completion):
    """Formate une paire Q/A selon template Mistral"""
    return f"""<s>[INST] {SYSTEM_PROMPT}

Question : {prompt} [/INST] {completion}</s>"""


def tokenize_dataset(data, tokenizer, max_length=512):
    """Tokenize dataset"""
    
    print(f"\n{'='*80}")
    print("TOKENIZATION")
    print(f"{'='*80}")
    print(f"Max length : {max_length} tokens")
    print(f"Samples : {len(data):,}")
    
    texts = [format_instruction(item['prompt'], item['completion']) for item in data]
    
    print(" Tokenization en cours...")
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    lengths = [(encodings['attention_mask'][i].sum().item()) for i in range(len(texts))]
    
    print(f" Tokenization terminée\n")
    print(f" Statistiques tokens :")
    print(f"   • Moyenne     : {np.mean(lengths):.1f}")
    print(f"   • Médiane     : {np.median(lengths):.1f}")
    print(f"   • Min         : {min(lengths)}")
    print(f"   • Max         : {max(lengths)}")
    print(f"   • Tronqués    : {sum(1 for l in lengths if l == max_length)} ({sum(1 for l in lengths if l == max_length)/len(lengths)*100:.1f}%)")
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': encodings['input_ids'].clone()
    })
    
    return dataset


# ==================== CHARGEMENT MODÈLE ====================

def load_model_and_tokenizer():
    """Charge Mistral 7B avec QLoRA"""
    
    print(f"\n{'='*80}")
    print("CHARGEMENT MODÈLE")
    print(f"{'='*80}")
    print(f"Modèle : {MODEL_NAME}")
    
    print("\n Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print(" Tokenizer chargé")
    
    print("\n Chargement modèle (QLoRA 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    print(" Modèle chargé (4-bit)")
    
    if torch.cuda.is_available():
        print(f"\n Mémoire GPU :")
        print(f"   • Allouée  : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   • Réservée : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"   • Totale   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return model, tokenizer


def setup_lora(model, rank=32, alpha=64):
    """Configure LoRA"""
    
    print(f"\n{'='*80}")
    print("CONFIGURATION LoRA")
    print(f"{'='*80}")
    print(f"Rank (r)     : {rank}")
    print(f"Alpha        : {alpha}")
    print(f"Dropout      : 0.05")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    
    model = get_peft_model(model, lora_config)
    
    # Stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n Paramètres :")
    print(f"   • Entraînables : {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   • Total        : {total_params:,}")
    
    return model


# ==================== CALLBACK CUSTOM ====================

class VRAMCallback(TrainerCallback):
    """Callback pour monitorer VRAM"""
    
    def __init__(self):
        self.vram_peak = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3
            if current > self.vram_peak:
                self.vram_peak = current


# ==================== TRAINING ====================

def train_model(config_name="phase1_validation"):
    """Training principal avec config spécifique"""
    
    config = OPTIMIZATION_CONFIGS[config_name]
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "FINE-TUNING OPTIMISÉ" + " "*33 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\n Configuration : {config['name']}")
    print(f" LoRA Rank     : {config['lora_rank']}")
    print(f" Context       : {config['max_length']} tokens")
    print(f" Validation    : {config.get('validation_split', 0)*100:.0f}%")
    print(f"  Early Stop   : {'Yes' if config.get('early_stopping') else 'No'}")
    
    timestamp_start = datetime.now()
    
    # 1. Charger données
    corpus_file = DATA_DIR / "training_data_final_12gb.jsonl"
    
    train_data, val_data, test_data = load_and_split_data(
        corpus_file,
        validation_split=config.get('validation_split'),
        test_split=0.10
    )
    
    # 2. Charger modèle
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. Setup LoRA
    model = setup_lora(model, rank=config['lora_rank'], alpha=config['lora_alpha'])
    
    # 4. Tokenize
    train_dataset = tokenize_dataset(train_data, tokenizer, max_length=config['max_length'])
    
    val_dataset = None
    if val_data:
        print(f"\n Tokenization validation set...")
        val_dataset = tokenize_dataset(val_data, tokenizer, max_length=config['max_length'])
    
    # 5. Training arguments
    output_dir = ADAPTERS_DIR / f"mistral-7b-{config_name}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,  # Réduit de 5→3 pour accélérer
        per_device_train_batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation', 4),
        learning_rate=config.get('learning_rate', 2e-4),
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        
        #  VALIDATION & EARLY STOPPING
        eval_strategy='steps' if val_dataset else 'no',
        eval_steps=100 if val_dataset else None,
        per_device_eval_batch_size=1,  # 1 = évite OOM avec context 1024
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model='eval_loss' if val_dataset else None,
        greater_is_better=False,
        
        optim='paged_adamw_8bit',  # Phase 2 : économise ~1 GB vs adamw_torch
        max_grad_norm=0.3,
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to='none',
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 7. Callbacks
    callbacks = [VRAMCallback()]
    
    if config.get('early_stopping'):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.get('early_stopping_patience', 3),
                early_stopping_threshold=config.get('early_stopping_threshold', 0.01)
            )
        )
        print(f"\n  Early Stopping activé (patience={config.get('early_stopping_patience')})")
    
    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # 9. Training
    print(f"\n{'='*80}")
    print("DÉMARRAGE TRAINING")
    print(f"{'='*80}\n")
    
    result = trainer.train()
    
    # 10. Sauvegarder
    print(f"\n{'='*80}")
    print("SAUVEGARDE MODÈLE")
    print(f"{'='*80}")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 11. Métriques finales
    timestamp_end = datetime.now()
    duration = (timestamp_end - timestamp_start).total_seconds() / 60
    
    vram_callback = [c for c in callbacks if isinstance(c, VRAMCallback)][0]
    
    summary = {
        "config_name": config_name,
        "config": config,
        "timestamp_start": timestamp_start.isoformat(),
        "timestamp_end": timestamp_end.isoformat(),
        "duration_minutes": duration,
        "train_samples": len(train_data),
        "val_samples": len(val_data) if val_data else 0,
        "test_samples": len(test_data),
        "vram_peak_gb": vram_callback.vram_peak,
        "final_loss": result.training_loss,
        "output_dir": str(output_dir)
    }
    
    # Sauvegarder summary
    summary_file = LOGS_DIR / f"training_{config_name}_{timestamp_end.strftime('%Y%m%d_%H%M')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(" TRAINING TERMINÉ")
    print(f"{'='*80}")
    print(f"  Durée         : {duration:.1f} minutes")
    print(f" VRAM Peak     : {vram_callback.vram_peak:.2f} GB")
    print(f" Loss Finale   : {result.training_loss:.4f}")
    print(f" Modèle        : {output_dir}")
    print(f" Summary       : {summary_file}")
    
    return summary


# ==================== MAIN ====================

if __name__ == "__main__":
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "FINE-TUNING OPTIMISÉ - MULTI-CONFIG" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\nConfigurations disponibles :")
    for i, (key, cfg) in enumerate(OPTIMIZATION_CONFIGS.items(), 1):
        print(f"   {i}. {key:20s} : {cfg['name']}")
    
    print(f"\n Configuration active : {ACTIVE_CONFIG}")
    print(f"   → Pour changer : Modifier ACTIVE_CONFIG dans le code\n")
    
    # Confirmer (sauf si --auto-confirm en argument)
    auto_confirm = '--auto-confirm' in sys.argv
    
    if auto_confirm:
        confirm = 'y'
        print(" Lancement automatique (--auto-confirm activé)\n")
    else:
        confirm = input("Lancer le training ? (y/n) : ").strip().lower()
    
    if confirm == 'y':
        summary = train_model(ACTIVE_CONFIG)
        
        print("\n NEXT STEPS :")
        print("   1. Évaluer : python code/evaluation/evaluate_model.py")
        print("   2. Comparer : python code/evaluation/compare_configs.py")
        print("   3. Si gain >2% PPL → Passer config suivante (Phase 2)")
    else:
        print("\n Training annulé")
