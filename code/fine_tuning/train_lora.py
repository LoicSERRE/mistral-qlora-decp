"""
FINE-TUNING MISTRAL 7B v0.3 - LoRA
===================================

Spécialisation pour accès données publiques françaises (DECP, RNE)

Matériel : RTX 4070 Ti 12GB VRAM
Technique : LoRA (Low-Rank Adaptation)
Corpus : 4,848 paires Q/A (488K tokens)

Hyperparamètres optimisés pour 12GB VRAM :
- LoRA rank (r) = 32
- LoRA alpha = 64
- LoRA dropout = 0.05
- Batch size = 8
- Learning rate = 2e-4
- Epochs = 3
- FP16 precision
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fine_tuning"
MODEL_DIR = PROJECT_ROOT / "models"
ADAPTERS_DIR = MODEL_DIR / "adapters"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

# Créer dossiers
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Modèle de base
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

# Hyperparamètres LoRA
LORA_CONFIG = {
    'r': 32,                    # Rank (dimension matrices low-rank)
    'lora_alpha': 64,           # Scaling factor (2x rank)
    'lora_dropout': 0.05,       # Dropout pour régularisation
    'bias': 'none',             # Pas de biais LoRA
    'task_type': TaskType.CAUSAL_LM,
    'target_modules': [         # Modules à adapter
        'q_proj',               # Query projection
        'k_proj',               # Key projection  
        'v_proj',               # Value projection
        'o_proj',               # Output projection
        'gate_proj',            # FFN gate
        'up_proj',              # FFN up
        'down_proj'             # FFN down
    ]
}

# Hyperparamètres entraînement
TRAINING_CONFIG = {
    'output_dir': str(ADAPTERS_DIR / "mistral-7b-lora-decp"),  # Nom fixe pour reprendre checkpoints
    'num_train_epochs': 3,
    'per_device_train_batch_size': 1,  # BATCH 1 pour éviter OOM complet
    'gradient_accumulation_steps': 8,  # Batch effectif = 1×8 = 8
    'learning_rate': 2e-4,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.03,
    'weight_decay': 0.01,
    'fp16': True,
    'logging_steps': 10,
    'save_strategy': 'steps',           # Sauvegarder par steps au lieu d'epochs
    'save_steps': 100,                  # Checkpoint tous les 100 steps (~10-11 min)
    'save_total_limit': 3,              # Garder seulement les 3 derniers checkpoints
    'load_best_model_at_end': False,    # Pas besoin, on garde le dernier
    'optim': 'adamw_torch',
    'seed': 42,
    'dataloader_num_workers': 0,
    'remove_unused_columns': False,
    'report_to': 'none',
    'gradient_checkpointing_kwargs': {'use_reentrant': False}  # Supprime warning PyTorch 2.5
    # Pas de gradient_checkpointing ici
}

# System prompt
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'accès aux données publiques françaises, notamment :
- DECP (Données Essentielles de la Commande Publique)
- RNE (Répertoire National des Élus)

Tu réponds avec précision en citant tes sources. Si une information n'est pas dans ton corpus, tu le dis clairement."""


# ==================== CHARGEMENT DONNÉES ====================

def load_training_data(file_path):
    """Charge corpus JSONL et prépare dataset"""
    
    print(f"\n CHARGEMENT CORPUS")
    print("="*80)
    print(f"Fichier : {file_path}")
    
    # Charger JSONL
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f" {len(data):,} paires chargées")
    
    # Statistiques
    sources = {}
    for item in data:
        source = item.get('source', 'UNKNOWN')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\n Répartition par source :")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(data)) * 100
        print(f"   • {source:30s} : {count:,} ({pct:.1f}%)")
    
    return data


def format_instruction(prompt, completion):
    """Formate une paire Q/A selon template instruction Mistral"""
    
    return f"""<s>[INST] {SYSTEM_PROMPT}

Question : {prompt} [/INST] {completion}</s>"""


def tokenize_dataset(data, tokenizer, max_length=512):
    """Tokenize dataset avec padding et truncation"""
    
    print(f"\n TOKENIZATION")
    print("="*80)
    print(f"Max length : {max_length} tokens")
    
    # Formater toutes les instructions
    texts = [
        format_instruction(item['prompt'], item['completion'])
        for item in data
    ]
    
    # Tokenizer
    print(" Tokenization en cours...")
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Calculer stats
    lengths = [
        (encodings['attention_mask'][i].sum().item())
        for i in range(len(texts))
    ]
    
    print(f" Tokenization terminée")
    print(f"\n Statistiques tokens :")
    print(f"   • Moyenne : {np.mean(lengths):.1f}")
    print(f"   • Médiane : {np.median(lengths):.1f}")
    print(f"   • Min : {min(lengths)}")
    print(f"   • Max : {max(lengths)}")
    print(f"   • Tronqués (>{max_length}) : {sum(1 for l in lengths if l == max_length)}")
    
    # Créer dataset Hugging Face
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': encodings['input_ids'].clone()  # Pour language modeling
    })
    
    return dataset


# ==================== CHARGEMENT MODÈLE ====================

def load_model_and_tokenizer():
    """Charge Mistral 7B et tokenizer"""
    
    print(f"\n CHARGEMENT MODÈLE")
    print("="*80)
    print(f"Modèle : {MODEL_NAME}")
    
    # Tokenizer
    print("\n Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print(" Tokenizer chargé")
    
    # Modèle avec quantization 4-bit (QLoRA)
    print("\n Chargement modèle...")
    
    # Configuration quantization 4-bit
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
    
    # Prépare le modèle pour QLoRA
    model = prepare_model_for_kbit_training(model)
    
    print(" Modèle chargé (4-bit)")
    
    # Stats mémoire
    if torch.cuda.is_available():
        print(f"\n Mémoire GPU :")
        print(f"   • Allouée : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   • Réservée : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"   • Totale disponible : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return model, tokenizer


def setup_lora(model):
    """Configure LoRA sur le modèle"""
    
    print(f"\n CONFIGURATION LoRA")
    print("="*80)
    
    # Config
    lora_config = LoraConfig(**LORA_CONFIG)
    
    print(f"Paramètres LoRA :")
    print(f"   • Rank (r) : {lora_config.r}")
    print(f"   • Alpha : {lora_config.lora_alpha}")
    print(f"   • Dropout : {lora_config.lora_dropout}")
    print(f"   • Target modules : {', '.join(lora_config.target_modules)}")
    
    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    
    # Compter paramètres
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n Paramètres modèle :")
    print(f"   • Total : {total_params:,}")
    print(f"   • Entraînables : {trainable_params:,}")
    print(f"   • % entraînable : {100 * trainable_params / total_params:.2f}%")
    
    # Mémoire LoRA
    lora_size_mb = (trainable_params * 2) / 1024**2  # FP16 = 2 bytes
    print(f"   • Taille adapters LoRA : {lora_size_mb:.1f} MB")
    
    return model


# ==================== CALLBACKS ====================

class MetricsCallback:
    """Callback pour tracker métriques durant entraînement"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.history = []
        
        # Header
        with open(log_file, 'w') as f:
            f.write(f"# Fine-tuning Mistral 7B LoRA - {self.start_time}\n")
            f.write(f"# Corpus: 4,848 paires | 488K tokens\n")
            f.write(f"# LoRA r={LORA_CONFIG['r']} alpha={LORA_CONFIG['lora_alpha']}\n\n")
    
    def on_log(self, trainer_state):
        """Appelé à chaque log"""
        
        if trainer_state.log_history:
            last_log = trainer_state.log_history[-1]
            
            # Formater log
            log_entry = {
                'step': last_log.get('step', 0),
                'epoch': last_log.get('epoch', 0),
                'loss': last_log.get('loss', None),
                'learning_rate': last_log.get('learning_rate', None),
                'elapsed': (datetime.now() - self.start_time).total_seconds()
            }
            
            self.history.append(log_entry)
            
            # Sauvegarder
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Afficher
            if log_entry['loss'] is not None:
                elapsed_min = log_entry['elapsed'] / 60
                print(f"Step {log_entry['step']:4d} | "
                      f"Epoch {log_entry['epoch']:.2f} | "
                      f"Loss {log_entry['loss']:.4f} | "
                      f"LR {log_entry['learning_rate']:.2e} | "
                      f"Time {elapsed_min:.1f}min")


# ==================== ENTRAÎNEMENT ====================

def train():
    """Fonction principale d'entraînement"""
    
    print("\n" + "="*80)
    print(" FINE-TUNING MISTRAL 7B - LoRA")
    print("="*80)
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    
    # 1. Charger données
    corpus_file = DATA_DIR / "training_data_final_12gb.jsonl"
    data = load_training_data(corpus_file)
    
    # 2. Charger modèle
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. Setup LoRA
    model = setup_lora(model)
    
    # 4. Tokenize dataset
    dataset = tokenize_dataset(data, tokenizer, max_length=512)
    
    # 5. Configuration training
    print(f"\n  CONFIGURATION ENTRAÎNEMENT")
    print("="*80)
    training_args = TrainingArguments(**TRAINING_CONFIG)
    
    print(f"Paramètres :")
    print(f"   • Epochs : {training_args.num_train_epochs}")
    print(f"   • Batch size : {training_args.per_device_train_batch_size}")
    print(f"   • Learning rate : {training_args.learning_rate}")
    print(f"   • LR scheduler : {training_args.lr_scheduler_type}")
    print(f"   • Warmup : {training_args.warmup_ratio}")
    print(f"   • Weight decay : {training_args.weight_decay}")
    print(f"   • FP16 : {training_args.fp16}")
    print(f"   • Output : {training_args.output_dir}")
    
    # Calculer steps
    total_steps = (len(dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    
    print(f"\n Steps :")
    print(f"   • Total : {total_steps:,}")
    print(f"   • Warmup : {warmup_steps:,}")
    print(f"   • Par epoch : {total_steps // training_args.num_train_epochs:,}")
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 7. Trainer
    print(f"\n INITIALISATION TRAINER")
    print("="*80)
    
    log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    metrics_callback = MetricsCallback(log_file)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("Trainer prêt")
    print(f"Logs : {log_file}")
    
    # Vérifier si un checkpoint existe pour reprendre
    checkpoint_dir = Path(training_args.output_dir)
    last_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.name.startswith('checkpoint-')])
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"\n CHECKPOINT DÉTECTÉ")
            print(f"   • Reprise depuis : {checkpoints[-1].name}")
            step_num = int(checkpoints[-1].name.split('-')[1])
            print(f"   • Step : {step_num}/{total_steps} ({step_num/total_steps*100:.1f}%)")
    
    # 8. Entraînement
    print(f"\n" + "="*80)
    if last_checkpoint:
        print(" REPRISE FINE-TUNING")
    else:
        print(" DÉBUT FINE-TUNING")
    print("="*80)
    print(f"Démarrage : {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = datetime.now()
    
    print("\n[*] Lancement de l'entraînement...")
    if not last_checkpoint:
        print("   (Le premier step peut prendre 30-60 secondes)")
    print("   [CHECKPOINT] Sauvegardes automatiques tous les 100 steps (~10 min)\n")
    
    try:
        # Reprendre depuis checkpoint si existe, sinon commencer de z\u00e9ro
        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        duration = (datetime.now() - start_time).total_seconds()
        duration_min = duration / 60
        
        print(f"\n" + "="*80)
        print(" FINE-TUNING TERMINÉ")
        print("="*80)
        print(f"Durée totale : {duration_min:.1f} minutes ({duration/60/60:.2f}h)")
        print(f"Steps/sec : {total_steps / duration:.2f}")
        
    except Exception as e:
        print(f"\n ERREUR DURANT ENTRAÎNEMENT")
        print(f"   {str(e)}")
        raise
    
    # 9. Sauvegarde adapters
    print(f"\n SAUVEGARDE ADAPTERS LoRA")
    print("="*80)
    
    output_dir = Path(training_args.output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f" Adapters sauvegardés : {output_dir}")
    
    # Taille adapters
    adapter_files = list(output_dir.glob("adapter_*.bin"))
    if adapter_files:
        total_size = sum(f.stat().st_size for f in adapter_files)
        print(f"Taille adapters : {total_size / 1024**2:.1f} MB")
    
    # 10. Métadonnées
    metadata = {
        'model_base': MODEL_NAME,
        'lora_config': LORA_CONFIG,
        'training_config': {k: v for k, v in TRAINING_CONFIG.items() if k != 'output_dir'},
        'corpus': {
            'file': str(corpus_file),
            'num_examples': len(data),
            'estimated_tokens': 488_514
        },
        'training': {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': duration_min,
            'total_steps': total_steps,
            'final_loss': trainer.state.log_history[-1].get('loss') if trainer.state.log_history else None
        },
        'hardware': {
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
    }
    
    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f" Métadonnées : {metadata_file}")
    
    # 11. Résumé final
    print(f"\n" + "="*80)
    print(" RÉSUMÉ FINAL")
    print("="*80)
    print(f"\n Modèle fine-tuné prêt :")
    print(f" Adapters LoRA : {output_dir}")
    print(f" Corpus : {len(data):,} paires (488K tokens)")
    print(f" Epochs : {training_args.num_train_epochs}")
    print(f" Durée : {duration_min:.1f} min")
    print(f" VRAM max : {metadata['hardware']['max_memory_allocated_gb']:.2f} GB")
    
    print(f"\n Prochaines étapes :")
    print(f"   1. Tester modèle fine-tuné (inférence)")
    print(f"   2. Évaluer performances (benchmarks)")
    print(f"   3. Comparer avec modèle de base")
    
    return output_dir


# ==================== MAIN ====================

if __name__ == "__main__":
    
    # Vérifier GPU
    if not torch.cuda.is_available():
        print("  ATTENTION : Pas de GPU détecté !")
        print("   Le fine-tuning sera très lent sur CPU")
        response = input("   Continuer quand même ? (o/n) : ")
        if response.lower() != 'o':
            exit(0)
    
    # Lancer entraînement
    output_dir = train()
    
    print(f"\nScript terminé avec succès")
    print(f"Adapters LoRA disponibles : {output_dir}")
