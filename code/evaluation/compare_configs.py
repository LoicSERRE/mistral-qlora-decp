"""
COMPARAISON CONFIGS - Mesure des Gains Réels
=============================================

Compare les performances de différentes configurations
pour identifier la meilleure.

Métriques comparées :
- Perplexity (↓ meilleur)
- Loss (↓ meilleur)
- Factual Accuracy (↑ meilleur)
- Tokens/sec (↑ meilleur)
- VRAM Peak (↓ meilleur)
- Training Time
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"


def evaluate_config(adapter_dir, test_questions, tokenizer_base):
    """Évalue une configuration sur test set"""
    
    print(f"\n Évaluation : {adapter_dir.name}")
    
    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.float16,
        device_map='cuda:0'
    )
    
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Évaluer sur questions test
    perplexities = []
    losses = []
    correct = 0
    total_time = 0
    
    for q in test_questions:
        prompt = q['prompt']
        expected = q['expected']
        
        inputs = tokenizer_base(prompt, return_tensors="pt").to(model.device)
        
        start = datetime.now()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer_base.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        end = datetime.now()
        duration = (end - start).total_seconds()
        total_time += duration
        
        # Response
        response = tokenizer_base.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Loss/PPL
        with torch.no_grad():
            loss_outputs = model(**inputs, labels=inputs['input_ids'])
            loss = loss_outputs.loss.item()
            ppl = torch.exp(loss_outputs.loss).item()
        
        losses.append(loss)
        perplexities.append(ppl)
        
        # Accuracy
        if expected.lower() in response.lower():
            correct += 1
    
    # Libérer mémoire
    del model
    torch.cuda.empty_cache()
    
    return {
        "perplexity": sum(perplexities) / len(perplexities),
        "loss": sum(losses) / len(losses),
        "factual_accuracy": (correct / len(test_questions)) * 100,
        "avg_time_per_response": total_time / len(test_questions),
        "tokens_per_sec": 200 / (total_time / len(test_questions))  # Approximation
    }


def compare_all_configs():
    """Compare toutes les configurations disponibles"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " "*22 + "COMPARAISON DES CONFIGURATIONS" + " "*26 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    # Trouver tous les adapters
    adapter_dirs = [d for d in ADAPTERS_DIR.iterdir() if d.is_dir() and d.name.startswith('mistral-7b')]
    
    if not adapter_dirs:
        print(" Aucun adapter trouvé dans models/adapters/")
        return
    
    print(f" {len(adapter_dirs)} configurations trouvées :\n")
    for i, d in enumerate(adapter_dirs, 1):
        print(f"   {i}. {d.name}")
    
    # Charger tokenizer
    print(f"\n Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Questions de test (échantillon)
    test_questions = [
        {
            "prompt": "Quel est le montant du marché de travaux de voirie rue des Sables à Montpellier en 2024 ?",
            "expected": "150,000"
        },
        {
            "prompt": "Qui est le maire de Toulouse élu en 2020 ?",
            "expected": "Jean-Luc Moudenc"
        },
        {
            "prompt": "Quelle procédure pour un marché de 35 000 euros ?",
            "expected": "procédure adaptée"
        },
        {
            "prompt": "Nombre de délibérations urbanisme Toulouse Métropole 2024 ?",
            "expected": "147"
        },
        {
            "prompt": "Montant seuil européen marchés publics ?",
            "expected": "90,000"
        }
    ]
    
    print(f"\n Test sur {len(test_questions)} questions\n")
    
    # Évaluer chaque config
    results = {}
    
    for adapter_dir in adapter_dirs:
        try:
            metrics = evaluate_config(adapter_dir, test_questions, tokenizer)
            results[adapter_dir.name] = metrics
            
            print(f"    {adapter_dir.name}")
            print(f"      PPL: {metrics['perplexity']:.2f} | Acc: {metrics['factual_accuracy']:.1f}% | Loss: {metrics['loss']:.4f}")
        
        except Exception as e:
            print(f"    {adapter_dir.name} : Erreur - {e}")
    
    # Générer rapport
    print(f"\n{'='*80}")
    print(" RAPPORT COMPARATIF")
    print(f"{'='*80}\n")
    
    # Tableau
    df = pd.DataFrame(results).T
    df = df.sort_values('perplexity')
    
    print(df.to_string())
    
    # Meilleure config
    best_config = df['perplexity'].idxmin()
    
    print(f"\n{'='*80}")
    print(f" MEILLEURE CONFIGURATION : {best_config}")
    print(f"{'='*80}")
    print(f"   Perplexity       : {df.loc[best_config, 'perplexity']:.2f}")
    print(f"   Factual Accuracy : {df.loc[best_config, 'factual_accuracy']:.1f}%")
    print(f"   Loss             : {df.loc[best_config, 'loss']:.4f}")
    
    # Gains vs baseline
    if 'mistral-7b-baseline' in df.index:
        baseline = df.loc['mistral-7b-baseline']
        best = df.loc[best_config]
        
        ppl_gain = ((baseline['perplexity'] - best['perplexity']) / baseline['perplexity']) * 100
        acc_gain = best['factual_accuracy'] - baseline['factual_accuracy']
        
        print(f"\n Gains vs Baseline :")
        print(f"   Perplexity : {ppl_gain:+.1f}%")
        print(f"   Accuracy   : {acc_gain:+.1f} points")
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = RESULTS_DIR / f"config_comparison_{timestamp}.json"
    
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "n_configs": len(results),
        "n_test_questions": len(test_questions),
        "results": results,
        "best_config": best_config,
        "dataframe": df.to_dict()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Rapport sauvegardé : {output_file}")
    
    # Graphique
    generate_comparison_charts(df, best_config)
    
    return df, best_config


def generate_comparison_charts(df, best_config):
    """Génère graphiques comparatifs"""
    
    print(f"\n Génération graphiques...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PPL
    df['perplexity'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Perplexity (↓ meilleur)')
    axes[0, 0].set_ylabel('PPL')
    axes[0, 0].axhline(y=df.loc[best_config, 'perplexity'], color='green', linestyle='--', label='Best')
    axes[0, 0].legend()
    
    # Accuracy
    df['factual_accuracy'].plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Factual Accuracy (↑ meilleur)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].axhline(y=df.loc[best_config, 'factual_accuracy'], color='green', linestyle='--', label='Best')
    axes[0, 1].legend()
    
    # Loss
    df['loss'].plot(kind='bar', ax=axes[1, 0], color='indianred')
    axes[1, 0].set_title('Loss (↓ meilleur)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].axhline(y=df.loc[best_config, 'loss'], color='green', linestyle='--', label='Best')
    axes[1, 0].legend()
    
    # Tokens/sec
    df['tokens_per_sec'].plot(kind='bar', ax=axes[1, 1], color='mediumseagreen')
    axes[1, 1].set_title('Tokens/sec (↑ meilleur)')
    axes[1, 1].set_ylabel('Tokens/s')
    axes[1, 1].axhline(y=df.loc[best_config, 'tokens_per_sec'], color='green', linestyle='--', label='Best')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_chart = RESULTS_DIR / f"config_comparison_{timestamp}.png"
    plt.savefig(output_chart, dpi=150, bbox_inches='tight')
    
    print(f"    Graphiques : {output_chart}")


if __name__ == "__main__":
    compare_all_configs()
