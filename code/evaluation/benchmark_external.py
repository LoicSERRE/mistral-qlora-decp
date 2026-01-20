"""
BENCHMARK EXTERNE - Comparaison LLM
===================================

Compare les performances de différents modèles sur les 20 questions standardisées :
- Modèle fine-tuné (local)
- Modèle base Mistral-7B (local)
- ChatGPT (OpenAI API) - optionnel
- Claude (Anthropic API) - optionnel
- Mistral API - optionnel

Usage:
    python benchmark_external.py --models finetuned,base
    python benchmark_external.py --models finetuned,chatgpt --openai-key YOUR_KEY
    python benchmark_external.py --all --openai-key KEY --anthropic-key KEY
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import local
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Charger questions
sys.path.insert(0, str(DATA_DIR))
from test_questions import TEST_QUESTIONS, calculate_score

# System prompt
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'accès aux données publiques françaises, notamment :
- DECP (Données Essentielles de la Commande Publique)
- RNE (Répertoire National des Élus)

Tu réponds avec précision en citant tes sources. Si une information n'est pas dans ton corpus, tu le dis clairement."""


# ==================== MODÈLES LOCAUX ====================

class LocalModelBenchmark:
    """Benchmark pour modèles locaux (Mistral base/fine-tuné)"""
    
    def __init__(self, model_type='base', adapter_path=None):
        self.model_type = model_type
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Charge le modèle"""
        print(f"\n Chargement modèle {self.model_type}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.model_type == 'base':
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.3",
                torch_dtype=torch.float16,
                device_map='cuda:0'
            )
        elif self.model_type == 'finetuned':
            if not self.adapter_path:
                raise ValueError("adapter_path requis pour modèle fine-tuné")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.3",
                torch_dtype=torch.float16,
                device_map='cuda:0'
            )
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        print(f" Modèle {self.model_type} chargé")
    
    def generate_response(self, question: str) -> str:
        """Génère réponse pour une question"""
        formatted_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nQuestion : {question} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire réponse après [/INST]
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response


# ==================== API EXTERNES ====================

class ChatGPTBenchmark:
    """Benchmark pour ChatGPT (OpenAI API)"""
    
    def __init__(self, api_key: str, model='gpt-4'):
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("pip install openai requis pour ChatGPT")
    
    def generate_response(self, question: str) -> str:
        """Génère réponse via API OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERREUR API] {str(e)}"


class ClaudeBenchmark:
    """Benchmark pour Claude (Anthropic API)"""
    
    def __init__(self, api_key: str, model='claude-3-sonnet-20240229'):
        self.api_key = api_key
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("pip install anthropic requis pour Claude")
    
    def generate_response(self, question: str) -> str:
        """Génère réponse via API Anthropic"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"[ERREUR API] {str(e)}"


# ==================== BENCHMARK ====================

def run_benchmark(models: List[Dict], questions: List[Dict]) -> pd.DataFrame:
    """
    Exécute benchmark sur tous les modèles
    
    Args:
        models: Liste de {name, instance}
        questions: Liste des questions test
        
    Returns:
        DataFrame avec résultats
    """
    results = []
    
    for model_info in models:
        model_name = model_info['name']
        model_instance = model_info['instance']
        
        print(f"\n{'='*80}")
        print(f" BENCHMARK : {model_name}")
        print(f"{'='*80}")
        
        for i, q in enumerate(questions, 1):
            question = q['question']
            expected = q['expected_answer']
            keywords = q['keywords']
            category = q['category']
            
            print(f"\n[{i}/{len(questions)}] {category}")
            print(f"Q: {question}")
            
            # Générer réponse
            start_time = time.time()
            response = model_instance.generate_response(question)
            elapsed = time.time() - start_time
            
            print(f"R: {response[:100]}..." if len(response) > 100 else f"R: {response}")
            
            # Calculer score
            score_details = calculate_score(response, expected, keywords)
            
            print(f"Score: {score_details['total']:.2f} ({score_details['keywords_found']} mots-clés)")
            
            results.append({
                'model': model_name,
                'question_id': q['id'],
                'category': category,
                'difficulty': q['difficulty'],
                'question': question,
                'expected': expected,
                'response': response,
                'score': score_details['total'],
                'exact_match': score_details['exact_match'],
                'keywords_ratio': score_details['keywords_ratio'],
                'time_seconds': round(elapsed, 2)
            })
    
    return pd.DataFrame(results)


def generate_comparison_report(df: pd.DataFrame, output_dir: Path):
    """Génère rapport comparatif avec graphiques"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Scores moyens par modèle
    avg_scores = df.groupby('model')['score'].mean().sort_values(ascending=False)
    
    print(f"\n{'='*80}")
    print(f" SCORES MOYENS PAR MODÈLE")
    print(f"{'='*80}")
    for model, score in avg_scores.items():
        print(f"   {model:30s} : {score:.3f}")
    
    # 2. Scores par catégorie
    category_scores = df.pivot_table(
        index='category',
        columns='model',
        values='score',
        aggfunc='mean'
    )
    
    print(f"\n{'='*80}")
    print(f" SCORES PAR CATÉGORIE")
    print(f"{'='*80}")
    print(category_scores.round(3).to_string())
    
    # 3. Temps de réponse moyen
    avg_time = df.groupby('model')['time_seconds'].mean()
    
    print(f"\n{'='*80}")
    print(f"  TEMPS MOYEN PAR RÉPONSE")
    print(f"{'='*80}")
    for model, t in avg_time.items():
        print(f"   {model:30s} : {t:.2f}s")
    
    # 4. Graphiques
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 4.1 Scores moyens
    avg_scores.plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Scores Moyens par Modèle', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Score (0-1)')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 4.2 Scores par catégorie
    category_scores.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('Scores par Catégorie', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score (0-1)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 4.3 Temps de réponse
    avg_time.plot(kind='barh', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('Temps Moyen par Réponse', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Temps (secondes)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4.4 Distribution scores
    for model in df['model'].unique():
        model_scores = df[df['model'] == model]['score']
        axes[1, 1].hist(model_scores, bins=10, alpha=0.5, label=model)
    axes[1, 1].set_title('Distribution des Scores', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Score (0-1)')
    axes[1, 1].set_ylabel('Nombre de questions')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_file = output_dir / f"benchmark_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n Graphiques sauvegardés : {plot_file}")
    
    # 5. Export JSON
    json_file = output_dir / f"benchmark_results_{timestamp}.json"
    df.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f" Résultats JSON : {json_file}")
    
    # 6. Export CSV
    csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f" Résultats CSV : {csv_file}")
    
    return avg_scores


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Benchmark externe - Comparaison LLM')
    parser.add_argument('--models', type=str, default='finetuned,base',
                       help='Modèles à tester (séparés par virgule): finetuned,base,chatgpt,claude')
    parser.add_argument('--adapter', type=str, default=None,
                       help='Chemin vers adapter fine-tuné (si non spécifié, utilise le dernier)')
    parser.add_argument('--openai-key', type=str, help='Clé API OpenAI')
    parser.add_argument('--anthropic-key', type=str, help='Clé API Anthropic')
    parser.add_argument('--all', action='store_true', help='Tester tous les modèles')
    
    args = parser.parse_args()
    
    # Modèles à tester
    requested_models = args.models.split(',') if not args.all else ['finetuned', 'base', 'chatgpt', 'claude']
    
    models = []
    
    # Modèle fine-tuné
    if 'finetuned' in requested_models:
        adapter_path = args.adapter
        if not adapter_path:
            # Trouver le dernier adapter
            adapters = list(ADAPTERS_DIR.glob("mistral-7b-*"))
            if adapters:
                adapter_path = max(adapters, key=lambda p: p.stat().st_mtime)
                print(f" Adapter auto-détecté : {adapter_path}")
            else:
                print(" Aucun adapter trouvé, skipping fine-tuned")
                requested_models.remove('finetuned')
        
        if adapter_path:
            ft_model = LocalModelBenchmark('finetuned', adapter_path)
            ft_model.load_model()
            models.append({'name': 'Fine-Tuned (Local)', 'instance': ft_model})
    
    # Modèle base
    if 'base' in requested_models:
        base_model = LocalModelBenchmark('base')
        base_model.load_model()
        models.append({'name': 'Mistral-7B Base', 'instance': base_model})
    
    # ChatGPT
    if 'chatgpt' in requested_models:
        if args.openai_key:
            try:
                chatgpt = ChatGPTBenchmark(args.openai_key)
                models.append({'name': 'ChatGPT (GPT-4)', 'instance': chatgpt})
            except ImportError as e:
                print(f"  ChatGPT skipped : {e}")
        else:
            print("  ChatGPT skipped : --openai-key manquant")
    
    # Claude
    if 'claude' in requested_models:
        if args.anthropic_key:
            try:
                claude = ClaudeBenchmark(args.anthropic_key)
                models.append({'name': 'Claude 3 Sonnet', 'instance': claude})
            except ImportError as e:
                print(f"  Claude skipped : {e}")
        else:
            print("  Claude skipped : --anthropic-key manquant")
    
    if not models:
        print(" Aucun modèle à tester")
        return
    
    # Benchmark
    print(f"\n{'='*80}")
    print(f" BENCHMARK EXTERNE - {len(models)} modèles × {len(TEST_QUESTIONS)} questions")
    print(f"{'='*80}")
    
    results_df = run_benchmark(models, TEST_QUESTIONS)
    
    # Rapport
    avg_scores = generate_comparison_report(results_df, RESULTS_DIR)
    
    print(f"\n{'='*80}")
    print(f" BENCHMARK TERMINÉ")
    print(f"{'='*80}")
    print(f"\n CLASSEMENT :")
    for i, (model, score) in enumerate(avg_scores.items(), 1):
        medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else f"{i}."
        print(f"   {medal} {model:30s} : {score:.3f}")


if __name__ == "__main__":
    main()
