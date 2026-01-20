"""
DASHBOARD COMPARAISON - Visualisation Résultats
===============================================

Dashboard interactif pour comparer :
- Baseline vs Fine-Tuné vs Phases d'optimisation
- Fine-Tuné vs ChatGPT vs Claude
- Évolution métrique par phase

Usage:
    python dashboard_results.py
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_training_logs():
    """Charge tous les logs d'entraînement"""
    logs = []
    
    for log_file in LOGS_DIR.glob("training_*.json*"):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                # Essayer de charger JSON standard
                try:
                    data = json.load(f)
                    logs.append(data)
                except:
                    # Si JSONL (une ligne par objet)
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
        except Exception as e:
            print(f"  Erreur lecture {log_file}: {e}")
    
    return logs


def extract_metrics_from_logs(logs):
    """Extrait métriques des logs"""
    
    metrics = []
    
    for log in logs:
        # Identifier config
        config_name = log.get('config', {}).get('name', 'unknown')
        timestamp = log.get('timestamp', 'unknown')
        
        # Métriques finales
        final_metrics = log.get('final_metrics', {})
        
        metrics.append({
            'config': config_name,
            'timestamp': timestamp,
            'perplexity': final_metrics.get('perplexity', None),
            'loss': final_metrics.get('train_loss', None),
            'accuracy': final_metrics.get('accuracy', None),
            'tokens_per_sec': final_metrics.get('tokens_per_sec', None),
            'training_time_min': final_metrics.get('training_time_seconds', 0) / 60,
            'vram_peak_gb': final_metrics.get('vram_peak_mb', 0) / 1024
        })
    
    return pd.DataFrame(metrics)


def load_benchmark_results():
    """Charge résultats benchmarks externes"""
    
    benchmark_files = list(BENCHMARKS_DIR.glob("benchmark_results_*.json"))
    
    if not benchmark_files:
        return None
    
    # Charger le plus récent
    latest = max(benchmark_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def plot_training_evolution(df_logs):
    """Graphique évolution métriques par phase"""
    
    if df_logs.empty:
        print("  Aucune donnée d'entraînement")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    configs = df_logs['config'].unique()
    colors = sns.color_palette("husl", len(configs))
    
    # 1. Perplexity
    for i, config in enumerate(configs):
        data = df_logs[df_logs['config'] == config]
        if not data['perplexity'].isna().all():
            axes[0, 0].plot(range(len(data)), data['perplexity'], 
                           marker='o', label=config, color=colors[i])
    
    axes[0, 0].set_title('Évolution Perplexity', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Perplexity (↓ mieux)')
    axes[0, 0].set_xlabel('Run')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Accuracy
    for i, config in enumerate(configs):
        data = df_logs[df_logs['config'] == config]
        if not data['accuracy'].isna().all():
            axes[0, 1].plot(range(len(data)), data['accuracy'], 
                           marker='s', label=config, color=colors[i])
    
    axes[0, 1].set_title('Évolution Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy % (↑ mieux)')
    axes[0, 1].set_xlabel('Run')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. VRAM Peak
    avg_vram = df_logs.groupby('config')['vram_peak_gb'].mean()
    avg_vram.plot(kind='barh', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('VRAM Peak Moyen', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('VRAM (GB)')
    axes[1, 0].axvline(12, color='red', linestyle='--', label='Limite 12GB')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. Training Time
    avg_time = df_logs.groupby('config')['training_time_min'].mean()
    avg_time.plot(kind='barh', ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('Temps Entraînement Moyen', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Temps (minutes)')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = RESULTS_DIR / f"dashboard_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f" Dashboard training : {output_file}")
    
    plt.show()


def plot_benchmark_comparison(df_bench):
    """Graphique comparaison benchmarks externes"""
    
    if df_bench is None or df_bench.empty:
        print("  Aucun benchmark externe disponible")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Scores moyens par modèle
    avg_scores = df_bench.groupby('model')['score'].mean().sort_values(ascending=False)
    avg_scores.plot(kind='barh', ax=axes[0, 0], color='forestgreen')
    axes[0, 0].set_title('Scores Moyens par Modèle', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Score (0-1)')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Scores par catégorie
    category_pivot = df_bench.pivot_table(
        index='category',
        columns='model',
        values='score',
        aggfunc='mean'
    )
    category_pivot.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('Scores par Catégorie', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score (0-1)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Distribution scores
    models = df_bench['model'].unique()
    for model in models:
        model_scores = df_bench[df_bench['model'] == model]['score']
        axes[1, 0].hist(model_scores, bins=10, alpha=0.5, label=model)
    
    axes[1, 0].set_title('Distribution Scores', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Score (0-1)')
    axes[1, 0].set_ylabel('Nombre de questions')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Temps réponse
    avg_time = df_bench.groupby('model')['time_seconds'].mean()
    avg_time.plot(kind='barh', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Temps Moyen par Réponse', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Temps (secondes)')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = RESULTS_DIR / f"dashboard_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f" Dashboard benchmark : {output_file}")
    
    plt.show()


def print_summary(df_logs, df_bench):
    """Affiche résumé textuel"""
    
    print("\n" + "="*80)
    print(" RÉSUMÉ PERFORMANCES")
    print("="*80)
    
    # Training
    if not df_logs.empty:
        print("\n ENTRAÎNEMENTS :")
        print(f"   • Nombre total : {len(df_logs)}")
        print(f"   • Configurations : {', '.join(df_logs['config'].unique())}")
        
        # Meilleure config
        best_ppl = df_logs.loc[df_logs['perplexity'].idxmin()]
        print(f"\n    MEILLEURE CONFIG (Perplexity) :")
        print(f"      • Config     : {best_ppl['config']}")
        print(f"      • Perplexity : {best_ppl['perplexity']:.3f}")
        print(f"      • Accuracy   : {best_ppl['accuracy']:.1f}%")
        print(f"      • VRAM Peak  : {best_ppl['vram_peak_gb']:.2f} GB")
        
        # Gains depuis baseline
        if 'baseline' in df_logs['config'].values:
            baseline = df_logs[df_logs['config'] == 'baseline'].iloc[0]
            gain_ppl = ((baseline['perplexity'] - best_ppl['perplexity']) / baseline['perplexity']) * 100
            gain_acc = best_ppl['accuracy'] - baseline['accuracy']
            
            print(f"\n    GAINS vs BASELINE :")
            print(f"      • Perplexity : {gain_ppl:+.1f}%")
            print(f"      • Accuracy   : {gain_acc:+.1f} pts")
    
    # Benchmarks
    if df_bench is not None and not df_bench.empty:
        print("\n\n BENCHMARKS EXTERNES :")
        
        avg_scores = df_bench.groupby('model')['score'].mean().sort_values(ascending=False)
        
        print(f"   • Modèles testés : {len(avg_scores)}")
        print(f"   • Questions : {len(df_bench['question_id'].unique())}")
        
        print("\n    CLASSEMENT :")
        for i, (model, score) in enumerate(avg_scores.items(), 1):
            medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else f"{i}."
            print(f"      {medal} {model:30s} : {score:.3f}")
        
        # Catégories faibles
        print("\n     CATÉGORIES FAIBLES (pour fine-tuné) :")
        if 'Fine-Tuned (Local)' in df_bench['model'].values:
            ft_scores = df_bench[df_bench['model'] == 'Fine-Tuned (Local)']
            category_scores = ft_scores.groupby('category')['score'].mean().sort_values()
            
            for category, score in category_scores.head(3).items():
                print(f"      • {category:40s} : {score:.3f}")
    
    print("\n" + "="*80)


def main():
    print("\n╔" + "="*78 + "╗")
    print("║" + " "*25 + "DASHBOARD RÉSULTATS" + " "*34 + "║")
    print("╚" + "="*78 + "╝")
    
    # Charger données
    print("\n Chargement données...")
    
    logs = load_training_logs()
    df_logs = extract_metrics_from_logs(logs)
    print(f"    {len(df_logs)} entraînements chargés")
    
    df_bench = load_benchmark_results()
    if df_bench is not None:
        print(f"    {len(df_bench)} résultats benchmark chargés")
    else:
        print(f"     Aucun benchmark externe (lancer benchmark_external.py)")
    
    # Résumé
    print_summary(df_logs, df_bench)
    
    # Graphiques
    print("\n Génération graphiques...")
    
    if not df_logs.empty:
        plot_training_evolution(df_logs)
    
    if df_bench is not None:
        plot_benchmark_comparison(df_bench)
    
    print("\n Dashboard généré !")


if __name__ == "__main__":
    main()
