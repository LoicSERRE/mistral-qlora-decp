"""
PIPELINE MASTER - Génération Corpus Complet
Exécute tous les scripts dans le bon ordre pour générer training_data_final_12gb.jsonl
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
CODE_DIR = BASE_DIR / "code" / "data_preparation"
DATA_DIR = BASE_DIR / "data" / "fine_tuning"
RAW_DIR = BASE_DIR / "data" / "raw"

# Configuration pipeline
SCRIPTS = [
    {
        'name': '01_collect_real_data.py',
        'description': 'Télécharge données réelles (RNE, DECP)',
        'required_outputs': ['data/raw/enrichment/elus_municipaux_sud_france.csv'],
        'skip_on_failure': False
    },
    {
        'name': '02_generate_varied_questions.py',
        'description': 'Génère questions variées sur données réelles',
        'required_outputs': ['data/fine_tuning/training_data_enriched_varied.jsonl'],
        'skip_on_failure': False
    },
    {
        'name': '03_clean_existing_corpus.py',
        'description': 'Nettoie corpus existant (37,696 paires)',
        'required_outputs': ['data/fine_tuning/training_data_cleaned.jsonl'],
        'skip_on_failure': True  # Peut ne pas avoir de corpus existant
    },
    {
        'name': '04_merge_and_deduplicate.py',
        'description': 'Fusionne et déduplique corpus',
        'required_outputs': ['data/fine_tuning/training_data_merged.jsonl'],
        'skip_on_failure': False
    },
    {
        'name': '05_optimize_for_12gb.py',
        'description': 'Optimise pour 12GB VRAM',
        'required_outputs': ['data/fine_tuning/training_data_final_12gb.jsonl'],
        'skip_on_failure': False
    }
]

class PipelineExecutor:
    """Orchestrateur pipeline complet"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = []
        self.failed_scripts = []
        
    def check_prerequisites(self):
        """Vérifie prérequis avant exécution"""
        print("\n" + "="*80)
        print("VÉRIFICATION PRÉREQUIS")
        print("="*80)
        
        prereqs_ok = True
        
        # Vérifier corpus de base (optionnel mais recommandé)
        corpus_base = DATA_DIR / "training_data_all.jsonl"
        if corpus_base.exists():
            print(f" Corpus de base trouvé : {corpus_base}")
        else:
            print(f"  Corpus de base absent : {corpus_base}")
            print(f"   Le pipeline fonctionnera en mode enrichissement uniquement")
        
        # Vérifier Python
        print(f" Python : {sys.version.split()[0]}")
        
        # Vérifier modules requis
        required_modules = ['requests', 'pandas', 'json']
        for module in required_modules:
            try:
                __import__(module)
                print(f" Module {module} : Installé")
            except ImportError:
                print(f" Module {module} : MANQUANT")
                prereqs_ok = False
        
        return prereqs_ok
    
    def run_script(self, script_info):
        """Exécute un script du pipeline"""
        script_path = CODE_DIR / script_info['name']
        
        print("\n" + "="*80)
        print(f"  SCRIPT : {script_info['name']}")
        print("="*80)
        print(f"Description : {script_info['description']}")
        print(f"Chemin : {script_path}")
        print(f"Heure : {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        if not script_path.exists():
            print(f" Script introuvable : {script_path}")
            return False
        
        try:
            # Exécuter script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=False,
                text=True,
                check=True
            )
            
            # Vérifier outputs attendus
            all_outputs_ok = True
            for output_path in script_info['required_outputs']:
                full_path = BASE_DIR / output_path
                if full_path.exists():
                    size_mb = full_path.stat().st_size / (1024**2)
                    print(f"    Output généré : {output_path} ({size_mb:.2f} MB)")
                else:
                    print(f"     Output manquant : {output_path}")
                    all_outputs_ok = False
            
            if all_outputs_ok:
                print(f"\n {script_info['name']} - SUCCÈS")
                return True
            else:
                print(f"\n  {script_info['name']} - SUCCÈS PARTIEL (outputs manquants)")
                return not script_info['skip_on_failure']
                
        except subprocess.CalledProcessError as e:
            print(f"\n {script_info['name']} - ÉCHEC")
            print(f"   Code retour : {e.returncode}")
            if script_info['skip_on_failure']:
                print(f"     Erreur ignorée (script optionnel)")
                return True
            return False
        
        except Exception as e:
            print(f"\n {script_info['name']} - ERREUR : {str(e)}")
            return False
    
    def validate_final_corpus(self):
        """Valide le corpus final généré"""
        print("\n" + "="*80)
        print("VALIDATION CORPUS FINAL")
        print("="*80)
        
        final_corpus = DATA_DIR / "training_data_final_12gb.jsonl"
        
        if not final_corpus.exists():
            print(f" Corpus final introuvable : {final_corpus}")
            return False
        
        # Charger et analyser
        pairs = []
        with open(final_corpus, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except:
                    pass
        
        total_chars = sum(len(p.get('prompt', '')) + len(p.get('completion', '')) for p in pairs)
        estimated_tokens = total_chars // 4
        
        print(f"\n MÉTRIQUES CORPUS FINAL :")
        print(f"   Paires Q/R : {len(pairs):,}")
        print(f"   Caractères : {total_chars:,}")
        print(f"   Tokens estimés : {estimated_tokens:,}")
        
        # Vérifications
        checks = {
            'Paires >= 500 (minimum viable)': len(pairs) >= 500,
            'Tokens < 600K (12GB safe)': estimated_tokens < 600000,
            'Aucun prompt vide': all(p.get('prompt') for p in pairs),
            'Aucune completion vide': all(p.get('completion') for p in pairs)
        }
        
        print(f"\n VALIDATIONS :")
        all_ok = True
        for check, status in checks.items():
            icon = '' if status else ''
            print(f"   {icon} {check}")
            if not status:
                all_ok = False
        
        # Hash SHA256
        sha256_hash = hashlib.sha256()
        with open(final_corpus, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        print(f"\n Hash SHA256 : {sha256_hash.hexdigest()[:32]}...")
        
        return all_ok
    
    def generate_report(self):
        """Génère rapport d'exécution"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("RAPPORT D'EXÉCUTION PIPELINE")
        print("="*80)
        
        print(f"\n  Durée totale : {duration:.0f} secondes ({duration/60:.1f} minutes)")
        print(f" Date exécution : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n SCRIPTS EXÉCUTÉS :")
        for i, script in enumerate(SCRIPTS, 1):
            status = '' if script['name'] not in self.failed_scripts else ''
            print(f"   {status} {i}. {script['name']}")
        
        success_rate = ((len(SCRIPTS) - len(self.failed_scripts)) / len(SCRIPTS)) * 100
        print(f"\n Taux de succès : {success_rate:.0f}%")
        
        if not self.failed_scripts:
            print("\n" + ""*20)
            print(" PIPELINE COMPLET TERMINÉ AVEC SUCCÈS !")
            print(""*20)
        else:
            print(f"\n  {len(self.failed_scripts)} script(s) en échec")
    
    def run(self):
        """Exécute pipeline complet"""
        print("="*80)
        print(" PIPELINE MASTER - GÉNÉRATION CORPUS COMPLET")
        print("="*80)
        print(f"\nDossier projet : {BASE_DIR}")
        print(f"Heure début : {self.start_time.strftime('%H:%M:%S')}")
        
        # Protection corpus existant
        final_corpus = DATA_DIR / "training_data_final_12gb.jsonl"
        force = "--force" in sys.argv
        if final_corpus.exists() and not force:
            with open(final_corpus, encoding="utf-8") as f:
                n_pairs = sum(1 for _ in f)
            print(f"\n Corpus de fine-tuning déjà présent : {n_pairs:,} paires")
            print(f"   Chemin : {final_corpus}")
            print("   Le pipeline ne régénère pas les données existantes.")
            print("   Utilisez --force pour forcer la régénération depuis les APIs.")
            print("\n Conseil : le corpus présent dans le repo est le corpus canonique")
            print("   utilisé pour entraîner les adapters dans models/adapters/.")
            return True
        
        if force:
            print("\n --force activé : régénération complète du corpus depuis les APIs")
        
        # Vérifier prérequis
        if not self.check_prerequisites():
            print("\n Prérequis non satisfaits. Arrêt du pipeline.")
            return False
        
        # Exécuter scripts séquentiellement
        for script_info in SCRIPTS:
            success = self.run_script(script_info)
            
            if not success:
                self.failed_scripts.append(script_info['name'])
                if not script_info['skip_on_failure']:
                    print(f"\n ARRÊT PIPELINE : Échec script critique {script_info['name']}")
                    self.generate_report()
                    return False
        
        # Valider corpus final
        if not self.validate_final_corpus():
            print("\n VALIDATION CORPUS FINAL ÉCHOUÉE")
            self.generate_report()
            return False
        
        # Générer rapport
        self.generate_report()
        
        return len(self.failed_scripts) == 0

if __name__ == "__main__":
    import sys
    
    print("\n" + ""*40)
    print("ATTENTION : Ce script va regénérer TOUT le corpus")
    print("Cela peut prendre 15-20 minutes")
    print(""*40)
    
    # Vérifier si mode automatique (argument --auto)
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("\n Mode automatique activé - pas de confirmation")
        executor = PipelineExecutor()
        success = executor.run()
        sys.exit(0 if success else 1)
    
    response = input("\nContinuer ? (o/n) : ").strip().lower()
    
    if response == 'o':
        executor = PipelineExecutor()
        success = executor.run()
        sys.exit(0 if success else 1)
    else:
        print("\n Pipeline annulé par l'utilisateur")
        sys.exit(0)
