"""
SCRIPT FINAL DE COLLECTE - Données Publiques Françaises
========================================================

Script unique et testé qui collecte TOUTES les données nécessaires.
Utilise uniquement des sources validées et fonctionnelles.

Auteur: Projet ADS Fine-tuning LLM
Date: Décembre 2024
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import time
from datasets import load_dataset

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

def log(msg, emoji=""):
    """Helper de logging"""
    print(f"{emoji} {msg}")

def header(title):
    """Afficher un header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def collect_piaf():
    """
    PIAF - Questions/Réponses en français
    Source: HuggingFace datasets
    ~3,835 exemples Q/R
    """
    header("1 PIAF - Dataset Q/R Français")
    
    output_dir = DATA_DIR / "piaf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        log("Téléchargement depuis HuggingFace...", "")
        dataset = load_dataset("piaf")
        
        # Convertir en JSONL
        output_file = output_dir / "piaf_train.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset["train"]:
                entry = {
                    "prompt": f"Question : {example['question']}\nContexte : {example['context'][:500]}...",
                    "completion": example['answers']['text'][0] if example['answers']['text'] else "",
                    "source": "PIAF"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Stats
        size_mb = output_file.stat().st_size / (1024*1024)
        log(f" {len(dataset['train'])} exemples | {size_mb:.1f} MB", "")
        
        # Métadonnées
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "source": "HuggingFace - PIAF",
                "examples": len(dataset["train"]),
                "size_mb": size_mb,
                "collected_at": datetime.now().isoformat()
            }, f, indent=2)
        
        return True
    except Exception as e:
        log(f" Erreur: {e}", "")
        return False

def collect_decp():
    """
    DECP - Marchés publics
    Source: data.gouv.fr XML consolidé
    ~600MB de données
    """
    header("2 DECP - Marchés Publics")
    
    output_dir = DATA_DIR / "decp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://files.data.gouv.fr/decp/dgfip-pes-decp.xml"
    output_file = output_dir / "decp_consolide.xml"
    
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024*1024)
        log(f" Déjà téléchargé: {size_mb:.1f} MB", "")
        return True
    
    try:
        log("  Fichier volumineux (~600MB), téléchargement en cours...", "")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = (downloaded/total)*100
                        print(f"   {pct:.1f}% ({downloaded/(1024*1024):.0f}/{total/(1024*1024):.0f} MB)", end='\r')
        
        size_mb = output_file.stat().st_size / (1024*1024)
        print()
        log(f" Téléchargé: {size_mb:.1f} MB", "")
        
        # Métadonnées
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "source": "data.gouv.fr - DECP",
                "url": url,
                "size_mb": size_mb,
                "format": "XML",
                "collected_at": datetime.now().isoformat(),
                "note": "Fichier volumineux - nécessite extraction"
            }, f, indent=2)
        
        return True
    except Exception as e:
        log(f" Erreur: {e}", "")
        return False

def collect_budgets():
    """
    Budgets communaux 2023
    Source: API data.economie.gouv.fr
    Grandes villes françaises
    """
    header("3 Budgets Municipaux 2023")
    
    output_dir = DATA_DIR / "budgets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/balances-comptables-des-communes-en-2023/records"
    
    cities = ["PARIS", "LYON", "MARSEILLE", "TOULOUSE", "NICE", 
              "NANTES", "MONTPELLIER", "STRASBOURG", "BORDEAUX", "LILLE"]
    
    total_records = 0
    
    for city in cities:
        try:
            params = {
                "where": f'lbudg like "{city}%"',
                "limit": 100
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            records = data.get("results", [])
            
            if records:
                city_file = output_dir / f"{city.lower()}_2023.json"
                with open(city_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                
                log(f"{city}: {len(records)} lignes budgétaires", "")
                total_records += len(records)
            else:
                log(f"{city}: Aucune donnée", "")
            
            time.sleep(0.5)
            
        except Exception as e:
            log(f"{city}: Erreur - {e}", "")
    
    # Métadonnées
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump({
            "source": "data.economie.gouv.fr - Balances comptables",
            "year": 2023,
            "cities": cities,
            "total_records": total_records,
            "collected_at": datetime.now().isoformat()
        }, f, indent=2)
    
    log(f"Total: {total_records} lignes budgétaires", "")
    return total_records > 0

def collect_deliberations():
    """
    Délibérations municipales
    Source: data.gouv.fr - Recherche datasets SCDL
    Format standardisé
    """
    header("4 Délibérations Municipales")
    
    output_dir = DATA_DIR / "deliberations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rechercher datasets de délibérations
    search_url = "https://www.data.gouv.fr/api/1/datasets/"
    params = {"q": "délibérations SCDL", "page_size": 20}
    
    try:
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        datasets_found = data.get("data", [])
        
        log(f"Trouvé {len(datasets_found)} datasets de délibérations", "")
        
        collected = 0
        for dataset in datasets_found[:5]:
            resources = dataset.get("resources", [])
            
            for resource in resources[:2]:
                fmt = resource.get("format", "").lower()
                if fmt in ["csv", "json"] and collected < 5:
                    try:
                        url = resource.get("url")
                        resp = requests.get(url, timeout=60, stream=True)
                        resp.raise_for_status()
                        
                        # Vérifier taille (max 15MB)
                        content_length = int(resp.headers.get('content-length', 0))
                        if content_length > 15 * 1024 * 1024:
                            log(f"Fichier trop gros ({content_length/(1024*1024):.1f}MB)", "")
                            continue
                        
                        filename = f"delib_{collected}.{fmt}"
                        filepath = output_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=1024*1024):
                                f.write(chunk)
                        
                        size_mb = filepath.stat().st_size / (1024*1024)
                        log(f"{filename}: {size_mb:.1f} MB", "")
                        collected += 1
                        time.sleep(1)
                        
                    except Exception as e:
                        log(f"Erreur fichier: {e}", "")
        
        # Métadonnées
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "source": "data.gouv.fr - Délibérations SCDL",
                "files_collected": collected,
                "collected_at": datetime.now().isoformat()
            }, f, indent=2)
        
        log(f"Total: {collected} fichiers collectés", "")
        return collected > 0
        
    except Exception as e:
        log(f" Erreur: {e}", "")
        return False

def main():
    """Point d'entrée principal"""
    print("\n" + "="*60)
    print("   COLLECTE FINALE DES DONNÉES")
    print("="*60)
    print(f" Destination: {DATA_DIR}\n")
    
    stats = {
        "start_time": datetime.now().isoformat(),
        "datasets": {}
    }
    
    # Collecte
    stats["datasets"]["piaf"] = "success" if collect_piaf() else "failed"
    stats["datasets"]["decp"] = "success" if collect_decp() else "failed"
    stats["datasets"]["budgets"] = "success" if collect_budgets() else "failed"
    stats["datasets"]["deliberations"] = "success" if collect_deliberations() else "failed"
    
    stats["end_time"] = datetime.now().isoformat()
    
    # Sauvegarder stats
    with open(DATA_DIR / "collection_final.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Résumé
    header(" COLLECTE TERMINÉE")
    
    success = sum(1 for v in stats["datasets"].values() if v == "success")
    total = len(stats["datasets"])
    
    print(f"\n Résultats: {success}/{total} datasets collectés")
    print(f" Données dans: {DATA_DIR}")
    
    # Volumétrie totale
    total_size = sum(
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        for p in DATA_DIR.iterdir() if p.is_dir()
    ) / (1024*1024)
    
    print(f" Taille totale: {total_size:.1f} MB")
    print(f"\n Prochaine étape: Extraction et formatage")
    print(f"   → cd ../data_preparation")
    
if __name__ == "__main__":
    main()
