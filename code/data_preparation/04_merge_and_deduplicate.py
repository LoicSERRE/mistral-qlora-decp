"""
04 - Fusion et déduplication du corpus
=======================================

Fusionne les différentes sources et déduplique via hash SHA-256
des prompts normalisés (minuscules, suppression ponctuation, espaces multiples).

En cas de doublons, conservation de la completion la plus détaillée.

Entrée : data/fine_tuning/training_data_cleaned.jsonl (étape 3)
Sortie : data/fine_tuning/training_data_merged.jsonl
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "fine_tuning"

INPUT_FILE = DATA_DIR / "training_data_cleaned.jsonl"
OUTPUT_FILE = DATA_DIR / "training_data_merged.jsonl"


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def normalize_prompt(prompt):
    """
    Normalise un prompt pour la détection de doublons.
    - Minuscules
    - Suppression ponctuation
    - Suppression espaces multiples
    - Suppression accents (optionnel - désactivé pour garder la distinction)
    """
    text = prompt.lower().strip()

    # Supprimer la ponctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prompt_hash(prompt):
    """Calcule le hash SHA-256 d'un prompt normalisé."""
    normalized = normalize_prompt(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def completion_quality_score(completion):
    """
    Score de qualité d'une completion.
    Plus le score est élevé, plus la completion est détaillée/utile.
    """
    if not completion:
        return 0

    score = 0

    # Longueur (favoriser les réponses plus longues, jusqu'à un plafond)
    length = len(completion)
    score += min(length / 10, 50)  # max 50 points pour la longueur

    # Présence d'annotations source
    if "[Source :" in completion:
        score += 20

    # Présence de contexte territorial
    if "[Territoire :" in completion:
        score += 15

    # Diversité du vocabulaire
    words = set(completion.lower().split())
    score += min(len(words) / 5, 30)  # max 30 points pour la diversité

    # Présence d'information structurée (dates, montants, codes)
    if re.search(r"\d{4}-\d{2}-\d{2}", completion):
        score += 5  # Date
    if re.search(r"\d+\s*(?:euros?|€|EUR)", completion, re.IGNORECASE):
        score += 5  # Montant
    if re.search(r"SIRET|SIREN", completion):
        score += 5  # Identifiant

    return score


def main():
    log("=" * 60)
    log("ÉTAPE 4 : FUSION ET DÉDUPLICATION")
    log("=" * 60)

    # Charger les paires
    pairs = []
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    else:
        log(f"Fichier d'entrée introuvable : {INPUT_FILE}", "ERROR")
        return False

    log(f"Corpus brut fusionné : {len(pairs)} paires")

    # Déduplication par hash SHA-256
    seen_hashes = {}  # hash -> (index, quality_score)
    duplicates = 0
    improvements = 0

    for i, pair in enumerate(pairs):
        prompt = pair.get("prompt", "")
        completion = pair.get("completion", "")
        h = prompt_hash(prompt)
        quality = completion_quality_score(completion)

        if h in seen_hashes:
            duplicates += 1
            existing_idx, existing_quality = seen_hashes[h]

            # Garder la meilleure completion
            if quality > existing_quality:
                seen_hashes[h] = (i, quality)
                improvements += 1
        else:
            seen_hashes[h] = (i, quality)

    # Reconstruire le corpus dédupliqué
    unique_indices = sorted(idx for idx, _ in seen_hashes.values())
    deduplicated = [pairs[i] for i in unique_indices]

    # Stats par source
    sources = {}
    for pair in deduplicated:
        src = pair.get("source", "UNKNOWN")
        sources[src] = sources.get(src, 0) + 1

    # Écrire
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in deduplicated:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Rapport
    dedup_rate = (duplicates / len(pairs) * 100) if pairs else 0

    log("=" * 60)
    log("RÉSUMÉ DÉDUPLICATION")
    log("=" * 60)
    log(f"  Corpus brut fusionné    : {len(pairs)} paires")
    log(f"  Doublons détectés       : {duplicates} ({dedup_rate:.1f}%)")
    log(f"  Améliorations remplacées: {improvements}")
    log(f"  Corpus dédupliqué       : {len(deduplicated)} paires uniques")
    log(f"  Taux déduplication      : {dedup_rate:.1f}%")

    log(f"\n  Répartition par source :")
    for src, count in sorted(sources.items()):
        log(f"    {src:25} : {count}")

    log(f"\nSortie : {OUTPUT_FILE}")

    # Hash
    h = hashlib.sha256()
    h.update(OUTPUT_FILE.read_bytes())
    log(f"Hash SHA-256 : {h.hexdigest()[:32]}...")

    return len(deduplicated) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
