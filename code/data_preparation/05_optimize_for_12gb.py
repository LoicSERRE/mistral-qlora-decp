"""
05 - Optimisation du corpus pour VRAM 12 GB
=============================================

Réduit le corpus dédupliqué pour respecter la contrainte QLoRA 12 GB VRAM.
Selon Dettmers et al. (2023), la limite pratique pour QLoRA 7B sur 12 GB
est d'environ 550K tokens (batch_size=4 en phase1, gradient checkpointing, 4-bit).

Stratégie : sélection par priorité de source, avec allocation proportionnelle.

Entrée : data/fine_tuning/training_data_merged.jsonl (étape 4)
Sortie : data/fine_tuning/training_data_final_12gb.jsonl
         data/fine_tuning/training_data_final_12gb_metadata.json
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime

# Reproductibilité
random.seed(42)

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "fine_tuning"

INPUT_FILE = DATA_DIR / "training_data_merged.jsonl"
OUTPUT_FILE = DATA_DIR / "training_data_final_12gb.jsonl"
METADATA_FILE = DATA_DIR / "training_data_final_12gb_metadata.json"

# Objectif tokens
TARGET_TOKENS = 550_000  # Limite QLoRA 7B sur 12 GB VRAM

# Priorités par source (ordre décroissant d'importance)
SOURCE_PRIORITIES = {
    "OUT_OF_SCOPE":       {"priority": 1, "allocation": 1.0,  "description": "Garde-fous essentiels"},
    "DECP_PROCEDURE":     {"priority": 2, "allocation": 0.18, "description": "Cible principale PoC"},
    "DECP_ACHETEUR":      {"priority": 2, "allocation": 0.18, "description": "Cible principale PoC"},
    "DECP_DATE":          {"priority": 2, "allocation": 0.18, "description": "Cible principale PoC"},
    "DECP":               {"priority": 2, "allocation": 0.18, "description": "Cible principale PoC"},
    "ELUS_CONSEILLERS":   {"priority": 3, "allocation": 0.10, "description": "Élus territoriaux 9 depts"},
    "RNE":                {"priority": 3, "allocation": 0.10, "description": "Élus territoriaux 9 depts"},
    "DELIBERATIONS":      {"priority": 3, "allocation": 0.10, "description": "Vocabulaire admin."},
    "DELIBERATIONS_TYPE": {"priority": 3, "allocation": 0.10, "description": "Vocabulaire admin."},
    "PROCEDURAL_SEUILS":  {"priority": 4, "allocation": 0.07, "description": "Réglementaire"},
    "PIAF":               {"priority": 5, "allocation": 0.05, "description": "Baseline français"},
    "BUDGETS":            {"priority": 5, "allocation": 0.02, "description": "Données financières"},
}


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def estimate_tokens(text):
    """Estimation du nombre de tokens (approximation 1 token = 4 caractères)."""
    return len(text) // 4


def pair_tokens(pair):
    """Nombre total de tokens estimé pour une paire Q/R."""
    prompt = pair.get("prompt", "")
    completion = pair.get("completion", "")
    return estimate_tokens(prompt + " " + completion)


def pair_quality_score(pair):
    """
    Score de qualité pour le tri intra-source.
    Favorise les paires avec réponses longues, vocabulaire diversifié, info complète.
    """
    completion = pair.get("completion", "")
    score = 0

    # Longueur réponse (favoriser les réponses détaillées)
    score += min(len(completion) / 5, 40)

    # Diversité vocabulaire
    words = set(completion.lower().split())
    score += min(len(words) / 3, 30)

    # Complétude informations (annotations)
    if "[Source :" in completion:
        score += 10
    if "[Territoire :" in completion:
        score += 10
    if any(c.isdigit() for c in completion):
        score += 10  # Contient des données chiffrées

    return score


def main():
    log("=" * 60)
    log("ÉTAPE 5 : OPTIMISATION POUR VRAM 12 GB")
    log("=" * 60)
    log(f"Objectif : ~{TARGET_TOKENS:,} tokens maximum")

    # Charger le corpus dédupliqué
    all_pairs = []
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_pairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    else:
        log(f"Fichier d'entrée introuvable : {INPUT_FILE}", "ERROR")
        return False

    total_tokens_before = sum(pair_tokens(p) for p in all_pairs)
    log(f"Corpus dédupliqué : {len(all_pairs)} paires, ~{total_tokens_before:,} tokens")

    # Grouper par source
    by_source = {}
    for pair in all_pairs:
        src = pair.get("source", "UNKNOWN")
        by_source.setdefault(src, []).append(pair)

    log(f"Sources détectées : {list(by_source.keys())}")

    # Phase 1 : Inclure toutes les paires critiques (OUT_OF_SCOPE)
    selected = []
    total_tokens_used = 0

    # OUT_OF_SCOPE : toujours inclus à 100%
    oos_pairs = by_source.pop("OUT_OF_SCOPE", [])
    selected.extend(oos_pairs)
    total_tokens_used += sum(pair_tokens(p) for p in oos_pairs)
    log(f"OUT_OF_SCOPE : {len(oos_pairs)} paires (100% inclus)")

    # Phase 2 : Conservation intégrale si dans le budget, sinon sélection par priorité
    remaining_budget = TARGET_TOKENS - total_tokens_used

    # Si le corpus dédupliqué est déjà dans les limites VRAM, tout conserver
    tokens_non_oos = total_tokens_before - total_tokens_used
    if tokens_non_oos <= remaining_budget:
        log(f"Corpus dans les limites ({total_tokens_before:,} <= {TARGET_TOKENS:,} tokens) - conservation intégrale")
        for source_name, pairs in sorted(by_source.items()):
            selected.extend(pairs)
            log(f"  {source_name:25} : {len(pairs):5} paires (100%)")
    else:
        log(f"Corpus dépasse le budget ({total_tokens_before:,} > {TARGET_TOKENS:,}) - sélection par priorité")

        # Trier les sources par priorité
        sorted_sources = sorted(
            by_source.items(),
            key=lambda x: SOURCE_PRIORITIES.get(x[0], {}).get("priority", 99)
        )

        for source_name, pairs in sorted_sources:
            if remaining_budget <= 0:
                log(f"  {source_name} : 0 paires (budget tokens épuisé)", "WARNING")
                continue

            config = SOURCE_PRIORITIES.get(source_name, {"allocation": 0.05})
            allocation = config.get("allocation", 0.05)

            # Trier par qualité décroissante
            pairs_sorted = sorted(pairs, key=pair_quality_score, reverse=True)

            # Calculer le nombre cible de paires
            target_tokens_source = int(TARGET_TOKENS * allocation)
            selected_source = []
            tokens_source = 0

            for pair in pairs_sorted:
                pt = pair_tokens(pair)
                if tokens_source + pt > target_tokens_source:
                    break
                if total_tokens_used + pt > TARGET_TOKENS:
                    break
                selected_source.append(pair)
                tokens_source += pt
                total_tokens_used += pt

            selected.extend(selected_source)
            pct = (len(selected_source) / len(pairs) * 100) if pairs else 0
            remaining_budget = TARGET_TOKENS - total_tokens_used

            log(f"  {source_name:25} : {len(selected_source):5} / {len(pairs):5} paires "
                f"({pct:5.1f}%) | ~{tokens_source:,} tokens")


    random.shuffle(selected)

    # Statistiques finales
    final_tokens = sum(pair_tokens(p) for p in selected)
    avg_prompt_len = sum(len(p.get("prompt", "")) for p in selected) / max(len(selected), 1)
    avg_completion_len = sum(len(p.get("completion", "")) for p in selected) / max(len(selected), 1)

    # Compter les garde-fous
    guardrails = sum(1 for p in selected if p.get("source") == "OUT_OF_SCOPE")

    # Départements couverts
    depts = set()
    for p in selected:
        d = p.get("departement", "")
        if d:
            depts.add(str(d))

    # Écrire le corpus final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in selected:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Hash SHA-256
    sha256_hash = hashlib.sha256()
    with open(OUTPUT_FILE, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(block)
    corpus_hash = sha256_hash.hexdigest()

    # Métadonnées
    # Split suggestion : 87.8% train / 12.2% réservé
    train_count = int(len(selected) * 0.878)
    test_reserved = len(selected) - train_count

    metadata = {
        "pipeline_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "target_tokens": TARGET_TOKENS,
        "corpus_stats": {
            "total_pairs": len(selected),
            "total_tokens_estimated": final_tokens,
            "token_utilization": f"{final_tokens / TARGET_TOKENS * 100:.1f}%",
            "avg_prompt_length_chars": round(avg_prompt_len),
            "avg_completion_length_chars": round(avg_completion_len),
            "guardrails_pairs": guardrails,
            "departments_covered": sorted(depts),
            "num_departments": len(depts),
        },
        "split_suggestion": {
            "train": train_count,
            "reserved_test": test_reserved,
            "note": "Le train set sera re-splitté en 80/10/10 (train/val/test) dans train_optimized.py"
        },
        "sources": {},
        "reduction": {
            "before_pairs": len(all_pairs) + len(oos_pairs),
            "after_pairs": len(selected),
            "reduction_percent": round(
                (1 - len(selected) / max(len(all_pairs) + len(oos_pairs), 1)) * 100, 1
            ),
        },
        "sha256_hash": corpus_hash,
        "seed": 42,
    }

    # Stats par source
    for pair in selected:
        src = pair.get("source", "UNKNOWN")
        metadata["sources"][src] = metadata["sources"].get(src, 0) + 1

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Rapport final
    log("=" * 60)
    log("RÉSUMÉ OPTIMISATION")
    log("=" * 60)
    log(f"  Paires avant réduction  : {len(all_pairs) + len(oos_pairs)}")
    log(f"  Paires après réduction  : {len(selected)}")
    log(f"  Réduction               : {metadata['reduction']['reduction_percent']}%")
    log(f"  Tokens estimés          : {final_tokens:,} ({final_tokens / TARGET_TOKENS * 100:.1f}% objectif)")
    log(f"  Long. moy. question     : {avg_prompt_len:.0f} caractères")
    log(f"  Long. moy. réponse      : {avg_completion_len:.0f} caractères")
    log(f"  Départements couverts   : {len(depts)}")
    log(f"  Garde-fous              : {guardrails} ({guardrails / max(len(selected), 1) * 100:.1f}%)")
    log(f"  Hash SHA-256            : {corpus_hash}")
    log(f"\nSortie : {OUTPUT_FILE}")
    log(f"Métadonnées : {METADATA_FILE}")

    return len(selected) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
