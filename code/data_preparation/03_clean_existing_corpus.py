"""
03 - Nettoyage du corpus existant
==================================

Amélioration qualitative du corpus existant :
  - Enrichissement des réponses "Non renseigné"
  - Ajout d'annotations de source [Source : DECP données réelles]
  - Ajout du contexte territorial [Territoire : Hérault (34)]
  - Correction encodage UTF-8

Entrée : data/fine_tuning/training_data_all.jsonl (corpus existant, optionnel)
         data/fine_tuning/training_data_enriched_varied.jsonl (étape 2)
Sortie : data/fine_tuning/training_data_cleaned.jsonl
"""

import json
import re
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "fine_tuning"

INPUT_EXISTING = DATA_DIR / "training_data_all.jsonl"
INPUT_ENRICHED = DATA_DIR / "training_data_enriched_varied.jsonl"
OUTPUT_FILE = DATA_DIR / "training_data_cleaned.jsonl"

# Départements pour annotation territoriale
DEPARTEMENTS = {
    "11": "Aude", "13": "Bouches-du-Rhône", "30": "Gard",
    "31": "Haute-Garonne", "33": "Gironde", "34": "Hérault",
    "66": "Pyrénées-Orientales", "69": "Rhône", "81": "Tarn",
}

# Régions associées
DEPT_TO_REGION = {
    "11": "Occitanie", "13": "PACA", "30": "Occitanie",
    "31": "Occitanie", "33": "Nouvelle-Aquitaine", "34": "Occitanie",
    "66": "Occitanie", "69": "Auvergne-Rhône-Alpes", "81": "Occitanie",
}


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def fix_encoding(text):
    """Corrige les problèmes d'encodage courants (UTF-8)."""
    if not text:
        return text

    # Normalisation Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Correction des séquences UTF-8 mal décodées
    replacements = {
        "Ã©": "é", "Ã¨": "è", "Ã ": "à", "Ã§": "ç",
        "Ã´": "ô", "Ã®": "î", "Ã¹": "ù", "Ã»": "û",
        "Ã¢": "â", "Ãª": "ê", "Ã«": "ë", "Ã¯": "ï",
        "â\x80\x99": "'", "â\x80\x93": "–", "â\x80\x94": "—",
        "â\x80\x9c": "\"", "â\x80\x9d": "\"",
        "\x00": "", "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def enrich_non_renseigne(completion):
    """
    Remplace les réponses 'Non renseigné' par un message plus informatif.
    Spécifiquement pour les données DECP où certains champs sont optionnels.
    """
    patterns = [
        (r"(?i)\bNon renseigné\b", "Non communiqué dans DECP (information non obligatoire)"),
        (r"(?i)\bNon disponible\b", "Non communiqué dans DECP (information non obligatoire)"),
        (r"(?i)\bN/A\b", "Non communiqué dans DECP (information non obligatoire)"),
    ]

    for pattern, replacement in patterns:
        completion = re.sub(pattern, replacement, completion)

    return completion


def add_source_annotation(pair):
    """Ajoute l'annotation [Source : ...] si absente."""
    completion = pair.get("completion", "")
    source = pair.get("source", "")

    # Déjà annoté ?
    if "[Source :" in completion:
        return completion

    # Déterminer la source
    source_map = {
        "DECP": "[Source : DECP données réelles]",
        "DECP_PROCEDURE": "[Source : DECP données réelles]",
        "DECP_ACHETEUR": "[Source : DECP données réelles]",
        "DECP_DATE": "[Source : DECP données réelles]",
        "ELUS_CONSEILLERS": "[Source : RNE données réelles]",
        "RNE": "[Source : RNE données réelles]",
        "DELIBERATIONS": "[Source : Délibérations SCDL données réelles]",
        "DELIBERATIONS_TYPE": "[Source : Délibérations SCDL données réelles]",
        "BUDGETS": "[Source : Budgets publics données réelles]",
        "PROCEDURAL_SEUILS": "[Source : Code de la commande publique]",
        "PIAF": "[Source : PIAF baseline français]",
        "OUT_OF_SCOPE": "",  # Pas d'annotation pour les garde-fous
    }

    annotation = source_map.get(source, "[Source : DECP données réelles]")
    if annotation and not completion.endswith(annotation):
        completion = completion.rstrip(". ") + ". " + annotation

    return completion


def add_territorial_context(pair):
    """Ajoute l'annotation [Territoire : ...] si pertinent."""
    completion = pair.get("completion", "")
    dept_code = pair.get("departement", "")

    # Déjà annoté ou pas de département ?
    if "[Territoire :" in completion or not dept_code:
        return completion

    dept_nom = DEPARTEMENTS.get(str(dept_code), "")
    region = DEPT_TO_REGION.get(str(dept_code), "")

    if dept_nom:
        annotation = f"[Territoire : {dept_nom} ({dept_code})"
        if region:
            annotation += f", {region}"
        annotation += "]"

        completion = completion.rstrip() + " " + annotation

    return completion


def clean_pair(pair):
    """Applique toutes les opérations de nettoyage à une paire."""
    prompt = pair.get("prompt", "")
    completion = pair.get("completion", "")

    # 1. Correction encodage UTF-8
    prompt = fix_encoding(prompt)
    completion = fix_encoding(completion)

    # 2. Enrichissement "Non renseigné"
    completion = enrich_non_renseigne(completion)

    # 3. Supprimer espaces multiples
    prompt = re.sub(r"\s+", " ", prompt).strip()
    completion = re.sub(r"\s+", " ", completion).strip()

    # 4. Vérifier que les paires ne sont pas vides
    if not prompt or not completion:
        return None

    # Reconstruire la paire
    cleaned = dict(pair)
    cleaned["prompt"] = prompt
    cleaned["completion"] = completion

    # 5. Ajouter annotations source
    cleaned["completion"] = add_source_annotation(cleaned)

    # 6. Ajouter contexte territorial
    cleaned["completion"] = add_territorial_context(cleaned)

    return cleaned


def main():
    log("=" * 60)
    log("ÉTAPE 3 : NETTOYAGE DU CORPUS")
    log("=" * 60)

    all_pairs = []
    stats = {
        "loaded_existing": 0,
        "loaded_enriched": 0,
        "encoding_fixed": 0,
        "non_renseigne_enriched": 0,
        "source_annotated": 0,
        "territory_annotated": 0,
        "empty_removed": 0,
    }

    # Charger le corpus existant (s'il existe)
    if INPUT_EXISTING.exists():
        log(f"Chargement corpus existant : {INPUT_EXISTING}")
        with open(INPUT_EXISTING, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    all_pairs.append(pair)
                    stats["loaded_existing"] += 1
                except json.JSONDecodeError:
                    continue
        log(f"  {stats['loaded_existing']} paires chargées depuis corpus existant")
    else:
        log("Pas de corpus existant trouvé (mode enrichissement uniquement)")

    # Charger le corpus enrichi (étape 2)
    if INPUT_ENRICHED.exists():
        log(f"Chargement corpus enrichi : {INPUT_ENRICHED}")
        with open(INPUT_ENRICHED, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    all_pairs.append(pair)
                    stats["loaded_enriched"] += 1
                except json.JSONDecodeError:
                    continue
        log(f"  {stats['loaded_enriched']} paires chargées depuis corpus enrichi")
    else:
        log("Pas de corpus enrichi trouvé", "WARNING")

    log(f"Total brut : {len(all_pairs)} paires")

    # Nettoyage
    cleaned_pairs = []
    for pair in all_pairs:
        original_completion = pair.get("completion", "")

        cleaned = clean_pair(pair)

        if cleaned is None:
            stats["empty_removed"] += 1
            continue

        # Compter les opérations
        if cleaned["completion"] != original_completion:
            if "Non communiqué" in cleaned["completion"] and "Non renseigné" in original_completion:
                stats["non_renseigne_enriched"] += 1
            if "[Source :" in cleaned["completion"] and "[Source :" not in original_completion:
                stats["source_annotated"] += 1
            if "[Territoire :" in cleaned["completion"] and "[Territoire :" not in original_completion:
                stats["territory_annotated"] += 1

        cleaned_pairs.append(cleaned)

    # Écrire le corpus nettoyé
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in cleaned_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Rapport
    log("=" * 60)
    log("RÉSUMÉ NETTOYAGE")
    log("=" * 60)
    log(f"  Paires corpus existant  : {stats['loaded_existing']}")
    log(f"  Paires corpus enrichi   : {stats['loaded_enriched']}")
    log(f"  Total avant nettoyage   : {len(all_pairs)}")
    log(f"  'Non renseigné' enrichi : {stats['non_renseigne_enriched']}")
    log(f"  Sources annotées        : {stats['source_annotated']}")
    log(f"  Territoires annotés     : {stats['territory_annotated']}")
    log(f"  Paires vides supprimées : {stats['empty_removed']}")
    log(f"  Total après nettoyage   : {len(cleaned_pairs)}")
    log(f"Sortie : {OUTPUT_FILE}")

    # Hash
    h = hashlib.sha256()
    h.update(OUTPUT_FILE.read_bytes())
    log(f"Hash SHA-256 : {h.hexdigest()[:32]}...")

    return len(cleaned_pairs) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
