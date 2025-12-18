"""
01 - Collecte des données réelles (RNE, DECP, Délibérations, Budgets)
=====================================================================

Télécharge les données publiques depuis data.gouv.fr et data.economie.gouv.fr
pour les 9 départements du Sud de la France.

Sources :
  - DECP : API data.economie.gouv.fr (marchés publics)
  - RNE  : data.gouv.fr (élus municipaux)
  - Délibérations : data.gouv.fr (SCDL)
  - Budgets : data.economie.gouv.fr (balances comptables)

Sortie : data/raw/enrichment/elus_municipaux_sud_france.csv
         data/raw/enrichment/decp_marches_sud_france.jsonl
         data/raw/enrichment/deliberations_sud_france.jsonl
         data/raw/enrichment/budgets_sud_france.jsonl
"""

import requests
import json
import csv
import time
import hashlib
from pathlib import Path
from datetime import datetime

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "enrichment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration - 9 départements Sud France
DEPARTEMENTS = {
    "11": "Aude",
    "13": "Bouches-du-Rhône",
    "30": "Gard",
    "31": "Haute-Garonne",
    "33": "Gironde",
    "34": "Hérault",
    "66": "Pyrénées-Orientales",
    "69": "Rhône",
    "81": "Tarn",
}

# Seeds pour reproductibilité
import random
import numpy as np
random.seed(42)
np.random.seed(42)


def log(msg, level="INFO"):
    """Logging horodaté."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# ============================================================
# 1. RNE - Répertoire National des Élus
# ============================================================

def collect_rne():
    """
    Collecte les élus municipaux des 9 départements depuis data.gouv.fr.
    Source : https://www.data.gouv.fr/fr/datasets/repertoire-national-des-elus-1/
    Sortie : CSV (nom, prénom, sexe, date_naissance, code_commune, libelle_commune,
             code_departement, libelle_departement, libelle_fonction)
    """
    log("=== Collecte RNE - Élus municipaux ===")

    # URL du fichier CSV des conseillers municipaux (mis à jour régulièrement)
    url = "https://www.data.gouv.fr/fr/datasets/r/d5f400de-ae3f-4966-8cb6-a85c70c6c24a"

    output_file = DATA_DIR / "elus_municipaux_sud_france.csv"

    try:
        log(f"Téléchargement RNE depuis data.gouv.fr...")
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        # Lire le CSV complet et filtrer par département
        content = response.content.decode("utf-8", errors="replace")
        lines = content.splitlines()

        if not lines:
            log("Fichier RNE vide", "ERROR")
            return False

        reader = csv.reader(lines, delimiter=";")
        header = next(reader)

        # Trouver les indices des colonnes utiles
        # Le CSV RNE a des colonnes comme :
        # Code du département, Libellé du département, Code de la commune, ...
        col_indices = {}
        for i, col in enumerate(header):
            col_lower = col.strip().lower()
            if "code" in col_lower and "département" in col_lower:
                col_indices["code_dept"] = i
            elif "libellé" in col_lower and "département" in col_lower:
                col_indices["lib_dept"] = i
            elif "code" in col_lower and "commune" in col_lower:
                col_indices["code_commune"] = i
            elif "libellé" in col_lower and "commune" in col_lower:
                col_indices["lib_commune"] = i
            elif "nom" in col_lower and "élu" in col_lower:
                col_indices["nom"] = i
            elif "prénom" in col_lower and "élu" in col_lower:
                col_indices["prenom"] = i
            elif "sexe" in col_lower:
                col_indices["sexe"] = i
            elif "date" in col_lower and "naissance" in col_lower:
                col_indices["date_naissance"] = i
            elif "libellé" in col_lower and "fonction" in col_lower:
                col_indices["fonction"] = i

        log(f"Colonnes RNE détectées : {list(col_indices.keys())}")

        # Filtrer par département
        dept_codes = set(DEPARTEMENTS.keys())
        elus = []

        for row in reader:
            try:
                code_dept_idx = col_indices.get("code_dept")
                if code_dept_idx is not None and code_dept_idx < len(row):
                    code_dept = row[code_dept_idx].strip()
                    if code_dept in dept_codes:
                        elu = {
                            "nom": row[col_indices["nom"]].strip() if "nom" in col_indices and col_indices["nom"] < len(row) else "",
                            "prenom": row[col_indices["prenom"]].strip() if "prenom" in col_indices and col_indices["prenom"] < len(row) else "",
                            "sexe": row[col_indices["sexe"]].strip() if "sexe" in col_indices and col_indices["sexe"] < len(row) else "",
                            "date_naissance": row[col_indices["date_naissance"]].strip() if "date_naissance" in col_indices and col_indices["date_naissance"] < len(row) else "",
                            "code_commune": row[col_indices["code_commune"]].strip() if "code_commune" in col_indices and col_indices["code_commune"] < len(row) else "",
                            "libelle_commune": row[col_indices["lib_commune"]].strip() if "lib_commune" in col_indices and col_indices["lib_commune"] < len(row) else "",
                            "code_departement": code_dept,
                            "libelle_departement": DEPARTEMENTS.get(code_dept, ""),
                            "libelle_fonction": row[col_indices["fonction"]].strip() if "fonction" in col_indices and col_indices["fonction"] < len(row) else "",
                        }
                        elus.append(elu)
            except (IndexError, KeyError):
                continue

        # Écrire le CSV filtré
        if elus:
            fieldnames = ["nom", "prenom", "sexe", "date_naissance", "code_commune",
                          "libelle_commune", "code_departement", "libelle_departement",
                          "libelle_fonction"]
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(elus)

            log(f"RNE : {len(elus)} élus extraits pour {len(dept_codes)} départements")

            # Stats par département
            for code, nom in sorted(DEPARTEMENTS.items()):
                count = sum(1 for e in elus if e["code_departement"] == code)
                log(f"  {nom} ({code}) : {count} élus")
        else:
            log("Aucun élu extrait - le format du CSV a peut-être changé", "WARNING")
            # Créer un fichier vide avec header pour que le pipeline continue
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                f.write("nom,prenom,sexe,date_naissance,code_commune,libelle_commune,"
                        "code_departement,libelle_departement,libelle_fonction\n")

        return True

    except requests.RequestException as e:
        log(f"Erreur téléchargement RNE : {e}", "ERROR")
        # Créer fichier vide pour ne pas bloquer le pipeline
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("nom,prenom,sexe,date_naissance,code_commune,libelle_commune,"
                    "code_departement,libelle_departement,libelle_fonction\n")
        return False


# ============================================================
# 2. DECP - Données Essentielles de la Commande Publique
# ============================================================

def collect_decp():
    """
    Collecte les marchés publics DECP pour les 9 départements Sud France.

    Stratégie en cascade (du plus précis au plus léger) :
      1. Si decp_sud_france.jsonl existe (fichier filtré issu du DECP complet 900 MB) → utiliser directement
      2. Si decp_complet.json existe (DECP complet ~900 MB) → filtrer localement
      3. Sinon → télécharger les fichiers JSON mensuels via data.gouv.fr (4 fichiers ~12 MB)

    Source complète : https://www.data.gouv.fr/fr/datasets/r/16962018-5c31-4296-9454-5998585496d2
    Source mensuelle : https://www.data.gouv.fr/fr/datasets/68caf6b135f19236a4f37a32/
    """
    log("=== Collecte DECP - Marchés publics ===")

    output_file = DATA_DIR / "decp_marches_sud_france.jsonl"
    decp_filtered_cache = DATA_DIR / "decp_sud_france.jsonl"        # 48 MB, jan. 2026
    decp_complet_cache  = DATA_DIR / "decp_complet.json"             # ~900 MB, jan. 2026
    dept_codes = set(DEPARTEMENTS.keys())

    # ----------------------------------------------------------------
    # Stratégie 1 : fichier filtré déjà présent (meilleure source)
    # ----------------------------------------------------------------
    if decp_filtered_cache.exists() and decp_filtered_cache.stat().st_size > 1_000_000:
        log(f"  Fichier filtré DECP trouvé : {decp_filtered_cache} "
            f"({decp_filtered_cache.stat().st_size / 1024**2:.1f} MB)")
        log("  Copie vers la sortie pipeline...")

        all_marches = []
        with open(decp_filtered_cache, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Normaliser les champs vers le format pipeline courant
                    # (le cache peut avoir des clés différentes selon la version)
                    lieu = rec.get("lieuExecution", {})
                    code_postal = str(lieu.get("code", "")) if isinstance(lieu, dict) else str(rec.get("code_postal", ""))
                    dept = next((c for c in dept_codes if code_postal.startswith(c)), rec.get("departement", ""))
                    if not dept:
                        continue
                    acheteur_raw = rec.get("acheteur", {})
                    acheteur_nom = (
                        acheteur_raw.get("nom", acheteur_raw.get("acheteur", "Non renseigné"))
                        if isinstance(acheteur_raw, dict)
                        else str(acheteur_raw or rec.get("acheteur", "Non renseigné"))
                    )
                    marche = {
                        "acheteur": acheteur_nom,
                        "objet": str(rec.get("objet", "")),
                        "montant": rec.get("montant", 0),
                        "procedure": str(rec.get("procedure", "")),
                        "date_notification": str(rec.get("dateNotification", rec.get("date_notification", ""))),
                        "code_postal": code_postal,
                        "titulaire": str(rec.get("titulaire", "")),
                        "id_marche": str(rec.get("id", rec.get("id_marche", ""))),
                        "nature": str(rec.get("nature", "Marché")),
                        "departement": dept,
                        "departement_nom": DEPARTEMENTS.get(dept, ""),
                    }
                    all_marches.append(marche)
                except Exception:
                    continue

        with open(output_file, "w", encoding="utf-8") as f:
            for m in all_marches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        log(f"DECP (cache filtré) : {len(all_marches)} marchés")
        return len(all_marches) > 0

    # ----------------------------------------------------------------
    # Stratégie 2 : fichier DECP complet (~900 MB) déjà téléchargé
    # ----------------------------------------------------------------
    if decp_complet_cache.exists() and decp_complet_cache.stat().st_size > 100_000_000:
        log(f"  Fichier DECP complet trouvé : {decp_complet_cache} "
            f"({decp_complet_cache.stat().st_size / 1024**2:.0f} MB) — filtrage en cours…")
        try:
            with open(decp_complet_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "marches" in data:
                raw_list = data["marches"]
                if isinstance(raw_list, dict):
                    raw_list = raw_list.get("marche", [])
            elif isinstance(data, list):
                raw_list = data
            else:
                raw_list = []

            all_marches = []
            for rec in raw_list:
                lieu = rec.get("lieuExecution", {})
                code_postal = str(lieu.get("code", "")) if isinstance(lieu, dict) else ""
                if not code_postal:
                    acheteur_cp = rec.get("acheteur", {})
                    code_postal = str(acheteur_cp.get("codePostal", "")) if isinstance(acheteur_cp, dict) else ""
                dept = next((c for c in dept_codes if code_postal.startswith(c)), None)
                if not dept:
                    continue
                acheteur_raw = rec.get("acheteur", {})
                acheteur_nom = (
                    acheteur_raw.get("nom", "Non renseigné")
                    if isinstance(acheteur_raw, dict)
                    else str(acheteur_raw or "Non renseigné")
                )
                titulaires = rec.get("titulaires", [])
                titulaire_nom = (
                    titulaires[0].get("denominationSociale", "")
                    if titulaires and isinstance(titulaires, list)
                    else ""
                )
                all_marches.append({
                    "acheteur": acheteur_nom,
                    "objet": str(rec.get("objet", "")),
                    "montant": rec.get("montant", 0),
                    "procedure": str(rec.get("procedure", "")),
                    "date_notification": str(rec.get("dateNotification", "")),
                    "code_postal": code_postal,
                    "titulaire": titulaire_nom,
                    "id_marche": str(rec.get("id", "")),
                    "nature": str(rec.get("nature", "Marché")),
                    "departement": dept,
                    "departement_nom": DEPARTEMENTS.get(dept, ""),
                })

            with open(output_file, "w", encoding="utf-8") as f:
                for m in all_marches:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            # Sauvegarder aussi le cache filtré pour les prochains runs
            with open(decp_filtered_cache, "w", encoding="utf-8") as f:
                for m in all_marches:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            log(f"DECP (complet filtré) : {len(all_marches)} marchés")
            return len(all_marches) > 0
        except Exception as e:
            log(f"  Erreur lecture decp_complet.json : {e} — repli sur fichiers mensuels", "WARNING")

    # ----------------------------------------------------------------
    # Stratégie 3 : fichiers JSON mensuels data.gouv.fr (fallback)
    # ----------------------------------------------------------------
    log("  Téléchargement des fichiers mensuels DECP (data.gouv.fr)…")
    DATASET_ID = "68caf6b135f19236a4f37a32"

    try:
        resp = requests.get(
            f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/",
            timeout=15,
        )
        resp.raise_for_status()
        resources = resp.json().get("resources", [])
        json_resources = [
            r for r in resources
            if r.get("format", "").lower() == "json"
            and any(y in r.get("title", "") for y in ["2024", "2025"])
        ]
        json_resources = sorted(json_resources, key=lambda r: r.get("title", ""), reverse=True)[:4]
        log(f"  Ressources sélectionnées : {[r.get('title') for r in json_resources]}")
    except requests.RequestException as e:
        log(f"Erreur récupération liste DECP : {e}", "ERROR")
        open(output_file, "w").close()
        return False

    all_marches = []
    for resource in json_resources:
        url = resource.get("url")
        title = resource.get("title", url)
        try:
            log(f"  Téléchargement {title}…")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            data = r.json()
            marches_data = data.get("marches", {})
            if isinstance(marches_data, dict):
                marches_list = marches_data.get("marche", [])
            elif isinstance(marches_data, list):
                marches_list = marches_data
            else:
                marches_list = []

            count = 0
            for rec in marches_list:
                lieu = rec.get("lieuExecution", {})
                code_postal = str(lieu.get("code", "")) if isinstance(lieu, dict) else ""
                dept = next((c for c in dept_codes if code_postal.startswith(c)), None)
                if not dept:
                    continue
                acheteur_raw = rec.get("acheteur", {})
                acheteur_nom = (
                    acheteur_raw.get("nom", "Non renseigné")
                    if isinstance(acheteur_raw, dict)
                    else str(acheteur_raw or "Non renseigné")
                )
                titulaires = rec.get("titulaires", [])
                titulaire_nom = (
                    titulaires[0].get("denominationSociale", "")
                    if titulaires and isinstance(titulaires, list)
                    else ""
                )
                all_marches.append({
                    "acheteur": acheteur_nom,
                    "objet": str(rec.get("objet", "")),
                    "montant": rec.get("montant", 0),
                    "procedure": str(rec.get("procedure", "")),
                    "date_notification": str(rec.get("dateNotification", "")),
                    "code_postal": code_postal,
                    "titulaire": titulaire_nom,
                    "id_marche": str(rec.get("id", "")),
                    "nature": str(rec.get("nature", "Marché")),
                    "departement": dept,
                    "departement_nom": DEPARTEMENTS.get(dept, ""),
                })
                count += 1
            log(f"    → {count} marchés Sud France (sur {len(marches_list)} total)")
            time.sleep(0.5)
        except Exception as e:
            log(f"  Erreur {title} : {e}", "WARNING")

    with open(output_file, "w", encoding="utf-8") as f:
        for marche in all_marches:
            f.write(json.dumps(marche, ensure_ascii=False) + "\n")
    log(f"DECP (mensuels) : {len(all_marches)} marchés extraits")
    return len(all_marches) > 0


# ============================================================
# 3. Délibérations municipales
# ============================================================

def collect_deliberations():
    """
    Collecte les délibérations municipales depuis data.gouv.fr (SCDL).
    Sortie : JSONL
    """
    log("=== Collecte Délibérations municipales ===")

    search_url = "https://www.data.gouv.fr/api/1/datasets/"
    output_file = DATA_DIR / "deliberations_sud_france.jsonl"

    all_delibs = []

    for code_dept, nom_dept in DEPARTEMENTS.items():
        params = {"q": f"délibérations {nom_dept}", "page_size": 5}
        try:
            resp = requests.get(search_url, params=params, timeout=30)
            resp.raise_for_status()
            datasets = resp.json().get("data", [])

            dept_count = 0
            for ds in datasets[:2]:
                for res in ds.get("resources", [])[:1]:
                    fmt = res.get("format", "").lower()
                    if fmt in ("csv", "json"):
                        try:
                            r = requests.get(res["url"], timeout=60)
                            r.raise_for_status()

                            if fmt == "json":
                                items = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
                                for item in items[:500]:
                                    delib = {
                                        "type": item.get("DELIB_MATIERE_NOM", item.get("type", "")),
                                        "date": item.get("DELIB_DATE", item.get("date", "")),
                                        "objet": item.get("DELIB_OBJET", item.get("objet", "")),
                                        "collectivite": item.get("COLL_NOM", nom_dept),
                                        "departement": code_dept,
                                    }
                                    all_delibs.append(delib)
                                    dept_count += 1

                        except Exception:
                            pass

            log(f"  {nom_dept} : {dept_count} délibérations")
            time.sleep(0.5)

        except requests.RequestException as e:
            log(f"  Erreur délibérations ({nom_dept}) : {e}", "WARNING")

    with open(output_file, "w", encoding="utf-8") as f:
        for d in all_delibs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    log(f"Délibérations : {len(all_delibs)} extraites au total")
    return True


# ============================================================
# 4. Budgets publics
# ============================================================

def collect_budgets():
    """
    Collecte les budgets communaux depuis data.economie.gouv.fr.
    Sortie : JSONL
    """
    log("=== Collecte Budgets publics ===")

    base_url = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/balances-comptables-des-communes-en-2023/records"
    output_file = DATA_DIR / "budgets_sud_france.jsonl"

    all_budgets = []

    # Grandes villes par département
    villes_ref = {
        "11": ["CARCASSONNE", "NARBONNE"],
        "13": ["MARSEILLE", "AIX-EN-PROVENCE"],
        "30": ["NIMES", "ALES"],
        "31": ["TOULOUSE"],
        "33": ["BORDEAUX"],
        "34": ["MONTPELLIER", "BEZIERS"],
        "66": ["PERPIGNAN"],
        "69": ["LYON", "VILLEURBANNE"],
        "81": ["ALBI", "CASTRES"],
    }

    for code_dept, villes in villes_ref.items():
        for ville in villes:
            try:
                params = {"where": f'lbudg LIKE "{ville}%"', "limit": 20}
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                records = resp.json().get("results", [])

                for rec in records:
                    budget = {
                        "commune": rec.get("lbudg", ville),
                        "compte": rec.get("compte", ""),
                        "libelle": rec.get("lcompte", ""),
                        "sd": rec.get("sd", 0),
                        "sc": rec.get("sc", 0),
                        "departement": code_dept,
                    }
                    all_budgets.append(budget)

                log(f"  {ville} : {len(records)} lignes budgétaires")
                time.sleep(0.3)

            except requests.RequestException as e:
                log(f"  Erreur budget ({ville}) : {e}", "WARNING")

    with open(output_file, "w", encoding="utf-8") as f:
        for b in all_budgets:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    log(f"Budgets : {len(all_budgets)} lignes extraites au total")
    return True


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 60)
    log("ÉTAPE 1 : COLLECTE DES DONNÉES RÉELLES")
    log("=" * 60)
    log(f"Départements cibles : {', '.join(f'{v} ({k})' for k, v in sorted(DEPARTEMENTS.items()))}")
    log(f"Sortie : {DATA_DIR}")

    results = {}

    results["rne"] = collect_rne()
    results["decp"] = collect_decp()
    results["deliberations"] = collect_deliberations()
    results["budgets"] = collect_budgets()

    # Résumé
    log("=" * 60)
    log("RÉSUMÉ COLLECTE")
    log("=" * 60)

    for source, ok in results.items():
        status = "OK" if ok else "ÉCHEC"
        log(f"  {source.upper():20} : {status}")

    # Hash de traçabilité
    h = hashlib.sha256()
    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    log(f"Hash SHA-256 collecte : {h.hexdigest()[:32]}...")

    success_count = sum(1 for v in results.values() if v)
    log(f"Résultat : {success_count}/{len(results)} sources collectées")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
