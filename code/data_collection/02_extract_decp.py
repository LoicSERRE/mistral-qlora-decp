"""
Extraction DECP filtrée : Montpellier + quelques grandes villes
Objectif : ~3-5M tokens pour RTX 4070 Ti
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "decp"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

xml_file = DATA_DIR / "decp_consolide.xml"

# Départements sélectionnés (élargir pour ~10-15k marchés)
REGIONS_CIBLES = {
    "HERAULT-34": ["34", "MONTPELLIER", "HERAULT", "HÉRAULT", "BEZIERS", "BÉZIERS", "SETE", "SÈTE"],
    "HAUTE-GARONNE-31": ["31", "TOULOUSE", "HAUTE-GARONNE"],
    "GARD-30": ["30", "NIMES", "NÎMES", "GARD", "ALES", "ALÈS"],
    "TARN-81": ["81", "ALBI", "TARN", "CASTRES"],
    "BOUCHES-DU-RHONE-13": ["13", "MARSEILLE", "BOUCHES-DU-RHONE", "BOUCHES-DU-RHÔNE", "AIX-EN-PROVENCE"],
    "RHONE-69": ["69", "LYON", "RHONE", "RHÔNE", "METROPOLE DE LYON", "VILLEURBANNE"],
    "GIRONDE-33": ["33", "BORDEAUX", "GIRONDE"],
    "PYRENEES-ORIENTALES-66": ["66", "PERPIGNAN", "PYRENEES-ORIENTALES", "PYRÉNÉES-ORIENTALES"],
    "AUDE-11": ["11", "AUDE", "CARCASSONNE", "NARBONNE"]
}

# Liste pour affichage
VILLES_CIBLES = list(REGIONS_CIBLES.keys())

print("="*70)
print("EXTRACTION MARCHES PUBLICS - DEPARTEMENTS SELECTIONNES")
print("="*70)
print(f"\nFocus : 9 departements (Herault, Haute-Garonne, Gard, Tarn,")
print(f"        Bouches-du-Rhone, Rhone, Gironde, Pyrenees-Orientales, Aude)")
print(f"Periode : 2023-2025")
print(f"\n Extraction en cours...\n")

marches_extraits = []
stats = {ville: 0 for ville in VILLES_CIBLES}
stats["AUTRES"] = 0
total_analyse = 0
total_2023plus = 0

try:
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'marche':
            total_analyse += 1
            
            # Extraire données basiques
            marche_data = {}
            for child in elem:
                if child.text and child.tag in ['objet', 'montant', 'nature', 'dureeMois', 
                                                  'datePublicationDonnees', 'dateNotification',
                                                  'procedure', 'codeCPV']:
                    marche_data[child.tag] = child.text.strip()
            
            # Extraire acheteur
            acheteur_elem = elem.find('acheteur')
            if acheteur_elem is not None:
                denom = acheteur_elem.find('denominationSociale')
                if denom is not None and denom.text:
                    marche_data['acheteur'] = denom.text.strip()
            
            # Filtre période (2023-2025)
            date_str = marche_data.get('datePublicationDonnees', marche_data.get('dateNotification', ''))
            annee = date_str[:4] if date_str else ''
            
            if annee not in ['2023', '2024', '2025']:
                elem.clear()
                continue
            
            total_2023plus += 1
            
            # Filtre région/département (recherche plus large)
            acheteur = marche_data.get('acheteur', '').upper()
            objet = marche_data.get('objet', '').upper()
            ville_trouvee = None
            
            # Chercher dans acheteur ET objet avec mots-clés multiples
            texte_complet = acheteur + " " + objet
            
            for ville, keywords in REGIONS_CIBLES.items():
                for keyword in keywords:
                    if keyword in texte_complet:
                        ville_trouvee = ville
                        break
                if ville_trouvee:
                    break
            
            if ville_trouvee:
                stats[ville_trouvee] += 1
                
                # Formater pour extraction
                marche_extrait = {
                    'departement': ville_trouvee,
                    'acheteur': marche_data.get('acheteur', 'Non renseigné'),
                    'objet': marche_data.get('objet', ''),
                    'montant': marche_data.get('montant', '0'),
                    'nature': marche_data.get('nature', 'Marché'),
                    'duree': marche_data.get('dureeMois', ''),
                    'procedure': marche_data.get('procedure', ''),
                    'cpv': marche_data.get('codeCPV', ''),
                    'date': date_str[:10] if date_str else ''
                }
                
                marches_extraits.append(marche_extrait)
            
            # Progression
            if total_analyse % 50000 == 0:
                print(f"   {total_analyse:,} marchés analysés | {len(marches_extraits):,} extraits", end='\r')
            
            elem.clear()
    
    print(f"\n Analyse terminée : {total_analyse:,} marchés")

except Exception as e:
    print(f"\n Erreur : {e}")
    import traceback
    traceback.print_exc()

# RÉSULTATS
print("\n" + "="*70)
print("RÉSULTATS EXTRACTION")
print("="*70)

print(f"\n STATISTIQUES")
print(f"   Total analysé       : {total_analyse:,} marchés")
print(f"   Période 2023-2025   : {total_2023plus:,} marchés ({total_2023plus/total_analyse*100:.1f}%)")
print(f"   Extraits (filtrés)  : {len(marches_extraits):,} marchés")

print(f"\n  RÉPARTITION PAR DÉPARTEMENT")
for dept in REGIONS_CIBLES.keys():
    count = stats[dept]
    pct = count/len(marches_extraits)*100 if marches_extraits else 0
    print(f"   {dept:25} : {count:6,} marchés ({pct:5.1f}%)")

print(f"\n VOLUMÉTRIE ESTIMÉE")
tokens_estimate = len(marches_extraits) * 200  # ~200 tokens/marché
tokens_mb = tokens_estimate / 1_000_000
print(f"   Marchés extraits    : {len(marches_extraits):,}")
print(f"   Tokens estimés      : {tokens_mb:.2f}M")
print(f"   Cible RTX 4070 Ti   : 5-10M tokens")
if tokens_mb < 5:
    print(f"    BON : Volumétrie adaptée (avec autres datasets)")
elif tokens_mb <= 10:
    print(f"    PARFAIT : Dans la cible optimale")
else:
    print(f"     TROP : Réduire les villes")

# Sauvegarder en JSONL
output_file = OUTPUT_DIR / "decp_filtered.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for marche in marches_extraits:
        f.write(json.dumps(marche, ensure_ascii=False) + '\n')

print(f"\n Données sauvegardées : {output_file}")
print(f"   Taille : {output_file.stat().st_size / (1024*1024):.1f} MB")

# Échantillon
print(f"\n ÉCHANTILLON (3 premiers marchés Hérault-34)")
herault_marches = [m for m in marches_extraits if m['departement'] == 'HERAULT-34'][:3]
for i, m in enumerate(herault_marches, 1):
    print(f"\n{i}. {m['objet'][:80]}...")
    print(f"   Acheteur : {m['acheteur'][:50]}")
    print(f"   Montant  : {m['montant']}€")
    print(f"   Date     : {m['date']}")

# Métadonnées
metadata = {
    "source": "DECP (data.gouv.fr)",
    "extraction_date": datetime.now().isoformat(),
    "filtres": {
        "regions": REGIONS_CIBLES,
        "periode": "2023-2025"
    },
    "statistiques": {
        "total_marches_xml": total_analyse,
        "marches_2023plus": total_2023plus,
        "marches_extraits": len(marches_extraits),
        "repartition_villes": stats,
        "tokens_estimes": tokens_mb
    }
}

metadata_file = OUTPUT_DIR / "decp_filtered_metadata.json"
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n Métadonnées : {metadata_file}")

print("\n" + "="*70)
print(" EXTRACTION TERMINÉE")
print("="*70)
print(f"\nPrêt pour l'étape suivante : formatage en Q/A pour fine-tuning")
