"""
06 - Génération Questions Variées sur Données Réelles
Crée 10-15 templates de questions différents sur VRAIES données publiques
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
import re
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ENRICHMENT_DIR = RAW_DIR / "enrichment"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "fine_tuning"

print("="*80)
print("GÉNÉRATION QUESTIONS VARIÉES SUR DONNÉES RÉELLES")
print("="*80)
print(f"\nObjectif : 10-15 templates questions × vraies données = corpus enrichi")
print(f"Sources : DECP, Élus, Délibérations (TOUT RÉEL)\n")


# ==================== CHARGEMENT DONNÉES RÉELLES ====================

def load_decp_complete():
    """Charge DECP filtré existant (Sud France)"""
    
    print("📂 Chargement DECP filtré Sud France...")
    
    # Essayer d'abord le fichier généré par 01_collect_real_data.py
    decp_file_new = ENRICHMENT_DIR / "decp_sud_france.jsonl"
    decp_file_old = PROCESSED_DIR / "decp_filtered.jsonl"
    
    # Choisir le fichier qui existe
    if decp_file_new.exists():
        decp_file = decp_file_new
    elif decp_file_old.exists():
        decp_file = decp_file_old
    else:
        print(f"   ⚠️ Fichier DECP non trouvé")
        print(f"   Cherché dans : {decp_file_new}")
        print(f"   Cherché dans : {decp_file_old}")
        return []
    
    marches = []
    
    with open(decp_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                marches.append(data)
            except:
                continue
            
            if (i+1) % 1000 == 0:
                print(f"   Progression : {i+1:,} marchés chargés...", end='\r')
    
    print(f"\n✅ {len(marches):,} marchés DECP chargés depuis {decp_file.name}")
    return marches


def load_existing_decp():
    """Compatibilité - redirige vers load_decp_complete"""
    return load_decp_complete()


def load_elus():
    """Charge élus municipaux réels"""
    
    print("\n📂 Chargement élus municipaux...")
    elus_file = ENRICHMENT_DIR / "elus_municipaux_sud_france.csv"
    
    if not elus_file.exists():
        print("   ⚠️ Fichier élus non trouvé")
        return []
    
    df = pd.read_csv(elus_file)
    print(f"✅ {len(df):,} élus chargés")
    
    return df.to_dict('records')


def load_deliberations():
    """Charge délibérations depuis training data"""
    
    print("\n📂 Chargement délibérations...")
    delib_file = OUTPUT_DIR / "training_data_deliberations.jsonl"
    
    if not delib_file.exists():
        print("   ⚠️ Fichier délibérations non trouvé")
        return []
    
    deliberations = []
    
    with open(delib_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extraire contexte depuis prompt
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                
                # Parser pour retrouver structure
                deliberations.append({
                    'titre': prompt,
                    'collectivite': 'Non extrait',
                    'date_seance': '',
                    'type': data.get('source', 'DELIBERATIONS')
                })
            except:
                continue
    
    print(f"✅ {len(deliberations):,} délibérations chargées")
    return deliberations


# ==================== TEMPLATES QUESTIONS VARIÉES (10-15 types) ====================

class VariedQuestionGenerator:
    """Génère 10-15 types de questions différents sur vraies données"""
    
    def __init__(self, marches_data, elus_data, deliberations_data):
        self.marches = marches_data
        self.elus = elus_data
        self.deliberations = deliberations_data
        self.qa_pairs = []
        
        # Index par ville pour questions croisées
        self.index_by_city()
    
    def index_by_city(self):
        """Index données par département pour questions contextuelles"""
        
        self.marches_by_dept = defaultdict(list)
        self.elus_by_city = defaultdict(list)
        self.delib_by_city = defaultdict(list)
        
        # Index marchés par département (lieuExecution.code)
        for marche in self.marches:
            lieu_exec = marche.get('lieuExecution', {})
            if isinstance(lieu_exec, dict):
                dept_code = lieu_exec.get('code', '')
                if dept_code:
                    self.marches_by_dept[dept_code].append(marche)
        
        # Index élus
        for elu in self.elus:
            ville = elu.get('libelle_commune', '')
            if ville:
                self.elus_by_city[ville].append(elu)
        
        # Index délibérations (pas de ville extraite facilement)
        for delib in self.deliberations:
            titre = delib.get('titre', '')
            # Tentative extraction ville du titre
            if 'Toulouse' in titre:
                self.delib_by_city['Toulouse'].append(delib)
            elif 'Montpellier' in titre:
                self.delib_by_city['Montpellier'].append(delib)
        
        print(f"\n📊 Index créés :")
        print(f"   • {len(self.marches_by_dept)} départements avec marchés")
        print(f"   • {len(self.elus_by_city)} communes avec élus")
        print(f"   • {len(self.delib_by_city)} collectivités avec délibérations")
    
    # ==================== TYPE 1 : Questions Factuelles Détaillées ====================
    
    def generate_detailed_marche_questions(self, n=3000):
        """Questions détaillées sur marchés (au-delà des 3 templates basiques)"""
        
        questions = []
        
        for marche in random.sample(self.marches, min(n, len(self.marches))):
            objet = marche.get('objet', '')
            montant = marche.get('montant', '')
            acheteur = marche.get('acheteur', '')
            date = marche.get('date', '')
            procedure = marche.get('procedure', marche.get('nature', ''))
            
            if not objet:
                continue
            
            # Question 1 : Procédure utilisée
            if procedure:
                questions.append({
                    'prompt': f"Quelle procédure a été utilisée pour le marché '{objet[:80]}...' ?",
                    'completion': f"Le marché '{objet}' a utilisé la procédure : {procedure}.\n\n"
                                f"Acheteur : {acheteur or 'Non renseigné'}\n"
                                f"Montant : {montant}€ HT\n"
                                f"Date notification : {date}\n\n"
                                f"[Source : DECP données réelles]",
                    'source': 'DECP_PROCEDURE'
                })
            
            # Question 2 : Acheteur du marché
            if acheteur:
                questions.append({
                    'prompt': f"Quelle collectivité a lancé le marché '{objet[:80]}...' ?",
                    'completion': f"Le marché '{objet}' a été lancé par : {acheteur}.\n\n"
                                f"Montant : {montant}€ HT\n"
                                f"Procédure : {procedure or 'Non renseignée'}\n"
                                f"Date : {date}\n\n"
                                f"[Source : DECP données réelles]",
                    'source': 'DECP_ACHETEUR'
                })
            
            # Question 3 : Date notification
            if date:
                questions.append({
                    'prompt': f"Quand a été notifié le marché '{objet[:80]}...' ?",
                    'completion': f"Le marché '{objet}' a été notifié le : {date}.\n\n"
                                f"Acheteur : {acheteur or 'Non renseigné'}\n"
                                f"Montant : {montant}€ HT\n"
                                f"Procédure : {procedure or 'Non renseignée'}\n\n"
                                f"[Source : DECP données réelles]",
                    'source': 'DECP_DATE'
                })
        
        return questions[:n]
    
    # ==================== TYPE 2 : Questions Élus Contextuelles ====================
    
    def generate_elus_questions(self, n=1500):
        """Questions sur élus réels"""
        
        questions = []
        
        # Maires par commune
        maires = [e for e in self.elus if e.get('libelle_fonction') == 'Maire']
        
        for maire in random.sample(maires, min(n//3, len(maires))):
            nom = f"{maire.get('prenom', '')} {maire.get('nom', '')}".strip()
            commune = maire.get('libelle_commune', '')
            dept = maire.get('code_departement', '')
            
            if nom and commune:
                # Q1: Qui est le maire ?
                questions.append({
                    'prompt': f"Qui est le maire de {commune} ({dept}) ?",
                    'completion': f"Le maire de {commune} ({dept}) est {nom}.\n\n"
                                f"[Source : Répertoire National des Élus - Données réelles]",
                    'source': 'ELUS_MAIRE'
                })
                
                # Q2: Dans quelle commune est maire X ?
                questions.append({
                    'prompt': f"Dans quelle commune {nom} est-il maire ?",
                    'completion': f"{nom} est maire de {commune} ({dept}).\n\n"
                                f"[Source : Répertoire National des Élus - Données réelles]",
                    'source': 'ELUS_COMMUNE'
                })
        
        # Conseillers municipaux
        conseillers = [e for e in self.elus if e.get('libelle_fonction') != 'Maire']
        
        # Compter conseillers par commune
        conseillers_by_commune = defaultdict(list)
        for c in conseillers:
            commune = c.get('libelle_commune', '')
            if commune:
                conseillers_by_commune[commune].append(c)
        
        for commune, liste_conseillers in list(conseillers_by_commune.items())[:n//3]:
            dept = liste_conseillers[0].get('code_departement', '')
            nb_conseillers = len(liste_conseillers)
            
            questions.append({
                'prompt': f"Combien de conseillers municipaux à {commune} ({dept}) ?",
                'completion': f"La commune de {commune} ({dept}) compte {nb_conseillers} conseillers municipaux.\n\n"
                            f"[Source : Répertoire National des Élus - Données réelles]",
                'source': 'ELUS_CONSEILLERS'
            })
        
        return questions[:n]
    
    # ==================== TYPE 3 : Questions Comparatives Territoriales ====================
    
    def generate_comparative_questions(self, n=1200):
        """Comparaisons réelles entre départements"""
        
        questions = []
        
        # Mapping codes départements -> noms
        dept_names = {
            '11': 'Aude', '12': 'Aveyron', '30': 'Gard', '31': 'Haute-Garonne',
            '34': 'Hérault', '48': 'Lozère', '66': 'Pyrénées-Orientales', '81': 'Tarn', '09': 'Ariège'
        }
        
        # Calculer stats par département
        stats_depts = {}
        for dept_code, marches_dept in self.marches_by_dept.items():
            if dept_code in dept_names:
                montants = [self.extract_montant(m) for m in marches_dept]
                montants = [m for m in montants if m > 0]
                
                stats_depts[dept_code] = {
                    'nom': dept_names[dept_code],
                    'nb_marches': len(marches_dept),
                    'montant_total': sum(montants) if montants else 0,
                    'montant_moyen': sum(montants) / len(montants) if montants else 0
                }
        
        # Générer comparaisons entre tous les départements
        dept_codes = list(stats_depts.keys())
        for i, dept1 in enumerate(dept_codes):
            for dept2 in dept_codes[i+1:]:
                s1 = stats_depts[dept1]
                s2 = stats_depts[dept2]
                
                questions.append({
                    'prompt': f"Comparer nombre marchés publics {s1['nom']} ({dept1}) vs {s2['nom']} ({dept2}) (données corpus DECP).",
                    'completion': f"Comparaison marchés publics entre départements :\n\n"
                                f"**{s1['nom']} ({dept1})** :\n"
                                f"- Nombre marchés : {s1['nb_marches']}\n"
                                f"- Montant total : {s1['montant_total']:,.0f}€\n"
                                f"- Montant moyen : {s1['montant_moyen']:,.0f}€\n\n"
                                f"**{s2['nom']} ({dept2})** :\n"
                                f"- Nombre marchés : {s2['nb_marches']}\n"
                                f"- Montant total : {s2['montant_total']:,.0f}€\n"
                                f"- Montant moyen : {s2['montant_moyen']:,.0f}€\n\n"
                                f"**Analyse** : "
                                f"Le {s1['nom']} compte {'plus' if s1['nb_marches'] > s2['nb_marches'] else 'moins'} "
                                f"de marchés que le {s2['nom']}.\n\n"
                                f"[Source : DECP corpus Sud France - Données réelles]",
                    'source': 'COMPARATIVE_DEPTS'
                })
        
        return questions[:n]
    
    # ==================== TYPE 4 : Questions Procédurales Basées sur Seuils Réels ====================
    
    def generate_procedural_questions(self, n=2000):
        """Questions procédurales basées sur montants réels du corpus"""
        
        questions = []
        
        # Seuils 2024-2025
        seuils = {
            'MAPA_sans_pub': 40000,
            'MAPA_avec_pub': 90000,
            'seuil_EU_fournitures': 140000,
            'seuil_EU_travaux': 215000,
        }
        
        # Extraire marchés autour des seuils
        for marche in random.sample(self.marches, min(n*2, len(self.marches))):
            montant = self.extract_montant(marche)
            objet = marche.get('objet', '')
            type_marche = self.infer_type_marche(objet) if objet else 'fournitures'
            
            if montant <= 0 or not objet:
                continue
            
            # Question : Quelle procédure pour ce montant ?
            procedure_requise = self.determine_procedure(montant, type_marche)
            
            questions.append({
                'prompt': f"Pour un marché de {type_marche} de {montant:,.0f}€ HT, quelle procédure appliquer ?",
                'completion': f"Pour un marché de {type_marche} de {montant:,.0f}€ HT :\n\n"
                            f"**Procédure applicable** : {procedure_requise}\n\n"
                            f"**Analyse seuils 2024-2025** :\n"
                            + self.explain_seuils(montant, type_marche) +
                            f"\n\n**Exemple corpus réel** :\n"
                            f"Marché '{objet[:100]}...' ({montant:,.0f}€ HT)\n\n"
                            f"[Source : Code Commande Publique + DECP données réelles]",
                'source': 'PROCEDURAL_SEUILS'
            })
            
            if len(questions) >= n:
                break
        
        return questions[:n]
    
    # ==================== TYPE 5 : Questions Délibérations Contextuelles ====================
    
    def generate_deliberations_questions(self, n=1000):
        """Questions sur délibérations réelles"""
        
        questions = []
        
        for delib in random.sample(self.deliberations, min(n, len(self.deliberations))):
            titre = self.get_field(delib, 'titre', 'objet', 'intitule')
            collectivite = self.get_field(delib, 'collectivite', 'nom_collectivite')
            date = self.get_field(delib, 'date_seance', 'date')
            type_delib = self.get_field(delib, 'type', 'theme', 'categorie')
            
            if not titre or not collectivite:
                continue
            
            # Q1 : Quel type de délibération ?
            questions.append({
                'prompt': f"Quel type de délibération pour '{titre[:80]}...' à {collectivite} ?",
                'completion': f"La délibération '{titre}' à {collectivite} est de type : {type_delib or 'Non catégorisé'}.\n\n"
                            f"Date séance : {date or 'Non renseignée'}\n\n"
                            f"[Source : Délibérations municipales - Données réelles]",
                'source': 'DELIBERATIONS_TYPE'
            })
            
            # Q2 : Quand votée ?
            if date:
                questions.append({
                    'prompt': f"Quand a été votée la délibération '{titre[:80]}...' à {collectivite} ?",
                    'completion': f"La délibération '{titre}' à {collectivite} a été votée le {date}.\n\n"
                                f"Type : {type_delib or 'Non catégorisé'}\n\n"
                                f"[Source : Délibérations municipales - Données réelles]",
                    'source': 'DELIBERATIONS_DATE'
                })
        
        return questions[:n]
    
    # ==================== TYPE 6 : Questions Croisées (Marchés + Élus) ====================
    
    def generate_cross_questions(self, n=500):
        """Questions croisant plusieurs sources de données"""
        
        questions = []
        
        # Pour chaque ville avec maire ET marchés
        for ville in self.marches_by_city.keys():
            if ville in self.elus_by_city:
                maires = [e for e in self.elus_by_city[ville] if e.get('libelle_fonction') == 'Maire']
                marches = self.marches_by_city[ville]
                
                if maires and marches:
                    maire = maires[0]
                    nom_maire = f"{maire.get('prenom', '')} {maire.get('nom', '')}".strip()
                    dept = maire.get('code_departement', '')
                    
                    # Statistiques marchés
                    nb_marches = len(marches)
                    montants = [self.extract_montant(m) for m in marches]
                    montants = [m for m in montants if m > 0]
                    montant_total = sum(montants)
                    
                    questions.append({
                        'prompt': f"Quel est le maire de {ville} et combien de marchés publics en cours ?",
                        'completion': f"**Maire** : {nom_maire} (maire de {ville}, {dept})\n\n"
                                    f"**Marchés publics** :\n"
                                    f"- Nombre marchés corpus : {nb_marches}\n"
                                    f"- Montant total : {montant_total:,.0f}€ HT\n"
                                    f"- Montant moyen : {montant_total/nb_marches:,.0f}€ HT\n\n"
                                    f"[Sources : RNE + DECP - Données réelles croisées]",
                        'source': 'CROSS_MAIRE_MARCHES'
                    })
        
        return questions[:n]
    
    # ==================== TYPE 7 : Refus Intelligents Hors Corpus ====================
    
    def generate_out_of_scope_refusals(self, n=500):
        """Exemples refus hors corpus pour garde-fous"""
        
        questions = []
        
        # Villes hors corpus
        villes_out = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Nice", "Strasbourg"]
        
        for ville in villes_out:
            questions.append({
                'prompt': f"Marchés publics {ville} 2024 ?",
                'completion': f"❌ **Données non disponibles pour {ville}**.\n\n"
                            f"Mon corpus couvre uniquement **9 départements Sud France** :\n"
                            f"- Hérault (34)\n"
                            f"- Haute-Garonne (31)\n"
                            f"- Gard (30)\n"
                            f"- Aude (11)\n"
                            f"- Pyrénées-Orientales (66)\n"
                            f"- Tarn (81)\n"
                            f"- Aveyron (12)\n"
                            f"- Lozère (48)\n"
                            f"- Ariège (09)\n\n"
                            f"**Pour {ville}**, consultez :\n"
                            f"- data.gouv.fr > DECP\n"
                            f"- Profil acheteur de la ville\n\n"
                            f"[Confiance : NULLE - Hors périmètre géographique]",
                'source': 'OUT_OF_SCOPE_GEO'
            })
            
            questions.append({
                'prompt': f"Qui est le maire de {ville} ?",
                'completion': f"❌ **Information non disponible pour {ville}**.\n\n"
                            f"Mon corpus élus couvre uniquement les 9 départements Sud France (période 2023-2025).\n\n"
                            f"Pour connaître le maire de {ville}, consultez :\n"
                            f"- Site officiel de la ville\n"
                            f"- data.gouv.fr > Répertoire National des Élus\n\n"
                            f"[Confiance : NULLE - Hors périmètre géographique]",
                'source': 'OUT_OF_SCOPE_ELUS'
            })
        
        # Années hors corpus
        annees_out = [2018, 2019, 2020, 2021, 2022]
        
        for annee in annees_out:
            questions.append({
                'prompt': f"Marchés publics Montpellier {annee} ?",
                'completion': f"❌ **Données temporelles limitées à 2023-2025**.\n\n"
                            f"Je ne dispose pas d'informations sur {annee}.\n\n"
                            f"Pour archives antérieures, consultez :\n"
                            f"- montpellier.fr > Marchés publics > Archives\n"
                            f"- BOAMP historique\n"
                            f"- data.gouv.fr > DECP (disponible depuis 2018)\n\n"
                            f"[Confiance : NULLE - Hors périmètre temporel]",
                'source': 'OUT_OF_SCOPE_TEMPORAL'
            })
        
        # Domaines hors scope
        domaines_out = [
            "analyse financière entreprise",
            "conseil juridique contentieux",
            "optimisation fiscale TVA",
            "droit du travail",
        ]
        
        for domaine in domaines_out:
            questions.append({
                'prompt': f"Question {domaine} marchés publics ?",
                'completion': f"❌ **Question hors périmètre spécialisation**.\n\n"
                            f"Je couvre :\n"
                            f"- Procédures marchés publics (Code Commande Publique)\n"
                            f"- Données factuelles DECP Sud France 2023-2025\n"
                            f"- Délibérations administratives\n"
                            f"- Élus territoriaux\n\n"
                            f"Pour {domaine}, consultez expert spécialisé.\n\n"
                            f"[Confiance : NULLE - Hors domaine expertise]",
                'source': 'OUT_OF_SCOPE_DOMAIN'
            })
        
        return questions[:n]
    
    # ==================== MÉTHODES UTILITAIRES ====================
    
    def get_field(self, obj, *field_names):
        """Récupère champ avec noms alternatifs"""
        if isinstance(obj, dict):
            for name in field_names:
                if name in obj and obj[name]:
                    return obj[name]
        else:
            for name in field_names:
                val = getattr(obj, name, None)
                if val:
                    return val
        return None
    
    def extract_montant(self, marche):
        """Extrait montant depuis différents formats"""
        montant = marche.get('montant', 0)
        
        if not montant:
            return 0
        
        if isinstance(montant, (int, float)):
            return float(montant)
        
        # Nettoyer string
        montant_str = str(montant).replace('€', '').replace(',', '.').replace(' ', '').strip()
        try:
            return float(montant_str)
        except:
            return 0
    
    def infer_type_marche(self, objet):
        """Inférer type marché depuis objet"""
        objet_lower = objet.lower()
        
        if any(word in objet_lower for word in ['travaux', 'construction', 'rénovation', 'voirie']):
            return 'travaux'
        elif any(word in objet_lower for word in ['fourniture', 'achat', 'matériel', 'équipement']):
            return 'fournitures'
        elif any(word in objet_lower for word in ['service', 'prestation', 'maintenance', 'nettoyage']):
            return 'services'
        else:
            return 'fournitures'
    
    def determine_procedure(self, montant, type_marche):
        """Détermine procédure applicable selon montant"""
        
        if type_marche == 'travaux':
            seuil_eu = 215000
        else:
            seuil_eu = 140000
        
        if montant < 40000:
            return "MAPA sans publicité (< 40K€)"
        elif montant < 90000:
            return "MAPA sans publicité obligatoire (40K-90K€)"
        elif montant < seuil_eu:
            return f"MAPA avec publicité obligatoire (90K-{seuil_eu/1000:.0f}K€)"
        else:
            return f"Appel d'offres ouvert/restreint (> {seuil_eu/1000:.0f}K€ seuil européen)"
    
    def explain_seuils(self, montant, type_marche):
        """Explique analyse seuils"""
        
        if type_marche == 'travaux':
            seuil_eu = 215000
        else:
            seuil_eu = 140000
        
        explication = ""
        
        if montant < 40000:
            explication = "- Montant < 40,000€ : MAPA sans publicité possible\n"
        elif montant < 90000:
            explication = "- Montant entre 40,000€ et 90,000€ : MAPA, publicité recommandée mais pas obligatoire\n"
        elif montant < seuil_eu:
            explication = f"- Montant entre 90,000€ et {seuil_eu:,}€ : MAPA, publicité OBLIGATOIRE (BOAMP ou équivalent)\n"
        else:
            explication = f"- Montant > {seuil_eu:,}€ : Seuil européen franchi, Appel d'offres ouvert ou restreint obligatoire\n"
        
        explication += f"- Seuil européen {type_marche} 2024-2025 : {seuil_eu:,}€ HT\n"
        
        return explication
    
    # ==================== GÉNÉRATION COMPLÈTE ====================
    
    def generate_all(self):
        """Génère TOUS les types de questions"""
        
        print("\n" + "="*80)
        print("GÉNÉRATION QUESTIONS VARIÉES")
        print("="*80)
        
        # Type 1 : Marchés détaillés
        print("\n📝 Type 1 : Questions marchés détaillées...")
        q1 = self.generate_detailed_marche_questions(3000)
        self.qa_pairs.extend(q1)
        print(f"   ✅ {len(q1)} questions générées")
        
        # Type 2 : Élus
        print("\n📝 Type 2 : Questions élus...")
        q2 = self.generate_elus_questions(1500)
        self.qa_pairs.extend(q2)
        print(f"   ✅ {len(q2)} questions générées")
        
        # Type 3 : Comparaisons
        print("\n📝 Type 3 : Comparaisons territoriales...")
        q3 = self.generate_comparative_questions(1200)
        self.qa_pairs.extend(q3)
        print(f"   ✅ {len(q3)} questions générées")
        
        # Type 4 : Procédures
        print("\n📝 Type 4 : Questions procédurales...")
        q4 = self.generate_procedural_questions(2000)
        self.qa_pairs.extend(q4)
        print(f"   ✅ {len(q4)} questions générées")
        
        # Type 5 : Délibérations
        print("\n📝 Type 5 : Questions délibérations...")
        q5 = self.generate_deliberations_questions(500)
        self.qa_pairs.extend(q5)
        print(f"   ✅ {len(q5)} questions générées")
        
        # Type 6 : Questions croisées
        print("\n📝 Type 6 : Questions croisées (maire + marchés)...")
        # Désactivé car nécessiterait indexation ville (pas disponible dans DECP)
        q6 = []
        self.qa_pairs.extend(q6)
        print(f"   ✅ {len(q6)} questions générées")
        
        # Type 7 : Refus hors corpus (CRITIQUE)
        print("\n📝 Type 7 : Refus intelligents hors corpus...")
        q7 = self.generate_out_of_scope_refusals(100)
        self.qa_pairs.extend(q7)
        print(f"   ✅ {len(q7)} questions générées")
        
        print(f"\n✅ TOTAL : {len(self.qa_pairs):,} paires Q/R variées générées")
        
        return self.qa_pairs
    
    def save_to_jsonl(self, output_file):
        """Sauvegarde en JSONL"""
        
        # Créer dossier parent si nécessaire
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in self.qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Sauvegardé : {output_file}")


# ==================== MAIN ====================

if __name__ == "__main__":
    
    # Chargement données
    print("\n" + "="*80)
    print("CHARGEMENT DONNÉES RÉELLES")
    print("="*80)
    
    marches = load_decp_complete()
    elus = load_elus()
    deliberations = load_deliberations()
    
    # Génération
    generator = VariedQuestionGenerator(marches, elus, deliberations)
    qa_pairs = generator.generate_all()
    
    # Sauvegarde
    output_file = OUTPUT_DIR / "training_data_enriched_varied.jsonl"
    generator.save_to_jsonl(output_file)
    
    # Stats
    print("\n" + "="*80)
    print("STATISTIQUES ENRICHISSEMENT")
    print("="*80)
    
    sources = defaultdict(int)
    for pair in qa_pairs:
        sources[pair['source']] += 1
    
    print("\n📊 Répartition par type :")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {source:30s} : {count:,} pairs")
    
    # Estimation tokens
    total_chars = sum(len(p['prompt']) + len(p['completion']) for p in qa_pairs)
    estimated_tokens = total_chars / 4
    
    print(f"\n📏 Estimation taille :")
    print(f"   • Total caractères : {total_chars:,}")
    print(f"   • Tokens estimés : {estimated_tokens:,.0f}")
    
    print("\n🎯 Prochaine étape : Fusionner avec corpus existant (37,696 pairs)")
    print(f"   Corpus final : ~{37696 + len(qa_pairs):,} pairs (~{(250112 + estimated_tokens):,.0f} tokens)")
