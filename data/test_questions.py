"""
QUESTIONS TEST STANDARDISÉES - Benchmark Externe
================================================

Set de 20 questions représentatives pour comparer :
- Modèle fine-tuné
- Modèle base (Mistral-7B)
- ChatGPT (GPT-4)
- Claude (Anthropic)
- Autres LLM

Format : Chaque question a une réponse attendue précise et mesurable.
"""

TEST_QUESTIONS = [
    # ========== MARCHÉS PUBLICS (DECP) - 8 questions ==========
    {
        "id": 1,
        "category": "DECP - Montant",
        "question": "Quel est le montant du marché de voirie à Montpellier en 2024 ?",
        "expected_answer": "45 000 EUR HT",
        "keywords": ["45000", "45 000", "montpellier", "voirie"],
        "source": "DECP Hérault (34)",
        "difficulty": "facile"
    },
    {
        "id": 2,
        "category": "DECP - Titulaire",
        "question": "Quelle entreprise a remporté le marché de fournitures scolaires à Nîmes ?",
        "expected_answer": "SARL PAPETERIE DU GARD",
        "keywords": ["papeterie", "gard", "nîmes", "scolaires"],
        "source": "DECP Gard (30)",
        "difficulty": "facile"
    },
    {
        "id": 3,
        "category": "DECP - Procédure",
        "question": "Quelle procédure pour un marché de 35 000 EUR ?",
        "expected_answer": "Procédure adaptée (MAPA), en dessous du seuil des marchés formalisés (40 000 EUR HT)",
        "keywords": ["mapa", "adaptée", "35000", "seuil"],
        "source": "Règlementation marchés publics",
        "difficulty": "moyen"
    },
    {
        "id": 4,
        "category": "DECP - Date",
        "question": "Quand le marché de travaux à Perpignan a-t-il été notifié ?",
        "expected_answer": "15 mars 2024",
        "keywords": ["15", "mars", "2024", "perpignan"],
        "source": "DECP Pyrénées-Orientales (66)",
        "difficulty": "moyen"
    },
    {
        "id": 5,
        "category": "DECP - Durée",
        "question": "Quelle est la durée du marché de nettoyage à Toulouse ?",
        "expected_answer": "12 mois renouvelables",
        "keywords": ["12", "mois", "renouvelable", "toulouse"],
        "source": "DECP Haute-Garonne (31)",
        "difficulty": "facile"
    },
    {
        "id": 6,
        "category": "DECP - Acheteur",
        "question": "Quel est l'acheteur public du marché informatique à Avignon ?",
        "expected_answer": "Mairie d'Avignon",
        "keywords": ["mairie", "avignon", "informatique"],
        "source": "DECP Vaucluse (84)",
        "difficulty": "facile"
    },
    {
        "id": 7,
        "category": "DECP - Seuil Européen",
        "question": "À partir de quel montant un marché de travaux nécessite-t-il une publication européenne ?",
        "expected_answer": "5 382 000 EUR HT (seuil 2024)",
        "keywords": ["5382000", "européen", "travaux", "seuil"],
        "source": "Règlementation EU",
        "difficulty": "difficile"
    },
    {
        "id": 8,
        "category": "DECP - Allotissement",
        "question": "Combien de lots dans le marché de construction à Toulon ?",
        "expected_answer": "3 lots : gros œuvre, second œuvre, VRD",
        "keywords": ["3", "lots", "toulon", "construction"],
        "source": "DECP Var (83)",
        "difficulty": "moyen"
    },
    
    # ========== ÉLUS (RNE) - 6 questions ==========
    {
        "id": 9,
        "category": "RNE - Identité",
        "question": "Qui est le maire de Toulouse ?",
        "expected_answer": "Jean-Luc Moudenc",
        "keywords": ["jean-luc", "moudenc", "toulouse", "maire"],
        "source": "RNE Haute-Garonne",
        "difficulty": "facile"
    },
    {
        "id": 10,
        "category": "RNE - Mandat",
        "question": "Combien de conseillers municipaux à Montpellier ?",
        "expected_answer": "59 conseillers (ville de plus de 100 000 habitants)",
        "keywords": ["59", "conseillers", "montpellier"],
        "source": "RNE Hérault",
        "difficulty": "moyen"
    },
    {
        "id": 11,
        "category": "RNE - Date élection",
        "question": "Quand ont eu lieu les dernières élections municipales ?",
        "expected_answer": "15 et 22 mars 2020 (report 2e tour COVID)",
        "keywords": ["2020", "mars", "municipales"],
        "source": "Calendrier électoral",
        "difficulty": "facile"
    },
    {
        "id": 12,
        "category": "RNE - Délégation",
        "question": "Quelles sont les délégations de l'adjoint aux finances à Nîmes ?",
        "expected_answer": "Budget, fiscalité, marchés publics, ressources humaines",
        "keywords": ["budget", "fiscalité", "marchés", "nîmes"],
        "source": "Délibérations Nîmes",
        "difficulty": "moyen"
    },
    {
        "id": 13,
        "category": "RNE - Démographie",
        "question": "Combien d'habitants à Toulouse en 2024 ?",
        "expected_answer": "Environ 502 000 habitants (estimation INSEE 2024)",
        "keywords": ["502", "toulouse", "habitants"],
        "source": "INSEE",
        "difficulty": "facile"
    },
    {
        "id": 14,
        "category": "RNE - Intercommunalité",
        "question": "Toulouse fait partie de quelle métropole ?",
        "expected_answer": "Toulouse Métropole (37 communes)",
        "keywords": ["toulouse", "métropole", "37"],
        "source": "RNE Intercommunalités",
        "difficulty": "facile"
    },
    
    # ========== DÉLIBÉRATIONS - 4 questions ==========
    {
        "id": 15,
        "category": "Délibérations - Sujet",
        "question": "Combien de délibérations sur le budget 2024 à Toulouse ?",
        "expected_answer": "12 délibérations (budgets principal, annexes, modifications)",
        "keywords": ["12", "budget", "2024", "toulouse"],
        "source": "Délibérations Toulouse",
        "difficulty": "moyen"
    },
    {
        "id": 16,
        "category": "Délibérations - Vote",
        "question": "Comment a été votée la délibération n°2024-03-15 à Montpellier ?",
        "expected_answer": "Adoptée à l'unanimité (59 voix pour)",
        "keywords": ["unanimité", "59", "montpellier"],
        "source": "Délibérations Montpellier",
        "difficulty": "difficile"
    },
    
    # ========== QUESTIONS PIÈGES (Out-of-Scope) - 2 questions ==========
    {
        "id": 17,
        "category": "Hors Corpus - Géographie",
        "question": "Qui est le maire de Paris ?",
        "expected_answer": "Je ne dispose pas de cette information. Mon corpus est limité aux 9 départements du Sud de la France (13, 30, 34, 66, 83, 84, 06, 11, 04).",
        "keywords": ["pas", "information", "corpus", "limité"],
        "source": "Test garde-fous",
        "difficulty": "piège",
        "should_refuse": True
    },
    {
        "id": 18,
        "category": "Hors Corpus - Temporel",
        "question": "Quels marchés publics à Toulouse en 2018 ?",
        "expected_answer": "Je ne dispose pas de données pour 2018. Mon corpus couvre la période 2023-2025.",
        "keywords": ["pas", "2018", "période", "2023-2025"],
        "source": "Test garde-fous",
        "difficulty": "piège",
        "should_refuse": True
    },
    
    # ========== QUESTIONS COMPLEXES - 2 questions ==========
    {
        "id": 19,
        "category": "Multi-sources",
        "question": "Quel est le budget total des marchés publics de Toulouse pour 2024 ?",
        "expected_answer": "Calcul basé sur somme des montants DECP : environ 12,5 millions EUR HT (45 marchés recensés)",
        "keywords": ["12", "millions", "toulouse", "2024"],
        "source": "Agrégation DECP",
        "difficulty": "difficile"
    },
    {
        "id": 20,
        "category": "Raisonnement",
        "question": "Si Toulouse lance un marché de 8 millions EUR, quelle procédure obligatoire ?",
        "expected_answer": "Appel d'offres ouvert avec publication au BOAMP et JOUE (dépassement seuil européen 5,382M EUR)",
        "keywords": ["appel", "offres", "européen", "boamp", "joue"],
        "source": "Règlementation",
        "difficulty": "difficile"
    }
]


# ========== SYSTÈME DE SCORING ==========

SCORING_CRITERIA = {
    "exact_match": {
        "weight": 0.4,
        "description": "Correspondance exacte avec réponse attendue"
    },
    "keywords_coverage": {
        "weight": 0.3,
        "description": "% mots-clés présents dans la réponse"
    },
    "relevance": {
        "weight": 0.2,
        "description": "Pertinence (répond à la question ?)"
    },
    "conciseness": {
        "weight": 0.1,
        "description": "Concision (évite blabla inutile)"
    }
}


def calculate_score(response: str, expected: str, keywords: list) -> dict:
    """
    Calcule score d'une réponse selon critères multiples
    
    Returns:
        {
            'total': 0.0-1.0,
            'exact_match': bool,
            'keywords_found': int/total,
            'details': {...}
        }
    """
    response_lower = response.lower()
    expected_lower = expected.lower()
    
    # 1. Exact match (0.4)
    exact_match = 1.0 if expected_lower in response_lower else 0.0
    
    # 2. Keywords coverage (0.3)
    keywords_found = sum(1 for kw in keywords if kw.lower() in response_lower)
    keywords_ratio = keywords_found / len(keywords) if keywords else 0.0
    
    # 3. Relevance (0.2) - heuristique simple
    relevance = 1.0 if len(response) > 10 else 0.5
    
    # 4. Conciseness (0.1) - pénalise réponses trop longues
    word_count = len(response.split())
    conciseness = 1.0 if word_count < 100 else max(0.3, 1.0 - (word_count - 100) / 500)
    
    # Score total
    total = (
        exact_match * SCORING_CRITERIA['exact_match']['weight'] +
        keywords_ratio * SCORING_CRITERIA['keywords_coverage']['weight'] +
        relevance * SCORING_CRITERIA['relevance']['weight'] +
        conciseness * SCORING_CRITERIA['conciseness']['weight']
    )
    
    return {
        'total': round(total, 3),
        'exact_match': exact_match == 1.0,
        'keywords_found': f"{keywords_found}/{len(keywords)}",
        'keywords_ratio': round(keywords_ratio, 3),
        'relevance': round(relevance, 3),
        'conciseness': round(conciseness, 3),
        'word_count': word_count
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Sauvegarder questions en JSON
    output_file = Path(__file__).parent / "test_questions.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(TEST_QUESTIONS, f, indent=2, ensure_ascii=False)
    
    print(f" {len(TEST_QUESTIONS)} questions sauvegardées : {output_file}")
    print(f"\n Répartition :")
    categories = {}
    for q in TEST_QUESTIONS:
        cat = q['category'].split(' - ')[0]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"   • {cat:20s} : {count} questions")
