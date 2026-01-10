"""
TEST MODÈLE FINE-TUNÉ - Inférence Interactive
==============================================

Script pour tester le modèle Mistral 7B fine-tuné avec LoRA.
Permet de comparer modèle base vs fine-tuné.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"

# System prompt
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'accès aux données publiques françaises, notamment :
- DECP (Données Essentielles de la Commande Publique)
- RNE (Répertoire National des Élus)

Tu réponds avec précision en citant tes sources. Si une information n'est pas dans ton corpus, tu le dis clairement."""


def load_base_model():
    """Charge modèle Mistral 7B de base (sans fine-tuning)"""
    
    print("\n Chargement modèle de base...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Modèle de base chargé")
    return model, tokenizer


def load_finetuned_model(adapter_path):
    """Charge modèle base + adapters LoRA fine-tunés"""
    
    print(f"\n Chargement modèle fine-tuné...")
    print(f"   Adapters : {adapter_path}")
    
    # Charger base
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    # Charger adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(" Modèle fine-tuné chargé")
    
    # Charger métadonnées
    metadata_file = Path(adapter_path) / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        print(f"\n Infos fine-tuning :")
        print(f"   • Corpus : {metadata['corpus']['num_examples']:,} paires")
        print(f"   • Durée : {metadata['training']['duration_minutes']:.1f} min")
        print(f"   • Loss finale : {metadata['training'].get('final_loss', 'N/A')}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Génère réponse du modèle"""
    
    # Formater avec template Mistral
    formatted_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nQuestion : {prompt} [/INST]"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(model.device)
    
    # Générer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Décoder (enlever prompt)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire seulement la réponse (après [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response


def compare_models(base_model, ft_model, tokenizer, prompts):
    """Compare réponses modèle base vs fine-tuné"""
    
    print("\n" + "="*80)
    print(" COMPARAISON MODÈLE BASE vs FINE-TUNÉ")
    print("="*80)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}/{len(prompts)} : {prompt}")
        print(f"{'─'*80}")
        
        # Modèle base
        print("\n Modèle BASE :")
        print("-" * 40)
        base_response = generate_response(base_model, tokenizer, prompt)
        print(base_response)
        
        # Modèle fine-tuné
        print("\n Modèle FINE-TUNÉ :")
        print("-" * 40)
        ft_response = generate_response(ft_model, tokenizer, prompt)
        print(ft_response)
    
    print("\n" + "="*80)


def interactive_mode(model, tokenizer, model_name="Fine-tuné"):
    """Mode interactif pour tester le modèle"""
    
    print("\n" + "="*80)
    print(f" MODE INTERACTIF - Modèle {model_name}")
    print("="*80)
    print("Tapez vos questions (ou 'quit' pour quitter)")
    print()
    
    while True:
        try:
            prompt = input(" Question : ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print(f"\n Réponse :")
            print("-" * 40)
            response = generate_response(model, tokenizer, prompt)
            print(response)
            print()
            
        except KeyboardInterrupt:
            break
    
    print("\n Au revoir !")


# ==================== EXEMPLES TESTS ====================

EXAMPLE_PROMPTS = [
    # Questions DECP (dans corpus)
    "Quel est le seuil pour un marché public sans publicité ni mise en concurrence ?",
    "Qu'est-ce qu'une procédure adaptée en marchés publics ?",
    "Combien de marchés publics dans le département de l'Hérault (34) ?",
    
    # Questions RNE (dans corpus)
    "Combien de conseillers municipaux à Montpellier ?",
    "Qui est le maire de Toulouse ?",
    
    # Questions hors corpus (test garde-fous)
    "Quelle est la capitale de l'Espagne ?",
    "Comment faire une tarte aux pommes ?",
]


# ==================== MAIN ====================

def main():
    
    print("\n" + "="*80)
    print(" TEST MODÈLE FINE-TUNÉ - Mistral 7B LoRA")
    print("="*80)
    
    # Vérifier GPU
    if torch.cuda.is_available():
        print(f" GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("  CPU uniquement (génération lente)")
    
    # Trouver dernier adapter
    adapter_dir = ADAPTERS_DIR / "mistral-7b-lora-decp"
    
    if not adapter_dir.exists():
        print("\n Aucun adapter LoRA trouvé !")
        print(f"   Cherché dans : {adapter_dir}")
        print(f"   Lancez d'abord train_lora.py")
        return
    
    latest_adapter = adapter_dir
    print(f"\n Adapter trouvé : {latest_adapter.name}")
    
    # Menu
    print("\n" + "="*80)
    print("OPTIONS")
    print("="*80)
    print("1. Tester modèle fine-tuné uniquement")
    print("2. Comparer base vs fine-tuné (exemples prédéfinis)")
    print("3. Comparer base vs fine-tuné (mode interactif)")
    print("4. Quitter")
    
    choice = input("\nChoix (1-4) : ").strip()
    
    if choice == '1':
        # Test fine-tuné uniquement
        model, tokenizer = load_finetuned_model(latest_adapter)
        interactive_mode(model, tokenizer, "Fine-tuné")
    
    elif choice == '2':
        # Comparaison avec exemples
        print("\n Chargement des 2 modèles...")
        base_model, base_tokenizer = load_base_model()
        ft_model, ft_tokenizer = load_finetuned_model(latest_adapter)
        
        compare_models(base_model, ft_model, ft_tokenizer, EXAMPLE_PROMPTS)
    
    elif choice == '3':
        # Comparaison interactive
        print("\n Chargement des 2 modèles...")
        base_model, base_tokenizer = load_base_model()
        ft_model, ft_tokenizer = load_finetuned_model(latest_adapter)
        
        print("\n Mode interactif - Les 2 réponses seront affichées")
        print()
        
        while True:
            try:
                prompt = input(" Question : ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print(f"\n MODÈLE BASE :")
                print("-" * 40)
                base_response = generate_response(base_model, base_tokenizer, prompt)
                print(base_response)
                
                print(f"\n MODÈLE FINE-TUNÉ :")
                print("-" * 40)
                ft_response = generate_response(ft_model, ft_tokenizer, prompt)
                print(ft_response)
                print()
                
            except KeyboardInterrupt:
                break
        
        print("\n Au revoir !")
    
    else:
        print("\n Au revoir !")


if __name__ == "__main__":
    main()
