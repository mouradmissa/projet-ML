"""
Script de test pour le modèle Random Forest de recommandation de priorité
Montre comment charger et utiliser le modèle sauvegardé pour faire des prédictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent

print("=" * 80)
print("TEST DU MODÈLE RANDOM FOREST - RECOMMANDATION DE PRIORITÉ")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DU MODÈLE ET DES MÉTADONNÉES
# ============================================================================
print("\n" + "=" * 80)
print("1. CHARGEMENT DU MODÈLE SAUVEGARDÉ")
print("=" * 80)

with open(_MODEL_DIR / 'rf_priority_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("[OK] Modele Random Forest charge")

with open(_MODEL_DIR / 'rf_priority_features.pkl', 'rb') as f:
    features = pickle.load(f)
print(f"[OK] Features chargees: {features}")

with open(_MODEL_DIR / 'rf_priority_meta.pkl', 'rb') as f:
    metadata = pickle.load(f)
print("[OK] Metadonnees chargees")

print(f"\nMapping des priorités:")
for key, value in metadata['priority_mapping'].items():
    print(f"  {value} → {key}")

# ============================================================================
# 2. PRÉDICTIONS SUR DES EXEMPLES
# ============================================================================
print("\n" + "=" * 80)
print("2. PRÉDICTIONS SUR DES EXEMPLES")
print("=" * 80)

# Créer des exemples de test
examples = pd.DataFrame({
    'Progress': [0.1, 0.5, 0.9, 0.3, 0.7],
    'Budget': [5000, 10000, 3000, 15000, 8000],
    'Planned_Duration_Days': [30, 60, 15, 90, 45]
})

print("\n--- Exemples de tâches ---")
print(examples.to_string(index=False))

# Faire les prédictions
predictions = model.predict(examples)
probabilities = model.predict_proba(examples)

# Mapper les prédictions aux labels
reverse_mapping = {v: k for k, v in metadata['priority_mapping'].items()}
predicted_labels = [reverse_mapping[pred] for pred in predictions]

print("\n--- Résultats des prédictions ---")
for i in range(len(examples)):
    print(f"\nTâche {i+1}:")
    print(f"  Progress: {examples.iloc[i]['Progress']}")
    print(f"  Budget: {examples.iloc[i]['Budget']}")
    print(f"  Duration: {examples.iloc[i]['Planned_Duration_Days']} jours")
    print(f"  → Priorité prédite: {predicted_labels[i]}")
    print(f"  → Probabilités: Low={probabilities[i][0]:.3f}, Medium={probabilities[i][1]:.3f}, High={probabilities[i][2]:.3f}")

# ============================================================================
# 3. FONCTION DE PRÉDICTION RÉUTILISABLE
# ============================================================================
print("\n" + "=" * 80)
print("3. FONCTION DE PRÉDICTION RÉUTILISABLE")
print("=" * 80)

def predict_priority(progress, budget, planned_duration_days):
    """
    Prédit la priorité d'une tâche
    
    Args:
        progress (float): Avancement de la tâche (0-1)
        budget (int): Budget alloué
        planned_duration_days (int): Durée planifiée en jours
    
    Returns:
        dict: {
            'priority': str (Low/Medium/High),
            'priority_code': int (0/1/2),
            'confidence': float (probabilité de la classe prédite),
            'probabilities': dict {Low: float, Medium: float, High: float}
        }
    """
    # Créer le DataFrame avec les features
    input_data = pd.DataFrame({
        'Progress': [progress],
        'Budget': [budget],
        'Planned_Duration_Days': [planned_duration_days]
    })
    
    # Prédiction
    pred_code = model.predict(input_data)[0]
    pred_label = reverse_mapping[pred_code]
    
    # Probabilités
    probs = model.predict_proba(input_data)[0]
    confidence = probs[pred_code]
    
    return {
        'priority': pred_label,
        'priority_code': int(pred_code),
        'confidence': float(confidence),
        'probabilities': {
            'Low': float(probs[0]),
            'Medium': float(probs[1]),
            'High': float(probs[2])
        }
    }

# Test de la fonction
print("\n--- Test de la fonction predict_priority() ---")
test_cases = [
    {'progress': 0.2, 'budget': 5000, 'duration': 30, 'description': 'Tâche peu avancée, petit budget, courte durée'},
    {'progress': 0.8, 'budget': 15000, 'duration': 90, 'description': 'Tâche avancée, gros budget, longue durée'},
    {'progress': 0.5, 'budget': 10000, 'duration': 60, 'description': 'Tâche moyenne sur tous les critères'}
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['description']}")
    result = predict_priority(test['progress'], test['budget'], test['duration'])
    print(f"  → Priorité: {result['priority']} (confiance: {result['confidence']:.1%})")
    print(f"  → Probabilités détaillées:")
    for priority, prob in result['probabilities'].items():
        print(f"     {priority}: {prob:.1%}")

# ============================================================================
# 4. IMPORTANCE DES FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("4. IMPORTANCE DES FEATURES")
print("=" * 80)

print("\nFacteurs influençant la priorité (par ordre d'importance):")
for item in metadata['feature_importance']:
    print(f"  {item['Feature']}: {item['Importance']:.1%}")

print("\nInterprétation:")
print("  • Budget (39%): Plus le budget est élevé, plus la priorité tend vers HIGH")
print("  • Duration (34%): Les tâches longues ont tendance à avoir une priorité plus élevée")
print("  • Progress (26%): L'avancement a un impact modéré sur la priorité")

# ============================================================================
# 5. RECOMMANDATIONS D'UTILISATION
# ============================================================================
print("\n" + "=" * 80)
print("5. RECOMMANDATIONS D'UTILISATION")
print("=" * 80)

print("""
Comment utiliser ce modèle en production:

1. CHARGEMENT DU MODÈLE
   - Charger une seule fois au démarrage de l'application
   - Garder en mémoire pour des prédictions rapides

2. PRÉDICTIONS
   - Utiliser la fonction predict_priority() pour chaque nouvelle tâche
   - Vérifier que les valeurs d'entrée sont valides:
     * Progress: entre 0 et 1
     * Budget: valeur positive
     * Planned_Duration_Days: entier positif

3. SEUILS DE CONFIANCE
   - Si confiance < 50%: la prédiction est incertaine
   - Considérer une révision manuelle ou des features supplémentaires

4. MISE À JOUR DU MODÈLE
   - Réentraîner périodiquement avec de nouvelles données
   - Surveiller la drift des performances au fil du temps

5. LIMITATIONS
   - Performances modérées (53% d'accuracy)
   - Biais vers la classe HIGH (classe majoritaire)
   - Faible performance sur les classes LOW et MEDIUM
   - Considérer comme un outil d'aide à la décision, pas une décision finale
""")

print("\n" + "=" * 80)
print("✓ TEST TERMINÉ AVEC SUCCÈS")
print("=" * 80)
