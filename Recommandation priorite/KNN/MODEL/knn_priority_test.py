"""
Script de test pour le modèle KNN de recommandation de priorité
Montre comment charger et utiliser le modèle sauvegardé pour faire des prédictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_RF_MODEL = _MODEL_DIR.parent.parent / "Random Forest" / "MODEL" / "rf_priority_model.pkl"

print("=" * 80)
print("TEST DU MODÈLE KNN - RECOMMANDATION DE PRIORITÉ")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DU MODÈLE ET DES MÉTADONNÉES
# ============================================================================
print("\n" + "=" * 80)
print("1. CHARGEMENT DU MODÈLE SAUVEGARDÉ")
print("=" * 80)

with open(_MODEL_DIR / 'knn_priority_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"[OK] Modele KNN charge (k={model.n_neighbors})")

with open(_MODEL_DIR / 'knn_priority_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("[OK] Scaler StandardScaler charge")

with open(_MODEL_DIR / 'knn_priority_features.pkl', 'rb') as f:
    features = pickle.load(f)
print(f"[OK] Features chargees: {features}")

with open(_MODEL_DIR / 'knn_priority_meta.pkl', 'rb') as f:
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

# Créer des exemples de test (MÊMES que pour Random Forest pour comparaison)
examples = pd.DataFrame({
    'Progress': [0.1, 0.5, 0.9, 0.3, 0.7],
    'Budget': [5000, 10000, 3000, 15000, 8000],
    'Planned_Duration_Days': [30, 60, 15, 90, 45]
})

print("\n--- Exemples de tâches ---")
print(examples.to_string(index=False))

# IMPORTANT: Scaler les données avant prédiction
examples_scaled = pd.DataFrame(
    scaler.transform(examples),
    columns=features
)

print("\n--- Données après scaling ---")
print(examples_scaled.to_string(index=False))

# Faire les prédictions
predictions = model.predict(examples_scaled)
probabilities = model.predict_proba(examples_scaled)

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

def predict_priority_knn(progress, budget, planned_duration_days):
    """
    Prédit la priorité d'une tâche avec KNN
    
    Args:
        progress (float): Avancement de la tâche (0-1)
        budget (int): Budget alloué
        planned_duration_days (int): Durée planifiée en jours
    
    Returns:
        dict: {
            'priority': str (Low/Medium/High),
            'priority_code': int (0/1/2),
            'confidence': float (probabilité de la classe prédite),
            'probabilities': dict {Low: float, Medium: float, High: float},
            'k_neighbors': int (nombre de voisins utilisés)
        }
    """
    # Créer le DataFrame avec les features
    input_data = pd.DataFrame({
        'Progress': [progress],
        'Budget': [budget],
        'Planned_Duration_Days': [planned_duration_days]
    })
    
    # ÉTAPE CRITIQUE: Scaler les données
    input_scaled = scaler.transform(input_data)
    
    # Prédiction
    pred_code = model.predict(input_scaled)[0]
    pred_label = reverse_mapping[pred_code]
    
    # Probabilités
    probs = model.predict_proba(input_scaled)[0]
    confidence = probs[pred_code]
    
    return {
        'priority': pred_label,
        'priority_code': int(pred_code),
        'confidence': float(confidence),
        'probabilities': {
            'Low': float(probs[0]),
            'Medium': float(probs[1]),
            'High': float(probs[2])
        },
        'k_neighbors': model.n_neighbors
    }

# Test de la fonction
print("\n--- Test de la fonction predict_priority_knn() ---")
test_cases = [
    {'progress': 0.2, 'budget': 5000, 'duration': 30, 'description': 'Tâche peu avancée, petit budget, courte durée'},
    {'progress': 0.8, 'budget': 15000, 'duration': 90, 'description': 'Tâche avancée, gros budget, longue durée'},
    {'progress': 0.5, 'budget': 10000, 'duration': 60, 'description': 'Tâche moyenne sur tous les critères'}
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['description']}")
    result = predict_priority_knn(test['progress'], test['budget'], test['duration'])
    print(f"  → Priorité: {result['priority']} (confiance: {result['confidence']:.1%})")
    print(f"  → k voisins utilisés: {result['k_neighbors']}")
    print(f"  → Probabilités détaillées:")
    for priority, prob in result['probabilities'].items():
        print(f"     {priority}: {prob:.1%}")

# ============================================================================
# 4. COMPARAISON KNN vs RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("4. COMPARAISON AVEC RANDOM FOREST")
print("=" * 80)

# Charger le modèle Random Forest pour comparaison
with open(_RF_MODEL, 'rb') as f:
    rf_model = pickle.load(f)

print("\n--- Prédictions sur le même exemple ---")
test_input = pd.DataFrame({
    'Progress': [0.3],
    'Budget': [12000],
    'Planned_Duration_Days': [45]
})

# KNN (avec scaling)
test_scaled = scaler.transform(test_input)
knn_pred = model.predict(test_scaled)[0]
knn_proba = model.predict_proba(test_scaled)[0]

# Random Forest (sans scaling)
rf_pred = rf_model.predict(test_input)[0]
rf_proba = rf_model.predict_proba(test_input)[0]

print(f"\nInput: Progress=0.3, Budget=12000, Duration=45 jours")
print(f"\nKNN:           {reverse_mapping[knn_pred]}")
print(f"  Probabilités: Low={knn_proba[0]:.2f}, Medium={knn_proba[1]:.2f}, High={knn_proba[2]:.2f}")
print(f"\nRandom Forest: {reverse_mapping[rf_pred]}")
print(f"  Probabilités: Low={rf_proba[0]:.2f}, Medium={rf_proba[1]:.2f}, High={rf_proba[2]:.2f}")

if knn_pred == rf_pred:
    print(f"\n✓ Les deux modèles sont d'accord: {reverse_mapping[knn_pred]}")
else:
    print(f"\n⚠ Les modèles sont en désaccord:")
    print(f"  KNN prédit: {reverse_mapping[knn_pred]}")
    print(f"  RF prédit:  {reverse_mapping[rf_pred]}")

# ============================================================================
# 5. AVANTAGES ET LIMITATIONS DU KNN
# ============================================================================
print("\n" + "=" * 80)
print("5. CARACTÉRISTIQUES DU MODÈLE KNN")
print("=" * 80)

print(f"""
CONFIGURATION:
  • k optimal: {metadata['best_k']} voisins
  • F1-score (CV): {metadata['best_k_score']:.4f}
  • Accuracy (test): {metadata['accuracy']:.4f}
  • F1-macro (test): {metadata['f1_macro']:.4f}

POINTS FORTS:
  ✓ Meilleur F1-macro que Random Forest (0.413 vs 0.404)
  ✓ Meilleures performances sur les classes LOW et MEDIUM
  ✓ Prédictions plus équilibrées entre les classes
  ✓ Bon pour détecter les tâches de faible priorité

LIMITATIONS:
  ✗ Accuracy inférieure à Random Forest (41.7% vs 53.3%)
  ✗ Nécessite un scaling OBLIGATOIRE des données
  ✗ Pas d'interprétabilité (pas d'importance des features)
  ✗ Mauvaise performance sur la classe HIGH
  ✗ Sensible au bruit et aux outliers
  ✗ Plus lent en prédiction (calcul des distances)

QUAND UTILISER KNN:
  → Si vous avez besoin de détecter les classes minoritaires (LOW, MEDIUM)
  → Si l'équité entre classes est plus importante que l'accuracy globale
  → Si vous pouvez garantir le scaling des données d'entrée
  → Pour comparer/ensembler avec Random Forest
""")

# ============================================================================
# 6. RECOMMANDATIONS D'UTILISATION
# ============================================================================
print("\n" + "=" * 80)
print("6. RECOMMANDATIONS D'UTILISATION EN PRODUCTION")
print("=" * 80)

print("""
WORKFLOW POUR UTILISER KNN EN PRODUCTION:

1. INITIALISATION (une seule fois)
   model = pickle.load('knn_priority_model.pkl')
   scaler = pickle.load('knn_priority_scaler.pkl')  # CRUCIAL!
   
2. POUR CHAQUE PRÉDICTION
   # Créer le DataFrame avec les features
   data = pd.DataFrame({
       'Progress': [value],
       'Budget': [value],
       'Planned_Duration_Days': [value]
   })
   
   # ÉTAPE OBLIGATOIRE: Scaler
   data_scaled = scaler.transform(data)
   
   # Prédire
   prediction = model.predict(data_scaled)
   probabilities = model.predict_proba(data_scaled)

3. VALIDATION DES DONNÉES
   ⚠ TOUJOURS vérifier avant prédiction:
   - Progress entre 0 et 1
   - Budget > 0
   - Planned_Duration_Days > 0
   - Pas de valeurs manquantes

4. GESTION DES ERREURS
   - Scaler AVANT de prédire (sinon les prédictions seront fausses)
   - Vérifier que les features sont dans le bon ordre
   - Traiter les cas limites (valeurs extrêmes)

5. MONITORING
   - Surveiller la distribution des prédictions
   - Vérifier que le scaler utilise les bonnes statistiques
   - Comparer avec Random Forest pour les cas incertains

6. ENSEMBLE AVEC RANDOM FOREST
   - Utiliser les deux modèles en parallèle
   - Si accord: confiance élevée
   - Si désaccord: révision manuelle ou vote pondéré
""")

print("\n" + "=" * 80)
print("✓ TEST TERMINÉ AVEC SUCCÈS")
print("=" * 80)
