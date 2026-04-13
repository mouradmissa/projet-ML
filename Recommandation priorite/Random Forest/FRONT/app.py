"""
Application Flask pour tester le modèle Random Forest de recommandation de priorité
Interface web interactive pour prédire la priorité des tâches
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Chemins des fichiers du modèle
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODEL')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_priority_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'rf_priority_features.pkl')
META_PATH = os.path.join(MODEL_DIR, 'rf_priority_meta.pkl')

# Charger le modèle au démarrage
print("Chargement du modele Random Forest...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

with open(META_PATH, 'rb') as f:
    metadata = pickle.load(f)

# Mapping inverse pour les prédictions
reverse_mapping = {v: k for k, v in metadata['priority_mapping'].items()}


def _to_jsonable_params(params):
    out = {}
    for k, v in (params or {}).items():
        if v is None:
            out[str(k)] = None
        elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            out[str(k)] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[str(k)] = float(v)
        elif hasattr(v, 'item'):
            out[str(k)] = v.item()
        else:
            out[str(k)] = v
    return out


def _probabilities_by_class(clf, X, rev_map):
    row = clf.predict_proba(X)[0]
    by_name = {}
    for i, cls in enumerate(clf.classes_):
        code = int(cls)
        by_name[rev_map[code]] = float(row[i])
    return {
        'Low': by_name.get('Low', 0.0),
        'Medium': by_name.get('Medium', 0.0),
        'High': by_name.get('High', 0.0),
    }


def _confidence_for_prediction(clf, X, pred_code):
    row = clf.predict_proba(X)[0]
    pred_code = int(pred_code)
    for i, cls in enumerate(clf.classes_):
        if int(cls) == pred_code:
            return float(row[i])
    return float(np.max(row))


print("[OK] Modele Random Forest charge")
print(f"[OK] Features: {features}")
print(f"[OK] Hyperparametres: {metadata['best_params']}")
print(f"[OK] F1-macro: {metadata['f1_macro']:.4f}")

@app.route('/')
def index():
    """Page d'accueil avec le formulaire de prédiction"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour faire une prédiction"""
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        progress = float(data.get('progress', 0))
        budget = float(data.get('budget', 0))
        planned_duration = int(data.get('planned_duration', 0))
        
        # Validation des données
        if not (0 <= progress <= 1):
            return jsonify({
                'error': 'Progress doit être entre 0 et 1'
            }), 400
        
        if budget <= 0:
            return jsonify({
                'error': 'Budget doit être positif'
            }), 400
        
        if planned_duration <= 0:
            return jsonify({
                'error': 'Planned Duration doit être positif'
            }), 400
        
        # Créer le DataFrame pour la prédiction
        input_data = pd.DataFrame({
            'Progress': [progress],
            'Budget': [budget],
            'Planned_Duration_Days': [planned_duration]
        })
        
        prediction_code = int(model.predict(input_data)[0])
        prediction_label = reverse_mapping[prediction_code]
        probs = _probabilities_by_class(model, input_data, reverse_mapping)
        confidence = _confidence_for_prediction(model, input_data, prediction_code)
        
        # Importance des features pour cette prédiction
        feature_contributions = {
            'Progress': float(progress),
            'Budget': float(budget),
            'Planned_Duration_Days': int(planned_duration)
        }
        
        # Préparer la réponse
        response = {
            'prediction': prediction_label,
            'prediction_code': prediction_code,
            'confidence': confidence,
            'probabilities': probs,
            'input': feature_contributions,
            'feature_importance': {
                item['Feature']: float(item['Importance'])
                for item in metadata['feature_importance']
            },
            'model_info': {
                'type': 'Random Forest',
                'accuracy': float(metadata['accuracy']),
                'f1_macro': float(metadata['f1_macro']),
                'hyperparameters': _to_jsonable_params(metadata.get('best_params')),
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Erreur lors de la prédiction: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint pour obtenir les informations sur le modèle"""
    acc = metadata['accuracy']
    f1 = metadata['f1_macro']
    if hasattr(acc, 'item'):
        acc = float(acc.item())
    else:
        acc = float(acc)
    if hasattr(f1, 'item'):
        f1 = float(f1.item())
    else:
        f1 = float(f1)
    return jsonify({
        'model_type': 'Random Forest',
        'features': list(features),
        'accuracy': acc,
        'f1_macro': f1,
        'hyperparameters': _to_jsonable_params(metadata.get('best_params')),
        'feature_importance': metadata['feature_importance'],
        'class_names': metadata['class_names'],
        'priority_mapping': metadata['priority_mapping']
    })

@app.route('/examples', methods=['GET'])
def examples():
    """Endpoint pour obtenir des exemples de prédictions"""
    examples_data = [
        {
            'name': 'Tâche peu avancée, petit budget, courte durée',
            'progress': 0.2,
            'budget': 5000,
            'planned_duration': 30
        },
        {
            'name': 'Tâche avancée, gros budget, longue durée',
            'progress': 0.8,
            'budget': 15000,
            'planned_duration': 90
        },
        {
            'name': 'Tâche moyenne sur tous les critères',
            'progress': 0.5,
            'budget': 10000,
            'planned_duration': 60
        },
        {
            'name': 'Tâche bloquée, budget élevé, durée moyenne',
            'progress': 0.1,
            'budget': 12000,
            'planned_duration': 45
        },
        {
            'name': 'Tâche presque terminée, petit budget, courte durée',
            'progress': 0.9,
            'budget': 3000,
            'planned_duration': 15
        }
    ]
    
    return jsonify(examples_data)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("APPLICATION FLASK - RANDOM FOREST RECOMMANDATION DE PRIORITE")
    print("="*80)
    print(f"\nModele: Random Forest")
    print(f"Accuracy: {float(metadata['accuracy']):.2%}")
    print(f"F1-macro: {float(metadata['f1_macro']):.4f}")
    print(f"\nOuvrez votre navigateur a: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
