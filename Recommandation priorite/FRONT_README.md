# Interfaces Front-End - Recommandation de Priorité

## 📱 Applications Web Interactives

Ce dossier contient deux applications Flask pour tester les modèles de recommandation de priorité via une interface web intuitive.

## 🚀 Démarrage Rapide

### Prérequis
```bash
pip install flask pandas numpy scikit-learn --break-system-packages
```

### Random Forest

```bash
cd "Random Forest/FRONT"
python3 app.py
```
Ouvrez votre navigateur à: **http://localhost:5000**

### KNN

```bash
cd "KNN/FRONT"
python3 app.py
```
Ouvrez votre navigateur à: **http://localhost:5001**

## ✨ Fonctionnalités

### Interface Random Forest
- 🎯 Prédiction interactive de la priorité (Low, Medium, High)
- 📊 Visualisation des probabilités pour chaque classe
- 🔍 Importance des features en temps réel
- 💡 Exemples pré-configurés pour tests rapides
- 📈 Statistiques du modèle (Accuracy: 53.3%, F1-macro: 0.404)

### Interface KNN
- 🎯 Prédiction interactive avec KNN (k=2 voisins)
- 📊 Visualisation des probabilités
- 📏 Affichage des valeurs normalisées (StandardScaler)
- 💡 Exemples pré-configurés
- 📈 Statistiques du modèle (Accuracy: 41.7%, F1-macro: 0.413)
- ⚠️ **Scaling automatique des données** (transparent pour l'utilisateur)

## 🎨 Captures d'Écran

### Random Forest Interface
- **Couleurs:** Violet/Bleu (#667eea, #764ba2)
- **Style:** Professionnel et moderne
- **Focus:** Interprétabilité (importance des features)

### KNN Interface
- **Couleurs:** Rose/Rouge (#f093fb, #f5576c)
- **Style:** Baseline avec scaling visible
- **Focus:** Transparence du preprocessing

## 📝 Utilisation

1. **Entrer les données de la tâche:**
   - **Progress:** Slider de 0 à 1 (avancement de la tâche)
   - **Budget:** Montant en euros
   - **Durée planifiée:** Nombre de jours

2. **Cliquer sur "Prédire la Priorité"**

3. **Voir les résultats:**
   - Priorité prédite (Low/Medium/High)
   - Niveau de confiance
   - Probabilités détaillées pour chaque classe
   - Importance des variables (RF uniquement)
   - Valeurs normalisées (KNN uniquement)

## 💡 Exemples Intégrés

Les deux interfaces incluent 5 exemples pré-configurés:

1. **Tâche peu avancée, petit budget, courte durée**
   - Progress: 0.2, Budget: 5000€, Durée: 30 jours

2. **Tâche avancée, gros budget, longue durée**
   - Progress: 0.8, Budget: 15000€, Durée: 90 jours

3. **Tâche moyenne sur tous les critères**
   - Progress: 0.5, Budget: 10000€, Durée: 60 jours

4. **Tâche bloquée, budget élevé, durée moyenne**
   - Progress: 0.1, Budget: 12000€, Durée: 45 jours

5. **Tâche presque terminée, petit budget, courte durée**
   - Progress: 0.9, Budget: 3000€, Durée: 15 jours

## 🔧 Structure des Fichiers

```
Random Forest/FRONT/
├── app.py              # Application Flask
└── templates/
    └── index.html      # Interface utilisateur

KNN/FRONT/
├── app.py              # Application Flask
└── templates/
    └── index.html      # Interface utilisateur
```

## 🌐 API Endpoints

### Random Forest (`http://localhost:5000`)

- `GET /` - Interface web
- `POST /predict` - Prédiction (JSON)
  ```json
  {
    "progress": 0.5,
    "budget": 10000,
    "planned_duration": 60
  }
  ```
- `GET /model_info` - Informations sur le modèle
- `GET /examples` - Exemples de données

### KNN (`http://localhost:5001`)

- Mêmes endpoints que Random Forest
- Scaling automatique transparent

## 📊 Différences Clés

| Aspect | Random Forest | KNN |
|--------|---------------|-----|
| **Port** | 5000 | 5001 |
| **Accuracy** | 53.3% | 41.7% |
| **F1-macro** | 0.404 | 0.413 |
| **Scaling** | Non requis | Automatique (StandardScaler) |
| **Interprétabilité** | Importance des features | Valeurs normalisées |
| **Points forts** | Classe High | Classes Low/Medium |
| **Couleurs** | Violet/Bleu | Rose/Rouge |

## ⚠️ Notes Importantes

### Random Forest
- Pas de preprocessing nécessaire
- Directement utilisable avec valeurs brutes
- Interprétable via importance des features

### KNN
- **Scaling obligatoire** (géré automatiquement)
- Les valeurs sont normalisées avant prédiction
- Interface affiche les valeurs normalisées pour transparence
- StandardScaler avec moyenne=0, écart-type=1

## 🎯 Comparaison en Direct

Pour comparer les deux modèles:

1. Ouvrir les deux interfaces dans des onglets séparés
2. Entrer les mêmes valeurs dans les deux
3. Comparer les prédictions et probabilités
4. Observer comment RF favorise High, KNN est plus équilibré

## 🔐 Sécurité

- Validation des entrées côté serveur
- Gestion d'erreurs robuste
- Messages d'erreur informatifs
- Pas de stockage de données

## 📱 Responsive Design

Les interfaces s'adaptent automatiquement:
- Desktop: 2 colonnes
- Tablette: 2 colonnes adaptées
- Mobile: 1 colonne

## 🚀 Déploiement

Pour déployer en production:

```python
# Modifier dans app.py
app.run(debug=False, host='0.0.0.0', port=5000)
```

Utiliser un serveur WSGI comme **Gunicorn** ou **uWSGI**.

## 📚 Ressources

- Documentation Flask: https://flask.palletsprojects.com/
- Modèles ML: Voir README.md principal
- Scripts de test: `rf_priority_test.py` et `knn_priority_test.py`

## 🤝 Support

Pour toute question ou problème:
1. Vérifier que tous les modèles sont chargés correctement
2. Vérifier que Flask et les dépendances sont installées
3. Consulter les logs dans le terminal

---

**Créé pour:** Projet ML - Recommandation de Priorité  
**Date:** Avril 2026  
**Frameworks:** Flask, Vanilla JavaScript, CSS3
