# Segmentation du Risque Projet — Classification Multi-Algorithmes

Prédiction du niveau de risque (**Low** / **Medium** / **High**) des tâches de gestion de projet à l'aide de 3 algorithmes de Machine Learning, avec une interface web pour chaque modèle et un comparateur unifié.

---

## Jeu de données

| Fichier | Description |
|---------|-------------|
| `Project-Management-2-original.csv` | Données brutes (300 lignes, 15 colonnes) |
| `Project-Management-2-enriched.csv` | Données enrichies (300 lignes, 20 colonnes) avec `Budget_Utilization`, `Assignee_Historical_Task_Count`, `Risk_Level` |

### Variable cible

**`Risk_Level`** — 3 classes :
- **Low** (124 tâches) — Situation sous contrôle
- **Medium** (145 tâches) — Vigilance requise
- **High** (31 tâches) — Situation critique

### Features principales

| Feature | Type | Description |
|---------|------|-------------|
| `Progress` | Float | Avancement de la tâche (0.0 à 1.0) |
| `Budget` | Int | Budget alloué en euros |
| `Actual Cost` | Int | Coût réel engagé |
| `Budget_Utilization` | Float | Ratio Actual Cost / Budget |
| `Planned_Duration_Days` | Int | Durée prévue en jours |
| `Hours Spent` | Int | Heures passées |
| `Assignee_Historical_Task_Count` | Int | Nombre de tâches historiques de la personne |
| `Project Type` | Catégoriel | Renovation, Construction, Infrastructure, etc. |
| `Project Status` | Catégoriel | Behind, On Track, On Hold, Completed |
| `Priority` | Ordinal | Low, Medium, High |
| `Task Status` | Catégoriel | Pending, In Progress, Completed |
| `Assigned To` | Catégoriel | Alice, Bob, Charlie, David, Eve, Frank, Grace, Ivy |

---

## Architecture du projet

```
ML/
├── README.md
├── Project-Management-2-enriched.csv
├── Project-Management-2-original.csv
├── Risk_Level_DecisionTree.ipynb          # Notebook complet Decision Tree (9 étapes)
│
├── knn/                                    # Algorithme 1 : K-Nearest Neighbors
│   ├── model/
│   │   ├── knn_risk_pipeline.py           # Entraînement + sauvegarde
│   │   ├── knn_risk_test.py               # Script de test
│   │   ├── knn_risk_model.pkl             # Modèle entraîné
│   │   ├── knn_risk_scaler.pkl            # StandardScaler
│   │   ├── knn_risk_pca.pkl               # PCA (95% variance)
│   │   └── knn_risk_label_encoder.pkl     # LabelEncoder cible
│   └── frontend/
│       ├── app_risk.py                    # API Flask (port 5011 par défaut)
│       └── frontend_risk.html             # Interface web
│
├── random_forest/                          # Algorithme 2 : Random Forest
│   ├── model/
│   │   ├── rf_risk_pipeline.py            # Entraînement + GridSearchCV + sauvegarde
│   │   ├── rf_risk_test.py                # Script de test
│   │   ├── rf_risk_model.pkl              # Modèle entraîné
│   │   ├── rf_risk_features.pkl           # Liste des features
│   │   ├── rf_risk_importances.pkl        # Importances des variables
│   │   └── rf_risk_label_encoder.pkl      # LabelEncoder cible
│   └── frontend/
│       ├── app_rf_risk.py                 # API Flask (port 5012 par défaut)
│       └── frontend_rf_risk.html          # Interface web
│
├── decision_tree/                          # Algorithme 3 : Decision Tree
│   ├── model/
│   │   ├── dt_risk_pipeline.py            # Entraînement + GridSearchCV + élagage + sauvegarde
│   │   ├── dt_risk_test.py                # Script de test
│   │   ├── dt_risk_model.pkl              # Modèle entraîné
│   │   ├── dt_risk_features.pkl           # Liste des features
│   │   ├── dt_risk_importances.pkl        # Importances des variables
│   │   └── dt_risk_label_encoder.pkl      # LabelEncoder cible
│   └── frontend/
│       ├── app_dt_risk.py                 # API Flask (port 5013 par défaut)
│       └── frontend_dt_risk.html          # Interface web
│
└── combined/                               # Comparateur unifié (3 modèles)
    ├── app_combined.py                    # API Flask unique (port 5005)
    └── frontend_combined.html             # Interface de comparaison
```

---

## Méthodologie commune (9 étapes)

Chaque algorithme suit rigoureusement la même méthodologie :

1. **Chargement et inspection** — `info()`, `head()`, `describe()`, `isnull()`, `duplicated()`
2. **Nettoyage** — Doublons, NaN, colonnes constantes, catégorielles >10 modalités, corrélations >0.95
3. **Encodage** — LabelEncoder (cible), One-Hot (nominales), pas de scaling pour RF/DT, StandardScaler+PCA pour KNN
4. **Séparation** — Features (X) / Cible (y)
5. **Division** — Train 80% / Test 20% (stratifié, `random_state=42`)
6. **Entraînement** — Modèle baseline
7. **Optimisation** — GridSearchCV (5-fold CV)
8. **Évaluation** — Accuracy, Confusion Matrix, Classification Report
9. **Résumé** — Récapitulatif structuré

---

## Résultats comparatifs

| Algorithme | Accuracy | F1 pondéré | Particularité |
|------------|----------|------------|---------------|
| **KNN** | 76.67% | ~76% | StandardScaler + PCA (95% variance), Best k optimisé |
| **Random Forest** | **96.67%** | ~96% | Ensemble de 100+ arbres, GridSearchCV |
| **Decision Tree** | 95.00% | ~95% | Arbre unique, élagage max_depth + ccp_alpha |

### Top 3 features (Decision Tree)

1. `Project Status_On Hold` (35.4%)
2. `Progress` (26.0%)
3. `Project Status_Behind` (23.4%)

---

## Prérequis

```
Python >= 3.10
```

### Dépendances

```
pandas
numpy
scikit-learn
flask
joblib
matplotlib
seaborn
```

Installation :

```bash
pip install pandas numpy scikit-learn flask joblib matplotlib seaborn
```

---

## Utilisation

### 1. Entraîner les modèles (si nécessaire)

```bash
python knn/model/knn_risk_pipeline.py
python random_forest/model/rf_risk_pipeline.py
python decision_tree/model/dt_risk_pipeline.py
```

### 2. Tester un modèle individuellement

```bash
python knn/model/knn_risk_test.py
python random_forest/model/rf_risk_test.py
python decision_tree/model/dt_risk_test.py
```

### 3. Lancer un frontend individuel

Ports par défaut choisis pour **ne pas entrer en conflit** avec les autres modules du dépôt (priorité : 5000–5001, avancement : 5002–5003). Variables d’environnement : `RISK_KNN_PORT`, `RISK_RF_PORT`, `RISK_DT_PORT`.

```bash
python knn/frontend/app_risk.py               # http://127.0.0.1:5011
python random_forest/frontend/app_rf_risk.py  # http://127.0.0.1:5012
python decision_tree/frontend/app_dt_risk.py   # http://127.0.0.1:5013
```

### 4. Lancer le comparateur unifié (recommandé)

```bash
python combined/app_combined.py               # http://127.0.0.1:5005
```

Ouvrir **http://127.0.0.1:5005** dans un navigateur.

### 5. Hub global (tout le projet ML)

Depuis la racine du dépôt, le dossier **`hub_global/`** propose une page unique (port **5080**) avec onglets : test individuel des trois algorithmes risque, comparateur, avancement, priorité. Voir le **`README.md`** à la racine `ML/`.

---

## Interfaces web

Chaque interface propose 2 onglets :

| Onglet | Contenu |
|--------|---------|
| **Prédiction** | Formulaire de saisie, scénarios rapides, résultat avec probabilités par classe |
| **Performance** | Accuracy, F1, Classification Report, Matrice de Confusion, graphiques spécifiques |

Le **comparateur unifié** (`/combined`) ajoute :
- Prédiction simultanée des 3 algorithmes côte à côte
- Vote majoritaire (consensus) avec indication d'unanimité
- Comparaison visuelle des performances des 3 modèles

---

## Notebook

Le fichier `Risk_Level_DecisionTree.ipynb` contient le pipeline complet du Decision Tree avec :
- Visualisations (distribution cible, matrice de corrélation, arbre graphique, importance des features)
- Courbes d'élagage (Accuracy vs max_depth, Accuracy vs ccp_alpha)
- Comparaison avant/après élagage
