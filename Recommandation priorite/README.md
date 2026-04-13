# Modèle 3 — Recommandation de priorité (`Priority`)

Deux classifieurs prédisent la **priorité** d’une tâche : **Low**, **Medium**, **High**, à partir de **Progress** (0–1), **Budget** et **Planned_Duration_Days**. Objectif : aider au tri et à la décision sur l’urgence.

Intégration au projet global : voir le **[README à la racine du dépôt](../README.md)** (hub **5080**, ports **5000** / **5001**).

---

## Données

- **Fichier utilisé par les pipelines** : `Project-Management-2-enriched.csv` à la racine du dossier **ML** (pas de chemin absolu fixe dans le code).
- **Cible** : `Priority` (déséquilibre notable : High majoritaire).
- **Features modèle** : `Progress`, `Budget`, `Planned_Duration_Days`.

---

## Modèles

| Modèle | Rôle | Scaling | Port Flask (FRONT) |
|--------|------|---------|-------------------|
| **Random Forest** | Modèle principal (accuracy plus élevée, bon recall High) | Non | **5000** |
| **KNN** | Baseline (F1-macro souvent comparable, meilleur sur Low/Medium) | **StandardScaler** (ajusté sur le train uniquement dans le pipeline) | **5001** |

Les performances indicatives (accuracy, F1-macro, etc.) sont dans les sorties console des scripts `MODEL/*_pipeline.py` et les fichiers `*_output.txt` / métadonnées `.pkl`.

---

## Structure

```
Recommandation priorite/
├── README.md
├── Random Forest/
│   ├── FRONT/                 # Flask — python app.py → :5000
│   │   ├── app.py
│   │   └── templates/index.html
│   └── MODEL/
│       ├── rf_priority_pipeline.py
│       ├── rf_priority_test.py
│       ├── rf_priority_model.pkl
│       ├── rf_priority_features.pkl
│       ├── rf_priority_meta.pkl
│       └── … (figures, logs)
├── KNN/
│   ├── FRONT/                 # Flask — python app.py → :5001
│   │   ├── app.py
│   │   └── templates/index.html
│   └── MODEL/
│       ├── knn_priority_pipeline.py
│       ├── knn_priority_test.py
│       ├── knn_priority_model.pkl
│       ├── knn_priority_scaler.pkl
│       ├── knn_priority_features.pkl
│       ├── knn_priority_meta.pkl
│       └── …
├── comparison_analysis.py
└── comparison_*.png
```

---

## Entraînement (PowerShell, depuis la racine `ML`)

```powershell
python "Recommandation priorite\Random Forest\MODEL\rf_priority_pipeline.py"
python "Recommandation priorite\KNN\MODEL\knn_priority_pipeline.py"
```

Les `.pkl` et graphiques sont écrits dans chaque dossier `MODEL/`.

## Tests unitaires / chargement

```powershell
python "Recommandation priorite\Random Forest\MODEL\rf_priority_test.py"
python "Recommandation priorite\KNN\MODEL\knn_priority_test.py"
```

## Comparaison graphique (scripts figés dans le dépôt)

```powershell
cd "Recommandation priorite"
python comparison_analysis.py
```

---

## Interfaces web

Même thème sombre type « Inter » que les autres modules. Les requêtes utilisent **`window.location.origin`** pour éviter les erreurs JSON lorsque la page est affichée dans le **hub** (`hub_global`).

```powershell
cd "Recommandation priorite\Random Forest\FRONT"
python app.py
```

```powershell
cd "Recommandation priorite\KNN\FRONT"
python app.py
```

---

## Inférence (rappel)

- **RF** : `predict` / `predict_proba` sur un `DataFrame` avec les colonnes attendues, **sans** scaler.
- **KNN** : **toujours** appliquer le **même** `StandardScaler` sauvegardé (`knn_priority_scaler.pkl`) que celui du pipeline, puis `predict` sur les données transformées.

---

## Dépendances

Identiques au reste du projet : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `flask`, fichiers pickle standards Python.

---

## Licence / usage

Projet pédagogique — voir le README racine pour le contexte global et le hub.
