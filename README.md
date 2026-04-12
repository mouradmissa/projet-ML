# Projet ML — Gestion de projet

Machine Learning sur des données de **gestion de projet** : classification du **niveau de risque** (`Risk_Level`) et régression sur l’**avancement** (`Progress`).

**Dépôt GitHub :** [https://github.com/mouradmissa/projet-ML](https://github.com/mouradmissa/projet-ML)

---

## Jeu de données (racine du projet)

| Fichier | Description |
|---------|-------------|
| `Project-Management-2-enriched.csv` | Données enrichies (~300 lignes, 20 colonnes) — **fichier principal** |
| `Project-Management-2-original.csv` | Données brutes |
| `Project-Management-2.csv` | Variante source |

Les scripts attendent le CSV enrichi à la racine du dossier `ML` (même niveau que ce `README.md`).

---

## Structure du dépôt

```
ML/
├── README.md
├── Project-Management-2-enriched.csv
├── Prediction avancement/          # Modèle 2 — régression sur Progress
│   ├── index.html                  # Liens vers les interfaces
│   ├── shared/
│   │   └── progress_inference.py   # Features communes aux FRONT
│   ├── Random Forest/
│   │   ├── FRONT/                  # Flask — port 5002
│   │   └── MODEL/                  # Pipeline + .pkl
│   └── Régression Linéaire/
│       ├── FRONT/                  # Flask — port 5003
│       └── MODEL/
├── Segmentation du risque/         # Modèle 1 — classification Risk_Level
│   ├── README.md                   # Détail KNN / RF / DT
│   ├── knn/, random_forest/, decision_tree/, combined/
│   └── ...
└── presentation/                   # Figures / support PPT (si présent)
```

---

## Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib flask
```

---

## 1. Prédiction de l’avancement (`Progress`)

### Entraînement

```powershell
# Random Forest
python "Prediction avancement\Random Forest\MODEL\rf_progress_pipeline.py"

# Régression linéaire (baseline, après le RF pour la comparaison dans les métadonnées)
python "Prediction avancement\Régression Linéaire\MODEL\lr_progress_pipeline.py"
```

### Interfaces web

Deux terminaux (ou arrière-plans distincts) :

```powershell
cd "Prediction avancement\Random Forest\FRONT"
python app.py
# → http://127.0.0.1:5002
```

```powershell
cd "Prediction avancement\Régression Linéaire\FRONT"
python app.py
# → http://127.0.0.1:5003
```

Page d’accueil locale : ouvrir `Prediction avancement\index.html` ou `http://127.0.0.1:5002/accueil_progress`.

---

## 2. Segmentation du risque (`Risk_Level`)

Voir le fichier détaillé **[Segmentation du risque/README.md](Segmentation%20du%20risque/README.md)** (KNN, Random Forest, arbre de décision, comparateur).

---

## Git — premier envoi sur GitHub

Depuis la racine `ML` :

```powershell
git init -b main
git add .
git commit -m "Initial commit: données, prédiction Progress (RF + LR), segmentation risque"
git remote add origin https://github.com/mouradmissa/projet-ML.git
git push -u origin main
```

Si le dépôt distant n’est pas vide, utiliser `git pull origin main --rebase` avant le push, ou forcer uniquement si vous assumez d’écraser l’historique distant (`git push -u origin main --force` — à éviter sur un dépôt partagé).

**Authentification GitHub :** PAT (Personal Access Token) ou [GitHub CLI](https://cli.github.com/) (`gh auth login`).

---

## Auteurs / licence

Projet pédagogique — usage selon les consignes du cours.
