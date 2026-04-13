# Projet ML — Gestion de projet

Machine Learning sur des données de **gestion de projet** : trois blocs complémentaires — **segmentation du risque** (`Risk_Level`), **prédiction de l’avancement** (`Progress`), **recommandation de priorité** (`Priority` : Low / Medium / High).

**Dépôt GitHub :** [https://github.com/mouradmissa/projet-ML](https://github.com/mouradmissa/projet-ML)

---

## Jeu de données (racine du projet)

| Fichier | Description |
|---------|-------------|
| `Project-Management-2-enriched.csv` | Données enrichies (~300 lignes, 20 colonnes) — **fichier principal** |
| `Project-Management-2-original.csv` | Données brutes |
| `Project-Management-2.csv` | Variante source |

Les pipelines et interfaces attendent le CSV enrichi à la racine du dossier `ML` (même niveau que ce `README.md`).

---

## Structure du dépôt

```
ML/
├── README.md
├── Project-Management-2-enriched.csv
├── hub_global/                     # Point d’entrée unique (hub web)
│   ├── app.py                    # Flask — port 5080
│   └── templates/index.html
├── Prediction avancement/        # Modèle 2 — régression sur Progress
│   ├── index.html
│   ├── shared/progress_inference.py
│   ├── Random Forest/            # FRONT port 5002 · MODEL/
│   └── Régression Linéaire/      # FRONT port 5003 · MODEL/
├── Segmentation du risque/       # Modèle 1 — classification Risk_Level
│   ├── README.md
│   ├── knn/frontend/             # KNN seul — port 5011
│   ├── random_forest/frontend/   # RF seul — port 5012
│   ├── decision_tree/frontend/   # DT seul — port 5013
│   └── combined/                 # Comparateur 3 algo — port 5005
└── Recommandation priorite/      # Modèle 3 — classification Priority
    ├── README.md
    ├── KNN/                      # FRONT port 5001 · MODEL/
    ├── Random Forest/            # FRONT port 5000 · MODEL/
    ├── comparison_analysis.py
    └── comparison_*.png          # figures RF vs KNN (générées par comparison_analysis.py)
```

---

## Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib flask
```

---

## Hub global (recommandé pour tester tout le projet)

Une seule page avec onglets : risque (test **individuel** KNN / RF / DT puis **comparer les 3**), avancement (deux iframes), priorité (deux iframes).

```powershell
cd hub_global
python app.py
# → http://127.0.0.1:5080
```

**À lancer en parallèle** (un terminal par serveur, ou en arrière-plan) :

| Service | Port | Commande (depuis la racine `ML`) |
|--------|------|-----------------------------------|
| Priorité — Random Forest | 5000 | `cd "Recommandation priorite\Random Forest\FRONT"; python app.py` |
| Priorité — KNN | 5001 | `cd "Recommandation priorite\KNN\FRONT"; python app.py` |
| Avancement — Random Forest | 5002 | `cd "Prediction avancement\Random Forest\FRONT"; python app.py` |
| Avancement — Régression linéaire | 5003 | `cd "Prediction avancement\Régression Linéaire\FRONT"; python app.py` |
| Risque — comparateur (3 algo) | 5005 | `cd "Segmentation du risque\combined"; python app_combined.py` |
| Risque — KNN seul | 5011 | `cd "Segmentation du risque\knn\frontend"; python app_risk.py` |
| Risque — RF seul | 5012 | `cd "Segmentation du risque\random_forest\frontend"; python app_rf_risk.py` |
| Risque — arbre de décision seul | 5013 | `cd "Segmentation du risque\decision_tree\frontend"; python app_dt_risk.py` |

Les ports **5011–5013** évitent les conflits avec priorité (5000–5001) et avancement (5002–5003). Vous pouvez les surcharger avec les variables d’environnement `RISK_KNN_PORT`, `RISK_RF_PORT`, `RISK_DT_PORT`.

Variables optionnelles pour le hub : `ML_HUB_BACKEND_HOST` (défaut `127.0.0.1`), `ML_HUB_PORT` (défaut `5080`).

Les interfaces web appellent les API en **`window.location.origin + '/…'`** pour rester cohérentes dans un iframe (hub) ou en accès direct.

---

## 1. Segmentation du risque (`Risk_Level`)

Documentation complémentaire : **[Segmentation du risque/README.md](Segmentation%20du%20risque/README.md)**.

**Entraînement** : exécuter les pipelines dans chaque dossier `knn/model`, `random_forest/model`, `decision_tree/model` (voir le README du dossier). Les interfaces Flask chargent les modèles sauvegardés (fichiers attendus dans chaque `model/`).

**Interfaces** : voir tableau des ports ci-dessus (individuel **5011–5013**, comparateur **5005**).

---

## 2. Prédiction de l’avancement (`Progress`)

### Entraînement

```powershell
python "Prediction avancement\Random Forest\MODEL\rf_progress_pipeline.py"
python "Prediction avancement\Régression Linéaire\MODEL\lr_progress_pipeline.py"
```

### Interfaces

Ports **5002** (Random Forest) et **5003** (régression linéaire). Page d’accueil des deux modèles : `Prediction avancement\index.html` ou `http://127.0.0.1:5002/accueil_progress`.

---

## 3. Recommandation de priorité (`Priority`)

Classification **Low / Medium / High** à partir notamment de **Progress**, **Budget**, **Planned_Duration_Days**.

Détail des modèles, métriques et bonnes pratiques d’inférence : **[Recommandation priorite/README.md](Recommandation%20priorite/README.md)**.

### Entraînement

```powershell
python "Recommandation priorite\Random Forest\MODEL\rf_priority_pipeline.py"
python "Recommandation priorite\KNN\MODEL\knn_priority_pipeline.py"
```

Les sorties (`.pkl`, figures) sont écrites dans chaque dossier `MODEL/`.

### Interfaces (Flask)

```powershell
cd "Recommandation priorite\Random Forest\FRONT"
python app.py
# → http://127.0.0.1:5000
```

```powershell
cd "Recommandation priorite\KNN\FRONT"
python app.py
# → http://127.0.0.1:5001  (StandardScaler côté serveur, obligatoire pour KNN)
```

---

## Git — premier envoi sur GitHub

Depuis la racine `ML` :

```powershell
git init -b main
git add .
git commit -m "Projet ML: risque, avancement, priorité, hub global"
git remote add origin https://github.com/mouradmissa/projet-ML.git
git push -u origin main
```

Si le dépôt distant n’est pas vide, utiliser `git pull origin main --rebase` avant le push. Éviter `--force` sur un dépôt partagé.

**Authentification GitHub :** PAT (Personal Access Token) ou [GitHub CLI](https://cli.github.com/) (`gh auth login`).

---

## Auteurs / licence

Projet pédagogique — usage selon les consignes du cours.
