"""
Hub global — les 3 familles de modèles ML (interfaces Flask existantes).

Lancer depuis ce dossier :
  python app.py

Puis ouvrir http://127.0.0.1:5080

Prérequis : démarrer chaque backend sur son port (voir panneau « Démarrage » sur la page).
"""
from __future__ import annotations

import os

from flask import Flask, render_template

app = Flask(__name__)

# Hôte des autres serveurs (changez si besoin, ex. autre machine)
BACKEND_HOST = os.environ.get("ML_HUB_BACKEND_HOST", "127.0.0.1")


def _u(port: int, path: str = "/") -> str:
    return f"http://{BACKEND_HOST}:{port}{path}"


@app.route("/")
def index():
    cfg = {
        "host": BACKEND_HOST,
        "model1": {
            "title": "Segmentation du risque",
            "subtitle": "Testez chaque algorithme seul (KNN, Random Forest, arbre de décision), puis ouvrez le comparateur pour les 3 côte à côte.",
            "modes": [
                {
                    "id": "knn",
                    "label": "KNN seul",
                    "url": _u(5011),
                    "port": 5011,
                    "cmd": 'cd "Segmentation du risque/knn/frontend" && python app_risk.py',
                },
                {
                    "id": "rf",
                    "label": "Random Forest seul",
                    "url": _u(5012),
                    "port": 5012,
                    "cmd": 'cd "Segmentation du risque/random_forest/frontend" && python app_rf_risk.py',
                },
                {
                    "id": "dt",
                    "label": "Arbre de décision seul",
                    "url": _u(5013),
                    "port": 5013,
                    "cmd": 'cd "Segmentation du risque/decision_tree/frontend" && python app_dt_risk.py',
                },
                {
                    "id": "compare",
                    "label": "Comparer les 3",
                    "url": _u(5005),
                    "port": 5005,
                    "cmd": 'cd "Segmentation du risque/combined" && python app_combined.py',
                },
            ],
        },
        "model2": {
            "title": "Prédiction d'avancement",
            "subtitle": "Cible Progress (régression) — deux algorithmes.",
            "frames": [
                {
                    "label": "Random Forest",
                    "url": _u(5002),
                    "port": 5002,
                    "cmd": 'cd "Prediction avancement/Random Forest/FRONT" && python app.py',
                },
                {
                    "label": "Régression linéaire",
                    "url": _u(5003),
                    "port": 5003,
                    "cmd": 'cd "Prediction avancement/Régression Linéaire/FRONT" && python app.py',
                },
            ],
        },
        "model3": {
            "title": "Recommandation de priorité",
            "subtitle": "Cible Priority (Low / Medium / High) — classification.",
            "frames": [
                {
                    "label": "Random Forest",
                    "url": _u(5000),
                    "port": 5000,
                    "cmd": 'cd "Recommandation priorite/Random Forest/FRONT" && python app.py',
                },
                {
                    "label": "KNN (baseline)",
                    "url": _u(5001),
                    "port": 5001,
                    "cmd": 'cd "Recommandation priorite/KNN/FRONT" && python app.py',
                },
            ],
        },
    }
    return render_template("index.html", cfg=cfg, hub_port=5080)


if __name__ == "__main__":
    port = int(os.environ.get("ML_HUB_PORT", "5080"))
    print("Hub global ML — http://127.0.0.1:%d" % port)
    print("Demarrez les backends (voir hub) : risque 5011-5013 + 5005, avancement 5002-5003, priorite 5000-5001.")
    app.run(debug=False, host="127.0.0.1", port=port, use_reloader=False)
