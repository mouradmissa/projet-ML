"""
Hub global — modèles ML gestion de projet (interfaces Flask).

Lancer depuis ce dossier :
  python app.py

Puis ouvrir http://127.0.0.1:5080

L’assignation (Random Forest) est intégrée au hub : pas de serveur séparé pour l’onglet 4
(prérequis : entraîner le modèle et fichiers .pkl dans Recommandation assignation/.../MODEL/).
Les autres modèles utilisent toujours leurs ports dédiés (voir page).
"""
from __future__ import annotations

import os

import assignation_api
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

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
        "model4": {
            "title": "Recommandation d'assignation",
            "subtitle": "Random Forest multi-classes sur Assigned To — classement Top‑K (probabilités), servi par ce hub.",
            "label": "RF — assignation (intégré)",
        },
    }
    cats = {}
    if assignation_api.metadata:
        cats = assignation_api.metadata.get("categorical_uniques") or {}
    return render_template(
        "index.html",
        cfg=cfg,
        hub_port=5080,
        assignation_ready=assignation_api.loaded,
        assignation_error=assignation_api.load_error,
        assignation_categories=cats,
    )


@app.route("/api/assignation/predict", methods=["POST"])
def assignation_predict():
    if not assignation_api.loaded:
        return (
            jsonify(
                {
                    "error": assignation_api.load_error
                    or "Modèle d'assignation non chargé. Exécutez rf_assignation_pipeline.py."
                }
            ),
            503,
        )
    try:
        data = request.get_json(force=True, silent=True) or {}
        return jsonify(assignation_api.predict_payload(data))
    except KeyError as e:
        return jsonify({"error": f"Champ manquant ou invalide: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assignation/model_info", methods=["GET"])
def assignation_model_info():
    if not assignation_api.loaded or not assignation_api.metadata:
        return (
            jsonify(
                {
                    "error": assignation_api.load_error or "Modèle non chargé",
                    "loaded": False,
                }
            ),
            503,
        )
    m = assignation_api.metadata
    return jsonify(
        {
            "loaded": True,
            "numeric_features": m.get("numeric_features"),
            "categorical_features": m.get("categorical_features"),
            "categorical_uniques": m.get("categorical_uniques"),
            "metrics": {
                "accuracy": m.get("accuracy"),
                "f1_macro": m.get("f1_macro"),
                "f1_weighted": m.get("f1_weighted"),
                "precision_macro": m.get("precision_macro"),
                "recall_macro": m.get("recall_macro"),
                "hit_at_1": m.get("hit_at_1"),
                "hit_at_3": m.get("hit_at_3"),
                "n_train": m.get("n_train"),
                "n_test": m.get("n_test"),
                "cv_best_f1_macro": m.get("cv_best_f1_macro"),
            },
            "best_params": m.get("best_params"),
        }
    )


@app.route("/api/assignation/performance", methods=["GET"])
def assignation_performance():
    if not assignation_api.loaded:
        return (
            jsonify(
                {
                    "error": assignation_api.load_error
                    or "Modèle non chargé",
                    "loaded": False,
                }
            ),
            503,
        )
    payload = assignation_api.performance_payload()
    if payload is None:
        return jsonify({"error": "Métadonnées absentes", "loaded": False}), 503
    return jsonify(payload)


@app.route("/api/assignation/assets/<path:filename>")
def assignation_asset(filename):
    allowed = {"rf_assignation_confusion_matrix.png"}
    if filename not in allowed:
        abort(404)
    path = assignation_api.ASSIGNATION_MODEL_DIR / filename
    if not path.is_file():
        abort(404)
    return send_from_directory(
        str(assignation_api.ASSIGNATION_MODEL_DIR), filename, mimetype="image/png"
    )


@app.route("/api/assignation/examples", methods=["GET"])
def assignation_examples():
    return jsonify(assignation_api.examples_json())


if __name__ == "__main__":
    port = int(os.environ.get("ML_HUB_PORT", "5080"))
    print("Hub global ML — http://127.0.0.1:%d" % port)
    if assignation_api.loaded:
        print("Assignation RF : chargee (onglet 4).")
    else:
        print("Assignation RF : non chargee —", assignation_api.load_error or "?")
    print(
        "Autres backends : risque 5011-5013 + 5005, avancement 5002-5003, priorite 5000-5001."
    )
    app.run(debug=False, host="127.0.0.1", port=port, use_reloader=False)
