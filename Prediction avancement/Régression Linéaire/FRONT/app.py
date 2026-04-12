"""
FRONT — Régression linéaire (Progress). Lancer : python app.py → http://localhost:5003
Modèles : ../MODEL/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


_HERE = Path(__file__).resolve()
FRONTEND_DIR = _HERE.parent
PREDICTION_AVANCEMENT = _HERE.parents[2]
MODEL_DIR = _HERE.parents[1] / "MODEL"

sys.path.insert(0, str(PREDICTION_AVANCEMENT / "shared"))
from progress_inference import build_progress_feature_frame, ml_project_root  # noqa: E402

ROOT = ml_project_root(_HERE)
DATA_PATH = ROOT / "Project-Management-2-enriched.csv"

app = Flask(__name__)

_REQUIRED = (
    MODEL_DIR / "lr_progress_model.pkl",
    MODEL_DIR / "lr_progress_features.pkl",
    MODEL_DIR / "lr_progress_meta.pkl",
)
_missing = [str(p) for p in _REQUIRED if not p.is_file()]
if _missing:
    raise FileNotFoundError(
        "Modele LR Progress introuvable dans MODEL/. Fichiers manquants :\n  - "
        + "\n  - ".join(_missing)
        + "\n\nLancez d'abord :\n  python \""
        + str(MODEL_DIR / "lr_progress_pipeline.py")
        + '"'
    )

pipe = joblib.load(MODEL_DIR / "lr_progress_model.pkl")
feature_cols: list[str] = joblib.load(MODEL_DIR / "lr_progress_features.pkl")
meta: dict = joblib.load(MODEL_DIR / "lr_progress_meta.pkl")

REFERENCE_DATE = pd.Timestamp(meta["REFERENCE_DATE"])
ORDER_PS = list(meta["ORDER_PROJECT_STATUS"])
ORDER_TS = list(meta["ORDER_TASK_STATUS"])


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


@app.route("/accueil_progress")
def accueil_progress():
    return send_from_directory(PREDICTION_AVANCEMENT, "index.html")


@app.route("/meta")
def meta_route():
    return jsonify(_json_safe(dict(meta)))


@app.route("/predict", methods=["POST"])
def predict():
    import traceback

    try:
        data = request.get_json(force=True, silent=True) or {}
        df_in = build_progress_feature_frame(
            data,
            feature_cols,
            REFERENCE_DATE,
            ORDER_PS,
            ORDER_TS,
            DATA_PATH,
        )
        pred = float(pipe.predict(df_in)[0])
        if not np.isfinite(pred):
            pred = 0.0
        pred = float(np.clip(pred, 0.0, 1.0))
        return jsonify(
            {
                "progress": round(pred, 4),
                "progress_percent": round(pred * 100, 2),
                "budget_utilization": round(float(df_in["Budget_Utilization"].iloc[0]), 4),
                "reference_date": str(REFERENCE_DATE.date()),
                "model": "linear_regression",
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "model": "linear_regression"}), 400


if __name__ == "__main__":
    _port = 5003
    print("LR Progress — MODEL_DIR =", MODEL_DIR)
    print("LR Progress — CSV =", DATA_PATH)
    try:
        print("LR Progress — http://127.0.0.1:%d" % _port)
        app.run(debug=False, host="127.0.0.1", port=_port, use_reloader=False)
    except OSError as e:
        winerr = getattr(e, "winerror", None)
        if winerr == 10048 or "address already in use" in str(e).lower():
            _port = 5033
            print("Port 5003 deja utilise — nouvel essai sur http://127.0.0.1:%d" % _port)
            app.run(debug=False, host="127.0.0.1", port=_port, use_reloader=False)
        else:
            raise
