"""
Flask — recommandation d'assignation (Random Forest, Top-K).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_DIR = Path(__file__).resolve().parent.parent / "MODEL"
MODEL_PATH = MODEL_DIR / "rf_assignation_model.pkl"
ENCODER_PATH = MODEL_DIR / "rf_assignation_label_encoder.pkl"
META_PATH = MODEL_DIR / "rf_assignation_meta.pkl"

app = Flask(__name__)

print("Chargement du modèle d'assignation...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

NUMERIC = metadata["numeric_features"]
CATEGORICAL = metadata["categorical_features"]
CAT_UNIQUES = metadata.get("categorical_uniques") or {}


def _row_from_json(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Progress": float(data["progress"]),
                "Budget": float(data["budget"]),
                "Planned_Duration_Days": float(data["planned_duration_days"]),
                "Hours Spent": float(data["hours_spent"]),
                "Budget_Utilization": float(data["budget_utilization"]),
                "Assignee_Historical_Task_Count": float(
                    data.get("assignee_historical_task_count", 0)
                ),
                "Project Type": str(data["project_type"]).strip(),
                "Location": str(data["location"]).strip(),
                "Project Status": str(data["project_status"]).strip(),
                "Priority": str(data["priority"]).strip(),
                "Task Status": str(data["task_status"]).strip(),
                "Risk_Level": str(data["risk_level"]).strip(),
            }
        ]
    )


def _top_k_probs(proba_row: np.ndarray, k: int) -> list[dict]:
    k = min(k, len(proba_row))
    order = np.argsort(-proba_row)[:k]
    names = label_encoder.inverse_transform(order)
    out = []
    for i, idx in enumerate(order):
        out.append(
            {
                "rank": i + 1,
                "assignee": str(names[i]),
                "probability": float(proba_row[idx]),
            }
        )
    return out


@app.route("/")
def index():
    return render_template(
        "index.html",
        meta=metadata,
        categories=CAT_UNIQUES,
        numeric_cols=NUMERIC,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True) or {}
        top_k = int(data.get("top_k", 5))
        top_k = max(1, min(top_k, len(label_encoder.classes_)))

        X = _row_from_json(data)
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_name = str(label_encoder.inverse_transform([pred_idx])[0])

        recommendations = _top_k_probs(proba, top_k)

        return jsonify(
            {
                "prediction": pred_name,
                "recommendations": recommendations,
                "metrics": {
                    "accuracy_train_report": float(metadata.get("accuracy", 0)),
                    "f1_macro": float(metadata.get("f1_macro", 0)),
                    "hit_at_1": float(metadata.get("hit_at_1", 0)),
                    "hit_at_3": float(metadata.get("hit_at_3", 0)),
                },
                "model_info": {
                    "type": "Random Forest (multi-class)",
                    "n_classes": len(label_encoder.classes_),
                    "classes": [str(c) for c in label_encoder.classes_],
                },
            }
        )
    except KeyError as e:
        return jsonify({"error": f"Champ manquant ou invalide: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify(
        {
            "numeric_features": NUMERIC,
            "categorical_features": CATEGORICAL,
            "categorical_uniques": CAT_UNIQUES,
            "metrics": {
                "accuracy": metadata.get("accuracy"),
                "f1_macro": metadata.get("f1_macro"),
                "hit_at_1": metadata.get("hit_at_1"),
                "hit_at_3": metadata.get("hit_at_3"),
            },
            "best_params": metadata.get("best_params"),
        }
    )


@app.route("/examples", methods=["GET"])
def examples():
    return jsonify(
        [
            {
                "name": "Tâche équilibrée",
                "data": {
                    "progress": 0.5,
                    "budget": 8000,
                    "planned_duration_days": 60,
                    "hours_spent": 20,
                    "budget_utilization": 0.45,
                    "assignee_historical_task_count": 2,
                    "project_type": "Renovation",
                    "location": "Texas",
                    "project_status": "On Track",
                    "priority": "Medium",
                    "task_status": "In Progress",
                    "risk_level": "Low",
                },
            },
            {
                "name": "Projet en retard, risque élevé",
                "data": {
                    "progress": 0.15,
                    "budget": 12000,
                    "planned_duration_days": 90,
                    "hours_spent": 40,
                    "budget_utilization": 0.2,
                    "assignee_historical_task_count": 5,
                    "project_type": "Maintenance",
                    "location": "California",
                    "project_status": "Behind",
                    "priority": "High",
                    "task_status": "Pending",
                    "risk_level": "High",
                },
            },
        ]
    )


if __name__ == "__main__":
    port = int(os.environ.get("ASSIGNATION_RF_PORT", "5020"))
    print(f"Réassignation RF — http://127.0.0.1:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
