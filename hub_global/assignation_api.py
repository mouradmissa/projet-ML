"""
Logique API recommandation d'assignation (Random Forest) — partagée par le hub.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_HUB_DIR = Path(__file__).resolve().parent
ML_ROOT = _HUB_DIR.parent
ASSIGNATION_MODEL_DIR = ML_ROOT / "Recommandation assignation" / "Random Forest" / "MODEL"

model = None
label_encoder = None
metadata: dict | None = None
loaded = False
load_error: str | None = None


def load_assignation_models() -> None:
    global model, label_encoder, metadata, loaded, load_error
    paths = (
        ASSIGNATION_MODEL_DIR / "rf_assignation_model.pkl",
        ASSIGNATION_MODEL_DIR / "rf_assignation_label_encoder.pkl",
        ASSIGNATION_MODEL_DIR / "rf_assignation_meta.pkl",
    )
    try:
        if not all(p.is_file() for p in paths):
            load_error = f"Fichiers .pkl attendus dans {ASSIGNATION_MODEL_DIR}"
            return
        with open(paths[0], "rb") as f:
            model = pickle.load(f)
        with open(paths[1], "rb") as f:
            label_encoder = pickle.load(f)
        with open(paths[2], "rb") as f:
            metadata = pickle.load(f)
        loaded = True
        load_error = None
    except Exception as e:
        load_error = str(e)
        loaded = False


def row_from_json(data: dict) -> pd.DataFrame:
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


def top_k_probs(proba_row: np.ndarray, k: int) -> list[dict]:
    assert label_encoder is not None
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


def performance_payload() -> dict | None:
    """Métriques détaillées, matrice, analyse — pour l’API hub."""
    if not loaded or metadata is None:
        return None
    m = metadata
    return {
        "loaded": True,
        "metrics": {
            "accuracy": m.get("accuracy"),
            "f1_macro": m.get("f1_macro"),
            "f1_weighted": m.get("f1_weighted"),
            "precision_macro": m.get("precision_macro"),
            "recall_macro": m.get("recall_macro"),
            "precision_weighted": m.get("precision_weighted"),
            "recall_weighted": m.get("recall_weighted"),
            "hit_at_1": m.get("hit_at_1"),
            "hit_at_3": m.get("hit_at_3"),
            "cv_best_f1_macro": m.get("cv_best_f1_macro"),
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
        },
        "per_class_metrics": m.get("per_class_metrics") or [],
        "confusion_matrix": m.get("confusion_matrix"),
        "labels": m.get("assignee_classes") or [],
        "performance_analysis": m.get("performance_analysis"),
        "classification_report_text": m.get("classification_report_text"),
        "best_params": m.get("best_params"),
        "feature_importance_top": m.get("feature_importance_top") or [],
        "confusion_matrix_png": m.get("confusion_matrix_png"),
        "has_confusion_image": (
            (ASSIGNATION_MODEL_DIR / (m.get("confusion_matrix_png") or "")).is_file()
            if m.get("confusion_matrix_png")
            else False
        ),
    }


def predict_payload(data: dict) -> dict:
    assert model is not None and label_encoder is not None and metadata is not None
    top_k = int(data.get("top_k", 5))
    top_k = max(1, min(top_k, len(label_encoder.classes_)))
    X = row_from_json(data)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_name = str(label_encoder.inverse_transform([pred_idx])[0])
    recommendations = top_k_probs(proba, top_k)
    return {
        "prediction": pred_name,
        "recommendations": recommendations,
        "metrics": {
            "accuracy_train_report": float(metadata.get("accuracy", 0)),
            "f1_macro": float(metadata.get("f1_macro", 0)),
            "f1_weighted": float(metadata.get("f1_weighted", 0)),
            "hit_at_1": float(metadata.get("hit_at_1", 0)),
            "hit_at_3": float(metadata.get("hit_at_3", 0)),
        },
        "model_info": {
            "type": "Random Forest (multi-class)",
            "n_classes": len(label_encoder.classes_),
            "classes": [str(c) for c in label_encoder.classes_],
        },
    }


def examples_json() -> list:
    return [
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


load_assignation_models()
