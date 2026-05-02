"""Test rapide du modèle sauvegardé (charge + une inférence factice)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent


def main() -> None:
    with open(MODEL_DIR / "rf_assignation_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "rf_assignation_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODEL_DIR / "rf_assignation_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    row = {
        "Progress": 0.45,
        "Budget": 5000.0,
        "Planned_Duration_Days": 60.0,
        "Hours Spent": 20.0,
        "Budget_Utilization": 0.3,
        "Assignee_Historical_Task_Count": 2.0,
        "Project Type": "Renovation",
        "Location": "Texas",
        "Project Status": "On Track",
        "Priority": "Medium",
        "Task Status": "In Progress",
        "Risk_Level": "Low",
    }
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[0]
    order = proba.argsort()[::-1][:5]
    names = le.inverse_transform(order)
    print("Classes:", list(le.classes_))
    print("Top-5 recommandations (nom, proba):")
    for i, idx in enumerate(order):
        print(f"  {i+1}. {names[i]}: {proba[idx]:.4f}")
    print("\nMeta accuracy:", meta.get("accuracy"), "Hit@3:", meta.get("hit_at_3"))


if __name__ == "__main__":
    main()
