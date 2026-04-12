"""Chargement du Pipeline LR + test de forme."""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

pipe = joblib.load(MODEL_DIR / "lr_progress_model.pkl")
features = joblib.load(MODEL_DIR / "lr_progress_features.pkl")

X0 = pd.DataFrame([[0.0] * len(features)], columns=features)
pred = float(pipe.predict(X0)[0])
print("Pipeline LR Progress charge. Prediction (zeros) : %.6f" % pred)
print("OK")
