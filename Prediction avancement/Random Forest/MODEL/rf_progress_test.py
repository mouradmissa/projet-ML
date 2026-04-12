"""Verification rapide : chargement du modele et prediction sur des zeros (forme des features)."""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

model = joblib.load(MODEL_DIR / "rf_progress_model.pkl")
feature_names = joblib.load(MODEL_DIR / "rf_progress_features.pkl")
meta = joblib.load(MODEL_DIR / "rf_progress_meta.pkl")

n = len(feature_names)
assert n == model.n_features_in_, "Nombre de features incoherent avec le modele."

X0 = pd.DataFrame([[0] * n], columns=feature_names)
pred = model.predict(X0)

print("Modele RF Progress charge.")
print("Meta :", meta)
print("Features (%d) : %s ..." % (n, feature_names[:5]))
print("Prediction (vecteur nul, test de forme) : %.6f" % float(pred[0]))
print("OK")
