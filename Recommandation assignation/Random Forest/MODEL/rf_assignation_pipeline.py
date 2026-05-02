"""
Random Forest — recommandation d'assignation de tâches (cible : Assigned To).
Top-K via predict_proba. Pipeline : features numériques + catégorielles (OneHotEncoder).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parent
ML_ROOT = MODEL_DIR.parent.parent.parent
DATA_CANDIDATES = [
    ML_ROOT / "Project-Management-2-enriched.csv",
    ML_ROOT / "Project-Management-2.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 2  # 3-fold plus lent ; 2 suffit pour un jeu ~300 lignes

NUMERIC_FEATURES = [
    "Progress",
    "Budget",
    "Planned_Duration_Days",
    "Hours Spent",
    "Budget_Utilization",
    "Assignee_Historical_Task_Count",
]
CATEGORICAL_FEATURES = [
    "Project Type",
    "Location",
    "Project Status",
    "Priority",
    "Task Status",
    "Risk_Level",
]

print("=" * 80)
print("RANDOM FOREST — RECOMMANDATION D'ASSIGNATION (Assigned To)")
print("=" * 80)

data_path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError(
        "CSV introuvable. Placez Project-Management-2-enriched.csv à la racine du dossier ML."
    )

df = pd.read_csv(data_path)
print(f"\nDataset: {len(df)} lignes, {len(df.columns)} colonnes")

df_clean = df.drop_duplicates().copy()
print(f"Après déduplication: {len(df_clean)} lignes")

for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
    if c not in df_clean.columns:
        raise KeyError(f"Colonne manquante: {c}")
if "Assigned To" not in df_clean.columns:
    raise KeyError("Colonne cible 'Assigned To' manquante.")

# Nettoyage basique des catégorielles
for c in CATEGORICAL_FEATURES:
    df_clean[c] = df_clean[c].astype(str).str.strip()

y_raw = df_clean["Assigned To"].astype(str).str.strip()
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)
print(f"\nAssignés (classes): {class_names}")
print(pd.Series(y_raw).value_counts())

X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
    ]
)

base_pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
        ),
    ]
)

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [12, None],
    "clf__min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring="f1_macro",
    n_jobs=1,
    verbose=1,
)

print("\nGridSearchCV (f1_macro)...")
grid.fit(X_train, y_train)
best: Pipeline = grid.best_estimator_

y_pred = best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
prec_macro, rec_macro, f1_macro_prf, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
prec_w, rec_w, f1_w_prf, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

print(f"\nMeilleurs params: {grid.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 macro: {f1_macro:.4f}")
print(f"F1 pondéré: {f1_weighted:.4f}")
print("\n--- Rapport ---\n")
report_str = classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0
)
print(report_str)
report_dict = classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
)

# Top-K (recommandation)
proba = best.predict_proba(X_test)
k_top = 3
hits = 0
hits1 = 0
for i in range(len(y_test)):
    true_idx = y_test[i]
    order = np.argsort(-proba[i])
    if order[0] == true_idx:
        hits1 += 1
    if true_idx in order[:k_top]:
        hits += 1
hit_at_1 = hits1 / len(y_test)
hit_at_k = hits / len(y_test)
print(f"\nHit@1 (assigné réel = meilleur score): {hit_at_1:.4f}")
print(f"Hit@{k_top} (assigné réel dans le top-{k_top}): {hit_at_k:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion (indices = ordre des classes du LabelEncoder):")
print(cm)

# Métriques par classe (liste sérialisable)
per_class_metrics = []
for name in class_names:
    if name not in report_dict:
        continue
    row = report_dict[name]
    per_class_metrics.append(
        {
            "assignee": name,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }
    )

CM_PNG = MODEL_DIR / "rf_assignation_confusion_matrix.png"
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
)
plt.ylabel("Vraie classe (assigné)")
plt.xlabel("Classe prédite")
plt.title("Matrice de confusion — assignation (jeu de test)")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=150)
plt.close()
print(f"\nFigure sauvegardée: {CM_PNG.name}")

analysis_lines = [
    f"Jeu de test : {len(y_test)} tâches, {len(class_names)} assignés.",
    f"Accuracy globale : {accuracy:.1%} — difficile avec des classes équilibrées mais des frontières floues entre profils.",
    f"F1 macro : {f1_macro:.3f} (moyenne non pondérée par classe) ; F1 pondéré : {f1_weighted:.3f}.",
    f"Pour la recommandation Top-K : Hit@1 = {hit_at_1:.1%}, Hit@{k_top} = {hit_at_k:.1%} (assigné réel souvent dans les {k_top} premières propositions).",
    "Interprétation : les confusions hors diagonale indiquent des tâches aux caractéristiques proches entre assignés.",
]
performance_analysis = "\n".join(analysis_lines)

# Importances (noms après preprocessing)
try:
    feat_names = best.named_steps["prep"].get_feature_names_out()
    imp = best.named_steps["clf"].feature_importances_
    importance_records = (
        pd.DataFrame({"Feature": feat_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .head(25)
        .to_dict("records")
    )
except Exception:
    importance_records = []

cat_uniques = {c: sorted(df_clean[c].unique().tolist()) for c in CATEGORICAL_FEATURES}

metadata = {
    "assignee_classes": class_names,
    "best_params": grid.best_params_,
    "cv_best_f1_macro": float(grid.best_score_),
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted),
    "precision_macro": float(prec_macro),
    "recall_macro": float(rec_macro),
    "precision_weighted": float(prec_w),
    "recall_weighted": float(rec_w),
    "hit_at_1": float(hit_at_1),
    "hit_at_3": float(hit_at_k),
    "top_k": k_top,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_png": str(CM_PNG.name),
    "per_class_metrics": per_class_metrics,
    "classification_report_text": report_str,
    "performance_analysis": performance_analysis,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "categorical_uniques": cat_uniques,
    "feature_importance_top": importance_records,
    "data_path": str(data_path),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}

with open(MODEL_DIR / "rf_assignation_model.pkl", "wb") as f:
    pickle.dump(best, f)
with open(MODEL_DIR / "rf_assignation_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open(MODEL_DIR / "rf_assignation_meta.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\nFichiers écrits:")
print(f"  {MODEL_DIR / 'rf_assignation_model.pkl'}")
print(f"  {MODEL_DIR / 'rf_assignation_label_encoder.pkl'}")
print(f"  {MODEL_DIR / 'rf_assignation_meta.pkl'}")
print("\nPipeline terminé.")
