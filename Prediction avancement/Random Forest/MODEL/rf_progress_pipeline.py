"""
Random Forest Regressor — prédiction de Progress (avancement) ∈ [0, 1].
Dataset : Project-Management-2-enriched.csv
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chemins (CSV à la racine du dépôt ML ; artefacts dans ce dossier MODEL)
# ---------------------------------------------------------------------------
import sys

_PREDICTION_AVANCEMENT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PREDICTION_AVANCEMENT / "shared"))
from progress_inference import ml_project_root  # noqa: E402

ROOT = ml_project_root(Path(__file__))
DATA_PATH = ROOT / "Project-Management-2-enriched.csv"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Progress"
# Date de référence fixe (reproductible) pour days_since_start / remaining_days
REFERENCE_DATE = pd.Timestamp("2026-04-12")

# Ordre métier pour encodage ordinal (LabelEncoder manuel)
ORDER_PROJECT_STATUS = ["On Hold", "Behind", "On Track", "Completed"]
ORDER_TASK_STATUS = ["Pending", "In Progress", "Completed"]

COLS_IQR = ["Hours Spent", "Budget", "Actual Cost"]
DROP_LEAKAGE_AND_IDS = [
    "Risk_Level",
    "Project ID",
    "Project Name",
    "Task Name",
    "Location",
]


def cap_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return series
    low, high = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=low, upper=high)


# =========================================================
# ETAPE 1 : CHARGEMENT ET INSPECTION
# =========================================================
print("=" * 60)
print("  ETAPE 1 : CHARGEMENT ET INSPECTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
n_init = len(df)
cols_init = df.shape[1]

print(df.info())
print()
print(df.head())
print()
print(df.describe())
print()
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())
print()
print("Nombre de doublons :", df.duplicated().sum())
print()

# =========================================================
# ETAPE 2 : NETTOYAGE
# =========================================================
print("=" * 60)
print("  ETAPE 2 : NETTOYAGE")
print("=" * 60)

df = df.drop_duplicates()
n_after_dup = len(df)
print("Lignes apres suppression des doublons :", n_after_dup, "(supprime : %d)" % (n_init - n_after_dup))

# Valeurs manquantes : <5% des lignes -> suppression des lignes concernees ; sinon imputation
n = len(df)
rows_with_na = df.isnull().any(axis=1).sum()
pct_na_rows = 100.0 * rows_with_na / n if n else 0.0
print("Lignes avec au moins un NaN : %d (%.2f%%)" % (rows_with_na, pct_na_rows))

if rows_with_na > 0:
    if pct_na_rows < 5.0:
        df = df.dropna()
        print("  -> Suppression des lignes avec NaN (< 5%%)")
    else:
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(include=["object"]).columns:
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "")
        print("  -> Imputation mediane (numerique) / mode (categoriel)")
else:
    print("  -> Aucune valeur manquante")

# Colonnes constantes
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    print("Colonnes constantes supprimees :", const_cols)
else:
    print("Colonnes constantes : aucune")

# Outliers IQR sur colonnes numeriques cles (plafonnement)
for col in COLS_IQR:
    if col in df.columns:
        before = df[col].copy()
        df[col] = cap_iqr(df[col])
        n_capped = (before != df[col]).sum()
        print("IQR plafonnement %s : %d valeurs ajustees" % (col, n_capped))

# Dates -> datetime (format JJ/MM/AAAA observe dans le CSV)
for c in ["Start Date", "End Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

# planned_duration_days : calcul depuis les dates ; on privilegie le calcul apres parsing
if "Start Date" in df.columns and "End Date" in df.columns:
    df["planned_duration_days"] = (df["End Date"] - df["Start Date"]).dt.days
    df["days_since_start"] = (REFERENCE_DATE - df["Start Date"]).dt.days
    df["remaining_days"] = (df["End Date"] - REFERENCE_DATE).dt.days
    df = df.drop(columns=["Start Date", "End Date"])
    if "Planned_Duration_Days" in df.columns:
        df = df.drop(columns=["Planned_Duration_Days"])

# Retirer colonnes fuite / identifiants / texte libre, puis forte cardinalite (>10)
to_drop = [c for c in DROP_LEAKAGE_AND_IDS if c in df.columns]
df = df.drop(columns=to_drop, errors="ignore")
if to_drop:
    print("Colonnes supprimees (fuites / IDs / texte) :", to_drop)

high_card = []
for c in df.select_dtypes(include=["object"]).columns:
    if c == TARGET:
        continue
    if df[c].nunique(dropna=False) > 10:
        high_card.append(c)
if high_card:
    df = df.drop(columns=high_card, errors="ignore")
    print("Colonnes > 10 modalites supprimees :", high_card)
else:
    print("Colonnes categorielles > 10 modalites : aucune (apres retrait IDs)")

n_after_clean = len(df)
print("Shape apres nettoyage (avant encodage) :", df.shape)
print()

# =========================================================
# ETAPE 3 : ENCODAGE DES VARIABLES CATEGORIELLES
# =========================================================
print("=" * 60)
print("  ETAPE 3 : ENCODAGE")
print("=" * 60)

if "Project Status" in df.columns:
    ps_map = {v: i for i, v in enumerate(ORDER_PROJECT_STATUS)}
    unk = len(ORDER_PROJECT_STATUS)
    df["Project_Status_ord"] = df["Project Status"].map(ps_map).fillna(unk).astype(int)
    df = df.drop(columns=["Project Status"])
    print("Project Status -> ordinal", ps_map, "| inconnu ->", unk)

if "Task Status" in df.columns:
    ts_map = {v: i for i, v in enumerate(ORDER_TASK_STATUS)}
    unk = len(ORDER_TASK_STATUS)
    df["Task_Status_ord"] = df["Task Status"].map(ts_map).fillna(unk).astype(int)
    df = df.drop(columns=["Task Status"])
    print("Task Status -> ordinal", ts_map, "| inconnu ->", unk)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print("One-Hot (drop_first=True), colonnes :", cat_cols)
    for c in cat_cols:
        print("  %s : %d modalites" % (c, df[c].nunique(dropna=False)))
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
else:
    print("Aucune categorielle nominale restante pour get_dummies")

# Cible : verifier [0, 1]
if TARGET not in df.columns:
    raise ValueError("Colonne cible '%s' absente apres preprocessing." % TARGET)

y_raw = df[TARGET].astype(float)
if y_raw.min() < 0 or y_raw.max() > 1:
    print("Attention : Progress hors [0,1] -> normalisation min-max sur la cible")
    ymin, ymax = y_raw.min(), y_raw.max()
    df[TARGET] = (y_raw - ymin) / (ymax - ymin) if ymax > ymin else y_raw.clip(0, 1)
else:
    df[TARGET] = y_raw.clip(0, 1)

print("Scaling des features : NON (Random Forest ; optionnel pour comparaison avec d'autres modeles)")
print("Shape apres encodage :", df.shape)
print()

# =========================================================
# ETAPE 4 : SEPARATION X / y
# =========================================================
print("=" * 60)
print("  ETAPE 4 : SEPARATION X / y")
print("=" * 60)

X = df.drop(columns=[TARGET])
y = df[TARGET]
feature_names_final = list(X.columns)
print("X :", X.shape, "| y :", y.shape)
print()

# =========================================================
# ETAPE 5 : DIVISION TRAIN / TEST
# =========================================================
print("=" * 60)
print("  ETAPE 5 : TRAIN / TEST (test_size=0.2, random_state=42)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train : %d | Test : %d" % (len(y_train), len(y_test)))
print()

# =========================================================
# ETAPE 6 : MODELE DE BASE
# =========================================================
print("=" * 60)
print("  ETAPE 6 : RANDOM FOREST (baseline n_estimators=100)")
print("=" * 60)

rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
rmse_base = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))
print("Baseline RMSE (test) : %.6f" % rmse_base)
print()

# =========================================================
# ETAPE 7 : GRID SEARCH (min RMSE en CV 3-fold sur train)
# =========================================================
print("=" * 60)
print("  ETAPE 7 : GRID SEARCH CV (scoring = RMSE negatif, cv=3)")
print("=" * 60)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_cv_rmse = -grid.best_score_
print("Meilleurs hyperparametres :", best_params)
print("RMSE moyenne validation croisee (train, 3-fold) : %.6f" % best_cv_rmse)
print()

# =========================================================
# ETAPE 8 : EVALUATION TEST
# =========================================================
print("=" * 60)
print("  ETAPE 8 : EVALUATION (jeu test)")
print("=" * 60)

model = grid.best_estimator_
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)

print("R2 score : %.6f" % r2)
print("RMSE     : %.6f" % rmse)
print("MAE      : %.6f" % mae)
print()

# =========================================================
# ETAPE 9 : IMPORTANCE DES VARIABLES (top 10)
# =========================================================
print("=" * 60)
print("  ETAPE 9 : IMPORTANCE DES VARIABLES (top 10)")
print("=" * 60)

importances = model.feature_importances_
names = X.columns.tolist()
ranked = sorted(zip(importances, names), key=lambda t: t[0], reverse=True)
for imp, name in ranked[:10]:
    print("  %.4f  %s" % (imp, name))
print()
top3 = [(n, round(float(i), 4)) for i, n in ranked[:3]]
print("Top 3 caracteristiques :", top3)
print()

# Graphique
fig, ax = plt.subplots(figsize=(9, 5))
top10 = ranked[:10][::-1]
ax.barh([n for _, n in top10], [i for i, _ in top10], color="steelblue", edgecolor="black")
ax.set_xlabel("Importance")
ax.set_title("Random Forest (régression Progress) — Top 10 des importances")
plt.tight_layout()
fig_path = MODEL_DIR / "rf_progress_feature_importance_top10.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("Figure enregistree :", fig_path)
print()

# =========================================================
# RESUME FINAL
# =========================================================
print("=" * 60)
print("         RESUME FINAL")
print("=" * 60)
print("- Lignes initiales : %d  |  Apres nettoyage (lignes) : %d" % (n_init, n_after_clean))
print("- Colonnes initiales : %d  |  Features finales : %d" % (cols_init, X.shape[1]))
print("- Features conservees :")
for f in feature_names_final:
    print("    %s" % f)
print("- Meilleurs hyperparametres : %s" % best_params)
print("- Performances test — R2: %.6f | RMSE: %.6f | MAE: %.6f" % (r2, rmse, mae))
print("- Top 3 importances : %s" % top3)
print("- Random Forest : non-linearites / interactions ; scaling non requis.")
print("=" * 60)

# Sauvegarde
joblib.dump(model, MODEL_DIR / "rf_progress_model.pkl")
joblib.dump(feature_names_final, MODEL_DIR / "rf_progress_features.pkl")
joblib.dump(
    {
        "ORDER_PROJECT_STATUS": ORDER_PROJECT_STATUS,
        "ORDER_TASK_STATUS": ORDER_TASK_STATUS,
        "REFERENCE_DATE": str(REFERENCE_DATE),
        "TARGET": TARGET,
        "r2_test": float(r2),
        "rmse_test": float(rmse),
        "mae_test": float(mae),
        "best_params": {k: (None if v is None else v) for k, v in best_params.items()},
        "n_samples_train": int(len(y_train)),
        "n_samples_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
    },
    MODEL_DIR / "rf_progress_meta.pkl",
)

print()
print("Fichiers sauvegardes dans :", MODEL_DIR)
print("  - rf_progress_model.pkl")
print("  - rf_progress_features.pkl")
print("  - rf_progress_meta.pkl")
