"""
Régression linéaire (baseline) — prédiction de Progress ∈ [0, 1].
Même prétraitement que rf_progress_pipeline + StandardScaler (obligatoire) + LinearRegression.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys

_PREDICTION_AVANCEMENT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PREDICTION_AVANCEMENT / "shared"))
from progress_inference import ml_project_root  # noqa: E402

ROOT = ml_project_root(Path(__file__))
DATA_PATH = ROOT / "Project-Management-2-enriched.csv"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Progress"
REFERENCE_DATE = pd.Timestamp("2026-04-12")
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
# ETAPE 1 : CHARGEMENT ET INSPECTION (identique RF)
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
# ETAPE 2 : NETTOYAGE (aligné RF + imputation si besoin)
# =========================================================
print("=" * 60)
print("  ETAPE 2 : NETTOYAGE")
print("=" * 60)

df = df.drop_duplicates()
print("Lignes apres suppression des doublons :", len(df))

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

const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    print("Colonnes constantes supprimees :", const_cols)
else:
    print("Colonnes constantes : aucune")

for col in COLS_IQR:
    if col in df.columns:
        before = df[col].copy()
        df[col] = cap_iqr(df[col])
        print("IQR plafonnement %s : %d valeurs ajustees" % (col, (before != df[col]).sum()))

for c in ["Start Date", "End Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

if "Start Date" in df.columns and "End Date" in df.columns:
    df["planned_duration_days"] = (df["End Date"] - df["Start Date"]).dt.days
    df["days_since_start"] = (REFERENCE_DATE - df["Start Date"]).dt.days
    df["remaining_days"] = (df["End Date"] - REFERENCE_DATE).dt.days
    df = df.drop(columns=["Start Date", "End Date"])
    if "Planned_Duration_Days" in df.columns:
        df = df.drop(columns=["Planned_Duration_Days"])

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
# ETAPE 3 : ENCODAGE (identique RF)
# =========================================================
print("=" * 60)
print("  ETAPE 3 : ENCODAGE")
print("=" * 60)

if "Project Status" in df.columns:
    ps_map = {v: i for i, v in enumerate(ORDER_PROJECT_STATUS)}
    unk = len(ORDER_PROJECT_STATUS)
    df["Project_Status_ord"] = df["Project Status"].map(ps_map).fillna(unk).astype(int)
    df = df.drop(columns=["Project Status"])

if "Task Status" in df.columns:
    ts_map = {v: i for i, v in enumerate(ORDER_TASK_STATUS)}
    unk = len(ORDER_TASK_STATUS)
    df["Task_Status_ord"] = df["Task Status"].map(ts_map).fillna(unk).astype(int)
    df = df.drop(columns=["Task Status"])

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

if TARGET not in df.columns:
    raise ValueError("Colonne cible '%s' absente." % TARGET)

y_raw = df[TARGET].astype(float)
if y_raw.min() < 0 or y_raw.max() > 1:
    ymin, ymax = y_raw.min(), y_raw.max()
    df[TARGET] = (y_raw - ymin) / (ymax - ymin) if ymax > ymin else y_raw.clip(0, 1)
else:
    df[TARGET] = y_raw.clip(0, 1)

print("Shape apres encodage :", df.shape)
print()

# =========================================================
# ETAPE 4 : X / y puis TRAIN / TEST (avant scaling)
# =========================================================
print("=" * 60)
print("  ETAPE 4-5 : X/y + TRAIN/TEST (test_size=0.2, random_state=42)")
print("=" * 60)

X = df.drop(columns=[TARGET])
y = df[TARGET]
feature_names_final = list(X.columns)
print("X :", X.shape, "| y :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train : %d | Test : %d" % (len(y_train), len(y_test)))
print()

# =========================================================
# ETAPE 6 : StandardScaler + LinearRegression (Pipeline)
# =========================================================
print("=" * 60)
print("  ETAPE 6 : StandardScaler + LinearRegression")
print("=" * 60)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ]
)
pipe.fit(X_train, y_train)

lr_step = pipe.named_steps["lr"]
print("Intercept : %.8f" % float(lr_step.intercept_))
print()
print("Coefficients (features dans l'ordre des colonnes) :")
for name, c in zip(feature_names_final, lr_step.coef_):
    print("  %+12.6f  %s" % (float(c), name))
print()

# =========================================================
# ETAPE 7 : EVALUATION TEST (apres scaling interne au pipeline)
# =========================================================
print("=" * 60)
print("  ETAPE 7 : EVALUATION (jeu test)")
print("=" * 60)

y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)

print("R2 score : %.6f" % r2)
print("RMSE     : %.6f" % rmse)
print("MAE      : %.6f" % mae)
print()

# =========================================================
# Comparaison Random Forest (si meta RF presente)
# =========================================================
rf_meta_path = _PREDICTION_AVANCEMENT / "Random Forest" / "MODEL" / "rf_progress_meta.pkl"
rf_r2 = rf_rmse = rf_mae = None
if rf_meta_path.is_file():
    rf_m = joblib.load(rf_meta_path)
    rf_r2 = rf_m.get("r2_test")
    rf_rmse = rf_m.get("rmse_test")
    rf_mae = rf_m.get("mae_test")

print("=" * 60)
print("  COMPARAISON AVEC RANDOM FOREST (meme split, memes features)")
print("=" * 60)
if rf_r2 is not None:
    print("              R2        RMSE       MAE")
    print("  Lineaire    %.6f  %.6f  %.6f" % (r2, rmse, mae))
    print("  RF (opt.)   %.6f  %.6f  %.6f" % (rf_r2, rf_rmse, rf_mae))
    print()
    print("Conclusion attendue : la baseline lineaire est en general moins bonne")
    print("(R2 plus faible, RMSE plus elevee) car Progress depend souvent de relations")
    print("non lineaires ; le RF capture mieux ces effets.")
else:
    print("Fichier rf_progress_meta.pkl introuvable — lancer Random Forest/MODEL/rf_progress_pipeline.py avant.")
print("=" * 60)
print()

# =========================================================
# RESUME FINAL
# =========================================================
print("         RESUME FINAL — REGRESSION LINEAIRE")
print("- Lignes apres nettoyage : %d" % n_after_clean)
print("- Variables (features) : %d" % len(feature_names_final))
for f in feature_names_final:
    print("    %s" % f)
print("- Performances test — R2: %.6f | RMSE: %.6f | MAE: %.6f" % (r2, rmse, mae))
print()

coef_list = [
    {"feature": str(n), "coef": float(c)} for n, c in zip(feature_names_final, lr_step.coef_)
]
coef_sorted = sorted(coef_list, key=lambda d: abs(d["coef"]), reverse=True)

joblib.dump(pipe, MODEL_DIR / "lr_progress_model.pkl")
joblib.dump(feature_names_final, MODEL_DIR / "lr_progress_features.pkl")
joblib.dump(
    {
        "ORDER_PROJECT_STATUS": ORDER_PROJECT_STATUS,
        "ORDER_TASK_STATUS": ORDER_TASK_STATUS,
        "REFERENCE_DATE": str(REFERENCE_DATE),
        "TARGET": TARGET,
        "r2_test": float(r2),
        "rmse_test": float(rmse),
        "mae_test": float(mae),
        "intercept": float(lr_step.intercept_),
        "coefficients": coef_list,
        "top_coef_abs": coef_sorted[:10],
        "n_samples_train": int(len(y_train)),
        "n_samples_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "rf_r2_test": float(rf_r2) if rf_r2 is not None else None,
        "rf_rmse_test": float(rf_rmse) if rf_rmse is not None else None,
        "rf_mae_test": float(rf_mae) if rf_mae is not None else None,
    },
    MODEL_DIR / "lr_progress_meta.pkl",
)

print("Fichiers sauvegardes : lr_progress_model.pkl, lr_progress_features.pkl, lr_progress_meta.pkl")
