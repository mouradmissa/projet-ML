import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

ROOT = r"C:\Users\MSI\Desktop\ML"
MODEL_DIR = os.path.join(ROOT, "random_forest", "model")
TARGET = "Risk_Level"

# =========================================================
# ETAPE 1 : CHARGEMENT ET INSPECTION
# =========================================================
print("=" * 60)
print("  ETAPE 1 : CHARGEMENT ET INSPECTION")
print("=" * 60)

df = pd.read_csv(os.path.join(ROOT, "Project-Management-2-enriched.csv"))
n_init = len(df)
cols_init = df.shape[1]

print(df.info())
print()
print(df.head())
print()
print(df.describe())
print()
print("Valeurs manquantes :")
print(df.isnull().sum())
print()
print("Doublons :", df.duplicated().sum())
print()
print("Cible : %s" % TARGET)
print(df[TARGET].value_counts())
print()

# =========================================================
# ETAPE 2 : NETTOYAGE
# =========================================================
print("=" * 60)
print("  ETAPE 2 : NETTOYAGE")
print("=" * 60)

# 2a. Doublons
df.drop_duplicates(inplace=True)
n_after_dup = len(df)
print("Doublons supprimes :", n_init - n_after_dup)

# 2b. Valeurs manquantes
pct_na = df.isnull().any(axis=1).sum() / len(df) * 100
print("Lignes avec NaN : %.2f%%" % pct_na)
if pct_na > 0 and pct_na < 5:
    df.dropna(inplace=True)
    print("  -> Lignes NaN supprimees (<5%%)")
elif pct_na >= 5:
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include="object").columns:
        df[c].fillna(df[c].mode()[0], inplace=True)
    print("  -> Imputation mediane/mode")
else:
    print("  -> Aucune valeur manquante")

# 2c. Colonnes constantes
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
if const_cols:
    df.drop(columns=const_cols, inplace=True)
    print("Colonnes constantes supprimees :", const_cols)
else:
    print("Colonnes constantes : aucune")

# 2d. Colonnes categorielles > 10 modalites
cols_sup_10 = []
for c in df.select_dtypes(include="object").columns:
    if c == TARGET:
        continue
    if df[c].nunique() > 10:
        cols_sup_10.append(c)
df.drop(columns=cols_sup_10, inplace=True)
print("Colonnes >10 modalites supprimees :", cols_sup_10)

# 2e. Colonnes correlees > 0.95
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[num_cols].corr().abs()
cols_corr_drop = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        if corr_matrix.iloc[i, j] > 0.95:
            c1, c2 = num_cols[i], num_cols[j]
            drop_col = c1
            if drop_col not in cols_corr_drop:
                cols_corr_drop.append(drop_col)
                print("  Paire >0.95 : %s vs %s -> suppression %s" % (c1, c2, drop_col))
df.drop(columns=cols_corr_drop, inplace=True)
if not cols_corr_drop:
    print("Paires correlees >0.95 : aucune")

print("Shape apres nettoyage :", df.shape)
print()

# =========================================================
# ETAPE 3 : ENCODAGE
# =========================================================
print("=" * 60)
print("  ETAPE 3 : ENCODAGE")
print("=" * 60)

# 3a. Cible : LabelEncoder
le_target = LabelEncoder()
le_target.fit(["High", "Low", "Medium"])
df[TARGET] = le_target.transform(df[TARGET])
mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print("Encodage cible :", mapping)
print("Distribution cible :", dict(df[TARGET].value_counts().sort_index()))

# 3b. One-Hot pour categorielles restantes
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Colonnes One-Hot :", cat_cols)
for c in cat_cols:
    print("  %s (%d modalites)" % (c, df[c].nunique()))
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# PAS DE SCALING pour Random Forest
print("Scaling : NON (Random Forest n'en a pas besoin)")
print("Shape apres encodage :", df.shape)
print()

# =========================================================
# ETAPE 4 : SEPARATION FEATURES / TARGET
# =========================================================
print("=" * 60)
print("  ETAPE 4 : SEPARATION X / y")
print("=" * 60)

X = df.drop(columns=[TARGET])
y = df[TARGET]
print("X :", X.shape, "| y :", y.shape)
print()

# =========================================================
# ETAPE 5 : DIVISION TRAIN / TEST
# =========================================================
print("=" * 60)
print("  ETAPE 5 : SPLIT TRAIN / TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train : %d | Test : %d" % (len(y_train), len(y_test)))
print("y_train :", dict(pd.Series(y_train).value_counts().sort_index()))
print("y_test  :", dict(pd.Series(y_test).value_counts().sort_index()))
print()

# =========================================================
# ETAPE 6 : ENTRAINEMENT RANDOM FOREST (baseline)
# =========================================================
print("=" * 60)
print("  ETAPE 6 : ENTRAINEMENT RANDOM FOREST (baseline)")
print("=" * 60)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
print("Baseline (n_estimators=100) -> Accuracy = %.4f" % acc_base)
print()

# =========================================================
# ETAPE 7 : OPTIMISATION HYPERPARAMETRES (GridSearchCV)
# =========================================================
print("=" * 60)
print("  ETAPE 7 : OPTIMISATION HYPERPARAMETRES")
print("=" * 60)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_cv_score = grid.best_score_
print("Meilleurs hyperparametres :", best_params)
print("Meilleure accuracy CV : %.4f" % best_cv_score)
print()

# =========================================================
# ETAPE 8 : EVALUATION MODELE FINAL
# =========================================================
print("=" * 60)
print("  ETAPE 8 : EVALUATION")
print("=" * 60)

model = grid.best_estimator_
y_pred = model.predict(X_test)
acc_final = accuracy_score(y_test, y_pred)

print("Accuracy : %.4f" % acc_final)
print()
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))

# Importances des variables
importances = model.feature_importances_
feature_names = X.columns.tolist()
top_features = sorted(zip(importances, feature_names), reverse=True)

print("=== Importances des variables (top 10) ===")
for imp, name in top_features[:10]:
    print("  %.4f  %s" % (imp, name))
print()
print("Top 3 features :", [(name, round(imp, 4)) for imp, name in top_features[:3]])
print()

# =========================================================
# ETAPE 9 : RESUME STRUCTURE
# =========================================================
print("=" * 60)
print("         RESUME DES ACTIONS")
print("=" * 60)
print("- Lignes initiales : %d  |  Apres nettoyage : %d" % (n_init, len(X)))
print("- Colonnes initiales : %d  |  Features finales : %d" % (cols_init, X.shape[1]))
print("- Colonnes supprimees :")
for c in cols_sup_10:
    print("    %s (>10 modalites)" % c)
for c in cols_corr_drop:
    print("    %s (correlation >0.95)" % c)
for c in const_cols:
    print("    %s (constante)" % c)
print("- Imputation : aucune (0 NaN)")
print("- Scaling : NON (Random Forest)")
print("- Hyperparametres finaux : %s" % best_params)
print("- Accuracy baseline : %.4f" % acc_base)
print("- Accuracy finale (optimise) : %.4f" % acc_final)
report_dict = classification_report(y_test, y_pred, target_names=le_target.classes_,
                                     output_dict=True, zero_division=0)
f1w = report_dict["weighted avg"]["f1-score"]
print("- F1-score pondere : %.4f" % f1w)
print("- Top 3 features : %s" % [(n, round(i, 4)) for i, n in top_features[:3]])
print("- Remarque : Random Forest = ensemble d'arbres, robuste au sur-apprentissage,")
print("  importance des variables estimee automatiquement.")
print("=" * 60)

# =========================================================
# SAUVEGARDE
# =========================================================
joblib.dump(model, os.path.join(MODEL_DIR, "rf_risk_model.pkl"))
joblib.dump(le_target, os.path.join(MODEL_DIR, "rf_risk_label_encoder.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "rf_risk_features.pkl"))
joblib.dump(top_features, os.path.join(MODEL_DIR, "rf_risk_importances.pkl"))

print()
print("Modele sauvegarde       : rf_risk_model.pkl")
print("LabelEncoder sauvegarde : rf_risk_label_encoder.pkl")
print("Features sauvegardees   : rf_risk_features.pkl")
print("Importances sauvegardees: rf_risk_importances.pkl")
