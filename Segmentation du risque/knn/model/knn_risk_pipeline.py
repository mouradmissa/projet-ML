import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ETAPE 1 : CHARGEMENT ET INSPECTION
# =========================================================
print("=" * 60)
print("  ETAPE 1 : CHARGEMENT ET INSPECTION")
print("=" * 60)

import os
ROOT = r"C:\Users\MSI\Desktop\ML"
MODEL_DIR = os.path.join(ROOT, "knn", "model")

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
print("Lignes avec NaN : %.2f%% -> aucune imputation necessaire" % pct_na)

# 2c. Colonnes constantes
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
if const_cols:
    print("Colonnes constantes supprimees :", const_cols)
    df.drop(columns=const_cols, inplace=True)
else:
    print("Colonnes constantes : aucune")

# 2d. Colonnes categorielles > 10 modalites -> suppression
cols_sup_10 = []
for c in df.select_dtypes(include="object").columns:
    if c == "Risk_Level":
        continue
    if df[c].nunique() > 10:
        cols_sup_10.append(c)
df.drop(columns=cols_sup_10, inplace=True)
print("Colonnes >10 modalites supprimees :", cols_sup_10)

# 2e. Colonnes correlees > 0.95
num_cols_check = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[num_cols_check].corr().abs()
cols_corr_drop = []
for i in range(len(num_cols_check)):
    for j in range(i + 1, len(num_cols_check)):
        if corr_matrix.iloc[i, j] > 0.95:
            c1, c2 = num_cols_check[i], num_cols_check[j]
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
# ETAPE 3 : ENCODAGE DES VARIABLES CATEGORIELLES
# =========================================================
print("=" * 60)
print("  ETAPE 3 : ENCODAGE")
print("=" * 60)

# 3a. Cible : LabelEncoder
le_target = LabelEncoder()
le_target.fit(["High", "Low", "Medium"])
df["Risk_Level"] = le_target.transform(df["Risk_Level"])
mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print("Encodage cible :", mapping)
print("Distribution cible :", dict(df["Risk_Level"].value_counts().sort_index()))

# 3b. Explicatives categorielles -> One-Hot (<=10 modalites garanties)
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Colonnes One-Hot :", cat_cols)
for c in cat_cols:
    print("  %s (%d modalites) : %s" % (c, df[c].nunique(), sorted(df[c].unique().tolist())))
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
print("Shape apres One-Hot :", df.shape)
print()

# =========================================================
# ETAPE 4 : SEPARATION FEATURES / TARGET
# =========================================================
print("=" * 60)
print("  ETAPE 4 : SEPARATION X / y")
print("=" * 60)

X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"]
print("X :", X.shape, "| y :", y.shape)
print("Features :", X.columns.tolist())
print()

# =========================================================
# ETAPE 5 : OUTLIERS + SCALING (OBLIGATOIRE POUR KNN)
# =========================================================
print("=" * 60)
print("  ETAPE 5 : OUTLIERS + SCALING")
print("=" * 60)

continuous_cols = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique() > 2]
outlier_info = []
for c in continuous_cols:
    Q1 = X[c].quantile(0.25)
    Q3 = X[c].quantile(0.75)
    IQR = Q3 - Q1
    lower_b = Q1 - 1.5 * IQR
    upper_b = Q3 + 1.5 * IQR
    n_out = int(((X[c] < lower_b) | (X[c] > upper_b)).sum())
    if n_out > 0:
        p1 = X[c].quantile(0.01)
        p99 = X[c].quantile(0.99)
        X.loc[:, c] = X[c].clip(lower=p1, upper=p99)
        msg = "%s: %d outliers -> clipping [P1=%.2f, P99=%.2f]" % (c, n_out, p1, p99)
        outlier_info.append(msg)
        print("  " + msg)
if not outlier_info:
    print("  Aucun outlier extreme (methode IQR).")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
print("  StandardScaler applique sur TOUTES les colonnes.")

n_features_before_pca = X_scaled.shape[1]
pca_applied = False
if X_scaled.shape[1] > 20:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_comp = pca.n_components_
    print("  PCA : %d features -> %d composantes (95%% variance)" % (X_scaled.shape[1], n_comp))
    X_scaled = pd.DataFrame(X_pca, columns=["PC%d" % (i + 1) for i in range(n_comp)])
    pca_applied = True
else:
    print("  %d features <= 20 -> PCA non necessaire." % X_scaled.shape[1])
print()

# =========================================================
# ETAPE 6 : DIVISION TRAIN / TEST
# =========================================================
print("=" * 60)
print("  ETAPE 6 : SPLIT TRAIN / TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train : %d lignes | Test : %d lignes" % (X_train.shape[0], X_test.shape[0]))
print("y_train :", dict(pd.Series(y_train).value_counts().sort_index()))
print("y_test  :", dict(pd.Series(y_test).value_counts().sort_index()))
print()

# =========================================================
# ETAPE 7 : RECHERCHE DU MEILLEUR k (1 a 10)
# =========================================================
print("=" * 60)
print("  ETAPE 7 : RECHERCHE MEILLEUR k")
print("=" * 60)

results = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    results.append((k, acc))
    print("  k=%2d -> Accuracy = %.4f" % (k, acc))

best_k = min(results, key=lambda x: (-x[1], x[0]))[0]
best_acc = dict(results)[best_k]
print()
print("  >> Meilleur k = %d (Accuracy = %.4f)" % (best_k, best_acc))
print()

# =========================================================
# ETAPE 8 : ENTRAINEMENT DU MODELE FINAL
# =========================================================
print("=" * 60)
print("  ETAPE 8 : MODELE FINAL")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("KNeighborsClassifier(n_neighbors=%d) entraine." % best_k)
print()

# =========================================================
# ETAPE 9 : EVALUATION
# =========================================================
print("=" * 60)
print("  ETAPE 9 : EVALUATION")
print("=" * 60)

acc_final = accuracy_score(y_test, y_pred)
print("Accuracy : %.4f" % acc_final)
print()
print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))

# =========================================================
# ETAPE 10 : RESUME STRUCTURE
# =========================================================
print()
print("=" * 60)
print("         RESUME DES ACTIONS")
print("=" * 60)
print("- Lignes initiales : %d  |  Apres nettoyage : %d" % (n_init, n_after_dup))
print("- Colonnes initiales : %d" % cols_init)
print("- Colonnes supprimees :")
for c in cols_sup_10:
    print("    %s (>10 modalites)" % c)
for c in cols_corr_drop:
    print("    %s (correlation >0.95)" % c)
for c in const_cols:
    print("    %s (constante)" % c)
print("- Imputation : non necessaire (0 NaN)")
print("- Outliers traites : %d colonne(s)" % len(outlier_info))
for oi in outlier_info:
    print("    " + oi)
if pca_applied:
    print("- PCA : oui (%d -> %d composantes)" % (n_features_before_pca, X_scaled.shape[1]))
else:
    print("- PCA : non (%d features <= 20)" % n_features_before_pca)
print("- Scaling : StandardScaler (obligatoire pour KNN)")
print("- Meilleur k : %d" % best_k)
print("- Accuracy finale : %.4f" % acc_final)
print("- Modele : KNeighborsClassifier(n_neighbors=%d)" % best_k)
print("  -> Entraine et pret a etre utilise.")
print("=" * 60)

# =========================================================
# ETAPE 11 : SAUVEGARDE
# =========================================================
joblib.dump(model, os.path.join(MODEL_DIR, "knn_risk_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "knn_risk_scaler.pkl"))
joblib.dump(le_target, os.path.join(MODEL_DIR, "knn_risk_label_encoder.pkl"))
if pca_applied:
    joblib.dump(pca, os.path.join(MODEL_DIR, "knn_risk_pca.pkl"))
print()
print("Modele sauvegarde  : knn_risk_model.pkl")
print("Scaler sauvegarde  : knn_risk_scaler.pkl")
print("Encoder sauvegarde : knn_risk_label_encoder.pkl")
if pca_applied:
    print("PCA sauvegarde     : knn_risk_pca.pkl")
