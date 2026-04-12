import joblib
import numpy as np
import pandas as pd
import os

ROOT = r"C:\Users\MSI\Desktop\ML"
MODEL_DIR = os.path.join(ROOT, "knn", "model")

model = joblib.load(os.path.join(MODEL_DIR, "knn_risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "knn_risk_scaler.pkl"))
le_target = joblib.load(os.path.join(MODEL_DIR, "knn_risk_label_encoder.pkl"))

print("Modele charge : KNN avec k =", model.n_neighbors)
print("Classes :", list(le_target.classes_))
print()

# --- EXEMPLE : tester avec le meme dataset ---
df = pd.read_csv(os.path.join(ROOT, "Project-Management-2-enriched.csv"))

# Reproduire le meme preprocessing que le pipeline
cols_to_drop = ["Project ID", "Project Name", "Location", "Start Date",
                "End Date", "Task Name", "Risk_Level"]
df_test = df.drop(columns=cols_to_drop)

cat_cols = df_test.select_dtypes(include="object").columns.tolist()
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)

# Clipping des outliers (memes seuils que le pipeline)
if "Actual Cost" in df_test.columns:
    df_test["Actual Cost"] = df_test["Actual Cost"].clip(lower=0, upper=9474.03)
if "Planned_Duration_Days" in df_test.columns:
    df_test["Planned_Duration_Days"] = df_test["Planned_Duration_Days"].clip(lower=4, upper=572.05)
if "Budget_Utilization" in df_test.columns:
    df_test["Budget_Utilization"] = df_test["Budget_Utilization"].clip(lower=0, upper=2.03)

# Scaling
X_test_scaled = scaler.transform(df_test)

# PCA (le modele a ete entraine avec PCA -> on doit aussi l'appliquer)
from sklearn.decomposition import PCA
pca = joblib.load(os.path.join(MODEL_DIR, "knn_risk_pca.pkl"))
X_test_pca = pca.transform(X_test_scaled)

# Prediction
y_pred_encoded = model.predict(X_test_pca)
y_pred_labels = le_target.inverse_transform(y_pred_encoded)

# Afficher les 20 premieres predictions
print("=" * 50)
print("  PREDICTIONS (20 premieres lignes)")
print("=" * 50)
for i in range(min(20, len(y_pred_labels))):
    vrai = df.iloc[i]["Risk_Level"]
    pred = y_pred_labels[i]
    ok = "OK" if vrai == pred else "FAUX"
    print("  Ligne %3d : Vrai = %-6s | Predit = %-6s  [%s]" % (i, vrai, pred, ok))

# Score global
from sklearn.metrics import accuracy_score
y_vrai = df["Risk_Level"]
acc = accuracy_score(y_vrai, y_pred_labels)
print()
print("Accuracy globale sur tout le dataset : %.4f (%.1f%%)" % (acc, acc * 100))
