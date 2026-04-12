import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score

ROOT = r"C:\Users\MSI\Desktop\ML"
MODEL_DIR = os.path.join(ROOT, "decision_tree", "model")

model = joblib.load(os.path.join(MODEL_DIR, "dt_risk_model.pkl"))
le_target = joblib.load(os.path.join(MODEL_DIR, "dt_risk_label_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "dt_risk_features.pkl"))

print("Modele : DecisionTree (max_depth=%s, criterion=%s, ccp_alpha=%.6f)" % (
    model.max_depth, model.criterion, model.ccp_alpha))
print("Profondeur effective : %d | Feuilles : %d" % (model.get_depth(), model.get_n_leaves()))
print("Classes :", list(le_target.classes_))
print()

df = pd.read_csv(os.path.join(ROOT, "Project-Management-2-enriched.csv"))
y_true = df["Risk_Level"].copy()

cols_drop = ["Project ID", "Project Name", "Location", "Start Date",
             "End Date", "Task Name", "Risk_Level"]
df_test = df.drop(columns=cols_drop)

cat_cols = df_test.select_dtypes(include="object").columns.tolist()
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)

for c in feature_cols:
    if c not in df_test.columns:
        df_test[c] = 0
df_test = df_test[feature_cols]

y_pred_encoded = model.predict(df_test)
y_pred_labels = le_target.inverse_transform(y_pred_encoded)

print("=" * 50)
print("  PREDICTIONS (20 premieres lignes)")
print("=" * 50)
for i in range(min(20, len(y_pred_labels))):
    vrai = y_true.iloc[i]
    pred = y_pred_labels[i]
    ok = "OK" if vrai == pred else "FAUX"
    print("  Ligne %3d : Vrai = %-6s | Predit = %-6s  [%s]" % (i, vrai, pred, ok))

acc = accuracy_score(y_true, y_pred_labels)
print()
print("Accuracy globale : %.4f (%.1f%%)" % (acc, acc * 100))
