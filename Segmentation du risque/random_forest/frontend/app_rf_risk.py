from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

_THIS = os.path.dirname(os.path.abspath(__file__))
_RF_ROOT = os.path.dirname(_THIS)
MODEL_DIR = os.path.join(_RF_ROOT, "model")
FRONTEND_DIR = _THIS
_SEG_ROOT = os.path.dirname(_RF_ROOT)
_ML_ROOT = os.path.dirname(_SEG_ROOT)
ROOT = _ML_ROOT

model = joblib.load(os.path.join(MODEL_DIR, "rf_risk_model.pkl"))
le_target = joblib.load(os.path.join(MODEL_DIR, "rf_risk_label_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "rf_risk_features.pkl"))
all_importances = joblib.load(os.path.join(MODEL_DIR, "rf_risk_importances.pkl"))


def compute_metrics():
    df = pd.read_csv(os.path.join(ROOT, "Project-Management-2-enriched.csv"))
    n_total = len(df)
    cols_sup_10 = [c for c in df.select_dtypes(include="object").columns
                   if c != "Risk_Level" and df[c].nunique() > 10]
    df.drop(columns=cols_sup_10, inplace=True)

    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])
    df["Risk_Level"] = le.transform(df["Risk_Level"])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df.drop(columns=["Risk_Level"])
    y = df["Risk_Level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf_base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    baseline_acc = round(accuracy_score(y_test, rf_base.predict(X_test)) * 100, 2)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None],
         "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    class_metrics = []
    for cls in le.classes_:
        m = report[str(cls)]
        class_metrics.append({"class": str(cls), "precision": round(m["precision"] * 100, 1),
                              "recall": round(m["recall"] * 100, 1), "f1": round(m["f1-score"] * 100, 1),
                              "support": int(m["support"])})

    conf = []
    for i, ct in enumerate(le.classes_):
        for j, cp in enumerate(le.classes_):
            conf.append({"true": str(ct), "predicted": str(cp), "count": int(cm[i][j])})

    imps = sorted(zip(best.feature_importances_, X.columns.tolist()), reverse=True)
    top_imp = [{"feature": n, "importance": round(float(v) * 100, 2)} for v, n in imps[:15]]

    return {
        "n_total": n_total, "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "best_params": {k: (str(v) if v is None else v) for k, v in grid.best_params_.items()},
        "best_accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "baseline_accuracy": baseline_acc,
        "class_metrics": class_metrics,
        "weighted_avg": {"precision": round(report["weighted avg"]["precision"] * 100, 1),
                         "recall": round(report["weighted avg"]["recall"] * 100, 1),
                         "f1": round(report["weighted avg"]["f1-score"] * 100, 1)},
        "confusion_matrix": conf,
        "classes": [str(c) for c in le.classes_],
        "importances": top_imp,
    }


CACHE = None


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "frontend_rf_risk.html"))


@app.route("/metrics")
def metrics():
    global CACHE
    if CACHE is None:
        CACHE = compute_metrics()
    return jsonify(CACHE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    row = {col: 0 for col in feature_cols}

    row["Hours Spent"] = float(data.get("hours_spent", 0))
    row["Budget"] = float(data.get("budget", 0))
    row["Actual Cost"] = float(data.get("actual_cost", 0))
    row["Progress"] = float(data.get("progress", 0))
    row["Planned_Duration_Days"] = float(data.get("planned_duration", 0))
    row["Assignee_Historical_Task_Count"] = float(data.get("history_count", 0))
    budget = row["Budget"]
    row["Budget_Utilization"] = (row["Actual Cost"] / budget) if budget > 0 else 0

    for prefix, field in [
        ("Project Type_", "project_type"), ("Project Status_", "project_status"),
        ("Priority_", "priority"), ("Task ID_", "task_id"),
        ("Task Status_", "task_status"), ("Assigned To_", "assigned_to"),
    ]:
        val = data.get(field, "")
        col = prefix + val
        if col in row:
            row[col] = 1

    df_in = pd.DataFrame([row], columns=feature_cols)
    pred = model.predict(df_in)[0]
    label = str(le_target.inverse_transform([pred])[0])
    probas = model.predict_proba(df_in)[0]
    proba_dict = {str(le_target.classes_[i]): round(float(probas[i]) * 100, 1)
                  for i in range(len(le_target.classes_))}

    return jsonify({"risk_level": label, "probabilities": proba_dict,
                     "budget_utilization": round(row["Budget_Utilization"], 4)})


if __name__ == "__main__":
    _port = int(os.environ.get("RISK_RF_PORT", "5012"))
    print("RF Risk — http://127.0.0.1:%d" % _port)
    app.run(debug=False, host="127.0.0.1", port=_port, use_reloader=False)
