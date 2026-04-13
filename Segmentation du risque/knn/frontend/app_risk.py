from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

_THIS = os.path.dirname(os.path.abspath(__file__))
_KNN_ROOT = os.path.dirname(_THIS)
MODEL_DIR = os.path.join(_KNN_ROOT, "model")
FRONTEND_DIR = _THIS
_SEG_ROOT = os.path.dirname(_KNN_ROOT)
_ML_ROOT = os.path.dirname(_SEG_ROOT)
ROOT = _ML_ROOT

model = joblib.load(os.path.join(MODEL_DIR, "knn_risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "knn_risk_scaler.pkl"))
le_target = joblib.load(os.path.join(MODEL_DIR, "knn_risk_label_encoder.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "knn_risk_pca.pkl"))

FEATURE_COLUMNS = [
    "Hours Spent", "Budget", "Actual Cost", "Progress",
    "Planned_Duration_Days", "Budget_Utilization", "Assignee_Historical_Task_Count",
    "Project Type_Construction", "Project Type_Infrastructure",
    "Project Type_Innovation", "Project Type_Maintenance",
    "Project Type_Other", "Project Type_Renovation",
    "Project Status_Behind", "Project Status_Completed",
    "Project Status_On Hold", "Project Status_On Track",
    "Priority_High", "Priority_Low", "Priority_Medium",
    "Task ID_T001", "Task ID_T002", "Task ID_T003",
    "Task Status_Completed", "Task Status_In Progress", "Task Status_Pending",
    "Assigned To_Alice", "Assigned To_Bob", "Assigned To_Charlie",
    "Assigned To_David", "Assigned To_Eve", "Assigned To_Frank",
    "Assigned To_Grace", "Assigned To_Ivy",
]

CLIP_RULES = {
    "Actual Cost": (0, 9474.03),
    "Planned_Duration_Days": (4, 572.05),
    "Budget_Utilization": (0, 2.03),
}


def compute_model_metrics():
    """Re-run the full pipeline to extract all metrics for the dashboard."""
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

    continuous = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique() > 2]
    for c in continuous:
        Q1, Q3 = X[c].quantile(0.25), X[c].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((X[c] < lo) | (X[c] > hi)).sum())
        if n_out > 0:
            X.loc[:, c] = X[c].clip(lower=X[c].quantile(0.01), upper=X[c].quantile(0.99))

    sc = StandardScaler()
    X_scaled = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    pca_m = PCA(n_components=0.95, random_state=42)
    X_pca = pca_m.fit_transform(X_scaled)
    n_components = pca_m.n_components_

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    k_results = []
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        k_results.append({"k": k, "accuracy": round(acc * 100, 2)})

    best_k = min(k_results, key=lambda x: (-x["accuracy"], x["k"]))["k"]
    best_acc = [r["accuracy"] for r in k_results if r["k"] == best_k][0]

    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    y_pred = final_knn.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)

    class_metrics = []
    for cls in le.classes_:
        m = report[str(cls)]
        class_metrics.append({
            "class": str(cls),
            "precision": round(m["precision"] * 100, 1),
            "recall": round(m["recall"] * 100, 1),
            "f1": round(m["f1-score"] * 100, 1),
            "support": int(m["support"]),
        })

    conf_matrix = []
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    for i, cls_true in enumerate(le.classes_):
        for j, cls_pred in enumerate(le.classes_):
            conf_matrix.append({
                "true": str(cls_true),
                "predicted": str(cls_pred),
                "count": int(cm[i][j]),
            })

    return {
        "n_total": n_total,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_raw": int(X.shape[1]),
        "n_components_pca": int(n_components),
        "k_results": k_results,
        "best_k": best_k,
        "best_accuracy": best_acc,
        "class_metrics": class_metrics,
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"] * 100, 1),
            "recall": round(report["weighted avg"]["recall"] * 100, 1),
            "f1": round(report["weighted avg"]["f1-score"] * 100, 1),
        },
        "confusion_matrix": conf_matrix,
        "classes": [str(c) for c in le.classes_],
    }


METRICS_CACHE = None


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "frontend_risk.html"))


@app.route("/metrics")
def metrics():
    global METRICS_CACHE
    if METRICS_CACHE is None:
        METRICS_CACHE = compute_model_metrics()
    return jsonify(METRICS_CACHE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    row = {col: 0 for col in FEATURE_COLUMNS}

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

    for col, (lo, hi) in CLIP_RULES.items():
        row[col] = max(lo, min(hi, row[col]))

    df_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(df_input)
    X_pca = pca.transform(X_scaled)
    pred_encoded = model.predict(X_pca)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]

    probas = model.predict_proba(X_pca)[0]
    proba_dict = {str(le_target.classes_[i]): round(float(probas[i]) * 100, 1)
                  for i in range(len(le_target.classes_))}

    return jsonify({
        "risk_level": str(pred_label),
        "probabilities": proba_dict,
        "budget_utilization": round(row["Budget_Utilization"], 4),
    })


if __name__ == "__main__":
    _port = int(os.environ.get("RISK_KNN_PORT", "5011"))
    print("KNN Risk — http://127.0.0.1:%d" % _port)
    app.run(debug=False, host="127.0.0.1", port=_port, use_reloader=False)
