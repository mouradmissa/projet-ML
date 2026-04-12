from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings("ignore")


def convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


app = Flask(__name__)

ROOT = r"C:\Users\MSI\Desktop\ML"
DT_DIR = os.path.join(ROOT, "decision_tree")
MODEL_DIR = os.path.join(DT_DIR, "model")
FRONTEND_DIR = os.path.join(DT_DIR, "frontend")

model = joblib.load(os.path.join(MODEL_DIR, "dt_risk_model.pkl"))
le_target = joblib.load(os.path.join(MODEL_DIR, "dt_risk_label_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "dt_risk_features.pkl"))
all_importances = joblib.load(os.path.join(MODEL_DIR, "dt_risk_importances.pkl"))


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

    dt_base = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    baseline_acc = round(accuracy_score(y_test, dt_base.predict(X_test)) * 100, 2)
    baseline_depth = dt_base.get_depth()
    baseline_leaves = dt_base.get_n_leaves()

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 7, 10, 15, None],
         "min_samples_split": [2, 5, 10, 20],
         "min_samples_leaf": [1, 2, 5, 10],
         "criterion": ["gini", "entropy"]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    path = best.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    best_alpha = 0.0
    best_alpha_acc = 0.0
    for alpha in ccp_alphas:
        dt_tmp = DecisionTreeClassifier(**grid.best_params_, ccp_alpha=alpha, random_state=42)
        dt_tmp.fit(X_train, y_train)
        acc_tmp = accuracy_score(y_test, dt_tmp.predict(X_test))
        if acc_tmp > best_alpha_acc:
            best_alpha_acc = acc_tmp
            best_alpha = alpha

    final_model = DecisionTreeClassifier(**grid.best_params_, ccp_alpha=best_alpha, random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

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

    imps = sorted(zip(final_model.feature_importances_, X.columns.tolist()), reverse=True)
    top_imp = [{"feature": n, "importance": round(float(v) * 100, 2)} for v, n in imps if v > 0][:15]

    depth_results = []
    for d in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        dt_d = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt_d.fit(X_train, y_train)
        acc_d = round(accuracy_score(y_test, dt_d.predict(X_test)) * 100, 2)
        depth_results.append({"depth": d, "accuracy": acc_d})

    best_params_str = {k: (str(v) if v is None else v) for k, v in grid.best_params_.items()}
    best_params_str["ccp_alpha"] = round(best_alpha, 6)

    result = {
        "n_total": int(n_total), "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "best_params": best_params_str,
        "best_accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "baseline_accuracy": float(baseline_acc),
        "baseline_depth": int(baseline_depth),
        "baseline_leaves": int(baseline_leaves),
        "final_depth": int(final_model.get_depth()),
        "final_leaves": int(final_model.get_n_leaves()),
        "class_metrics": class_metrics,
        "weighted_avg": {"precision": round(float(report["weighted avg"]["precision"]) * 100, 1),
                         "recall": round(float(report["weighted avg"]["recall"]) * 100, 1),
                         "f1": round(float(report["weighted avg"]["f1-score"]) * 100, 1)},
        "confusion_matrix": conf,
        "classes": [str(c) for c in le.classes_],
        "importances": top_imp,
        "depth_results": depth_results,
    }
    return convert_numpy(result)


CACHE = None


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "frontend_dt_risk.html"))


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
    print("Serveur Decision Tree Risk demarre sur http://localhost:5003")
    app.run(debug=False, port=5003)
