from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def convert_numpy(obj):
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

# Racine "Segmentation du risque" puis sous-dossiers knn / random_forest / decision_tree
_SEG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_ROOT = os.path.dirname(_SEG_ROOT)
COMBINED_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== LOAD KNN =====================
KNN_MODEL_DIR = os.path.join(_SEG_ROOT, "knn", "model")
knn_model = joblib.load(os.path.join(KNN_MODEL_DIR, "knn_risk_model.pkl"))
knn_scaler = joblib.load(os.path.join(KNN_MODEL_DIR, "knn_risk_scaler.pkl"))
knn_le = joblib.load(os.path.join(KNN_MODEL_DIR, "knn_risk_label_encoder.pkl"))
knn_pca = joblib.load(os.path.join(KNN_MODEL_DIR, "knn_risk_pca.pkl"))

KNN_FEATURES = [
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
KNN_CLIP = {"Actual Cost": (0, 9474.03), "Planned_Duration_Days": (4, 572.05), "Budget_Utilization": (0, 2.03)}

# ===================== LOAD RANDOM FOREST =====================
RF_MODEL_DIR = os.path.join(_SEG_ROOT, "random_forest", "model")
rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "rf_risk_model.pkl"))
rf_le = joblib.load(os.path.join(RF_MODEL_DIR, "rf_risk_label_encoder.pkl"))
rf_features = joblib.load(os.path.join(RF_MODEL_DIR, "rf_risk_features.pkl"))

# ===================== LOAD DECISION TREE =====================
DT_MODEL_DIR = os.path.join(_SEG_ROOT, "decision_tree", "model")
dt_model = joblib.load(os.path.join(DT_MODEL_DIR, "dt_risk_model.pkl"))
dt_le = joblib.load(os.path.join(DT_MODEL_DIR, "dt_risk_label_encoder.pkl"))
dt_features = joblib.load(os.path.join(DT_MODEL_DIR, "dt_risk_features.pkl"))

print("3 modeles charges : KNN, Random Forest, Decision Tree")


def build_row(data, feature_cols):
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
    return row


# ===================== ROUTES =====================
@app.route("/")
def index():
    return send_file(os.path.join(COMBINED_DIR, "frontend_combined.html"))


@app.route("/predict_all", methods=["POST"])
def predict_all():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(convert_numpy(_predict_all_core(data)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _predict_all_core(data):
    results = {}

    # --- KNN ---
    row_knn = build_row(data, KNN_FEATURES)
    for col, (lo, hi) in KNN_CLIP.items():
        row_knn[col] = max(lo, min(hi, row_knn[col]))
    df_knn = pd.DataFrame([row_knn], columns=KNN_FEATURES)
    X_knn = knn_pca.transform(knn_scaler.transform(df_knn))
    pred_knn = knn_model.predict(X_knn)[0]
    probas_knn = knn_model.predict_proba(X_knn)[0]
    results["knn"] = {
        "risk_level": str(knn_le.inverse_transform([pred_knn])[0]),
        "probabilities": {str(knn_le.classes_[i]): round(float(probas_knn[i]) * 100, 1)
                          for i in range(len(knn_le.classes_))}
    }

    # --- RANDOM FOREST ---
    row_rf = build_row(data, rf_features)
    df_rf = pd.DataFrame([row_rf], columns=rf_features)
    pred_rf = rf_model.predict(df_rf)[0]
    probas_rf = rf_model.predict_proba(df_rf)[0]
    results["rf"] = {
        "risk_level": str(rf_le.inverse_transform([pred_rf])[0]),
        "probabilities": {str(rf_le.classes_[i]): round(float(probas_rf[i]) * 100, 1)
                          for i in range(len(rf_le.classes_))}
    }

    # --- DECISION TREE ---
    row_dt = build_row(data, dt_features)
    df_dt = pd.DataFrame([row_dt], columns=dt_features)
    pred_dt = dt_model.predict(df_dt)[0]
    probas_dt = dt_model.predict_proba(df_dt)[0]
    results["dt"] = {
        "risk_level": str(dt_le.inverse_transform([pred_dt])[0]),
        "probabilities": {str(dt_le.classes_[i]): round(float(probas_dt[i]) * 100, 1)
                          for i in range(len(dt_le.classes_))}
    }

    bu = row_knn["Budget_Utilization"]
    results["budget_utilization"] = round(bu, 4)

    # Consensus vote
    votes = [results["knn"]["risk_level"], results["rf"]["risk_level"], results["dt"]["risk_level"]]
    from collections import Counter
    consensus = Counter(votes).most_common(1)[0][0]
    unanimity = len(set(votes)) == 1
    results["consensus"] = {"risk_level": consensus, "unanimous": unanimity, "votes": votes}

    return results


METRICS_CACHE = {}


def compute_metrics_all():
    df_raw = pd.read_csv(os.path.join(_ML_ROOT, "Project-Management-2-enriched.csv"))
    n_total = len(df_raw)

    cols_sup_10 = [c for c in df_raw.select_dtypes(include="object").columns
                   if c != "Risk_Level" and df_raw[c].nunique() > 10]
    df = df_raw.drop(columns=cols_sup_10)

    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])
    df["Risk_Level"] = le.transform(df["Risk_Level"])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df.drop(columns=["Risk_Level"])
    y = df["Risk_Level"]

    results = {"n_total": n_total, "classes": [str(c) for c in le.classes_]}

    # ---------- KNN ----------
    continuous = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique() > 2]
    X_knn = X.copy()
    for c in continuous:
        X_knn.loc[:, c] = X_knn[c].clip(lower=X_knn[c].quantile(0.01), upper=X_knn[c].quantile(0.99))
    sc = StandardScaler()
    X_scaled = pd.DataFrame(sc.fit_transform(X_knn), columns=X_knn.columns)
    pca_m = PCA(n_components=0.95, random_state=42)
    X_pca = pca_m.fit_transform(X_scaled)
    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    best_k, best_k_acc = 1, 0
    k_results = []
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        acc = round(accuracy_score(y_te, knn.predict(X_te)) * 100, 2)
        k_results.append({"k": k, "accuracy": acc})
        if acc > best_k_acc:
            best_k, best_k_acc = k, acc

    knn_final = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr)
    y_pred_knn = knn_final.predict(X_te)
    rep_knn = classification_report(y_te, y_pred_knn, target_names=le.classes_, output_dict=True, zero_division=0)
    cm_knn = confusion_matrix(y_te, y_pred_knn)

    results["knn"] = {
        "accuracy": best_k_acc,
        "best_k": best_k,
        "n_components_pca": int(pca_m.n_components_),
        "k_results": k_results,
        "class_metrics": [{"class": str(c), "precision": round(rep_knn[str(c)]["precision"]*100,1),
                           "recall": round(rep_knn[str(c)]["recall"]*100,1),
                           "f1": round(rep_knn[str(c)]["f1-score"]*100,1),
                           "support": int(rep_knn[str(c)]["support"])} for c in le.classes_],
        "weighted_f1": round(rep_knn["weighted avg"]["f1-score"]*100,1),
        "confusion_matrix": [[int(cm_knn[i][j]) for j in range(3)] for i in range(3)],
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    }

    # ---------- RANDOM FOREST ----------
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_base_acc = round(accuracy_score(y_te2, RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr2, y_tr2).predict(X_te2))*100, 2)
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
        {"n_estimators": [50,100,200], "max_depth": [10,20,None], "min_samples_split": [2,5,10], "min_samples_leaf": [1,2,4]},
        cv=5, scoring="accuracy", n_jobs=-1).fit(X_tr2, y_tr2)
    y_pred_rf = grid_rf.best_estimator_.predict(X_te2)
    rep_rf = classification_report(y_te2, y_pred_rf, target_names=le.classes_, output_dict=True, zero_division=0)
    cm_rf = confusion_matrix(y_te2, y_pred_rf)
    imp_rf = sorted(zip(grid_rf.best_estimator_.feature_importances_, X.columns.tolist()), reverse=True)

    results["rf"] = {
        "accuracy": round(accuracy_score(y_te2, y_pred_rf)*100,2),
        "baseline_accuracy": rf_base_acc,
        "best_params": {k: (str(v) if v is None else v) for k,v in grid_rf.best_params_.items()},
        "class_metrics": [{"class": str(c), "precision": round(rep_rf[str(c)]["precision"]*100,1),
                           "recall": round(rep_rf[str(c)]["recall"]*100,1),
                           "f1": round(rep_rf[str(c)]["f1-score"]*100,1),
                           "support": int(rep_rf[str(c)]["support"])} for c in le.classes_],
        "weighted_f1": round(rep_rf["weighted avg"]["f1-score"]*100,1),
        "confusion_matrix": [[int(cm_rf[i][j]) for j in range(3)] for i in range(3)],
        "importances": [{"feature": n, "importance": round(float(v)*100,2)} for v,n in imp_rf[:10]],
        "n_train": int(len(y_tr2)), "n_test": int(len(y_te2)),
    }

    # ---------- DECISION TREE ----------
    dt_base = DecisionTreeClassifier(random_state=42).fit(X_tr2, y_tr2)
    dt_base_acc = round(accuracy_score(y_te2, dt_base.predict(X_te2))*100,2)
    grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
        {"max_depth": [3,5,7,10,15,None], "min_samples_split": [2,5,10,20], "min_samples_leaf": [1,2,5,10], "criterion": ["gini","entropy"]},
        cv=5, scoring="accuracy", n_jobs=-1).fit(X_tr2, y_tr2)

    path = grid_dt.best_estimator_.cost_complexity_pruning_path(X_tr2, y_tr2)
    ba, baa = 0.0, 0.0
    for alpha in path.ccp_alphas:
        tmp = DecisionTreeClassifier(**grid_dt.best_params_, ccp_alpha=alpha, random_state=42).fit(X_tr2, y_tr2)
        a = accuracy_score(y_te2, tmp.predict(X_te2))
        if a > baa:
            baa, ba = a, alpha
    dt_final = DecisionTreeClassifier(**grid_dt.best_params_, ccp_alpha=ba, random_state=42).fit(X_tr2, y_tr2)
    y_pred_dt = dt_final.predict(X_te2)
    rep_dt = classification_report(y_te2, y_pred_dt, target_names=le.classes_, output_dict=True, zero_division=0)
    cm_dt = confusion_matrix(y_te2, y_pred_dt)
    imp_dt = sorted(zip(dt_final.feature_importances_, X.columns.tolist()), reverse=True)

    depth_results = []
    for d in [2,3,4,5,6,7,8,10,12,15]:
        dtd = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr2, y_tr2)
        depth_results.append({"depth": d, "accuracy": round(accuracy_score(y_te2, dtd.predict(X_te2))*100,2)})

    bp_dt = {k: (str(v) if v is None else v) for k,v in grid_dt.best_params_.items()}
    bp_dt["ccp_alpha"] = round(float(ba), 6)

    results["dt"] = {
        "accuracy": round(accuracy_score(y_te2, y_pred_dt)*100,2),
        "baseline_accuracy": dt_base_acc,
        "best_params": bp_dt,
        "final_depth": int(dt_final.get_depth()),
        "final_leaves": int(dt_final.get_n_leaves()),
        "depth_results": depth_results,
        "class_metrics": [{"class": str(c), "precision": round(rep_dt[str(c)]["precision"]*100,1),
                           "recall": round(rep_dt[str(c)]["recall"]*100,1),
                           "f1": round(rep_dt[str(c)]["f1-score"]*100,1),
                           "support": int(rep_dt[str(c)]["support"])} for c in le.classes_],
        "weighted_f1": round(rep_dt["weighted avg"]["f1-score"]*100,1),
        "confusion_matrix": [[int(cm_dt[i][j]) for j in range(3)] for i in range(3)],
        "importances": [{"feature": n, "importance": round(float(v)*100,2)} for v,n in imp_dt if v > 0][:10],
        "n_train": int(len(y_tr2)), "n_test": int(len(y_te2)),
    }

    return convert_numpy(results)


@app.route("/metrics")
def metrics():
    global METRICS_CACHE
    if not METRICS_CACHE:
        METRICS_CACHE = compute_metrics_all()
    return jsonify(METRICS_CACHE)


if __name__ == "__main__":
    print("Serveur combine (KNN + RF + DT) sur http://localhost:5005")
    app.run(debug=False, port=5005)
