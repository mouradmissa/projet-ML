"""
Évaluation comparative Random Forest vs KNN — Recommandation de priorité (multiclasse).
Même prétraitement et même split train/test que les pipelines (random_state=42, stratify).
Utilise les modèles et le scaler sauvegardés pour reproduire les résultats des pipelines.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

EVAL_DIR = Path(__file__).resolve().parent
RP_ROOT = EVAL_DIR.parent
ML_ROOT = RP_ROOT.parent
RF_MODEL_DIR = RP_ROOT / "Random Forest" / "MODEL"
KNN_MODEL_DIR = RP_ROOT / "KNN" / "MODEL"

DATA_CANDIDATES = [
    ML_ROOT / "Project-Management-2-enriched.csv",
    ML_ROOT / "Project-Management-2.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COLS = ["Progress", "Budget", "Planned_Duration_Days"]
CLASS_LABELS = ["Low", "Medium", "High"]
PRIORITY_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> None:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    lo = Q1 - 1.5 * iqr
    hi = Q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lo, upper=hi)


def load_and_prepare_xy() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    data_path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(
            "CSV introuvable. Placez Project-Management-2-enriched.csv à la racine ML."
        )
    df = pd.read_csv(data_path)
    df_clean = df.drop_duplicates().copy()
    for col in FEATURE_COLS:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    if df_clean["Priority"].isnull().sum() > 0:
        df_clean["Priority"].fillna(df_clean["Priority"].mode()[0], inplace=True)
    const_cols = [c for c in df_clean.columns if df_clean[c].nunique() == 1]
    if const_cols:
        df_clean.drop(columns=const_cols, inplace=True)
    for col in ["Budget", "Planned_Duration_Days"]:
        cap_outliers_iqr(df_clean, col)
    df_clean["Priority_encoded"] = df_clean["Priority"].map(PRIORITY_MAPPING)
    X = df_clean[FEATURE_COLS].copy()
    y = df_clean["Priority_encoded"].copy()
    return df_clean, X, y


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(title)
    plt.ylabel("Vraie classe")
    plt.xlabel("Classe prédite")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_f1_per_class(f1_rf: np.ndarray, f1_knn: np.ndarray, out_path: Path) -> None:
    x = np.arange(len(CLASS_LABELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, f1_rf, w, label="Random Forest", color="#2E86AB")
    ax.bar(x + w / 2, f1_knn, w, label="KNN", color="#F77F00")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score par classe de priorité")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    for i in range(len(CLASS_LABELS)):
        ax.text(i - w / 2, f1_rf[i] + 0.02, f"{f1_rf[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f1_knn[i] + 0.02, f"{f1_knn[i]:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_ovr(
    y_test: np.ndarray,
    proba_rf: np.ndarray,
    proba_knn: np.ndarray,
    out_path: Path,
) -> dict[str, float]:
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#E63946", "#F77F00", "#06A77D"]
    aucs_rf: list[float] = []
    aucs_knn: list[float] = []

    for i in range(n_classes):
        fpr_rf, tpr_rf, _ = roc_curve(y_bin[:, i], proba_rf[:, i])
        fpr_kn, tpr_kn, _ = roc_curve(y_bin[:, i], proba_knn[:, i])
        a_rf = auc(fpr_rf, tpr_rf)
        a_kn = auc(fpr_kn, tpr_kn)
        aucs_rf.append(a_rf)
        aucs_knn.append(a_kn)
        ax.plot(
            fpr_rf,
            tpr_rf,
            color=colors[i],
            linestyle="-",
            lw=2,
            label=f"RF — {CLASS_LABELS[i]} (AUC = {a_rf:.3f})",
        )
        ax.plot(
            fpr_kn,
            tpr_kn,
            color=colors[i],
            linestyle="--",
            lw=2,
            label=f"KNN — {CLASS_LABELS[i]} (AUC = {a_kn:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("ROC one-vs-rest — Random Forest vs KNN")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    macro_rf = float(np.mean(aucs_rf))
    macro_kn = float(np.mean(aucs_knn))
    return {
        "auc_ovr_macro_rf": macro_rf,
        "auc_ovr_macro_knn": macro_kn,
        "auc_sklearn_macro_rf": roc_auc_score(
            y_test, proba_rf, multi_class="ovr", average="macro"
        ),
        "auc_sklearn_macro_knn": roc_auc_score(
            y_test, proba_knn, multi_class="ovr", average="macro"
        ),
    }


def plot_rf_feature_importance(rf_model, out_path: Path) -> None:
    imp = np.asarray(rf_model.feature_importances_)
    order = np.argsort(imp)
    plt.figure(figsize=(8, 4))
    plt.barh(np.array(FEATURE_COLS)[order], imp[order], color="#2E86AB")
    plt.xlabel("Importance (Gini)")
    plt.title("Importance des variables — Random Forest (priorité)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    df_clean, X, y = load_and_prepare_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_test_arr = np.asarray(y_test)

    with open(RF_MODEL_DIR / "rf_priority_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(KNN_MODEL_DIR / "knn_priority_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    with open(KNN_MODEL_DIR / "knn_priority_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    y_pred_rf = rf_model.predict(X_test)
    X_test_scaled = scaler.transform(X_test.values)
    y_pred_knn = knn_model.predict(X_test_scaled)

    proba_rf = rf_model.predict_proba(X_test)
    proba_knn = knn_model.predict_proba(X_test_scaled)

    lines: list[str] = []
    lines.append("Évaluation — Recommandation de priorité (RF vs KNN)\n")
    lines.append("Distribution globale de la cible (dataset nettoyé):\n")
    lines.append(str(df_clean["Priority"].value_counts()))
    lines.append("")
    lines.append("Distribution sur le jeu de test (stratifié):\n")
    vc = pd.Series(y_test_arr).value_counts().sort_index()
    for code, name in enumerate(CLASS_LABELS):
        n = int(vc.get(code, 0))
        lines.append(f"  {name}: {n} ({100 * n / len(y_test_arr):.1f}%)")

    def row(model_name: str, y_pred: np.ndarray) -> dict:
        return {
            "Modèle": model_name,
            "Accuracy": accuracy_score(y_test_arr, y_pred),
            "Précision (macro)": precision_score(
                y_test_arr, y_pred, average="macro", zero_division=0
            ),
            "Rappel (macro)": recall_score(
                y_test_arr, y_pred, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(y_test_arr, y_pred, average="macro", zero_division=0),
            "Précision (weighted)": precision_score(
                y_test_arr, y_pred, average="weighted", zero_division=0
            ),
            "Rappel (weighted)": recall_score(
                y_test_arr, y_pred, average="weighted", zero_division=0
            ),
            "F1 (weighted)": f1_score(
                y_test_arr, y_pred, average="weighted", zero_division=0
            ),
        }

    r_rf = row("Random Forest", y_pred_rf)
    r_kn = row("KNN", y_pred_knn)
    comp = pd.DataFrame([r_kn, r_rf])
    lines.append("\nTableau comparatif:\n")
    lines.append(comp.to_string(index=False))

    lines.append("\n\nRapport Random Forest:\n")
    lines.append(
        classification_report(
            y_test_arr,
            y_pred_rf,
            target_names=CLASS_LABELS,
            zero_division=0,
        )
    )
    lines.append("\nRapport KNN:\n")
    lines.append(
        classification_report(
            y_test_arr,
            y_pred_knn,
            target_names=CLASS_LABELS,
            zero_division=0,
        )
    )

    cm_rf = confusion_matrix(y_test_arr, y_pred_rf)
    cm_kn = confusion_matrix(y_test_arr, y_pred_knn)
    plot_confusion(cm_rf, "Matrice de confusion — Random Forest", EVAL_DIR / "eval_priority_confusion_rf.png")
    plot_confusion(cm_kn, "Matrice de confusion — KNN", EVAL_DIR / "eval_priority_confusion_knn.png")

    _, _, f1_rf, _ = precision_recall_fscore_support(
        y_test_arr, y_pred_rf, average=None, zero_division=0
    )
    _, _, f1_knn, _ = precision_recall_fscore_support(
        y_test_arr, y_pred_knn, average=None, zero_division=0
    )
    plot_f1_per_class(f1_rf, f1_knn, EVAL_DIR / "eval_priority_f1_par_classe.png")

    auc_info = plot_roc_ovr(y_test_arr, proba_rf, proba_knn, EVAL_DIR / "eval_priority_roc_ovr.png")
    lines.append("\nAUC ROC (one-vs-rest, moyenne des AUC par classe — calcul manuel):\n")
    lines.append(f"  Random Forest: {auc_info['auc_ovr_macro_rf']:.4f}")
    lines.append(f"  KNN:           {auc_info['auc_ovr_macro_knn']:.4f}")
    lines.append("\nAUC ROC (sklearn, multi_class='ovr', average='macro'):\n")
    lines.append(f"  Random Forest: {auc_info['auc_sklearn_macro_rf']:.4f}")
    lines.append(f"  KNN:           {auc_info['auc_sklearn_macro_knn']:.4f}")

    plot_rf_feature_importance(rf_model, EVAL_DIR / "eval_priority_rf_feature_importance.png")

    out_txt = EVAL_DIR / "evaluation_priorite_output.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nFigures et rapport sauvegardés dans: {EVAL_DIR}")


if __name__ == "__main__":
    main()
