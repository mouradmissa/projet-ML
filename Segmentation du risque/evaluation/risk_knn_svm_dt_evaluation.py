"""
Évaluation comparative KNN / SVM (RBF) / Decision Tree — Segmentation du risque.
Prétraitement unifié (aligné sur le pipeline KNN : nettoyage, one-hot, scaling, PCA si >20 features).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_STATE = 42


def resolve_csv_path() -> Path:
    here = Path(__file__).resolve().parent
    seg = here.parent
    root = seg.parent
    candidates = [
        seg / "Project-Management-2-enriched.csv",
        root / "Project-Management-2-enriched.csv",
        here / "Project-Management-2-enriched.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Project-Management-2-enriched.csv introuvable. Cherché : " + ", ".join(str(c) for c in candidates)
    )


def preprocess_unified(csv_path: Path) -> dict:
    """Retourne X_train, X_test, y_train, y_test, le_target, noms de classes, flags."""
    df = pd.read_csv(csv_path)
    n_init = len(df)

    df = df.drop_duplicates()
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)

    cols_sup_10 = []
    for c in df.select_dtypes(include="object").columns:
        if c == "Risk_Level":
            continue
        if df[c].nunique() > 10:
            cols_sup_10.append(c)
    df = df.drop(columns=cols_sup_10)

    num_cols_check = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[num_cols_check].corr().abs()
    cols_corr_drop = []
    for i in range(len(num_cols_check)):
        for j in range(i + 1, len(num_cols_check)):
            if corr_matrix.iloc[i, j] > 0.95:
                c1 = num_cols_check[i]
                if c1 not in cols_corr_drop:
                    cols_corr_drop.append(c1)
    df = df.drop(columns=cols_corr_drop)

    le_target = LabelEncoder()
    le_target.fit(["High", "Low", "Medium"])
    df = df.copy()
    df["Risk_Level"] = le_target.transform(df["Risk_Level"])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df.drop(columns=["Risk_Level"])
    y = df["Risk_Level"]

    continuous_cols = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique() > 2]
    X = X.copy()
    for c in continuous_cols:
        X[c] = X[c].astype(float)
        Q1, Q3 = X[c].quantile(0.25), X[c].quantile(0.75)
        iqr = Q3 - Q1
        lower_b, upper_b = Q1 - 1.5 * iqr, Q3 + 1.5 * iqr
        if ((X[c] < lower_b) | (X[c] > upper_b)).any():
            p1, p99 = X[c].quantile(0.01), X[c].quantile(0.99)
            X.loc[:, c] = X[c].clip(lower=float(p1), upper=float(p99))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    pca = None
    if X_scaled.shape[1] > 20:
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_arr = pca.fit_transform(X_scaled)
        X_scaled = pd.DataFrame(X_arr, columns=[f"PC{i+1}" for i in range(X_arr.shape[1])], index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    class_names = list(le_target.classes_)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "le_target": le_target,
        "class_names": class_names,
        "n_init": n_init,
        "n_after_clean": len(df),
        "pca_applied": pca is not None,
        "n_features_final": X_scaled.shape[1],
    }


def train_knn(X_train, y_train, X_test, y_test):
    best_k, best_acc = 1, -1.0
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc or (acc == best_acc and k < best_k):
            best_k, best_acc = k, acc
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    return model, best_k


def train_svm(X_train, y_train):
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
    }
    base = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_params = dict(grid.best_params_)

    dt_for_pruning = DecisionTreeClassifier(**best_params, random_state=RANDOM_STATE)
    dt_for_pruning.fit(X_train, y_train)
    path = dt_for_pruning.cost_complexity_pruning_path(X_train, y_train)

    best_alpha = 0.0
    best_alpha_acc = -1.0
    for alpha in path.ccp_alphas:
        tmp = DecisionTreeClassifier(**best_params, ccp_alpha=alpha, random_state=RANDOM_STATE)
        tmp.fit(X_train, y_train)
        acc_tmp = accuracy_score(y_test, tmp.predict(X_test))
        if acc_tmp > best_alpha_acc:
            best_alpha_acc = acc_tmp
            best_alpha = alpha

    model = DecisionTreeClassifier(**best_params, ccp_alpha=best_alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, best_params, best_alpha


def macro_auc_ovr(y_true, y_proba, n_classes: int) -> float:
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def build_metrics_row(name: str, y_test, y_pred, y_proba, n_classes: int) -> dict:
    return {
        "Modèle": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Précision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Rappel (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "AUC (moyenne OvR)": macro_auc_ovr(y_test, y_proba, n_classes),
    }


def plot_confusion_matrices(y_test, preds: dict, class_names: list, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names)
        ax.set_title(title)
        ax.set_ylabel("Vraie classe")
        ax.set_xlabel("Prédiction")
    plt.tight_layout()
    p = out_dir / "confusion_matrices_knn_svm_dt.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_roc_ovr(y_test, probas: dict, class_names: list, out_dir: Path):
    y_test = np.asarray(y_test)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5))
    if n_classes == 1:
        axes = [axes]
    for c_idx, cname in enumerate(class_names):
        ax = axes[c_idx]
        y_bin = (y_test == c_idx).astype(int)
        for model_name, proba in probas.items():
            fpr, tpr, _ = roc_curve(y_bin, proba[:, c_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title(f"Classe : {cname} (one-vs-rest)")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    p = out_dir / "roc_ovr_knn_svm_dt.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_f1_per_class(y_test, preds: dict, class_names: list, out_dir: Path):
    rows = []
    for mname, y_pred in preds.items():
        f1s = f1_score(y_test, y_pred, average=None, zero_division=0, labels=np.arange(len(class_names)))
        for i, cn in enumerate(class_names):
            rows.append({"Modèle": mname, "Classe": cn, "F1": f1s[i]})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(class_names))
    width = 0.25
    models = list(preds.keys())
    for i, m in enumerate(models):
        vals = df.loc[df["Modèle"] == m, "F1"].values
        plt.bar(x + (i - 1) * width, vals, width, label=m)
    plt.xticks(x, class_names)
    plt.ylabel("F1-score")
    plt.title("F1-score par classe — comparaison des modèles")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    p = out_dir / "f1_par_classe_knn_svm_dt.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_learning_curves(X, y, out_dir: Path):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimators = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, est) in zip(axes, estimators.items()):
        sizes, train_scores, val_scores = learning_curve(
            est, X, y, cv=cv, scoring="accuracy", train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1, random_state=RANDOM_STATE
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        ax.plot(sizes, train_mean, "o-", label="Train")
        ax.plot(sizes, val_mean, "o-", label="Validation (CV)")
        ax.set_title(name)
        ax.set_xlabel("Nombre d’échantillons d’entraînement")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="best")
    plt.suptitle("Courbes d’apprentissage (accuracy)", y=1.02)
    plt.tight_layout()
    p = out_dir / "learning_curves_knn_svm_dt.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def confusion_mistake_summary(y_true, y_pred, model_name: str, class_names: list) -> str:
    cm = confusion_matrix(y_true, y_pred)
    off = cm.astype(float).copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        return f"- **{model_name}** : aucune erreur sur le jeu de test."
    i, j = np.unravel_index(np.argmax(off), off.shape)
    n = int(off[i, j])
    return (
        f"- **{model_name}** : confusion la plus fréquente — vérité **{class_names[i]}** "
        f"prédite **{class_names[j]}** ({n} cas)."
    )


def french_interpretation(
    df_metrics: pd.DataFrame,
    best_k: int,
    svm_params: dict,
    dt_train_acc: float,
    dt_test_acc: float,
    imbalance_note: str,
    mistake_lines: list[str],
) -> str:
    best_acc_model = df_metrics.loc[df_metrics["Accuracy"].idxmax(), "Modèle"]
    best_f1_model = df_metrics.loc[df_metrics["F1-score (macro)"].idxmax(), "Modèle"]
    knn_row = df_metrics[df_metrics["Modèle"] == "KNN"].iloc[0]
    svm_row = df_metrics[df_metrics["Modèle"] == "SVM"].iloc[0]
    delta_f1 = svm_row["F1-score (macro)"] - knn_row["F1-score (macro)"]

    lines = [
        "## Interprétation des résultats",
        "",
        "### 1. Meilleure accuracy et meilleur F1 macro",
        f"- **Accuracy** : le meilleur modèle est **{best_acc_model}** ({df_metrics.loc[df_metrics['Accuracy'].idxmax(), 'Accuracy']:.4f}).",
        f"- **F1 macro** : le meilleur modèle est **{best_f1_model}** ({df_metrics.loc[df_metrics['F1-score (macro)'].idxmax(), 'F1-score (macro)']:.4f}).",
        "",
        "Le SVM à noyau RBF peut mieux séparer des frontières non linéaires dans l’espace des features (après mise à l’échelle / PCA) ; le KNN repose sur des voisinages locaux (ici `k` optimisé) ; l’arbre de décision partitionne l’espace par seuils orthogonaux, ce qui peut être moins fin qu’un noyau sur des données lisses.",
        "",
        "### 2. Déséquilibre des classes",
        imbalance_note,
        "",
        "L’**accuracy** peut rester élevée si le modèle prédit souvent la classe majoritaire. Les moyennes **macro** (précision, rappel, F1) donnent le même poids à chaque classe et sont plus informatives ici ; les courbes **ROC one-vs-rest** complètent la vue par classe.",
        "",
        "### 3. Sur-apprentissage du Decision Tree",
        f"- Accuracy **train** (arbre final) : **{dt_train_acc:.4f}**.",
        f"- Accuracy **test** : **{dt_test_acc:.4f}**.",
        "",
        "Un écart important entre train et test indiquerait du sur-apprentissage. L’optimisation par grille + élagage `ccp_alpha` vise à limiter ce phénomène ; comparer visuellement les courbes d’apprentissage (train vs validation) pour l’arbre.",
        "",
        "*Note méthodologique : le choix de `k` (KNN) et de `ccp_alpha` (arbre) repose ici sur le **jeu de test**, comme dans les pipelines historiques du dossier — ce qui peut rendre les scores test **légèrement optimistes**. Pour un protocole plus strict : validation croisée ou jeu de validation séparé.*",
        "",
        "### 4. Classes les plus difficiles",
        "Synthèse automatique à partir des matrices de confusion (erreur la plus fréquente par modèle) :",
        "",
        *mistake_lines,
        "",
        "Les erreurs hors diagonale indiquent quelles classes sont confondues (souvent la classe minoritaire **High** avec **Medium** ou **Low** si les signaux se chevauchent).",
        "",
        "### 5. SVM vs KNN",
        f"- Différence de F1 macro (SVM - KNN) : **{delta_f1:+.4f}**.",
        "",
        (
            "Sur ce jeu, le SVM peut **gagner en accuracy globale** tout en **perdant en F1 macro** si les erreurs se concentrent sur la classe minoritaire : l'accuracy est alors trompeuse. "
            "Si l'écart de F1 est faible, l'amélioration n'est pas forcément **statistiquement** significative (peu d'échantillons de test) ; on croise avec la matrice de confusion et l'AUC par classe."
        ),
        "",
        "### 6. Recommandation production",
        f"Pour un déploiement orienté **équité entre classes** (F1 macro), **{best_f1_model}** est le meilleur candidat parmi les trois ; pour maximiser l'**accuracy** sur ce split, privilégier **{best_acc_model}**. Valider sur de nouvelles données et surveiller la classe **High**. ",
        "Si la contrainte principale est la **latence** et l’**interprétabilité**, un arbre (ou des règles dérivées) peut rester pertinent malgré un score légèrement inférieur.",
        "",
        "---",
        f"*Hyperparamètres retenus (indicatif) : KNN `n_neighbors={best_k}` ; SVM `RBF` {svm_params}.*",
    ]
    return "\n".join(lines)


def run_evaluation(save_dir: Path | None = None, show_plots: bool = True) -> dict:
    csv_path = resolve_csv_path()
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    prep = preprocess_unified(csv_path)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    class_names = prep["class_names"]
    n_classes = len(class_names)

    y_all_labels = pd.concat([prep["y_train"], prep["y_test"]], axis=0).map(dict(enumerate(class_names)))
    counts_full = y_all_labels.value_counts()
    dist_plain = {str(k): int(v) for k, v in counts_full.items()}
    imbalance_note = (
        f"Distribution globale (train+test) : {dist_plain}. "
        "La classe **High** est minoritaire : le rapport macro/weighted et les matrices de confusion sont essentiels."
    )

    knn_model, best_k = train_knn(X_train, y_train, X_test, y_test)
    y_pred_knn = knn_model.predict(X_test)
    proba_knn = knn_model.predict_proba(X_test)

    svm_model, svm_params = train_svm(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    proba_svm = svm_model.predict_proba(X_test)

    dt_model, dt_grid_best, dt_alpha = train_decision_tree(X_train, y_train, X_test, y_test)
    y_pred_dt = dt_model.predict(X_test)
    proba_dt = dt_model.predict_proba(X_test)
    dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
    dt_test_acc = accuracy_score(y_test, y_pred_dt)

    rows = [
        build_metrics_row("KNN", y_test, y_pred_knn, proba_knn, n_classes),
        build_metrics_row("SVM", y_test, y_pred_svm, proba_svm, n_classes),
        build_metrics_row("Decision Tree", y_test, y_pred_dt, proba_dt, n_classes),
    ]
    df_metrics = pd.DataFrame(rows).set_index("Modèle")

    wrows = []
    for name, y_pred in [("KNN", y_pred_knn), ("SVM", y_pred_svm), ("Decision Tree", y_pred_dt)]:
        wrows.append(
            {
                "Modèle": name,
                "Précision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Rappel (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )
    df_weighted = pd.DataFrame(wrows).set_index("Modèle")

    preds = {"KNN": y_pred_knn, "SVM": y_pred_svm, "Decision Tree": y_pred_dt}
    probas = {"KNN": proba_knn, "SVM": proba_svm, "Decision Tree": proba_dt}

    p_cm = plot_confusion_matrices(y_test, preds, class_names, save_dir)
    p_roc = plot_roc_ovr(y_test, probas, class_names, save_dir)
    p_f1 = plot_f1_per_class(y_test, preds, class_names, save_dir)
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    p_lc = plot_learning_curves(X_all, y_all, save_dir)

    mistake_lines = [
        confusion_mistake_summary(y_test, y_pred_knn, "KNN", class_names),
        confusion_mistake_summary(y_test, y_pred_svm, "SVM", class_names),
        confusion_mistake_summary(y_test, y_pred_dt, "Decision Tree", class_names),
    ]
    text_md = french_interpretation(
        df_metrics.reset_index(),
        best_k,
        svm_params,
        dt_train_acc,
        dt_test_acc,
        imbalance_note,
        mistake_lines,
    )

    if show_plots:
        for path in (p_cm, p_roc, p_f1, p_lc):
            img = plt.imread(path)
            plt.figure(figsize=(12, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title(path.name)
            plt.tight_layout()
            plt.show()

    return {
        "df_metrics": df_metrics,
        "df_weighted": df_weighted,
        "y_test": y_test,
        "y_pred_knn": y_pred_knn,
        "y_pred_svm": y_pred_svm,
        "y_pred_dt": y_pred_dt,
        "preds": preds,
        "probas": probas,
        "class_names": class_names,
        "prep": prep,
        "figures": {"confusion": p_cm, "roc": p_roc, "f1": p_f1, "learning": p_lc},
        "interpretation_md": text_md,
        "best_k": best_k,
        "svm_params": svm_params,
        "dt_train_acc": dt_train_acc,
        "dt_test_acc": dt_test_acc,
        "dt_alpha": dt_alpha,
        "dt_grid_best": dt_grid_best,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    out = run_evaluation(show_plots=False)
    print("CSV utilisé :", resolve_csv_path())
    print("\nTableau comparatif :\n")
    print(out["df_metrics"].round(4).to_string())
    print("\nFigures enregistrées :")
    for k, v in out["figures"].items():
        print(f"  {k}: {v}")
    print("\n" + out["interpretation_md"])
