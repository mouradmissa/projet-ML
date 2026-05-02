"""
Évaluation comparative Random Forest Regressor vs Régression linéaire — prédiction de Progress.
Prétraitement aligné sur rf_progress_pipeline.py et lr_progress_pipeline.py (même split random_state=42).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

_EVAL = Path(__file__).resolve().parent
_PRED = _EVAL.parent
sys.path.insert(0, str(_PRED / "shared"))
from progress_inference import ml_project_root  # noqa: E402

ROOT = ml_project_root(Path(__file__))
DATA_PATH = ROOT / "Project-Management-2-enriched.csv"
TARGET = "Progress"
REFERENCE_DATE = pd.Timestamp("2026-04-12")
ORDER_PROJECT_STATUS = ["On Hold", "Behind", "On Track", "Completed"]
ORDER_TASK_STATUS = ["Pending", "In Progress", "Completed"]
COLS_IQR = ["Hours Spent", "Budget", "Actual Cost"]
DROP_LEAKAGE_AND_IDS = [
    "Risk_Level",
    "Project ID",
    "Project Name",
    "Task Name",
    "Location",
]
RANDOM_STATE = 42


def cap_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return series
    low, high = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=low, upper=high)


def preprocess_progress(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """Retourne X, y, noms de features, meta (effectifs)."""
    df = pd.read_csv(csv_path)
    n_init = len(df)

    df = df.drop_duplicates()

    n = len(df)
    rows_with_na = df.isnull().any(axis=1).sum()
    pct_na_rows = 100.0 * rows_with_na / n if n else 0.0
    if rows_with_na > 0:
        if pct_na_rows < 5.0:
            df = df.dropna()
        else:
            for c in df.select_dtypes(include=[np.number]).columns:
                df[c] = df[c].fillna(df[c].median())
            for c in df.select_dtypes(include=["object"]).columns:
                mode = df[c].mode()
                df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "")

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)

    for col in COLS_IQR:
        if col in df.columns:
            df[col] = cap_iqr(df[col])

    for c in ["Start Date", "End Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

    if "Start Date" in df.columns and "End Date" in df.columns:
        df["planned_duration_days"] = (df["End Date"] - df["Start Date"]).dt.days
        df["days_since_start"] = (REFERENCE_DATE - df["Start Date"]).dt.days
        df["remaining_days"] = (df["End Date"] - REFERENCE_DATE).dt.days
        df = df.drop(columns=["Start Date", "End Date"])
        if "Planned_Duration_Days" in df.columns:
            df = df.drop(columns=["Planned_Duration_Days"])

    to_drop = [c for c in DROP_LEAKAGE_AND_IDS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    high_card = []
    for c in df.select_dtypes(include=["object"]).columns:
        if c == TARGET:
            continue
        if df[c].nunique(dropna=False) > 10:
            high_card.append(c)
    if high_card:
        df = df.drop(columns=high_card, errors="ignore")

    if "Project Status" in df.columns:
        ps_map = {v: i for i, v in enumerate(ORDER_PROJECT_STATUS)}
        unk = len(ORDER_PROJECT_STATUS)
        df["Project_Status_ord"] = df["Project Status"].map(ps_map).fillna(unk).astype(int)
        df = df.drop(columns=["Project Status"])

    if "Task Status" in df.columns:
        ts_map = {v: i for i, v in enumerate(ORDER_TASK_STATUS)}
        unk = len(ORDER_TASK_STATUS)
        df["Task_Status_ord"] = df["Task Status"].map(ts_map).fillna(unk).astype(int)
        df = df.drop(columns=["Task Status"])

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if TARGET not in df.columns:
        raise ValueError("Colonne cible absente.")

    y_raw = df[TARGET].astype(float)
    if y_raw.min() < 0 or y_raw.max() > 1:
        ymin, ymax = y_raw.min(), y_raw.max()
        df[TARGET] = (y_raw - ymin) / (ymax - ymin) if ymax > ymin else y_raw.clip(0, 1)
    else:
        df[TARGET] = y_raw.clip(0, 1)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    meta = {
        "n_init": n_init,
        "n_final": len(df),
        "n_features": X.shape[1],
        "csv": str(csv_path),
    }
    return X, y, feature_names, meta


def safe_mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.abs(y) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100.0)


def regression_metrics_row(model_name: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "Modèle": model_name,
        "R²": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": safe_mape(y_true, y_pred),
    }


def train_rf(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_lr(X_train, y_train):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def plot_scatter_truth_pred(y_test, y_pred_rf, y_pred_lr, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    lims = (min(y_test.min(), y_pred_rf.min(), y_pred_lr.min()) - 0.02, max(y_test.max(), y_pred_rf.max(), y_pred_lr.max()) + 0.02)
    for ax, title, yp in zip(axes, ["Régression linéaire", "Random Forest"], [y_pred_lr, y_pred_rf]):
        ax.scatter(y_test, yp, alpha=0.65, edgecolors="k", linewidths=0.3, s=40)
        ax.plot(lims, lims, "k--", lw=1.2, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y_test (Progress réel)")
        ax.set_ylabel("y_pred")
        ax.set_title(title)
        ax.legend(loc="upper left")
    plt.tight_layout()
    p = out_dir / "scatter_ytest_vs_ypred_rf_lr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_residual_hist(y_test, y_pred_rf, y_pred_lr, out_dir: Path) -> Path:
    res_lr = np.asarray(y_test) - np.asarray(y_pred_lr)
    res_rf = np.asarray(y_test) - np.asarray(y_pred_rf)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, title, res in zip(axes, ["Régression linéaire", "Random Forest"], [res_lr, res_rf]):
        sns.histplot(res, kde=True, ax=ax, color="steelblue", edgecolor="black")
        ax.axvline(0, color="crimson", ls="--", lw=1.2)
        ax.axvline(np.mean(res), color="green", ls="-", lw=1.2, label=f"Moyenne = {np.mean(res):.4f}")
        ax.set_title(title)
        ax.set_xlabel("Résidu (y_test − y_pred)")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    p = out_dir / "residuals_hist_kde_rf_lr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_residual_boxplot(y_test, y_pred_rf, y_pred_lr, out_dir: Path) -> Path:
    df = pd.DataFrame(
        {
            "Résidu": np.concatenate(
                [np.asarray(y_test) - np.asarray(y_pred_lr), np.asarray(y_test) - np.asarray(y_pred_rf)]
            ),
            "Modèle": ["Régression linéaire"] * len(y_test) + ["Random Forest"] * len(y_test),
        }
    )
    plt.figure(figsize=(6, 4.5))
    sns.boxplot(data=df, x="Modèle", y="Résidu", palette="Set2")
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.title("Comparaison des résidus (jeu de test)")
    plt.tight_layout()
    p = out_dir / "residuals_boxplot_rf_lr.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_rf_top3_importance(rf_model, feature_names: list[str], out_dir: Path) -> Path:
    imp = rf_model.feature_importances_
    ranked = sorted(zip(imp, feature_names), key=lambda t: t[0], reverse=True)[:3]
    names = [n for _, n in ranked][::-1]
    vals = [float(i) for i, _ in ranked][::-1]
    plt.figure(figsize=(7, 3.5))
    plt.barh(names, vals, color="darkgreen", edgecolor="black")
    plt.xlabel("Importance (Random Forest)")
    plt.title("Top 3 variables — Random Forest (Progress)")
    plt.tight_layout()
    p = out_dir / "rf_progress_top3_importance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def plot_residuals_vs_fitted_lr(y_test, y_pred_lr, out_dir: Path) -> Path:
    """Aide à discuter hétéroscédasticité / tendance des résidus (régression linéaire)."""
    res = np.asarray(y_test) - np.asarray(y_pred_lr)
    pred = np.asarray(y_pred_lr)
    plt.figure(figsize=(5.5, 4.2))
    plt.scatter(pred, res, alpha=0.65, edgecolors="k", linewidths=0.3)
    plt.axhline(0, color="crimson", ls="--", lw=1)
    plt.xlabel("ŷ (régression linéaire)")
    plt.ylabel("Résidu (y − ŷ)")
    plt.title("Résidus vs valeurs prédites — Régression linéaire")
    plt.tight_layout()
    p = out_dir / "residuals_vs_fitted_lr.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def french_interpretation(
    df_metrics: pd.DataFrame,
    n_test: int,
    res_lr_mean: float,
    spearman_abs_res_vs_pred: float,
    spearman_p: float,
    delta_r2: float,
    delta_rmse: float,
    top3_features: list[tuple[str, float]],
) -> str:
    lr = df_metrics.loc["Régression linéaire"]
    rf = df_metrics.loc["Random Forest"]
    best_r2 = df_metrics["R²"].idxmax()
    best_rmse = df_metrics["RMSE"].idxmin()

    het = (
        f"Corrélation de Spearman entre |résidu| et ŷ (LR) : **{spearman_abs_res_vs_pred:.3f}** (p ≈ {spearman_p:.3g}). "
        "Une corrélation **positive** marquée et **significative** suggère que l’erreur absolue augmente avec ŷ (**hétéroscédasticité**). "
        "Une corrélation **négative** indique plutôt des erreurs plus fortes pour les **faibles** prédictions."
    )
    if abs(spearman_abs_res_vs_pred) < 0.2 or spearman_p > 0.1:
        het += " Sur ce split, l’indicateur reste **peu conclusif** (peu de points de test) : prioriser le graphique résidus vs ŷ."

    top3_txt = ", ".join(f"**{n}** ({v:.3f})" for n, v in top3_features)

    lines = [
        "## Interprétation des résultats",
        "",
        "### 1. Meilleur R² et plus faible RMSE",
        f"- **R²** : meilleur modèle **{best_r2}** ({df_metrics.loc[best_r2, 'R²']:.4f}).",
        f"- **RMSE** : meilleur modèle **{best_rmse}** ({df_metrics.loc[best_rmse, 'RMSE']:.4f}).",
        "",
        "La **forêt aléatoire** agrège des arbres sur des sous-échantillons et peut capturer des **non-linéarités** et des **interactions** entre variables (statuts de tâche, temps écoulé, coûts, etc.), ce qui aide souvent pour un indicateur d’**avancement** qui n’est pas strictement une fonction affine des entrées.",
        "La **régression linéaire** impose une structure additive linéaire dans l’espace des variables (après encodage / scaling) : elle sert de **baseline** interprétable mais est souvent moins flexible.",
        "",
        "### 2. Résidus de la régression linéaire",
        f"- **Moyenne des résidus** (test) : **{res_lr_mean:.6f}** (un biais proche de 0 est attendu si le modèle est bien spécifié ; un léger écart peut venir du petit échantillon ou de la non-linéarité).",
        f"- **Hétéroscédasticité / tendance** : {het}",
        "Consulter aussi le graphique *résidus vs ŷ* : un motif en entonnoir ou une courbe résiduelle indiquerait des hypothèses classiques du MCO mal respectées.",
        "",
        "### 3. La Random Forest améliore-t-elle nettement ?",
        f"- **Δ R²** (RF − LR) : **{delta_r2:+.4f}**.",
        f"- **Δ RMSE** (LR − RF, gain si positif) : **{delta_rmse:+.6f}** (réduction d’erreur quadratique moyenne en passant au RF).",
        "",
        "Si Δ R² est substantiel et le RMSE du RF clairement plus bas, le gain est **pratiquement** notable pour ce jeu de données. Sur seulement quelques dizaines de points de test, il faut rester prudent sur la **variabilité** d’estimation.",
        "",
        "### 4. Importance des variables (Random Forest)",
        f"Top 3 : {top3_txt}.",
        "",
        "Les variables liées au **calendrier** (ex. temps écoulé / restant), au **travail effectué** (heures) et au **statut** de la tâche ou du projet sont souvent cohérentes avec l’intuition : elles résument « où en est la tâche » dans son déroulement.",
        "",
        "### 5. Limites de l’évaluation",
        f"- **Taille du jeu de test** : **{n_test}** observations — les métriques (surtout MAPE avec des **Progress** proches de 0) peuvent être **instables**.",
        "- **MAPE** : lorsque la vraie valeur est très petite, le ratio d’erreur relative explose ; le calcul exclut les |y| quasi nuls (seuil numérique).",
        "- **Valeurs extrêmes** : le plafonnement IQR sur certaines colonnes réduit l’influence des outliers mais peut aussi **compresser** l’information.",
        "",
        "---",
        "*Prétraitement et split identiques aux scripts `rf_progress_pipeline.py` et `lr_progress_pipeline.py` (test_size=0.2, random_state=42).*",
    ]
    return "\n".join(lines)


def run_evaluation(save_dir: Path | None = None, show_plots: bool = False) -> dict:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"CSV introuvable : {DATA_PATH}")

    if save_dir is None:
        save_dir = _EVAL / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, meta = preprocess_progress(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    n_test = len(y_test)

    lr_pipe = train_lr(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)

    rf_model, rf_params = train_rf(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rows = [
        regression_metrics_row("Régression linéaire", y_test, y_pred_lr),
        regression_metrics_row("Random Forest", y_test, y_pred_rf),
    ]
    df_metrics = pd.DataFrame(rows).set_index("Modèle")

    res_lr = np.asarray(y_test) - np.asarray(y_pred_lr)
    res_lr_mean = float(np.mean(res_lr))
    abs_res = np.abs(res_lr)
    spearman_abs_res_vs_pred, spearman_p = spearmanr(abs_res, np.asarray(y_pred_lr))

    delta_r2 = float(df_metrics.loc["Random Forest", "R²"] - df_metrics.loc["Régression linéaire", "R²"])
    delta_rmse = float(
        df_metrics.loc["Régression linéaire", "RMSE"] - df_metrics.loc["Random Forest", "RMSE"]
    )

    ranked = sorted(zip(rf_model.feature_importances_, feature_names), key=lambda t: t[0], reverse=True)
    top3 = [(n, float(i)) for i, n in ranked[:3]]

    p_scatter = plot_scatter_truth_pred(y_test, y_pred_rf, y_pred_lr, save_dir)
    p_hist = plot_residual_hist(y_test, y_pred_rf, y_pred_lr, save_dir)
    p_box = plot_residual_boxplot(y_test, y_pred_rf, y_pred_lr, save_dir)
    p_imp = plot_rf_top3_importance(rf_model, feature_names, save_dir)
    p_resfit = plot_residuals_vs_fitted_lr(y_test, y_pred_lr, save_dir)

    text_md = french_interpretation(
        df_metrics,
        n_test=n_test,
        res_lr_mean=res_lr_mean,
        spearman_abs_res_vs_pred=float(spearman_abs_res_vs_pred),
        spearman_p=float(spearman_p) if spearman_p == spearman_p else 1.0,
        delta_r2=delta_r2,
        delta_rmse=delta_rmse,
        top3_features=top3,
    )

    if show_plots:
        for path in (p_scatter, p_hist, p_box, p_imp, p_resfit):
            img = plt.imread(path)
            plt.figure(figsize=(11, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title(path.name)
            plt.tight_layout()
            plt.show()

    return {
        "df_metrics": df_metrics,
        "y_test": y_test,
        "y_pred_lr": y_pred_lr,
        "y_pred_rf": y_pred_rf,
        "meta": meta,
        "rf_params": rf_params,
        "figures": {
            "scatter": p_scatter,
            "residual_hist": p_hist,
            "residual_box": p_box,
            "rf_top3": p_imp,
            "residuals_vs_fitted_lr": p_resfit,
        },
        "interpretation_md": text_md,
        "res_lr_mean": res_lr_mean,
        "top3_importance": top3,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out = run_evaluation(show_plots=False)
    print("CSV :", out["meta"]["csv"])
    print("Effectifs : init", out["meta"]["n_init"], "| final", out["meta"]["n_final"], "| features", out["meta"]["n_features"])
    print("\nHyperparamètres RF :", out["rf_params"])
    print("\nTableau comparatif :\n")
    print(out["df_metrics"].round(6).to_string())
    print("\nFigures :")
    for k, v in out["figures"].items():
        print(f"  {k}: {v}")
    print("\n" + out["interpretation_md"])
