# -*- coding: utf-8 -*-
"""
Figures 8 et 9 — Boxplots Budget avant / après plafonnement IQR (k = 1,5).
Pipeline Progress : Hours Spent, Budget, Actual Cost (ici : Budget uniquement).
Pipeline Priorité : Budget, Planned_Duration_Days (ici : Budget uniquement).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ML_ROOT / "Project-Management-2-enriched.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"

COLS_IQR_PROGRESS = ["Hours Spent", "Budget", "Actual Cost"]
FEATURE_PRIO = ["Progress", "Budget", "Planned_Duration_Days"]


def cap_iqr_series(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return series
    low, high = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=low, upper=high)


def cap_iqr_inplace_df(df: pd.DataFrame, column: str, k: float = 1.5) -> None:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    lo = Q1 - k * iqr
    hi = Q3 + k * iqr
    df[column] = df[column].clip(lower=lo, upper=hi)


def _prepare_progress_pre_iqr() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
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
    return df


def _prepare_priority_pre_iqr() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df_clean = df.drop_duplicates().copy()
    for col in FEATURE_PRIO:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    if "Priority" in df_clean.columns and df_clean["Priority"].isnull().sum() > 0:
        df_clean["Priority"].fillna(df_clean["Priority"].mode()[0], inplace=True)
    const_cols = [c for c in df_clean.columns if df_clean[c].nunique() == 1]
    if const_cols:
        df_clean.drop(columns=const_cols, inplace=True)
    return df_clean


def _plot_before_after_boxplot(
    avant: pd.Series,
    apres: pd.Series,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    data_a = avant.dropna().astype(float)
    data_b = apres.dropna().astype(float)

    colors = ("#457B9D", "#2A9D8F")
    for ax, data, label, color in zip(
        axes,
        (data_a, data_b),
        ("Avant IQR", "Après IQR (plafonnement)"),
        colors,
    ):
        bp = ax.boxplot(
            [data.values],
            vert=True,
            tick_labels=[label],
            patch_artist=True,
            widths=0.45,
            showfliers=True,
            flierprops=dict(marker="o", markersize=4, alpha=0.65),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_ylabel("Budget", fontsize=11)
        ax.grid(axis="y", alpha=0.35)
        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9.5, color="#444444")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    # --- Figure 8 : Progress ---
    df_p = _prepare_progress_pre_iqr()
    if "Budget" not in df_p.columns:
        raise ValueError("Colonne Budget absente.")
    budget_avant_prog = df_p["Budget"].copy()
    budget_apres_prog = cap_iqr_series(budget_avant_prog.copy())
    # Alignement strict avec la boucle du pipeline (même résultat pour Budget seul)
    _plot_before_after_boxplot(
        budget_avant_prog,
        budget_apres_prog,
        "Figure 8 — Pipeline Progress : Budget avant / après IQR (k = 1,5)",
        "Même logique que cap_iqr sur Hours Spent, Budget, Actual Cost (régression / RF avancement).",
        OUT_DIR / "pretraitement_outliers_budget_progress.png",
    )

    # --- Figure 9 : Priorité ---
    df_pr = _prepare_priority_pre_iqr()
    budget_avant_prio = df_pr["Budget"].copy()
    df_capped = df_pr.copy()
    cap_iqr_inplace_df(df_capped, "Budget")
    budget_apres_prio = df_capped["Budget"]

    _plot_before_after_boxplot(
        budget_avant_prio,
        budget_apres_prio,
        "Figure 9 — Pipeline Priorité : Budget avant / après IQR (k = 1,5)",
        "Même logique que rf_priority_pipeline / knn_priority_pipeline (Budget et Planned_Duration_Days).",
        OUT_DIR / "pretraitement_outliers_budget_priorite.png",
    )

    print("Figures enregistrées :")
    print(f"  {OUT_DIR / 'pretraitement_outliers_budget_progress.png'}")
    print(f"  {OUT_DIR / 'pretraitement_outliers_budget_priorite.png'}")


if __name__ == "__main__":
    main()
