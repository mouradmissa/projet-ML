# -*- coding: utf-8 -*-
"""
Figures 10 et 11 — Avant / après StandardScaler (z-score).
Données : Progress, Budget, Planned_Duration_Days après nettoyage pipeline Priorité (comme KNN).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ML_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ML_ROOT / "Project-Management-2-enriched.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"

FEATURE_COLS = ["Progress", "Budget", "Planned_Duration_Days"]


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> None:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    lo = Q1 - 1.5 * iqr
    hi = Q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lo, upper=hi)


def load_x_priority_style() -> pd.DataFrame:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
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
    return df_clean[FEATURE_COLS].copy()


def _fig10_before(X: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colors = ("#E63946", "#457B9D", "#2A9D8F")
    short = ["Progress\n[0–1]", "Budget", "Durée planifiée\n(jours)"]
    for ax, col, lab, c in zip(axes, FEATURE_COLS, short, colors):
        data = X[col].astype(float).dropna()
        bp = ax.boxplot(
            [data.values],
            vert=True,
            tick_labels=[col.replace("_", " ")],
            patch_artist=True,
            widths=0.5,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylabel("Valeur brute", fontsize=10)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle(
        "Figure 10 — Avant standardisation (échelles très différentes)",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    fig.text(
        0.5,
        -0.02,
        "Les variables ne sont pas comparables en distance (KNN) ni stables pour la régression linéaire sans mise à l’échelle.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig11_after(X: pd.DataFrame, out_path: Path) -> None:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.values)
    colnames = [c.replace("_", " ") for c in FEATURE_COLS]
    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    positions = np.arange(1, len(FEATURE_COLS) + 1)
    bp = ax.boxplot(
        [Z[:, i] for i in range(Z.shape[1])],
        positions=positions,
        vert=True,
        tick_labels=colnames,
        patch_artist=True,
        widths=0.55,
    )
    colors = ("#E63946", "#457B9D", "#2A9D8F")
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_ylabel("Valeur après StandardScaler (z-score)", fontsize=11)
    ax.grid(axis="y", alpha=0.35)
    stats_txt = " ; ".join(
        f"{colnames[i]}: μ≈{means[i]:.2f}, σ≈{stds[i]:.2f}" for i in range(len(colnames))
    )
    fig.suptitle(
        "Figure 11 — Après StandardScaler (moyennes ≈ 0, écarts-types ≈ 1)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.text(0.5, -0.06, stats_txt, ha="center", fontsize=9, color="#444444")
    fig.text(
        0.5,
        -0.12,
        "Formule : z = (x − μ) / σ (ajustée sur le jeu d’entraînement ; ici illustré sur tout le jeu nettoyé).",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    X = load_x_priority_style()
    _fig10_before(X, OUT_DIR / "pretraitement_scaling_avant.png")
    _fig11_after(X, OUT_DIR / "pretraitement_scaling_apres.png")
    print("Figures enregistrées :")
    print(f"  {OUT_DIR / 'pretraitement_scaling_avant.png'}")
    print(f"  {OUT_DIR / 'pretraitement_scaling_apres.png'}")


if __name__ == "__main__":
    main()
