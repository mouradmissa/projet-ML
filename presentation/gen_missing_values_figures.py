# -*- coding: utf-8 -*-
"""
Figures 5 et 6 — Valeurs manquantes : avant / après traitement.
Aligné sur le pipeline Priorité (médiane : Progress, Budget, Planned_Duration_Days ; mode : Priority).
Le CSV enrichi réel n’a pas de NaN : des lacunes sont simulées (seed fixe) pour illustrer la diapo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ML_ROOT / "Project-Management-2-enriched.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"

COLS_NUM = ["Progress", "Budget", "Planned_Duration_Days"]
COL_CAT = "Priority"
RNG_SEED = 42


def _impute_priority_style(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COLS_NUM:
        if col in out.columns and out[col].isnull().any():
            med = out[col].median()
            out[col] = out[col].fillna(med)
    if COL_CAT in out.columns and out[COL_CAT].isnull().any():
        mode = out[COL_CAT].mode()
        fill = mode.iloc[0] if len(mode) else ""
        out[COL_CAT] = out[COL_CAT].fillna(fill)
    return out


def _inject_demo_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Introduit des NaN reproductibles sur une copie (démonstration uniquement)."""
    demo = df.copy()
    rng = np.random.default_rng(RNG_SEED)
    n = len(demo)
    # ~5 indices par variable numérique
    for col in COLS_NUM:
        if col not in demo.columns:
            continue
        pick = rng.choice(n, size=min(6, n), replace=False)
        demo.loc[pick, col] = np.nan
    if COL_CAT in demo.columns:
        pick_p = rng.choice(n, size=min(4, n), replace=False)
        demo.loc[pick_p, COL_CAT] = np.nan
    return demo


def _plot_nan_bars(
    counts: pd.Series,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    cols = counts.index.tolist()
    vals = counts.values.astype(float)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#C1121F" if v > 0 else "#2A9D8F" for v in vals]
    y = np.arange(len(cols))
    ax.barh(y, vals, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(cols, fontsize=11)
    ax.set_xlabel("Nombre de valeurs manquantes (NaN)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9, color="#444444", wrap=True)
    ax.set_xlim(left=0)
    m = float(vals.max()) if len(vals) else 0.0
    ax.set_xlim(0, max(m * 1.15, 1.0))
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 0.08, i, str(int(v)), va="center", fontsize=10, color="#1a1a1a")
        else:
            ax.text(0.15, i, "0", va="center", fontsize=10, color="#1a1a1a", fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).drop_duplicates()
    demo_with_nan = _inject_demo_nans(df)

    track_cols = [c for c in COLS_NUM + [COL_CAT] if c in demo_with_nan.columns]
    before_counts = demo_with_nan[track_cols].isnull().sum()
    after_df = _impute_priority_style(demo_with_nan)
    after_counts = after_df[track_cols].isnull().sum()

    assert after_counts.sum() == 0, "Après imputation, attendu 0 NaN sur les colonnes suivies."

    note = (
        "Démonstration : NaN injectés aléatoirement (seed=42) sur une copie du jeu — "
        "le fichier enrichi livré ne contient pas de lacunes. "
        "Règles : médiane pour Progress, Budget, Planned_Duration_Days ; mode pour Priority."
    )

    _plot_nan_bars(
        before_counts,
        "Figure 5 — Avant traitement",
        note + " | Comptage des NaN par colonne ciblée.",
        OUT_DIR / "pretraitement_nan_avant.png",
    )
    _plot_nan_bars(
        after_counts,
        "Figure 6 — Après traitement",
        note + " | Aucune valeur manquante restante sur ces colonnes.",
        OUT_DIR / "pretraitement_nan_apres.png",
    )

    print("Figures enregistrées :")
    print(f"  {OUT_DIR / 'pretraitement_nan_avant.png'}")
    print(f"  {OUT_DIR / 'pretraitement_nan_apres.png'}")


if __name__ == "__main__":
    main()
