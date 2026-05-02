# -*- coding: utf-8 -*-
"""Figures 3 et 4 : encodage catégoriel (ordinal + one-hot) — aligné sur le pipeline Progress."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ML_ROOT / "Project-Management-2-enriched.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"

ORDER_PROJECT_STATUS = ["On Hold", "Behind", "On Track", "Completed"]
ORDER_TASK_STATUS = ["Pending", "In Progress", "Completed"]

N_ROWS = 8
MAX_AFTER_COLS = 12  # lisibilité sur la diapo


def _df_to_table_fig(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: Path,
    *,
    figsize: tuple[float, float] = (14, 4.5),
    fontsize: int = 10,
    col_scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table_df = df.copy()
    table_df.columns = [str(c).replace("_", " ") for c in table_df.columns]

    table = ax.table(
        cellText=table_df.astype(str).values,
        colLabels=list(table_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(col_scale, 1.9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F8F9FA" if row % 2 else "white")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.88, subtitle, ha="center", fontsize=9.5, color="#444444")
    plt.tight_layout(rect=[0, 0.02, 1, 0.84])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def encode_like_progress_pipeline(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Ordinal (Project / Task Status) puis get_dummies(drop_first=True) sur le reste des objets."""
    work = df_slice.copy()

    if "Project Status" in work.columns:
        ps_map = {v: i for i, v in enumerate(ORDER_PROJECT_STATUS)}
        unk = len(ORDER_PROJECT_STATUS)
        work["Project_Status_ord"] = (
            work["Project Status"].map(ps_map).fillna(unk).astype(int)
        )
        work = work.drop(columns=["Project Status"])

    if "Task Status" in work.columns:
        ts_map = {v: i for i, v in enumerate(ORDER_TASK_STATUS)}
        unk = len(ORDER_TASK_STATUS)
        work["Task_Status_ord"] = work["Task Status"].map(ts_map).fillna(unk).astype(int)
        work = work.drop(columns=["Task Status"])

    cat_cols = work.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        work = pd.get_dummies(work, columns=cat_cols, drop_first=True)

    ord_front = []
    for c in ["Project_Status_ord", "Task_Status_ord"]:
        if c in work.columns:
            ord_front.append(c)
    rest = [c for c in work.columns if c not in ord_front]
    rest.sort()
    return work[ord_front + rest]


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).drop_duplicates().head(N_ROWS)

    cols_before = [
        "Project Status",
        "Task Status",
        "Project Type",
        "Assigned To",
        "Priority",
    ]
    missing = [c for c in cols_before if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    before = df[cols_before].copy()

    after_full = encode_like_progress_pipeline(before)
    if after_full.shape[1] > MAX_AFTER_COLS:
        after = after_full.iloc[:, :MAX_AFTER_COLS]
        extra = after_full.shape[1] - MAX_AFTER_COLS
        sub_after = (
            f"Extrait des {after_full.shape[1]} colonnes numériques (+ {extra} autres colonnes dummy non affichées)."
        )
    else:
        after = after_full
        sub_after = (
            "Ordinal : Project_Status_ord, Task_Status_ord — One-Hot : pd.get_dummies(..., drop_first=True)."
        )

    _df_to_table_fig(
        before,
        "Figure 3 — Avant encodage",
        "Statuts et catégories au format texte (extrait du jeu enrichi).",
        OUT_DIR / "pretraitement_encodage_avant.png",
        figsize=(13, 4.6),
        fontsize=11,
        col_scale=1.05,
    )
    _df_to_table_fig(
        after,
        "Figure 4 — Après encodage",
        sub_after,
        OUT_DIR / "pretraitement_encodage_apres.png",
        figsize=(16, 4.8),
        fontsize=9,
        col_scale=1.0,
    )

    print("Figures enregistrées :")
    print(f"  {OUT_DIR / 'pretraitement_encodage_avant.png'}")
    print(f"  {OUT_DIR / 'pretraitement_encodage_apres.png'}")


if __name__ == "__main__":
    main()
