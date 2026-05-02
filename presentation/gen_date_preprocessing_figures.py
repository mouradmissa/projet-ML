# -*- coding: utf-8 -*-
"""Génère les figures « avant / après » : conversion des dates en durées (jours)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ML_ROOT / "Project-Management-2-enriched.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
REFERENCE_DATE = pd.Timestamp("2026-04-12")
N_ROWS = 8


def _df_to_table_fig(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: Path,
    col_width_scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table_df = df.copy()
    table_df.columns = [c.replace("_", " ") for c in table_df.columns]

    table = ax.table(
        cellText=table_df.astype(str).values,
        colLabels=list(table_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(col_width_scale, 1.85)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F8F9FA" if row % 2 else "white")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.88, subtitle, ha="center", fontsize=10, color="#444444", wrap=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.84])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).drop_duplicates().head(N_ROWS)

    before_cols = ["Start Date", "End Date", "Planned_Duration_Days"]
    before = df[before_cols].copy()

    start = pd.to_datetime(df["Start Date"], dayfirst=True, errors="coerce")
    end = pd.to_datetime(df["End Date"], dayfirst=True, errors="coerce")
    after = pd.DataFrame(
        {
            "planned_duration_days": (end - start).dt.days,
            "days_since_start": (REFERENCE_DATE - start).dt.days,
            "remaining_days": (end - REFERENCE_DATE).dt.days,
        },
        index=df.index,
    )

    _df_to_table_fig(
        before,
        "Avant transformation",
        "Dates au format texte dans le CSV (ex. jj/mm/aaaa) — colonne Planned_Duration_Days conservée à ce stade.",
        OUT_DIR / "pretraitement_dates_avant.png",
        col_width_scale=1.15,
    )
    _df_to_table_fig(
        after,
        "Après transformation",
        f"Durées en jours (référence : {REFERENCE_DATE.date()}) — Start/End supprimées, Planned_Duration_Days retirée pour éviter la redondance.",
        OUT_DIR / "pretraitement_dates_apres.png",
        col_width_scale=1.05,
    )

    print(f"Figures enregistrées :\n  {OUT_DIR / 'pretraitement_dates_avant.png'}\n  {OUT_DIR / 'pretraitement_dates_apres.png'}")


if __name__ == "__main__":
    main()
