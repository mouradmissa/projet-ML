"""
Construction du vecteur de features Progress (aligné sur les pipelines MODEL).
Partagé par les dossiers FRONT (Flask) de chaque algorithme.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS_IQR = ["Hours Spent", "Budget", "Actual Cost"]


def project_root() -> Path:
    """Racine du dépôt ML (fichier CSV à la racine)."""
    return Path(__file__).resolve().parents[2]


def ml_project_root(from_file: Path | None = None) -> Path:
    """
    Racine du depot (dossier contenant Project-Management-2-enriched.csv).
    Permet de lancer les apps Flask depuis frontend/<algo>/ sans casser les chemins.
    """
    start = (from_file or Path(__file__)).resolve()
    cur = start.parent
    for _ in range(10):
        if (cur / "Project-Management-2-enriched.csv").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(__file__).resolve().parents[2]  # shared → Prediction avancement → ML


_DF_REF_CACHE: dict[str, pd.DataFrame] = {}


def _ref_df(csv_path: Path) -> pd.DataFrame:
    key = str(csv_path.resolve())
    if key not in _DF_REF_CACHE:
        _DF_REF_CACHE[key] = pd.read_csv(csv_path).drop_duplicates()
    return _DF_REF_CACHE[key]


def _delta_days(ts_a: pd.Timestamp, ts_b: pd.Timestamp) -> int:
    """Nombre de jours entre deux dates ; 0 si date invalide (NaT) ou erreur."""
    if pd.isna(ts_a) or pd.isna(ts_b):
        return 0
    try:
        td = ts_a - ts_b
        if pd.isna(td):
            return 0
        return int(td.days)
    except (ValueError, TypeError, OverflowError):
        return 0


def iqr_clip_value(val: float, series: pd.Series) -> float:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return float(val)
    low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    return float(np.clip(val, low, high))


def build_progress_feature_frame(
    data: dict,
    feature_cols: list[str],
    reference_date: pd.Timestamp,
    order_ps: list[str],
    order_ts: list[str],
    data_path: Path,
) -> pd.DataFrame:
    df_ref = _ref_df(data_path)
    start = pd.to_datetime(
        data.get("start_date", "01/01/2024"), dayfirst=True, errors="coerce"
    )
    end = pd.to_datetime(
        data.get("end_date", "31/12/2024"), dayfirst=True, errors="coerce"
    )
    planned_duration_days = _delta_days(end, start)
    days_since_start = _delta_days(reference_date, start)
    remaining_days = _delta_days(end, reference_date)

    budget = float(data.get("budget", 0) or 0)
    actual_cost = float(data.get("actual_cost", 0) or 0)
    hours = float(data.get("hours_spent", 0) or 0)
    hist = float(data.get("assignee_historical_task_count", 0) or 0)

    for col in COLS_IQR:
        if col not in df_ref.columns:
            continue
        if col == "Hours Spent":
            hours = iqr_clip_value(hours, df_ref[col])
        elif col == "Budget":
            budget = iqr_clip_value(budget, df_ref[col])
        elif col == "Actual Cost":
            actual_cost = iqr_clip_value(actual_cost, df_ref[col])

    bu = (actual_cost / budget) if budget > 0 else 0.0

    ps_map = {v: i for i, v in enumerate(order_ps)}
    ts_map = {v: i for i, v in enumerate(order_ts)}
    project_status = data.get("project_status", order_ps[0])
    task_status = data.get("task_status", order_ts[0])
    project_status_ord = int(ps_map.get(project_status, len(order_ps)))
    task_status_ord = int(ts_map.get(task_status, len(order_ts)))

    row = {c: 0.0 for c in feature_cols}
    row["Hours Spent"] = hours
    row["Budget"] = budget
    row["Actual Cost"] = actual_cost
    row["Budget_Utilization"] = bu
    row["Assignee_Historical_Task_Count"] = hist
    row["planned_duration_days"] = float(planned_duration_days)
    row["days_since_start"] = float(days_since_start)
    row["remaining_days"] = float(remaining_days)
    row["Project_Status_ord"] = float(project_status_ord)
    row["Task_Status_ord"] = float(task_status_ord)

    pt = data.get("project_type", "Construction")
    if pt != "Construction":
        c = "Project Type_%s" % pt
        if c in row:
            row[c] = 1.0

    pr = data.get("priority", "High")
    if pr == "Low" and "Priority_Low" in row:
        row["Priority_Low"] = 1.0
    elif pr == "Medium" and "Priority_Medium" in row:
        row["Priority_Medium"] = 1.0

    tid = data.get("task_id", "T001")
    if tid == "T002" and "Task ID_T002" in row:
        row["Task ID_T002"] = 1.0
    elif tid == "T003" and "Task ID_T003" in row:
        row["Task ID_T003"] = 1.0

    assignee = data.get("assigned_to", "Alice")
    if assignee != "Alice":
        c = "Assigned To_%s" % assignee
        if c in row:
            row[c] = 1.0

    return pd.DataFrame([row], columns=feature_cols)
