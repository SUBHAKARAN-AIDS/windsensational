# app/db/anomaly_json_store.py
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# Stored alongside the backend code, not in a DB
STORE_PATH = Path(__file__).parent / "anomaly_daily_scores.json"


def _load_store() -> dict:
    """
    Structure on disk:
    {
      "DZH1": {
        "2026-01-22": {"avg_score": 0.12, "max_score": 0.34, "anomaly_count": 2, "total_predictions": 96},
        "2026-01-23": {...}
      },
      "DZH2": {...}
    }
    """
    if not STORE_PATH.exists():
        return {}
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read anomaly score store, treating as empty: {e}")
        return {}


def _save_store(data: dict):
    """
    Atomic write: write to a temp file, then rename over the real file.
    Prevents a crash mid-write from corrupting the store.
    """
    tmp_path = STORE_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, STORE_PATH)
    except OSError as e:
        logger.error(f"Failed to write anomaly score store: {e}", exc_info=True)


def save_daily_anomaly_scores(df: pd.DataFrame, ensemble_score_norm, ensemble_pred):
    """
    Aggregates row-level ensemble scores by turbine_id + day and upserts
    into the JSON store. Called from the live endpoint (write-on-read)
    and the scheduled job.
    """
    if df.empty or 'turbine_id' not in df.columns or 'time' not in df.columns:
        return

    work = df[['turbine_id', 'time']].copy()
    work['score'] = ensemble_score_norm
    work['is_anomaly'] = ensemble_pred
    work['day'] = pd.to_datetime(work['time']).dt.date.astype(str)

    grouped = work.groupby(['turbine_id', 'day']).agg(
        avg_score=('score', 'mean'),
        max_score=('score', 'max'),
        anomaly_count=('is_anomaly', 'sum'),
        total=('score', 'count'),
    ).reset_index()

    store = _load_store()

    for _, row in grouped.iterrows():
        turbine_id = str(row['turbine_id'])
        day = row['day']
        store.setdefault(turbine_id, {})
        store[turbine_id][day] = {
            "avg_score": float(row['avg_score']),
            "max_score": float(row['max_score']),
            "anomaly_count": int(row['anomaly_count']),
            "total_predictions": int(row['total']),
            "updated_at": datetime.now().isoformat(),
        }

    _save_store(store)
    logger.info(f"Persisted {len(grouped)} turbine-day rows to JSON store")


def get_daily_anomaly_scores(turbine_id: str, lookback_days: int) -> list:
    """
    Returns daily-averaged ensemble anomaly scores for a turbine,
    oldest to newest, within the lookback window — anchored to the
    most recent date actually present in the data, not wall-clock time,
    since this dataset is historical rather than live-streaming.
    """
    store = _load_store()
    turbine_data = store.get(turbine_id, {})

    if not turbine_data:
        return []

    sorted_days = sorted(turbine_data.keys())  # "YYYY-MM-DD" strings sort correctly
    latest_day = datetime.strptime(sorted_days[-1], "%Y-%m-%d").date()
    cutoff = latest_day - timedelta(days=lookback_days)

    scores = [
        turbine_data[day]["avg_score"]
        for day in sorted_days
        if datetime.strptime(day, "%Y-%m-%d").date() >= cutoff
    ]
    return scores