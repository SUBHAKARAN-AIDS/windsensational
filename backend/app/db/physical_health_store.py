"""Physical, per-turbine health history used by RUL prediction."""

from collections import defaultdict
from datetime import datetime
from math import hypot

import numpy as np

from app.core.config import (
    RUL_BASELINE_FRACTION,
    RUL_BASELINE_VARIATION_FLOOR,
    RUL_TEMPERATURE_Z_THRESHOLD,
    RUL_VIBRATION_Z_THRESHOLD,
)
from app.core.config import RUL_LOOKBACK_DAYS
from app.db.database_pipeline import fetch_rul_physical_history


def _number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _physical_rows(records):
    rows = []
    for record in records:
        timestamp = record.get("time")
        vibration_lateral = _number(record.get("vibration_lateral"))
        vibration_axial = _number(record.get("vibration_axial"))
        temperature = _number(record.get("main_bearing_temperature"))
        if timestamp is None or vibration_lateral is None or vibration_axial is None or temperature is None:
            continue
        rows.append({
            "time": timestamp,
            "day": timestamp.date() if hasattr(timestamp, "date") else datetime.fromisoformat(str(timestamp)).date(),
            "vibration": hypot(vibration_lateral, vibration_axial),
            "temperature": temperature,
        })
    return rows


def build_daily_physical_health(records, lookback_days=RUL_LOOKBACK_DAYS):
    """Return daily average health using a baseline from this turbine only."""
    rows = _physical_rows(records)
    if len(rows) < 5:
        return []

    baseline_count = max(5, int(len(rows) * RUL_BASELINE_FRACTION))
    baseline = rows[:baseline_count]
    vibration_values = np.array([row["vibration"] for row in baseline], dtype=float)
    temperature_values = np.array([row["temperature"] for row in baseline], dtype=float)

    vibration_mean = float(np.mean(vibration_values))
    temperature_mean = float(np.mean(temperature_values))
    vibration_scale = max(float(np.std(vibration_values)), abs(vibration_mean) * RUL_BASELINE_VARIATION_FLOOR)
    temperature_scale = max(float(np.std(temperature_values)), abs(temperature_mean) * RUL_BASELINE_VARIATION_FLOOR)
    vibration_limit = vibration_mean + RUL_VIBRATION_Z_THRESHOLD * vibration_scale
    temperature_limit = temperature_mean + RUL_TEMPERATURE_Z_THRESHOLD * temperature_scale

    latest_day = max(row["day"] for row in rows)
    cutoff_day = latest_day.fromordinal(latest_day.toordinal() - lookback_days)
    daily = defaultdict(list)
    for row in rows:
        vibration_degradation = (row["vibration"] - vibration_mean) / max(vibration_limit - vibration_mean, np.finfo(float).eps)
        temperature_degradation = (row["temperature"] - temperature_mean) / max(temperature_limit - temperature_mean, np.finfo(float).eps)
        degradation = float(np.clip(max(vibration_degradation, temperature_degradation), 0.0, 1.0))
        if row["day"] >= cutoff_day:
            daily[row["day"]].append(1.0 - degradation)

    return [float(np.mean(daily[day])) for day in sorted(daily)]


def get_daily_physical_health(turbine_id: str, lookback_days: int):
    records = fetch_rul_physical_history(turbine_id)
    return build_daily_physical_health(records, lookback_days)