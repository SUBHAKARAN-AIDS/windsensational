# app/analytics/rul_pipeline.py
import numpy as np
from scipy.optimize import curve_fit
from app.core.config import RUL_CRITICAL_DAYS, RUL_FAILURE_THRESHOLD, RUL_MONITOR_DAYS

def _exp_growth(t, a, b):
    """Anomaly score growth model: score rises toward 1.0 as bearing degrades."""
    return 1 - a * np.exp(-b * t)

def compute_rul(turbine_id: str, health_history: list[float]) -> dict | None:
    """
    health_history: daily physical health values, oldest -> newest, 1 (healthy)
    to 0 (failed).
    """
    health = np.clip(np.asarray(health_history, dtype=float), 0.0, 1.0)
    if len(health) < 5:
        return None

    days = np.arange(len(health))

    try:
        popt, _ = curve_fit(lambda t, a, b: a * np.exp(-b * t), days, health,
                             p0=[1.0, 0.05], maxfev=5000)
        a, b = popt
        if a <= RUL_FAILURE_THRESHOLD or b <= 0:
            return {
                "turbine_id": turbine_id,
                "rul_days": None,
                "confidence_pct": 0,
                "health_history": health.tolist(),
                "projected_health": [],
                "failure_threshold": RUL_FAILURE_THRESHOLD,
                "status": "stable",
            }

        t_failure = -np.log(RUL_FAILURE_THRESHOLD / a) / b
        rul_days = max(t_failure - days[-1], 0)

        residuals = health - (a * np.exp(-b * days))
        confidence = round(max(1 - np.std(residuals) * 5, 0) * 100, 1)

        projected_days = np.arange(days[-1], days[-1] + rul_days + 5)
        projected_health = (a * np.exp(-b * projected_days)).tolist()

        return {
            "turbine_id": turbine_id,
            "rul_days": round(float(rul_days), 1),
            "confidence_pct": confidence,
            "health_history": health.tolist(),
            "projected_health": projected_health,
            "failure_threshold": RUL_FAILURE_THRESHOLD,
            "status": (
                "critical" if rul_days < RUL_CRITICAL_DAYS
                else "monitor" if rul_days < RUL_MONITOR_DAYS
                else "stable"
            )
        }
    except (RuntimeError, ValueError, FloatingPointError):
        return None