# app/analytics/fallbacks.py
#
# generate_fallback_power_curve() moved VERBATIM from app/main.py during
# restructuring. Behavior unchanged.
#
# NOTE (deferred): synthetic fallback data returned as if real is flagged for
# the later "fallback data honesty" phase. Not changed here.

import logging
from app.core.config import RUL_FAILURE_THRESHOLD

logger = logging.getLogger(__name__)


def generate_fallback_power_curve():
    """Generate fallback power curve data for testing"""
    logger.info("Generating fallback power curve data")
    result = []
    for ws in range(1, 26):
        if ws < 3:
            actual = 0
            theoretical = 0
        elif ws >= 15:
            actual = 6500 * (0.92 + 0.08 * (ws - 15) / 10)
            theoretical = 6500
        else:
            theoretical = 6500 * ((ws - 3) / 12.0) ** 3
            variation = 0.85 + (0.15 * (ws - 3) / 12)
            actual = theoretical * variation
        
        result.append({
            "windSpeed": float(ws),
            "actualPower": round(actual, 1),
            "theoreticalPower": round(theoretical, 1)
        })
    return result


def get_fallback_rul(turbine_id: str):
    """Generate fallback RUL response when insufficient anomaly-score history exists"""
    logger.info(f"Generating fallback RUL data for {turbine_id}")
    return {
        "turbine_id": turbine_id,
        "rul_days": None,
        "confidence_pct": 0,
        "health_history": [],
        "projected_health": [],
        "failure_threshold": RUL_FAILURE_THRESHOLD,
        "status": "no_data"
    }