# app/api/routes/rul.py
import logging
from fastapi import APIRouter, HTTPException
from app.db.physical_health_store import get_daily_physical_health
from app.analytics.rul_pipeline import compute_rul
from app.analytics.fallbacks import get_fallback_rul
from app.core.config import RUL_LOOKBACK_DAYS

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{turbine_id}")
async def rul_prediction(turbine_id: str):
    health_history = get_daily_physical_health(turbine_id, RUL_LOOKBACK_DAYS)
    logger.info("[%s] fetched %s daily physical health rows", turbine_id, len(health_history))

    if not health_history or len(health_history) < 5:
        return get_fallback_rul(turbine_id)

    result = compute_rul(turbine_id, health_history)
    if result is None:
        raise HTTPException(422, "Could not fit degradation curve")
    return result