# api/routes/thermal.py
from fastapi import APIRouter, Query, HTTPException
import logging

from app.pipeline import run_thermal_inference

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{turbine_id}")
async def get_thermal_data(
    turbine_id: str,
    days: int = Query(7, ge=1, le=3650),
    limit: int = Query(300, ge=1, le=5000)
):
    """Get thermal data for a specific turbine"""
    try:
        from db.database_pipeline import get_turbine_thermal_data
        records = get_turbine_thermal_data(turbine_id, days, limit)
    except Exception as e:
        logger.error(f"Database error fetching thermal data for {turbine_id}: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

    if records is None:
        # get_turbine_thermal_data returns None only when the DB connection itself failed
        raise HTTPException(status_code=503, detail="Database unavailable")

    # records == [] just means "no rows for this turbine/date range" —
    # run_thermal_inference already handles that and returns a clean
    # "No data available" response instead of crashing.
    return run_thermal_inference(turbine_id, records)