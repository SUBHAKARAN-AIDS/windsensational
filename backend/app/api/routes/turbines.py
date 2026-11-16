# app/api/routes/turbines.py - Active production turbine routes

import logging
from datetime import datetime

from fastapi import APIRouter, Query

from app.db.database_pipeline import get_all_turbine_ids, get_turbine_latest_data
from app.db.db_telemetry import get_live_fleet_telemetry
from app.db.data_mode import check_data_mode
from app.api.routes.ml import get_ensemble_anomaly_detection
from app.core.config import DEFAULT_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/list")
@router.get("/api/v1/turbines/list")
def get_turbine_list():
    """Get list of all turbine IDs from the database"""
    try:
        turbines = get_all_turbine_ids()
        logger.info(f"Returning turbine list: {len(turbines)} turbines")
        return {
            "turbines": turbines,
            "count": len(turbines),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in turbine list: {e}")
        return {
            "error": str(e),
            "turbines": [],
            "count": 0
        }


@router.get("/live")
@router.get("/api/v1/turbines/live")
def get_live_turbines():
    """Get live telemetry from all turbines"""
    try:
        logger.info("Fetching live turbine telemetry...")
        turbines = get_live_fleet_telemetry()
        
        if not turbines:
            logger.warning("No live telemetry data available")
            return {"turbines": [], "message": "No live data available"}
        
        logger.info(f"Returning {len(turbines)} turbines")
        return {"turbines": turbines}
    except Exception as e:
        logger.error(f"Error in live turbines: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "turbines": []}


@router.get("/{turbine_id}/latest")
@router.get("/api/v1/turbines/{turbine_id}/latest")
def get_turbine_latest(turbine_id: str):
    """Get the latest SCADA reading for a specific turbine"""
    try:
        record = get_turbine_latest_data(turbine_id)
        if record:
            return record
        else:
            return {
                "turbine_id": turbine_id,
                "status": "No data found",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching latest data for {turbine_id}: {e}")
        return {"error": str(e)}


@router.get("/{turbine_id}/status")
@router.get("/api/v1/turbines/{turbine_id}/status")
async def get_turbine_status(
    turbine_id: str,
    days: int = Query(DEFAULT_LOOKBACK_DAYS, description="Look back N days")
):
    """Get status for a specific turbine based on ML predictions."""
    try:
        result = await get_ensemble_anomaly_detection(
            turbine_id=turbine_id,
            days=days,
            limit=100,
            min_score=0.75
        )
        
        if result.get("error") or result.get("status") == "error":
            return {
                "turbine_id": turbine_id,
                "status": "Unknown",
                "health_score": 0,
                "has_anomaly": False,
                "error": result.get("message") or result.get("error")
            }
        
        anomalies = result.get("anomalies", [])
        anomaly_count = len(anomalies)
        total_samples = result.get("total_predictions", 0)
        voting_models = result.get("voting_models", [])
        score_only_models = result.get("score_only_models", [])
        
        if total_samples > 0:
            anomaly_ratio = anomaly_count / total_samples
            health_score = max(0, 100 - (anomaly_ratio * 100))
        else:
            anomaly_ratio = 0
            health_score = 100
        
        if anomaly_ratio > 0.3:
            status = "Critical"
        elif anomaly_ratio > 0.1:
            status = "Warning"
        elif anomaly_count > 5:
            status = "Monitor"
        else:
            status = "Healthy"
        
        return {
            "turbine_id": turbine_id,
            "status": status,
            "health_score": int(health_score),
            "anomaly_count": anomaly_count,
            "total_samples": total_samples,
            "anomaly_ratio": float(anomaly_ratio),
            "has_anomaly": anomaly_count > 5,
            "voting_models": voting_models,
            "score_only_models": score_only_models,
            "total_voters": result.get("total_voters", len(voting_models)),
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting status for {turbine_id}: {e}")
        return {
            "turbine_id": turbine_id,
            "status": "Unknown",
            "health_score": 0,
            "has_anomaly": False,
            "error": str(e)
        }


@router.get("/all-status")
@router.get("/api/v1/turbines/all-status")
async def get_all_turbines_status(
    days: int = Query(DEFAULT_LOOKBACK_DAYS, description="Look back N days")
):
    """Get status for ALL turbines."""
    try:
        turbines = get_all_turbine_ids()
        if not turbines:
            return {"turbines": [], "total": 0}
        
        data_mode = check_data_mode()
        
        results = []
        for tid in turbines:
            try:
                status = await get_turbine_status(tid, days=days)
                status['data_mode'] = data_mode
                results.append(status)
            except Exception as e:
                logger.error(f"Failed to get status for {tid}: {e}")
                results.append({
                    "turbine_id": tid,
                    "status": "Unknown",
                    "health_score": 0,
                    "has_anomaly": False,
                    "data_mode": data_mode,
                    "error": str(e)
                })
        
        summary = {
            "healthy": len([t for t in results if t.get('status') == 'Healthy']),
            "monitor": len([t for t in results if t.get('status') == 'Monitor']),
            "warning": len([t for t in results if t.get('status') == 'Warning']),
            "critical": len([t for t in results if t.get('status') == 'Critical']),
            "unknown": len([t for t in results if t.get('status') == 'Unknown'])
        }
        
        return {
            "turbines": results,
            "total": len(results),
            "summary": summary,
            "data_mode": data_mode
        }
        
    except Exception as e:
        logger.error(f"Error getting all statuses: {e}")
        return {"turbines": [], "total": 0, "error": str(e)}