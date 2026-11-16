# app/api/routes/model_performance.py
"""
Real model performance endpoints using ONLY database data.
NO MOCK DATA.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta
import logging

from app.analytics.model_backtest import backtest_models_against_real_faults
from app.db.event_queries import get_real_fault_event_summary

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/v1/ml/performance")
async def get_model_performance(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(30, description="Number of days to backtest"),
    lookback_hours: int = Query(24, description="Hours before fault to check for detection")
):
    """
    Get REAL model performance metrics against actual fault events.
    
    Returns:
        - Precision, Recall, F1 for each model
        - Based on REAL fault events from database
        - NO MOCK DATA
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"📊 Backtesting models against real faults from {start_date} to {end_date}")
        
        # Run backtest against REAL faults
        results = backtest_models_against_real_faults(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            turbine_id=turbine_id,
            lookback_hours=lookback_hours
        )
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_source": "real_database_events",
            "total_faults": results.get("total_faults", 0),
            "total_scada_records": results.get("total_scada_records", 0),
            "models": results.get("models", {}),
            "period": results.get("period", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Performance API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/ml/performance/summary")
async def get_performance_summary(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine")
):
    """
    Get summary of REAL fault events available for performance testing.
    """
    try:
        summary = get_real_fault_event_summary(turbine_id=turbine_id)
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return {
            "status": "success",
            "data_source": "real_database_events",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"❌ Summary API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))