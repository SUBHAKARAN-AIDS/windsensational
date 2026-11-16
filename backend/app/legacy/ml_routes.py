# backend/app/api/ml_routes.py

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import pandas as pd
# LEGACY (quarantined, not wired into the app). Imports repointed to the new
# module locations after the restructuring move; logic is otherwise unchanged.
from app.legacy.ensemble_service import EnsembleAnomalyDetector
from app.db.database_pipeline import get_db_connection, get_all_turbine_ids
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

# Initialize ensemble detector
ensemble_detector = EnsembleAnomalyDetector()

@router.get("/ensemble-anomaly-detection")
async def ensemble_anomaly_detection(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back days"),
    limit: int = Query(500, description="Max records"),
    min_score: float = Query(0.7, description="Minimum anomaly score")
):
    """
    Ensemble anomaly detection using ALL 7 models
    """
    try:
        logger.info("="*60)
        logger.info("🚀 ENSEMBLE ANOMALY DETECTION (ALL 7 MODELS)")
        logger.info(f"Parameters: turbine_id={turbine_id}, days={days}, limit={limit}")
        logger.info("="*60)
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT 
                time,
                name as turbine_id,
                wind_speed_m_s,
                generator_active_power_kw as active_power,
                generator_speed_r_min as generator_speed,
                rotor_speed_r_min as rotor_speed,
                torque_nm,
                pitch_angle_1,
                pitch_angle_2,
                pitch_angle_3,
                gearbox_oil_temperature,
                gearbox_de_bearing_temperature,
                gearbox_nde_bearing_temperature,
                main_bearing_temperature,
                generator_stator_temp_u,
                generator_stator_temp_v,
                generator_stator_temp_w,
                generator_drive_end_bearing_temperature,
                generator_non_drive_end_bearing_temperature,
                temperature_outside_nacelle,
                temperature_inside_nacelle_,
                vibration_lateral_mm_s,
                vibration_axial_mm_s
            FROM site_dzh.scada_raw
            WHERE time >= NOW() - INTERVAL %s DAY
        """
        params = [days]
        
        if turbine_id:
            query += " AND name = %s"
            params.append(turbine_id)
        
        query += " ORDER BY time DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return {
                "anomalies": [],
                "message": "No data found",
                "total_predictions": 0
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(records, columns=[
            'time', 'turbine_id', 'wind_speed_ms', 'active_power',
            'generator_speed', 'rotor_speed', 'torque_nm',
            'pitch_angle_1deg', 'pitch_angle_2deg', 'pitch_angle_3deg',
            'gearbox_oil_temperature', 'gearbox_de_bearing_temperature',
            'gearbox_nde_bearing_temperature', 'main_bearing_temperature',
            'generator_stator_temp_u', 'generator_stator_temp_v',
            'generator_stator_temp_w', 'generator_drive_end_bearing_temperature',
            'generator_non_drive_end_bearing_temperature',
            'temperature_outside_nacelle', 'temperature_inside_nacelle',
            'vibration_lateral_mms', 'vibration_axial_mms'
        ])
        
        # Clean data
        numeric_cols = ['wind_speed_ms', 'active_power', 'generator_speed', 
                       'rotor_speed', 'torque_nm', 'pitch_angle_1deg',
                       'pitch_angle_2deg', 'pitch_angle_3deg',
                       'gearbox_oil_temperature', 'gearbox_de_bearing_temperature',
                       'gearbox_nde_bearing_temperature', 'main_bearing_temperature',
                       'generator_stator_temp_u', 'generator_stator_temp_v',
                       'generator_stator_temp_w', 
                       'generator_drive_end_bearing_temperature',
                       'generator_non_drive_end_bearing_temperature',
                       'temperature_outside_nacelle', 'temperature_inside_nacelle',
                       'vibration_lateral_mms', 'vibration_axial_mms']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add missing columns if needed
        if 'temperature_outside_nacelle' not in df.columns:
            df['temperature_outside_nacelle'] = 0
        
        logger.info(f"📊 Data prepared: {len(df)} rows, {len(df.columns)} columns")
        
        # Run ensemble detection
        results = ensemble_detector.detect_anomalies_ensemble(df)
        
        # Filter by min_score
        if min_score > 0:
            results['anomalies'] = [
                a for a in results['anomalies'] 
                if a['score'] >= min_score
            ]
        
        logger.info(f"✅ Ensemble detection complete: {len(results['anomalies'])} anomalies")
        
        return {
            "anomalies": results['anomalies'],
            "total_predictions": len(df),
            "models_used": list(results['individual_predictions'].keys()),
            "ensemble_votes": results['ensemble'].get('vote_sum', []),
            "ensemble_scores": results['ensemble'].get('scores', []),
            "anomaly_count": len(results['anomalies']),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ensemble anomaly detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "anomalies": [],
            "total_predictions": 0
        }

@router.get("/turbines/{turbine_id}/ensemble-status")
async def get_turbine_ensemble_status(turbine_id: str, days: int = 7):
    """
    Get turbine status using ALL 7 models
    """
    try:
        # Reuse the ensemble detection
        result = await ensemble_anomaly_detection(turbine_id=turbine_id, days=days, limit=100)
        
        if result.get('error'):
            return {
                "turbine_id": turbine_id,
                "status": "Unknown",
                "health_score": 0,
                "has_anomaly": False,
                "error": result['error']
            }
        
        anomaly_count = len(result.get('anomalies', []))
        total_predictions = result.get('total_predictions', 0)
        
        if total_predictions > 0:
            anomaly_ratio = anomaly_count / total_predictions
            health_score = max(0, 100 - (anomaly_ratio * 100))
        else:
            anomaly_ratio = 0
            health_score = 0
        
        # Determine status
        if anomaly_ratio > 0.3:
            status = "Critical"
        elif anomaly_ratio > 0.1:
            status = "Warning"
        elif anomaly_count > 0:
            status = "Monitor"
        else:
            status = "Healthy"
        
        return {
            "turbine_id": turbine_id,
            "status": status,
            "health_score": int(health_score),
            "anomaly_count": anomaly_count,
            "total_samples": total_predictions,
            "anomaly_ratio": float(anomaly_ratio),
            "has_anomaly": anomaly_count > 0,
            "models_used": result.get('models_used', []),
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting ensemble status for {turbine_id}: {e}")
        return {
            "turbine_id": turbine_id,
            "status": "Unknown",
            "health_score": 0,
            "has_anomaly": False,
            "error": str(e)
        }