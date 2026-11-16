# app/analytics/model_backtest.py
"""
Real model backtesting against ACTUAL fault events from database.
NO MOCK DATA - only real data.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging

from app.db.event_queries import get_real_fault_events
from app.ml.model_state import ML_MODELS

logger = logging.getLogger(__name__)


def backtest_models_against_real_faults(
    start_date: str = None,
    end_date: str = None,
    turbine_id: str = None,
    lookback_hours: int = 24
) -> dict:
    """
    Backtest all models against REAL fault events from database.
    
    Args:
        start_date: Start date for backtest period
        end_date: End date for backtest period
        turbine_id: Optional turbine filter
        lookback_hours: How many hours before fault to check for detection
    
    Returns:
        Dictionary with per-model performance metrics (Precision, Recall, F1)
    """
    
    # ── 1. Fetch REAL Fault Events ──────────────────────────────────
    faults = get_real_fault_events(
        turbine_id=turbine_id,
        start_date=start_date,
        end_date=end_date
    )
    
    if faults.empty:
        return {
            "error": "No real fault events found in database",
            "total_faults": 0
        }
    
    logger.info(f"Found {len(faults)} real fault events for backtesting")
    
    # ── 2. Fetch SCADA data for the same period ─────────────────────
    # (This would use your existing scada query function)
    scada_data = get_scada_data_for_period(start_date, end_date, turbine_id)
    if scada_data.empty:
        return {
            "error": "No SCADA data found for the same period",
            "total_faults": len(faults)
        }
    
    logger.info(f"Found {len(scada_data)} SCADA records for backtesting")
    
    # ── 3. Run each model on the SCADA data ─────────────────────────
    model_results = {}
    
    for model_name in ['isolation_forest', 'elliptic_envelope', 'xgboost', 'lstm_autoencoder']:
        if model_name not in ML_MODELS or ML_MODELS[model_name] is None:
            continue
        
        try:
            # Run model inference (use existing logic)
            predictions = run_model_inference(model_name, scada_data)
            
            # Calculate performance against real faults
            metrics = calculate_model_performance(
                predictions,
                faults,
                scada_data,
                lookback_hours
            )
            
            model_results[model_name] = metrics
            
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
            model_results[model_name] = {"error": str(e)}
    
    return {
        "total_faults": len(faults),
        "total_scada_records": len(scada_data),
        "models": model_results,
        "period": {
            "start": start_date,
            "end": end_date
        }
    }


def run_model_inference(model_name: str, scada_data: pd.DataFrame) -> pd.Series:
    """Run a single model on SCADA data and return predictions."""
    # This uses your existing model inference logic
    from app.api.routes.ml import run_ml_inference
    return run_ml_inference(model_name, scada_data)


def calculate_model_performance(
    predictions: pd.Series,
    faults: pd.DataFrame,
    scada_data: pd.DataFrame,
    lookback_hours: int = 24
) -> dict:
    """
    Calculate real performance metrics against actual fault events.
    
    Metrics:
    - True Positives: Model flagged a fault within lookback window
    - False Positives: Model flagged when no fault occurred
    - False Negatives: Fault occurred but model didn't flag
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    if predictions.empty or faults.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_faults": len(faults)
        }
    
    # Ensure timestamps are datetime
    scada_data['timestamp'] = pd.to_datetime(scada_data['timestamp'])
    faults['event_time'] = pd.to_datetime(faults['event_time'])
    
    # Track detections
    detected_faults = set()
    false_positives = 0
    
    # For each SCADA prediction, check if there's a fault nearby
    for idx, row in scada_data.iterrows():
        if predictions.iloc[idx] == 1:  # Model flagged anomaly
            # Check if there's a fault within lookback window
            found_fault = False
            for fault_idx, fault in faults.iterrows():
                time_diff = abs((row['timestamp'] - fault['event_time']).total_seconds() / 3600)
                if time_diff <= lookback_hours:
                    detected_faults.add(fault_idx)
                    found_fault = True
                    break
            
            if not found_fault:
                false_positives += 1
    
    # Calculate metrics
    total_faults = len(faults)
    true_positives = len(detected_faults)
    false_negatives = total_faults - true_positives
    
    # Prevent division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / total_faults if total_faults > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_faults": total_faults,
        "detection_rate": round(true_positives / total_faults * 100, 2)
    }


def get_scada_data_for_period(start_date, end_date, turbine_id):
    """Fetch SCADA data for the specified period."""
    # Use your existing SCADA data function
    from app.db.database_pipeline import get_scada_data
    return get_scada_data(
        turbine_id=turbine_id,
        start_date=start_date,
        end_date=end_date
    )