# app/api/routes/ml.py - Production version with Real SCADA DB & Saved ML Models

import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Query, HTTPException

from app.db.connection import get_db_connection
from app.db.database_pipeline import get_all_turbine_ids
from app.db.data_mode import check_data_mode, get_date_condition
from app.ml.model_state import ML_MODELS, load_ml_models
from app.db.training_status_store import get_training_history
from app.ml.retrain_watcher import _load_state, get_retrain_advice, is_training, start_training
from app.core.config import (
    VAE_ENABLED, MAIN_FEATURES, AE_FEATURES,
    VOTING_MODELS, SCORE_ONLY_MODELS,
    LSTM_SEQ_LEN, LSTM_INPUT_WIDTH, AE_RECON_PERCENTILE,
    SEVERITY_HIGH, SEVERITY_MEDIUM,
    DEFAULT_LOOKBACK_DAYS, DEFAULT_LIMIT, DEFAULT_MIN_SCORE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models/status")
def get_ml_model_status():
    """Returns the status of loaded ML models."""
    if not ML_MODELS.get("loaded"):
        load_ml_models()

    return {
        "models_loaded": ML_MODELS.get("loaded", False),
        "models": {
            "isolation_forest": ML_MODELS.get("isolation_forest") is not None,
            "elliptic_envelope": ML_MODELS.get("elliptic_envelope") is not None,
            "xgboost": ML_MODELS.get("xgboost") is not None,
            "lstm_autoencoder": ML_MODELS.get("lstm_autoencoder") is not None,
            "vae_autoencoder": ML_MODELS.get("vae_autoencoder") is not None,
        },
        "scaler": ML_MODELS.get("scaler") is not None,
        "scaler_ae": ML_MODELS.get("scaler_ae") is not None,
        "feature_columns": len(ML_MODELS.get("feature_columns") or []),
        "validation": ML_MODELS.get("validation", {}),
        "participation_policy": {
            "voting_models": VOTING_MODELS,
            "score_only_models": SCORE_ONLY_MODELS,
            "excluded_models": {"vae_autoencoder": "input_dim=3 mapping unrecorded"},
        },
        "model_path": "backend/saved_models/",
        "scoring_notes": _scoring_readiness_notes(),
    }


@router.post("/models/reload")
def reload_ml_models():
    """Reload ML models from disk"""
    success = load_ml_models()
    return {
        "success": success,
        "models_loaded": ML_MODELS.get("loaded", False)
    }


# ─── CHART ENDPOINTS WITH REAL DATA ────────────────────────────

@router.get("/isolation-forest-chart")
async def get_isolation_forest_chart(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back N days"),
    limit: int = Query(500, description="Max records for chart")
):
    """Get Isolation Forest scatter plot data - REAL SCADA DATA"""
    try:
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "normal_points": [],
                "anomaly_points": [],
                "scores": [],
                "total_normal": 0,
                "total_anomalies": 0,
                "threshold": 0.5
            }
        
        scaler = ML_MODELS.get("scaler")
        iso = ML_MODELS.get("isolation_forest")

        if scaler is None or iso is None:
            load_ml_models()
            scaler = ML_MODELS.get("scaler")
            iso = ML_MODELS.get("isolation_forest")

        if scaler is None or iso is None:
            raise HTTPException(
                status_code=503,
                detail="Isolation Forest model or Scaler not loaded."
            )

        if hasattr(scaler, "feature_names_in_"):
            feature_cols = list(scaler.feature_names_in_)
        else:
            feature_cols = list(AE_FEATURES)

        df_clean = df[[c for c in feature_cols if c in df.columns]].fillna(0).replace([np.inf, -np.inf], 0)
        for col in feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
        df_clean = df_clean[feature_cols]

        X_scaled = scaler.transform(df_clean)
        
        scores = iso.score_samples(X_scaled)
        predictions = iso.predict(X_scaled)
        
        # Normalize scores (0-1 scale, higher = more anomalous)
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            norm_scores = 1 - (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.zeros_like(scores)
        
        # Get x and y axes for scatter
        x_col = 'wind_speed_ms' if 'wind_speed_ms' in df.columns else 'active_power'
        y_col = 'active_power' if 'active_power' in df.columns else 'wind_speed_ms'
        
        x_axis = df[x_col].fillna(0).values
        y_axis = df[y_col].fillna(0).values
        turbine_ids = df['turbine_id'].tolist() if 'turbine_id' in df.columns else [f'WT-{i}' for i in range(len(df))]
        
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        normal_indices = np.where(normal_mask)[0]
        if len(normal_indices) > 200:
            normal_indices = np.random.choice(normal_indices, 200, replace=False)
        
        anomaly_indices = np.where(anomaly_mask)[0]
        
        normal_points = [
            {
                'x': float(x_axis[i]),
                'y': float(y_axis[i]),
                'turbine_id': str(turbine_ids[i])
            }
            for i in normal_indices
        ]
        
        anomaly_points = [
            {
                'x': float(x_axis[i]),
                'y': float(y_axis[i]),
                'score': float(norm_scores[i]),
                'turbine_id': str(turbine_ids[i])
            }
            for i in anomaly_indices
        ]
        
        score_sample = np.random.choice(len(scores), min(100, len(scores)), replace=False)
        score_points = [
            {
                'x': float(x_axis[i]),
                'y': float(y_axis[i]),
                'score': float(norm_scores[i])
            }
            for i in score_sample
        ]
        
        return {
            "status": "success",
            "normal_points": normal_points,
            "anomaly_points": anomaly_points,
            "scores": score_points,
            "total_normal": int(np.sum(normal_mask)),
            "total_anomalies": int(np.sum(anomaly_mask)),
            "threshold": 0.5
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Isolation Forest chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/autoencoder-chart")
async def get_autoencoder_chart(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back N days"),
    limit: int = Query(500, description="Max records for chart")
):
    """Get Autoencoder reconstruction error data - REAL SCADA DATA"""
    try:
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "reconstruction_errors": [],
                "threshold": 0.0,
                "anomaly_indices": [],
                "timestamps": [],
                "total_anomalies": 0
            }
        
        lstm = ML_MODELS.get("lstm_autoencoder")
        scaler_ae = ML_MODELS.get("scaler_ae")

        if lstm is None or scaler_ae is None:
            load_ml_models()
            lstm = ML_MODELS.get("lstm_autoencoder")
            scaler_ae = ML_MODELS.get("scaler_ae")

        if lstm is None or scaler_ae is None:
            raise HTTPException(
                status_code=503,
                detail=("LSTM Autoencoder is not loaded. "
                        + ("The LSTM file could not be read by this backend (it was saved with a newer Keras). "
                           "Run the 'export LSTM' cell in the notebook and copy lstm_autoencoder_weights.npz "
                           "into saved_models. " if lstm is None else "")
                        + ("The autoencoder scaler (scaler_ae) is missing. " if scaler_ae is None else "")).strip()
            )
        
        ae_feature_cols = list(AE_FEATURES)
        df_ae = df[[c for c in ae_feature_cols if c in df.columns]].fillna(0).replace([np.inf, -np.inf], 0)
        for col in ae_feature_cols:
            if col not in df_ae.columns:
                df_ae[col] = 0.0
        df_ae = df_ae[ae_feature_cols]

        X_scaled_ae = scaler_ae.transform(df_ae.values)
        
        SEQ_LEN = LSTM_SEQ_LEN
        n_samples_ae = len(X_scaled_ae)
        n_sequences = n_samples_ae // SEQ_LEN
        
        if n_sequences == 0:
            return {
                "status": "success",
                "reconstruction_errors": [],
                "threshold": 0.0,
                "anomaly_indices": [],
                "timestamps": [],
                "total_anomalies": 0,
                "message": f"Need at least {SEQ_LEN} SCADA records for LSTM sequence calculation."
            }
        
        X_reshaped = X_scaled_ae[:n_sequences * SEQ_LEN].reshape(n_sequences, SEQ_LEN, LSTM_INPUT_WIDTH)
        reconstructed = lstm.predict(X_reshaped, verbose=0)
        reconstruction_errors = np.mean(np.square(X_reshaped - reconstructed), axis=(1, 2))
        
        padded_errors = np.zeros(n_samples_ae)
        for seq_idx, err in enumerate(reconstruction_errors):
            start_row = seq_idx * SEQ_LEN
            for i in range(SEQ_LEN):
                row_idx = start_row + i
                if row_idx < n_samples_ae:
                    padded_errors[row_idx] = float(err)
        
        padded_errors = padded_errors[:len(df)]
        
        threshold = float(np.percentile(reconstruction_errors, AE_RECON_PERCENTILE)) if len(reconstruction_errors) > 0 else 0.0
        anomaly_indices = [i for i, err in enumerate(padded_errors) if err > threshold]
        
        timestamps = df['time'].dt.strftime('%Y-%m-%d %H:%M').tolist() if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']) else [str(t) for t in df.get('time', [])]
        
        return {
            "status": "success",
            "reconstruction_errors": padded_errors.tolist(),
            "threshold": threshold,
            "anomaly_indices": anomaly_indices[:30],
            "timestamps": timestamps,
            "total_anomalies": len(anomaly_indices)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Autoencoder chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mahalanobis-chart")
async def get_mahalanobis_chart(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back N days"),
    limit: int = Query(500, description="Max records for chart")
):
    """Get Mahalanobis distance data - REAL SCADA DATA"""
    try:
        from scipy.spatial.distance import mahalanobis
        from scipy.stats import chi2
        
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "distances": [],
                "threshold": 0.0,
                "anomaly_indices": [],
                "labels": [],
                "total_anomalies": 0
            }
        
        mahal_features = ['wind_speed_ms', 'active_power', 'vibration_lateral_mms', 
                          'gearbox_oil_temperature', 'main_bearing_temperature']
        available_features = [f for f in mahal_features if f in df.columns]
        
        if len(available_features) < 3:
            return {
                "status": "no_data",
                "distances": [],
                "threshold": 0.0,
                "anomaly_indices": [],
                "labels": [],
                "total_anomalies": 0,
                "message": "Insufficient feature columns in dataset"
            }
        
        mahal_data = df[available_features].fillna(0).values
        
        if len(mahal_data) < 5:
            return {
                "status": "no_data",
                "distances": [],
                "threshold": 0.0,
                "anomaly_indices": [],
                "labels": [],
                "total_anomalies": 0,
                "message": "Insufficient data points for Mahalanobis calculation"
            }
        
        baseline_size = max(3, int(len(mahal_data) * 0.7))
        baseline = mahal_data[:baseline_size]
        mean = np.mean(baseline, axis=0)
        cov = np.cov(baseline, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        
        distances = [float(mahalanobis(point, mean, cov_inv)) for point in mahal_data]
        
        threshold = float(chi2.ppf(0.999, df=len(available_features)))
        anomaly_indices = [i for i, d in enumerate(distances) if d > threshold]
        
        return {
            "status": "success",
            "distances": distances,
            "threshold": threshold,
            "anomaly_indices": anomaly_indices[:30],
            "labels": list(range(len(distances))),
            "total_anomalies": len(anomaly_indices)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mahalanobis chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tsne-chart")
async def get_tsne_chart(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back N days"),
    limit: int = Query(300, description="Max records for chart")
):
    """Get t-SNE embedding data - REAL SCADA DATA"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "normal_points": [],
                "anomaly_points": [],
                "total_normal": 0,
                "total_anomalies": 0
            }
        
        scaler = ML_MODELS.get("scaler")
        if hasattr(scaler, "feature_names_in_"):
            feature_cols = list(scaler.feature_names_in_)
        else:
            feature_cols = list(AE_FEATURES)

        df_clean = df[[c for c in feature_cols if c in df.columns]].fillna(0).replace([np.inf, -np.inf], 0)
        for col in feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
        df_clean = df_clean[feature_cols]
        X = df_clean.values
        
        sample_size = min(300, len(X))
        if sample_size < 5:
            return {
                "status": "no_data",
                "normal_points": [],
                "anomaly_points": [],
                "total_normal": 0,
                "total_anomalies": 0
            }
        
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
        
        pca = PCA(n_components=min(10, X_sample.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_sample)
        
        perplexity = min(30, max(2, sample_size - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_pca)
        
        iso = ML_MODELS.get("isolation_forest")
        scaler = ML_MODELS.get("scaler")
        
        anomaly_indices = []
        if iso is not None and scaler is not None:
            X_scaled = scaler.transform(df_clean.iloc[indices])
            iso_scores = iso.score_samples(X_scaled)
            threshold = np.percentile(iso_scores, 5)
            anomaly_indices = [i for i, s in enumerate(iso_scores) if s <= threshold]
        
        normal_points = [
            {'x': float(X_tsne[i][0]), 'y': float(X_tsne[i][1])}
            for i in range(len(X_tsne)) if i not in anomaly_indices
        ]
        
        anomaly_points = [
            {'x': float(X_tsne[i][0]), 'y': float(X_tsne[i][1])}
            for i in range(len(X_tsne)) if i in anomaly_indices
        ]
        
        return {
            "status": "success",
            "normal_points": normal_points[:200],
            "anomaly_points": anomaly_points,
            "total_normal": len(normal_points),
            "total_anomalies": len(anomaly_points)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"t-SNE chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cusum-chart")
async def get_cusum_chart(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(7, description="Look back N days"),
    limit: int = Query(500, description="Max records for chart")
):
    """Get CUSUM change detection data - REAL SCADA DATA"""
    try:
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "signal": [],
                "anomaly_indices": [],
                "labels": [],
                "threshold": 3.0,
                "total_anomalies": 0
            }
        
        vib_col = 'vibration_lateral_mms' if 'vibration_lateral_mms' in df.columns else None
        if vib_col is None:
            return {
                "status": "no_data",
                "signal": [],
                "anomaly_indices": [],
                "labels": [],
                "threshold": 3.0,
                "total_anomalies": 0
            }
        
        signal = df[vib_col].fillna(0).values
        
        if len(signal) < 5:
            return {
                "status": "no_data",
                "signal": [],
                "anomaly_indices": [],
                "labels": [],
                "threshold": 3.0,
                "total_anomalies": 0
            }
        
        mean = np.mean(signal[:max(1, int(len(signal) * 0.7))])
        
        cusum_positive = np.zeros(len(signal))
        cusum_negative = np.zeros(len(signal))
        
        k = 0.3
        h = 3.0
        
        pos = 0.0
        neg = 0.0
        for i in range(len(signal)):
            pos = max(0.0, pos + (signal[i] - mean) - k)
            neg = min(0.0, neg + (signal[i] - mean) + k)
            cusum_positive[i] = pos
            cusum_negative[i] = neg
        
        cusum = cusum_positive + abs(cusum_negative)
        anomaly_indices = [i for i in range(len(cusum)) if cusum_positive[i] > h or cusum_negative[i] < -h]
        
        return {
            "status": "success",
            "signal": cusum.tolist(),
            "anomaly_indices": anomaly_indices[:30],
            "labels": list(range(len(cusum))),
            "threshold": float(h),
            "total_anomalies": len(anomaly_indices)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CUSUM chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


_NORMAL_CLASS_NAMES = {"normal", "healthy", "no_fault", "no fault", "none", "ok", "nominal"}


def _scoring_readiness_notes():
    """Plain-language list of things that make the live scoring differ from
    training (missing extra files). Empty list = everything matches."""
    notes = []
    if ML_MODELS.get("isolation_forest") is not None and ML_MODELS.get("if_threshold") is None:
        notes.append("Isolation Forest: saved cut-off not found, using the model's default rule (may flag more rows than in training).")
    if ML_MODELS.get("elliptic_envelope") is not None:
        if ML_MODELS.get("power_transformer") is None or not ML_MODELS.get("ee_input_spec") or ML_MODELS.get("scaler_normal_ee") is None:
            notes.append("Elliptic Envelope: the data-reshaping files from training are missing, so it is fed the wrong kind of numbers. Run the notebook export cell.")
        elif ML_MODELS.get("ee_threshold") is None:
            notes.append("Elliptic Envelope: saved cut-off not found, using the model's default rule.")
    if ML_MODELS.get("xgboost") is not None and not ML_MODELS.get("xgb_classes"):
        notes.append("XGBoost: class names file missing, assuming the first class means 'normal' (may be wrong). Run the notebook export cell.")
    return notes


def _xgb_normal_index(xgb):
    """Find which XGBoost class means 'normal'. Returns (index, note_or_None)."""
    names = ML_MODELS.get("xgb_classes")
    if isinstance(names, dict):
        names = names.get("classes")
    if names:
        for i, n in enumerate(names):
            if str(n).strip().lower() in _NORMAL_CLASS_NAMES:
                return i, None
        return 0, f"XGBoost: no class named 'normal' found in {list(names)}; assuming the first class is normal."
    return 0, "XGBoost: class names file missing; assuming the first class is normal."


def _build_ee_input(df: pd.DataFrame):
    """Rebuild the exact numbers the Elliptic Envelope was trained on:
    Yeo-Johnson on the skewed sensors + plain scaling on the rest, taken from the
    RAW (unscaled) sensor values. Returns None if the needed files are missing."""
    pt = ML_MODELS.get("power_transformer")
    sc = ML_MODELS.get("scaler_normal_ee")
    spec = ML_MODELS.get("ee_input_spec")
    if pt is None or sc is None or not spec:
        return None
    order = list(spec["feature_order"])
    skewed = list(spec["skewed_features"])
    normal = list(spec["normal_features"])
    raw = pd.DataFrame(index=df.index)
    for c in order:
        raw[c] = df[c] if c in df.columns else 0.0
    raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
    X = np.zeros((len(raw), len(order)))
    if skewed:
        X[:, [order.index(f) for f in skewed]] = pt.transform(raw[skewed])
    if normal:
        X[:, [order.index(f) for f in normal]] = sc.transform(raw[normal])
    return X


def _run_ensemble_scoring(df: pd.DataFrame):
    """
    Core ensemble scoring, factored out so both the live endpoint and the
    daily-persistence job compute the identical thing.

    Each voting model now scores the way it was trained:
      - Isolation Forest: flag when decision_function < saved cut-off
      - Elliptic Envelope: Yeo-Johnson input + saved cut-off
      - XGBoost: raw (unscaled) input, 'normal' class found by name
    If an extra file is missing the old behaviour is used and a note is added.
    """
    scaler = ML_MODELS.get("scaler")
    if scaler is None:
        return None

    notes = []

    if hasattr(scaler, "feature_names_in_"):
        feature_columns = list(scaler.feature_names_in_)
    else:
        feature_columns = list(AE_FEATURES)

    df_clean = df[[c for c in feature_columns if c in df.columns]].fillna(0).replace([np.inf, -np.inf], 0)
    for col in feature_columns:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
    df_clean = df_clean[feature_columns]

    X_scaled = scaler.transform(df_clean)

    model_results = {}
    all_scores = []
    all_votes = []

    iso = ML_MODELS.get("isolation_forest")
    if iso is not None:
        try:
            iso_scores = iso.score_samples(X_scaled)
            score_range = iso_scores.max() - iso_scores.min()
            iso_scores_norm = np.zeros_like(iso_scores) if score_range == 0 else 1 - (iso_scores - iso_scores.min()) / score_range
            if_thr = ML_MODELS.get("if_threshold")
            if if_thr is not None:
                iso_anomaly = iso.decision_function(X_scaled) < if_thr
            else:
                iso_anomaly = (iso.predict(X_scaled) == -1)
                notes.append("Isolation Forest used its default rule (saved cut-off missing).")
            model_results["isolation_forest"] = {"anomaly": iso_anomaly.tolist(), "scores": iso_scores_norm.tolist(), "name": "Isolation Forest"}
            all_scores.append(iso_scores_norm)
            all_votes.append(iso_anomaly.astype(int))
        except Exception as e:
            logger.error(f"Isolation Forest prediction error: {e}")

    ee = ML_MODELS.get("elliptic_envelope")
    if ee is not None:
        try:
            X_ee = None
            try:
                X_ee = _build_ee_input(df)
            except Exception as e:
                logger.error(f"Could not rebuild Elliptic Envelope input: {e}")
            if X_ee is None:
                X_ee = X_scaled
                notes.append("Elliptic Envelope was fed plain-scaled numbers (training files missing) - its flags are unreliable.")
            ee_scores = ee.score_samples(X_ee)
            ee_scores_norm = 1 - (ee_scores - ee_scores.min()) / (ee_scores.max() - ee_scores.min() + 1e-10)
            ee_thr = ML_MODELS.get("ee_threshold")
            if ee_thr is not None:
                ee_anomaly = ee.decision_function(X_ee) < ee_thr
            else:
                ee_anomaly = (ee.predict(X_ee) == -1)
            model_results["elliptic_envelope"] = {"anomaly": ee_anomaly.tolist(), "scores": ee_scores_norm.tolist(), "name": "Elliptic Envelope"}
            all_scores.append(ee_scores_norm)
            all_votes.append(ee_anomaly.astype(int))
        except Exception as e:
            logger.error(f"Elliptic Envelope prediction error: {e}")

    xgb = ML_MODELS.get("xgboost")
    if xgb is not None:
        try:
            # trained on RAW sensor values, so do not scale
            xgb_pred = xgb.predict(df_clean)
            xgb_proba = xgb.predict_proba(df_clean)
            normal_idx, nnote = _xgb_normal_index(xgb)
            if nnote:
                notes.append(nnote)
            if xgb_proba.shape[1] > 1 and normal_idx < xgb_proba.shape[1]:
                xgb_scores = 1 - xgb_proba[:, normal_idx]
                xgb_anomaly = np.argmax(xgb_proba, axis=1) != normal_idx
            else:
                xgb_scores = xgb_proba.flatten()
                xgb_anomaly = (xgb_pred != 0)
            model_results["xgboost"] = {"anomaly": xgb_anomaly.tolist(), "scores": xgb_scores.tolist(), "name": "XGBoost Classifier"}
            all_scores.append(xgb_scores)
            all_votes.append(xgb_anomaly.astype(int))
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")

    if not model_results:
        return None

    all_votes_array = np.array(all_votes)
    total_voters = len(all_votes)
    threshold_votes = max(1, total_voters // 2 + 1)
    vote_sum = np.sum(all_votes_array, axis=0)
    ensemble_pred = vote_sum >= threshold_votes

    all_scores_array = np.array(all_scores)
    ensemble_score = np.mean(all_scores_array, axis=0)
    srange = ensemble_score.max() - ensemble_score.min()
    ensemble_score_norm = (ensemble_score - ensemble_score.min()) / (srange + 1e-10)

    return {
        "ensemble_score_norm": ensemble_score_norm,
        "ensemble_pred": ensemble_pred,
        "vote_sum": vote_sum,
        "total_voters": total_voters,
        "threshold_votes": threshold_votes,
        "model_results": model_results,
        "notes": notes,
    }


@router.get("/ensemble-anomaly-detection")
async def get_ensemble_anomaly_detection(
    turbine_id: Optional[str] = Query(None, description="Filter by turbine"),
    days: int = Query(DEFAULT_LOOKBACK_DAYS, description="Look back N days"),
    limit: int = Query(DEFAULT_LIMIT, description="Max records"),
    min_score: float = Query(DEFAULT_MIN_SCORE, description="Minimum anomaly score threshold")
):
    """Ensemble anomaly detection across all loaded models using REAL SCADA DATA"""
    try:
        if hasattr(days, 'default'):
            days = days.default
        days = int(days) if days is not None else 30

        if hasattr(limit, 'default'):
            limit = limit.default
        limit = int(limit) if limit is not None else 500

        if hasattr(min_score, 'default'):
            min_score = min_score.default
        min_score = float(min_score) if min_score is not None else 0.0

        if hasattr(turbine_id, 'default'):
            turbine_id = turbine_id.default

        if not ML_MODELS.get("loaded"):
            load_ml_models()

        data_mode = check_data_mode()
        df = await _fetch_scada_data(turbine_id, days, limit)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": "No SCADA data available for the specified parameters",
                "anomalies": [],
                "total_predictions": 0,
                "models_loaded": ML_MODELS.get("loaded", False),
                "data_mode": data_mode
            }

        n_samples = len(df)

        scoring = _run_ensemble_scoring(df)
        if scoring is None:
            return {
                "status": "error",
                "message": "No trained ML models could be executed",
                "anomalies": [],
                "total_predictions": n_samples
            }

        ensemble_score_norm = scoring["ensemble_score_norm"]
        ensemble_pred = scoring["ensemble_pred"]
        vote_sum = scoring["vote_sum"]
        total_voters = scoring["total_voters"]
        threshold_votes = scoring["threshold_votes"]
        model_results = scoring["model_results"]

        # Persist daily aggregate for RUL and other pages (non-fatal on failure)
        try:
            from app.db.anomaly_json_store import save_daily_anomaly_scores
            save_daily_anomaly_scores(df, ensemble_score_norm, ensemble_pred)
        except Exception as e:
            logger.warning(f"Anomaly score persistence failed (non-fatal): {e}")

        anomalies = []
        for idx in np.where(ensemble_pred)[0]:
            if idx >= n_samples:
                continue
            score = float(ensemble_score_norm[idx])
            if score < min_score:
                continue
            
            if score > SEVERITY_HIGH:
                severity = "high"
            elif score > SEVERITY_MEDIUM:
                severity = "medium"
            else:
                severity = "low"
            
            anomalies.append({
                "turbine_id": str(df.iloc[idx]['turbine_id']) if 'turbine_id' in df.columns else f"WT-{idx}",
                "timestamp": df.iloc[idx]['time'].isoformat() if 'time' in df.columns and hasattr(df.iloc[idx]['time'], 'isoformat') else str(df.iloc[idx].get('time', datetime.now().isoformat())),
                "score": score,
                "severity": severity,
                "ensemble_votes": int(vote_sum[idx]),
                "total_voters": total_voters,
            })
        
        anomalies.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "status": "success",
            "anomalies": anomalies[:limit],
            "total_predictions": n_samples,
            "anomaly_count": len(anomalies),
            "models_loaded": ML_MODELS.get("loaded", False),
            "total_voters": total_voters,
            "ensemble_vote_threshold": threshold_votes,
            "voting_models": list(model_results.keys()),
            "score_only_models": SCORE_ONLY_MODELS,
            "scoring_notes": scoring.get("notes", []),
            "data_mode": data_mode,
            "lookback_days": days,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Fatal error in ensemble detection: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "anomalies": [],
            "total_predictions": 0
        }


# ─── HELPER: Fetch SCADA Data from Database ──────────────────────────

async def _fetch_scada_data(turbine_id: Optional[str], days: int, limit: int) -> pd.DataFrame:
    """Helper to fetch raw SCADA data from site_dzh.scada_raw table."""
    try:
        if hasattr(days, 'default'):
            days = days.default
        days = int(days) if days is not None else 30

        if hasattr(limit, 'default'):
            limit = limit.default
        limit = int(limit) if limit is not None else 500

        if hasattr(turbine_id, 'default'):
            turbine_id = turbine_id.default
        if turbine_id is not None:
            turbine_id = str(turbine_id)

        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed")
            return pd.DataFrame()
        
        cursor = conn.cursor()
        
        query = """
            SELECT 
                "time",
                "name" as turbine_id,
                "wind_speed_m_s" as wind_speed_ms,
                "generator_active_power_kw" as active_power,
                "generator_speed_r_min" as generator_speed,
                "rotor_speed_r_min" as rotor_speed,
                "torque_nm" as torque_nm,
                "pitch_angle_1" as pitch_angle_1deg,
                "pitch_angle_2" as pitch_angle_2deg,
                "pitch_angle_3" as pitch_angle_3deg,
                "gearbox_oil_temperature" as gearbox_oil_temperature,
                "gearbox_de_bearing_temperature" as gearbox_de_bearing_temperature,
                "gearbox_nde_bearing_temperature" as gearbox_nde_bearing_temperature,
                "main_bearing_temperature" as main_bearing_temperature,
                "generator_stator_temp_u" as generator_stator_temp_u,
                "generator_stator_temp_v" as generator_stator_temp_v,
                "generator_stator_temp_w" as generator_stator_temp_w,
                "generator_drive_end_bearing_temperature" as generator_drive_end_bearing_temperature,
                "generator_non_drive_end_bearing_temperature" as generator_non_drive_end_bearing_temperature,
                "temperature_outside_nacelle" as temperature_outside_nacelle,
                "temperature_inside_nacelle" as temperature_inside_nacelle,
                "vibration_lateral_mm_s" as vibration_lateral_mms,
                "vibration_axial_mm_s" as vibration_axial_mms,
                "q" as reactive_power,
                "gearbox_oil_inlet_temperature" as gearbox_oil_inlet_temperature
            FROM site_dzh.scada_raw
            WHERE name IS NOT NULL AND name != ''
        """
        params = []
        
        if turbine_id:
            query += " AND name = %s"
            params.append(turbine_id)
        
        query += " ORDER BY time DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        colnames = [desc[0] for desc in cursor.description]
        conn.close()
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records, columns=colnames)
        
        # Calculate derived feature columns matching MAIN_FEATURES
        if 'gearbox_oil_temperature' in df.columns and 'gearbox_oil_inlet_temperature' in df.columns:
            df['gearbox_temp_delta'] = pd.to_numeric(df['gearbox_oil_temperature'], errors='coerce') - pd.to_numeric(df['gearbox_oil_inlet_temperature'], errors='coerce')
        else:
            df['gearbox_temp_delta'] = 0.0

        if 'pitch_angle_1deg' in df.columns and 'pitch_angle_2deg' in df.columns and 'pitch_angle_3deg' in df.columns:
            p1 = pd.to_numeric(df['pitch_angle_1deg'], errors='coerce').fillna(0)
            p2 = pd.to_numeric(df['pitch_angle_2deg'], errors='coerce').fillna(0)
            p3 = pd.to_numeric(df['pitch_angle_3deg'], errors='coerce').fillna(0)
            df['pitch_imbalance'] = np.maximum(np.abs(p1 - p2), np.maximum(np.abs(p2 - p3), np.abs(p3 - p1)))
        else:
            df['pitch_imbalance'] = 0.0

        if 'generator_stator_temp_u' in df.columns and 'generator_stator_temp_v' in df.columns and 'generator_stator_temp_w' in df.columns:
            tu = pd.to_numeric(df['generator_stator_temp_u'], errors='coerce').fillna(0)
            tv = pd.to_numeric(df['generator_stator_temp_v'], errors='coerce').fillna(0)
            tw = pd.to_numeric(df['generator_stator_temp_w'], errors='coerce').fillna(0)
            df['stator_imbalance'] = np.maximum(np.abs(tu - tv), np.maximum(np.abs(tv - tw), np.abs(tw - tu)))
        else:
            df['stator_imbalance'] = 0.0

        if 'gearbox_de_bearing_temperature' in df.columns and 'gearbox_nde_bearing_temperature' in df.columns:
            de = pd.to_numeric(df['gearbox_de_bearing_temperature'], errors='coerce').fillna(0)
            nde = pd.to_numeric(df['gearbox_nde_bearing_temperature'], errors='coerce').fillna(0)
            df['bearing_de_nde_diff'] = np.abs(de - nde)
        else:
            df['bearing_de_nde_diff'] = 0.0

        if 'pitch_angle_1deg' in df.columns:
            df['pitch_angle_1_deg'] = pd.to_numeric(df['pitch_angle_1deg'], errors='coerce').fillna(0)
        if 'pitch_angle_2deg' in df.columns:
            df['pitch_angle_2_deg'] = pd.to_numeric(df['pitch_angle_2deg'], errors='coerce').fillna(0)
        if 'pitch_angle_3deg' in df.columns:
            df['pitch_angle_3_deg'] = pd.to_numeric(df['pitch_angle_3deg'], errors='coerce').fillna(0)

        numeric_cols = [c for c in df.columns if c not in ('time', 'turbine_id')]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching SCADA data: {e}")
        return pd.DataFrame()


# ─── TRAINING STATUS / TRIGGER ───────────────────────────────────────
# NOTE: paths are relative. The router is mounted with prefix /api/v1/ml,
# so these are served at /api/v1/ml/training/status and /api/v1/ml/training/trigger.
# The single "training is running" flag lives in retrain_watcher.py so that
# every way of starting training shares it.

@router.get("/training/status")
def training_status():
    try:
        state = _load_state() or {}
    except Exception as e:
        logger.warning(f"Could not load training state: {e}")
        state = {}

    try:
        history = get_training_history(limit=10)
    except Exception as e:
        logger.warning(f"Could not load training history: {e}")
        history = []

    advice = get_retrain_advice()

    return {
        "is_training": is_training(),
        "last_trained_at": state.get("last_trained_at"),
        "last_trained_row_count": state.get("last_trained_row_count"),
        "current_row_count": advice["current_row_count"],
        "new_rows": advice["new_rows"],
        "row_threshold": advice["row_threshold"],
        "retrain_recommended": advice["retrain_recommended"],
        "history": history,
    }


@router.post("/training/trigger")
def trigger_training():
    if not start_training(trigger="manual"):
        return {"status": "already_running", "message": "Training is already in progress"}
    return {"status": "started", "message": "Training started in background, check /training/status for progress"}