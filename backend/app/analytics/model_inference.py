"""
app/analytics/model_inference.py

Real-time inference for the ML Models page — ONLY the models that are
actually trained and verified against the training notebook:

  - XGBoost classifier          (26 features, scaler.pkl)
  - Isolation Forest            (26 features, scaler.pkl)
  - Elliptic Envelope           (26 features, scaler.pkl)
  - LSTM Autoencoder            (18 features, scaler_lstm.pkl, seq_len=20)

Deliberately EXCLUDED (do not wire these up as "live"):
  - VAE (vae_autoencoder.pth)   — verified trained on synthetic demo data
                                  (temp_sensor/vibe_sensor/pres_sensor),
                                  not real SCADA data. Retrain before using.
  - Prophet, Gaussian Process Regression, CUSUM+SPRT — never implemented.
"""

import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor  # type: ignore

from app.db.connection import get_db_connection

logger = logging.getLogger(__name__)

# ============================================================
# SAVED_MODELS PATH — resolved so it works regardless of which
# directory `uvicorn`/`python` was launched from.
#
# Priority:
#   1. SAVED_MODELS_DIR env var, if set
#   2. saved_models/ next to this file's package root (backend/saved_models)
#   3. saved_models/ in the current working directory (fallback)
# ============================================================

def _resolve_models_dir() -> str:
    env_dir = os.environ.get("SAVED_MODELS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # this file is at .../backend/app/analytics/model_inference.py
    # -> go up to backend/, then look for saved_models/ there
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidate = os.path.join(backend_root, "saved_models")
    if os.path.isdir(candidate):
        return candidate

    # fallback: relative to whatever directory the process was launched from
    cwd_candidate = os.path.join(os.getcwd(), "saved_models")
    return cwd_candidate


MODELS_DIR = _resolve_models_dir()
logger.info(f"model_inference: using MODELS_DIR = {MODELS_DIR}")

# ============================================================
# DB COLUMN -> MODEL FEATURE NAME MAPPING
# ============================================================
COL_SPECS = [
    ("wind_speed_m_s", -5, 50, "wind_speed_ms"),
    ("generator_active_power_kw", -500, 8000, "active_power"),
    ("q", -5000, 5000, "reactive_power"),
    ("rotor_speed_r_min", 0, 30, "rotor_speed"),
    ("generator_speed_r_min", 0, 2000, "generator_speed"),
    ("torque_nm", 0, 1000000, "torque_nm"),
    ("pitch_angle_1", -5, 95, "pitch_angle_1deg"),
    ("pitch_angle_2", -5, 95, "pitch_angle_2deg"),
    ("pitch_angle_3", -5, 95, "pitch_angle_3deg"),
    ("gearbox_oil_temperature", -10, 150, "gearbox_oil_temperature"),
    ("gearbox_oil_inlet_temperature", -10, 150, "gearbox_oil_inlet_temperature"),
    ("gearbox_de_bearing_temperature", -10, 150, "gearbox_de_bearing_temperature"),
    ("gearbox_nde_bearing_temperature", -10, 150, "gearbox_nde_bearing_temperature"),
    ("main_bearing_temperature", -10, 150, "main_bearing_temperature"),
    ("generator_stator_temp_u", -10, 200, "generator_stator_temp_u"),
    ("generator_stator_temp_v", -10, 200, "generator_stator_temp_v"),
    ("generator_stator_temp_w", -10, 200, "generator_stator_temp_w"),
    ("generator_drive_end_bearing_temperature", -10, 150, "generator_drive_end_bearing_temperature"),
    ("generator_non_drive_end_bearing_temperature", -10, 150, "generator_non_drive_end_bearing_temperature"),
    ("temperature_outside_nacelle", -40, 60, "temperature_outside_nacelle"),
    ("vibration_lateral_mm_s", 0, 50, "vibration_lateral_mms"),
    ("vibration_axial_mm_s", 0, 50, "vibration_axial_mms"),
]
DB_TO_MODEL = {db_col: model_name for db_col, _, _, model_name in COL_SPECS}
_ALL_DB_COLS = sorted(DB_TO_MODEL.keys())

# ============================================================
# FEATURE LISTS
# ============================================================
FEATURES_26 = [
    "gearbox_oil_temperature", "gearbox_oil_inlet_temperature",
    "gearbox_de_bearing_temperature", "gearbox_nde_bearing_temperature",
    "main_bearing_temperature", "generator_stator_temp_u",
    "generator_stator_temp_v", "generator_stator_temp_w",
    "generator_drive_end_bearing_temperature",
    "generator_non_drive_end_bearing_temperature",
    "vibration_lateral_mms", "vibration_axial_mms", "active_power",
    "reactive_power", "rotor_speed", "generator_speed", "torque_nm",
    "pitch_angle_1deg", "pitch_angle_2deg", "pitch_angle_3deg",
    "temperature_outside_nacelle", "wind_speed_ms",
    "gearbox_temp_delta", "pitch_imbalance", "stator_imbalance",
    "bearing_de_nde_diff",
]

FEATURES_18_LSTM = [
    "wind_speed_ms", "active_power", "vibration_lateral_mms",
    "gearbox_oil_temperature", "gearbox_oil_inlet_temperature",
    "gearbox_de_bearing_temperature", "gearbox_nde_bearing_temperature",
    "main_bearing_temperature", "generator_stator_temp_u",
    "generator_stator_temp_v", "generator_stator_temp_w",
    "generator_drive_end_bearing_temperature",
    "generator_non_drive_end_bearing_temperature",
    "temperature_outside_nacelle", "pitch_angle_1deg",
    "pitch_angle_2deg", "pitch_angle_3deg", "vibration_axial_mms",
]

SEQ_LEN = 20
LSTM_CONTAMINATION = 0.05


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gearbox_temp_delta"] = (
        df["gearbox_oil_temperature"] - df["gearbox_oil_inlet_temperature"]
    )
    df["pitch_imbalance"] = df[
        ["pitch_angle_1deg", "pitch_angle_2deg", "pitch_angle_3deg"]
    ].std(axis=1)
    df["stator_imbalance"] = df[
        ["generator_stator_temp_u", "generator_stator_temp_v", "generator_stator_temp_w"]
    ].std(axis=1)
    df["bearing_de_nde_diff"] = (
        df["gearbox_de_bearing_temperature"] - df["gearbox_nde_bearing_temperature"]
    )
    return df


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    artifacts = {}

    def _try_joblib(name, filename):
        path = os.path.join(MODELS_DIR, filename)
        try:
            artifacts[name] = joblib.load(path)
            logger.info(f"Loaded {name} from {path}")
        except Exception as e:
            logger.error(f"Failed to load {name} from {path}: {e}")

    _try_joblib("scaler_26", "scaler.pkl")
    _try_joblib("isolation_forest", "isolation_forest.pkl")
    _try_joblib("elliptic_envelope", "elliptic_envelope.pkl")
    _try_joblib("xgb_classifier", "xgb_classifier.pkl")
    _try_joblib("scaler_lstm", "scaler_lstm.pkl")

    try:
        import tensorflow as tf
        path = os.path.join(MODELS_DIR, "lstm_autoencoder.keras")
        artifacts["lstm_autoencoder"] = tf.keras.models.load_model(path)
        logger.info(f"Loaded lstm_autoencoder from {path}")
    except Exception as e:
        logger.error(f"Failed to load lstm_autoencoder: {e}")

    return artifacts


def fetch_latest_rows(turbine_id: str, n_rows: int = SEQ_LEN) -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        logger.error("Model inference: database connection failed")
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            select_cols = ", ".join(f'"{c}"' for c in _ALL_DB_COLS)
            query = f"""
                SELECT time, name as turbine_id, {select_cols}
                FROM site_dzh.scada_raw
                WHERE name = %(turbine_id)s
                ORDER BY time DESC
                LIMIT %(n_rows)s
            """
            cursor.execute(query, {"turbine_id": turbine_id, "n_rows": n_rows})
            records = cursor.fetchall()
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.rename(columns=DB_TO_MODEL)

            for model_name in DB_TO_MODEL.values():
                if model_name in df.columns:
                    df[model_name] = pd.to_numeric(
                        df[model_name].astype(str).str.strip().replace({"": None}),
                        errors="coerce",
                    )

            df = add_derived_features(df)
            df = df.sort_values("time").reset_index(drop=True)
            return df
    except Exception as e:
        logger.error(f"Error fetching latest rows for {turbine_id}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def _prepare_26_feature_row(df: pd.DataFrame):
    if df.empty:
        return None
    latest = df.iloc[[-1]]
    missing = [c for c in FEATURES_26 if c not in latest.columns]
    if missing:
        logger.warning(f"Missing 26-feature columns: {missing}")
        return None
    return latest[FEATURES_26].fillna(0).values


def predict_xgb(df, artifacts) -> dict:
    model = artifacts.get("xgb_classifier")
    scaler = artifacts.get("scaler_26")
    if model is None or scaler is None:
        return {"status": "load_failed"}
    X = _prepare_26_feature_row(df)
    if X is None:
        return {"status": "no_data"}
    X_scaled = scaler.transform(X)
    pred_class = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    return {
        "status": "ok",
        "predicted_class": pred_class,
        "anomaly_probability": round(float(proba[1]), 4) if len(proba) > 1 else None,
    }


def predict_isolation_forest(df, artifacts) -> dict:
    model = artifacts.get("isolation_forest")
    scaler = artifacts.get("scaler_26")
    if model is None or scaler is None:
        return {"status": "load_failed"}
    X = _prepare_26_feature_row(df)
    if X is None:
        return {"status": "no_data"}
    X_scaled = scaler.transform(X)
    raw_pred = int(model.predict(X_scaled)[0])
    score = float(model.decision_function(X_scaled)[0])
    return {"status": "ok", "is_anomaly": raw_pred == -1, "anomaly_score": round(score, 4)}


def predict_elliptic_envelope(df, artifacts) -> dict:
    model = artifacts.get("elliptic_envelope")
    scaler = artifacts.get("scaler_26")
    if model is None or scaler is None:
        return {"status": "load_failed"}
    X = _prepare_26_feature_row(df)
    if X is None:
        return {"status": "no_data"}
    X_scaled = scaler.transform(X)
    raw_pred = int(model.predict(X_scaled)[0])
    score = float(model.decision_function(X_scaled)[0])
    return {"status": "ok", "is_anomaly": raw_pred == -1, "anomaly_score": round(score, 4)}


def predict_lstm(df, artifacts) -> dict:
    model = artifacts.get("lstm_autoencoder")
    scaler = artifacts.get("scaler_lstm")
    if model is None or scaler is None:
        return {"status": "load_failed"}
    if len(df) < SEQ_LEN:
        return {"status": "no_data", "detail": f"need {SEQ_LEN} rows, got {len(df)}"}
    missing = [c for c in FEATURES_18_LSTM if c not in df.columns]
    if missing:
        logger.warning(f"Missing 18-feature LSTM columns: {missing}")
        return {"status": "load_failed"}

    window = df.iloc[-SEQ_LEN:][FEATURES_18_LSTM].fillna(0).values
    window_scaled = scaler.transform(window)
    sequence = window_scaled.reshape(1, SEQ_LEN, len(FEATURES_18_LSTM))

    reconstructed = model.predict(sequence, verbose=0)
    recon_error = float(np.mean(np.square(sequence - reconstructed)))

    return {
        "status": "ok",
        "reconstruction_error": round(recon_error, 5),
        "threshold": None,
        "threshold_note": (
            f"No saved threshold for this model. Training used the "
            f"{1 - LSTM_CONTAMINATION:.0%} quantile of reconstruction error "
            f"as a cutoff, but that value was never persisted."
        ),
    }


def build_ml_models_payload(turbine_id: str) -> dict:
    artifacts = load_artifacts()
    df = fetch_latest_rows(turbine_id, n_rows=SEQ_LEN)

    xgb_result = predict_xgb(df, artifacts)
    if_result = predict_isolation_forest(df, artifacts)
    ee_result = predict_elliptic_envelope(df, artifacts)
    lstm_result = predict_lstm(df, artifacts)

    not_trained = {
        "vae": {
            "status": "not_trained",
            "reason": "Saved artifact was trained on synthetic demo data, not real SCADA data. Retrain before use.",
        },
        "prophet": {"status": "not_trained", "reason": "No trained artifact exists yet."},
        "gaussian_process_regression": {"status": "not_trained", "reason": "No trained artifact exists yet."},
        "cusum_sprt": {"status": "not_trained", "reason": "No trained artifact exists yet."},
    }

    return {
        "turbine_id": turbine_id,
        "models_dir_used": MODELS_DIR,
        "models": {
            "xgboost": xgb_result,
            "isolation_forest": if_result,
            "elliptic_envelope": ee_result,
            "lstm_autoencoder": lstm_result,
            **not_trained,
        },
    }