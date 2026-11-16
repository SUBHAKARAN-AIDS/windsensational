# app/core/config.py
#
# Central configuration. All environment-specific and previously hard-coded
# values live here and are overridable via environment variables (.env).

import os
from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
# Credentials come from the environment. There is deliberately NO hard-coded
# password fallback (the previous 'Bala12345' default was removed). Non-secret
# connection fields fall back to sensible values; the password must be provided
# via WINDSENTINEL_DB_PASSWORD. get_db_connection() validates presence and fails
# with a clear message rather than silently using a baked-in credential.
DB_CONFIG = {
    "host": os.environ.get("WINDSENTINEL_DB_HOST", ""),
    "port": int(os.environ.get("WINDSENTINEL_DB_PORT", "5432")),
    "database": os.environ.get("WINDSENTINEL_DB_NAME", ""),
    "user": os.environ.get("WINDSENTINEL_DB_USER", ""),
    "password": os.environ.get("WINDSENTINEL_DB_PASSWORD", ""),
}

# Fields that must be non-empty for a connection to be attempted.
DB_REQUIRED_FIELDS = ("host", "database", "user", "password")


# ---------------------------------------------------------------------------
# Data window (historical mode)
# ---------------------------------------------------------------------------
# Previously hard-coded inline in SQL and in get_date_condition(). Kept as the
# same default values (behavior preserved) but now overridable.
DATA_HISTORICAL_START = os.environ.get("DATA_HISTORICAL_START", "2025-01-01 00:00:00")
DATA_HISTORICAL_END = os.environ.get("DATA_HISTORICAL_END", "2026-03-31 23:59:59")


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ORIGINS = _split_csv(
    os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:5174,*",
    )
)


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
# Directory holding the saved model artifacts. Defaults to the package-local
# app/saved_models/ (resolved in ml/model_state.py); override if externalized.
SAVED_MODELS_DIR = os.environ.get("SAVED_MODELS_DIR", "")

# The saved VAE (vae_autoencoder.pth) has input_dim=3, but the 3 training
# features are NOT recorded in model_config.pkl (which only carries the 26-item
# features_list and vae_threshold) or anywhere else in the codebase. Feeding it
# any other feature set would produce meaningless reconstruction errors, so the
# VAE is loaded (for validation) but EXCLUDED from ensemble scoring by default.
# Set VAE_ENABLED=1 only once a correct 3-feature mapping + scaler is supplied.
VAE_ENABLED = os.environ.get("VAE_ENABLED", "0") in ("1", "true", "True", "yes")


# ---------------------------------------------------------------------------
# Canonical feature contracts (single source of truth)
# ---------------------------------------------------------------------------
# These lists were previously duplicated inline in api/routes/ml.py (and again
# in the quarantined legacy code). They are consolidated here so exactly one
# definition drives inference. Order is significant: it is the order the
# artifacts were trained/scaled with.
#
# MAIN_FEATURES is validated at model-load time against model_config.pkl's
# "features_list"; a mismatch is logged as an error (contract drift).
MAIN_FEATURES = [
    "gearbox_oil_temperature",
    "gearbox_oil_inlet_temperature",
    "gearbox_de_bearing_temperature",
    "gearbox_nde_bearing_temperature",
    "main_bearing_temperature",
    "generator_stator_temp_u",
    "generator_stator_temp_v",
    "generator_stator_temp_w",
    "generator_drive_end_bearing_temperature",
    "generator_non_drive_end_bearing_temperature",
    "vibration_lateral_mms",
    "vibration_axial_mms",
    "active_power",
    "reactive_power",
    "rotor_speed",
    "generator_speed",
    "torque_nm",
    "pitch_angle_1deg",
    "pitch_angle_2deg",
    "pitch_angle_3deg",
    "temperature_outside_nacelle",
    "wind_speed_ms",
    "gearbox_temp_delta",
    "pitch_imbalance",
    "stator_imbalance",
    "bearing_de_nde_diff",
]

# 18-feature contract used by the LSTM autoencoder (scaler_ae.pkl was fit on 18).
AE_FEATURES = [
    "active_power",
    "generator_speed",
    "rotor_speed",
    "torque_nm",
    "pitch_angle_1deg",
    "pitch_angle_2deg",
    "pitch_angle_3deg",
    "gearbox_oil_temperature",
    "gearbox_de_bearing_temperature",
    "gearbox_nde_bearing_temperature",
    "main_bearing_temperature",
    "generator_stator_temp_u",
    "generator_stator_temp_v",
    "generator_stator_temp_w",
    "vibration_lateral_mms",
    "vibration_axial_mms",
    "wind_speed_ms",
    "temperature_outside_nacelle",
]


# ---------------------------------------------------------------------------
# Database object names (single source of truth)
# ---------------------------------------------------------------------------
# Previously hard-coded inline in every query. Kept as the same value but
# centralized so schema/table renames happen in one place.
SCADA_SCHEMA = os.environ.get("SCADA_SCHEMA", "site_dzh")
SCADA_TABLE = os.environ.get("SCADA_TABLE", f"{SCADA_SCHEMA}.scada_raw")


# ---------------------------------------------------------------------------
# Model contract (dimensions / architecture — validated at load, not forced)
# ---------------------------------------------------------------------------
MAIN_FEATURE_COUNT = len(MAIN_FEATURES)          # 26
LSTM_SEQ_LEN = int(os.environ.get("LSTM_SEQ_LEN", "20"))
LSTM_INPUT_WIDTH = len(AE_FEATURES)              # 18  -> LSTM contract (20, 18)
VAE_INPUT_DIM = int(os.environ.get("VAE_INPUT_DIM", "3"))  # checkpoint contract
AE_RECON_PERCENTILE = float(os.environ.get("AE_RECON_PERCENTILE", "95"))


# ---------------------------------------------------------------------------
# Ensemble policy (which models VOTE vs are SCORE-ONLY) + thresholds
# ---------------------------------------------------------------------------
# Only models with a trained ABSOLUTE decision boundary vote. The autoencoders
# produce a reconstruction-error SCORE but have no saved absolute threshold
# (only a per-batch percentile, which is non-deterministic), so they are
# advisory/score-only and never counted as ensemble voters. See the ensemble
# route for how this is enforced.
VOTING_MODELS = ["isolation_forest", "elliptic_envelope", "xgboost"]
SCORE_ONLY_MODELS = ["lstm_autoencoder"]   # LSTM AE: advisory score, non-voting
# VAE is neither: excluded entirely until its 3-feature mapping is recovered.

SEVERITY_HIGH = float(os.environ.get("SEVERITY_HIGH", "0.85"))
SEVERITY_MEDIUM = float(os.environ.get("SEVERITY_MEDIUM", "0.75"))

# API defaults (were literals in the route signature / SQL).
DEFAULT_LOOKBACK_DAYS = int(os.environ.get("DEFAULT_LOOKBACK_DAYS", "7"))
DEFAULT_LIMIT = int(os.environ.get("DEFAULT_LIMIT", "200"))
DEFAULT_MIN_SCORE = float(os.environ.get("DEFAULT_MIN_SCORE", "0.75"))

RUL_FAILURE_THRESHOLD = float(os.environ.get("RUL_FAILURE_THRESHOLD", "0.2"))
RUL_LOOKBACK_DAYS = int(os.environ.get("RUL_LOOKBACK_DAYS", "30"))
RUL_BASELINE_FRACTION = float(os.environ.get("RUL_BASELINE_FRACTION", "0.2"))
RUL_VIBRATION_Z_THRESHOLD = float(os.environ.get("RUL_VIBRATION_Z_THRESHOLD", "3.0"))
RUL_TEMPERATURE_Z_THRESHOLD = float(os.environ.get("RUL_TEMPERATURE_Z_THRESHOLD", "3.0"))
RUL_BASELINE_VARIATION_FLOOR = float(os.environ.get("RUL_BASELINE_VARIATION_FLOOR", "0.05"))
RUL_CRITICAL_DAYS = float(os.environ.get("RUL_CRITICAL_DAYS", "21"))
RUL_MONITOR_DAYS = float(os.environ.get("RUL_MONITOR_DAYS", "90"))
RUL_PRIMARY_COMPONENT_MAP = {
    # optional: map which feature drove the anomaly score to a component label
    "vibration_lateral": "Main Bearing Wear",
    "gearbox_bearing_temp": "Gearbox Bearing Wear",
}