# app/ml/train_runner.py
import os
import shutil
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from app.db.connection import get_db_connection
from app.db.training_status_store import record_training_run
from app.ml.retrain_watcher import mark_trained
from app.ml import model_state

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
STAGING_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models_staging")

# ── Matches Cell 3's sanity_rules keys exactly ──────────────────
SANITY_RULES = {
    'wind_speed_ms': (0.1, 50),
    'active_power': (10, 8000),
    'vibration_lateral_mms': (0.01, 50),
    'gearbox_oil_temperature': (20, 150),
    'gearbox_oil_inlet_temperature': (20, 150),
    'gearbox_de_bearing_temperature': (20, 150),
    'gearbox_nde_bearing_temperature': (20, 150),
    'main_bearing_temperature': (20, 150),
    'generator_stator_temp_u': (30, 200),
    'generator_stator_temp_v': (30, 200),
    'generator_stator_temp_w': (30, 200),
    'generator_drive_end_bearing_temperature': (20, 150),
    'generator_non_drive_end_bearing_temperature': (20, 150),
    'temperature_outside_nacelle': (-40, 60),
    'pitch_angle_1_deg': (-5, 95),
    'pitch_angle_2_deg': (-5, 95),
    'pitch_angle_3_deg': (-5, 95),
    'vibration_axial_mms': (0.01, 50),
}
BASE_FEATURES = list(SANITY_RULES.keys())  # 18

# Extra raw columns that live in your 26-feature production contract
# but weren't in sanity_rules — add them here if your scaler expects 26
EXTRA_RAW_FEATURES = ['reactive_power', 'rotor_speed', 'generator_speed', 'torque_nm']

DERIVED_FEATURES = ['gearbox_temp_delta', 'pitch_imbalance', 'stator_imbalance', 'bearing_de_nde_diff']

# Skewed features that get Yeo-Johnson-transformed before Elliptic Envelope (Cell 10B)
SKEWED_FEATURES = [
    'vibration_lateral_mms', 'vibration_axial_mms', 'reactive_power',
    'pitch_angle_1_deg', 'pitch_angle_2_deg', 'pitch_angle_3_deg',
    'gearbox_de_bearing_temperature', 'wind_speed_ms',
    'gearbox_oil_temperature', 'gearbox_nde_bearing_temperature',
]

# LSTM config — matches Cell 12B
SEQ_LEN = 20
MAX_GAP_MINUTES = 20
LSTM_UNITS = 32
EPOCHS = 15
BATCH_SIZE = 128
LSTM_CONTAMINATION = 0.05

RPM_CUTIN = 1300.0
PITCH_FEATHER_CUTOFF = 60.0


# ═════════════════════════════════════════════════════════════
# 1. LOAD — mirrors Cell 2, but window is rolling instead of fixed dates
# ═════════════════════════════════════════════════════════════
def _load_training_data(lookback_days: int = 90) -> pd.DataFrame:
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Could not connect to database for training data load")
    try:
        query = f"""
            WITH ranked AS (
                SELECT
                    name AS turbine_id,
                    time AS date_time,
                    wind_speed_m_s::double precision AS wind_speed_ms,
                    generator_active_power_kw::double precision AS active_power,
                    q::double precision AS reactive_power,
                    rotor_speed_r_min::double precision AS rotor_speed,
                    generator_speed_r_min::double precision AS generator_speed,
                    torque_nm::double precision AS torque_nm,
                    pitch_angle_1::double precision AS pitch_angle_1_deg,
                    pitch_angle_2::double precision AS pitch_angle_2_deg,
                    pitch_angle_3::double precision AS pitch_angle_3_deg,
                    gearbox_oil_temperature::double precision AS gearbox_oil_temperature,
                    gearbox_oil_inlet_temperature::double precision AS gearbox_oil_inlet_temperature,
                    gearbox_de_bearing_temperature::double precision AS gearbox_de_bearing_temperature,
                    gearbox_nde_bearing_temperature::double precision AS gearbox_nde_bearing_temperature,
                    main_bearing_temperature::double precision AS main_bearing_temperature,
                    generator_stator_temp_u::double precision AS generator_stator_temp_u,
                    generator_stator_temp_v::double precision AS generator_stator_temp_v,
                    generator_stator_temp_w::double precision AS generator_stator_temp_w,
                    generator_drive_end_bearing_temperature::double precision AS generator_drive_end_bearing_temperature,
                    generator_non_drive_end_bearing_temperature::double precision AS generator_non_drive_end_bearing_temperature,
                    temperature_outside_nacelle::double precision AS temperature_outside_nacelle,
                    vibration_lateral_mm_s::double precision AS vibration_lateral_mms,
                    vibration_axial_mm_s::double precision AS vibration_axial_mms,
                    ROW_NUMBER() OVER (PARTITION BY name, time ORDER BY time) AS rn
                FROM site_dzh.scada_raw
                WHERE time >= NOW() - INTERVAL '{lookback_days} days'
                    AND name IS NOT NULL
                    AND generator_active_power_kw::double precision > 50.0
                    AND generator_speed_r_min::double precision > {RPM_CUTIN}
                    AND wind_speed_m_s::double precision > 3.0
                    AND torque_nm::double precision > 0
                    AND pitch_angle_1::double precision < {PITCH_FEATHER_CUTOFF}
                    AND rotor_speed_r_min::double precision > 2.0
            )
            SELECT * FROM ranked WHERE rn = 1 ORDER BY turbine_id, date_time
        """
        df = pd.read_sql(query, conn, parse_dates=["date_time"])
        return df
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════
# 2. CLEAN + FEATURE ENGINEER — mirrors Cell 2 (baseline flag) + Cell 3
# ═════════════════════════════════════════════════════════════
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # physical bounds (Cell 3)
    for col, (lo, hi) in SANITY_RULES.items():
        if col in df_clean.columns:
            bad = ~df_clean[col].between(lo, hi)
            df_clean.loc[bad, col] = np.nan

    all_numeric_cols = BASE_FEATURES + EXTRA_RAW_FEATURES

    # per-turbine median fill, then fleet-median fallback
    for col in all_numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('turbine_id')[col].transform(lambda x: x.fillna(x.median()))
    for col in all_numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # derived features (Cell 3)
    df_clean['gearbox_temp_delta'] = df_clean['gearbox_oil_temperature'] - df_clean['gearbox_oil_inlet_temperature']
    df_clean['pitch_imbalance'] = df_clean[['pitch_angle_1_deg', 'pitch_angle_2_deg', 'pitch_angle_3_deg']].std(axis=1)
    df_clean['stator_imbalance'] = df_clean[['generator_stator_temp_u', 'generator_stator_temp_v', 'generator_stator_temp_w']].std(axis=1)
    df_clean['bearing_de_nde_diff'] = df_clean['gearbox_de_bearing_temperature'] - df_clean['gearbox_nde_bearing_temperature']

    # steady-baseline flag (Cell 2) — this is what IF/EE/LSTM fit on
    df_clean['is_steady_baseline'] = (
        df_clean['wind_speed_ms'].between(5.0, 12.0) &
        df_clean['active_power'].between(200.0, 2000.0) &
        df_clean['generator_speed'].between(1350.0, 1750.0) &
        df_clean['pitch_angle_1_deg'].between(-2.0, 5.0)
    )

    return df_clean


# ═════════════════════════════════════════════════════════════
# 3. FIT — mirrors Cells 11 (scaler), 10B (Yeo-Johnson), 12 (IF), 13 (EE), 12B (LSTM)
# ═════════════════════════════════════════════════════════════
def _fit_models(df_clean: pd.DataFrame, main_features: list):
    baseline_mask = df_clean['is_steady_baseline'].values

    # ── main scaler + Isolation Forest (Cell 12) ──
    scaler = StandardScaler()
    X_baseline = df_clean.loc[baseline_mask, main_features].fillna(0)
    scaler.fit(X_baseline)
    X_all = scaler.transform(df_clean[main_features].fillna(0))
    X_baseline_scaled = scaler.transform(X_baseline)

    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso.fit(X_baseline_scaled)
    if_baseline_scores = iso.decision_function(X_baseline_scaled)
    if_threshold = float(np.percentile(if_baseline_scores, 2))

    # ── Yeo-Johnson transform + Elliptic Envelope (Cells 10B, 13) ──
    skewed_present = [f for f in SKEWED_FEATURES if f in main_features]
    normal_present = [f for f in main_features if f not in skewed_present]

    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_skewed_all = pt.fit_transform(df_clean[skewed_present].fillna(0))
    sc_normal = StandardScaler()
    X_normal_all = sc_normal.fit_transform(df_clean[normal_present].fillna(0))

    X_stats_input = np.zeros((len(df_clean), len(main_features)))
    skewed_idx = [main_features.index(f) for f in skewed_present]
    normal_idx = [main_features.index(f) for f in normal_present]
    X_stats_input[:, skewed_idx] = X_skewed_all
    X_stats_input[:, normal_idx] = X_normal_all

    X_ee_baseline = X_stats_input[baseline_mask]
    ee = EllipticEnvelope(contamination=0.05, support_fraction=0.75, random_state=42)
    ee.fit(X_ee_baseline)
    ee_baseline_scores = ee.decision_function(X_ee_baseline)
    ee_threshold = float(np.percentile(ee_baseline_scores, 2))

    return {
        "scaler": scaler,
        "isolation_forest": iso,
        "if_threshold": if_threshold,
        "power_transformer": pt,
        "scaler_normal_ee": sc_normal,
        "elliptic_envelope": ee,
        "ee_threshold": ee_threshold,
        "skewed_features": skewed_present,
        "normal_features": normal_present,
    }


def _fit_lstm_autoencoder(df_clean: pd.DataFrame, lstm_features: list):
    """Mirrors Cell 12B. Trained on ALL production rows (not baseline-only),
    matching your notebook's current behavior."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    scaler_lstm = StandardScaler()
    X_scaled = scaler_lstm.fit_transform(df_clean[lstm_features].fillna(0))
    df_scaled = pd.DataFrame(X_scaled, columns=lstm_features, index=df_clean.index)
    df_scaled['turbine_id'] = df_clean['turbine_id'].values
    df_scaled['date_time'] = df_clean['date_time'].values

    sequences = []
    for tid, grp in df_scaled.groupby('turbine_id'):
        grp = grp.sort_values('date_time')
        time_diffs = grp['date_time'].diff().dt.total_seconds() / 60.0
        segment_id = (time_diffs > MAX_GAP_MINUTES).cumsum()
        grp = grp.assign(segment_id=segment_id)
        for _, seg in grp.groupby('segment_id'):
            seg_features = seg[lstm_features].values
            if len(seg_features) < SEQ_LEN:
                continue
            for start in range(len(seg_features) - SEQ_LEN + 1):
                sequences.append(seg_features[start:start + SEQ_LEN])

    sequences = np.array(sequences)
    if len(sequences) < 50:
        raise ValueError(f"Not enough sequences to train LSTM: {len(sequences)}")

    split = int(len(sequences) * 0.8)
    X_train_seq, X_val_seq = sequences[:split], sequences[split:]

    n_features = len(lstm_features)
    inputs = layers.Input(shape=(SEQ_LEN, n_features))
    encoded = layers.LSTM(LSTM_UNITS, activation='tanh', return_sequences=False)(inputs)
    repeated = layers.RepeatVector(SEQ_LEN)(encoded)
    decoded = layers.LSTM(LSTM_UNITS, activation='tanh', return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train_seq, X_train_seq, validation_data=(X_val_seq, X_val_seq),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    reconstructed = autoencoder.predict(sequences, batch_size=BATCH_SIZE, verbose=0)
    recon_error = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
    threshold = float(np.quantile(recon_error, 1 - LSTM_CONTAMINATION))

    return autoencoder, scaler_lstm, threshold


# ═════════════════════════════════════════════════════════════
# 4. ORCHESTRATE + SAFE SWAP
# ═════════════════════════════════════════════════════════════
def _copy_untouched_files(live_dir: str, staging_dir: str, include_lstm: bool):
    """Copy every file from the CURRENT models folder that this training run did
    not rebuild (XGBoost, VAE, model_config.pkl, ...) into the NEW folder, so the
    swap does not lose them. Files the training rebuilt are never overwritten,
    and old LSTM files are skipped when a new LSTM was trained (otherwise the
    loader could pick the old one first)."""
    if not os.path.isdir(live_dir):
        return
    new_files = set(os.listdir(staging_dir))
    for name in os.listdir(live_dir):
        if name in new_files:
            continue
        if include_lstm and (name.startswith("lstm_autoencoder") or name == "scaler_ae.pkl"):
            continue
        src = os.path.join(live_dir, name)
        dst = os.path.join(staging_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"[train_runner] kept existing file: {name}")


def run_training_job(trigger: str = "manual", row_count_at_trigger: int | None = None,
                      main_features: list | None = None, lstm_features: list | None = None,
                      include_lstm: bool = True):
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[train_runner] training started (trigger={trigger})")

    # Use the SAME folder the live app loads its models from, so a retrain can
    # never update a different folder than the one being used.
    live_dir = os.path.abspath(model_state.find_models_dir() or SAVED_MODELS_DIR)
    staging_dir = live_dir + "_staging"
    backup_dir = live_dir + "_backup"

    # Defaults — override with your real MAIN_FEATURES / AE_FEATURES from app.core.config
    main_features = main_features or (BASE_FEATURES + EXTRA_RAW_FEATURES + DERIVED_FEATURES)
    lstm_features = lstm_features or BASE_FEATURES

    try:
        df = _load_training_data()
        if len(df) < 500:
            raise ValueError(f"Not enough fresh data to retrain safely: {len(df)} rows")

        df_clean = _engineer_features(df)
        fit_result = _fit_models(df_clean, main_features)

        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir)

        joblib.dump(fit_result["scaler"], os.path.join(staging_dir, "scaler.pkl"))
        # the live loader looks for scaler_if.pkl first, so save the new scaler under that name too
        joblib.dump(fit_result["scaler"], os.path.join(staging_dir, "scaler_if.pkl"))
        joblib.dump(fit_result["isolation_forest"], os.path.join(staging_dir, "isolation_forest.pkl"))
        joblib.dump(fit_result["elliptic_envelope"], os.path.join(staging_dir, "elliptic_envelope.pkl"))
        joblib.dump(fit_result["power_transformer"], os.path.join(staging_dir, "power_transformer.pkl"))
        joblib.dump(fit_result["scaler_normal_ee"], os.path.join(staging_dir, "scaler_normal_ee.pkl"))

        if include_lstm:
            lstm_model, scaler_lstm, lstm_threshold = _fit_lstm_autoencoder(df_clean, lstm_features)
            lstm_model.save(os.path.join(staging_dir, "lstm_autoencoder.keras"))
            # plain-numpy copy of the weights, so the app can run the LSTM without TensorFlow
            import numpy as _np
            _np.savez(os.path.join(staging_dir, "lstm_autoencoder_weights.npz"),
                      **{f"w{i}": w for i, w in enumerate(lstm_model.get_weights())})
            joblib.dump(scaler_lstm, os.path.join(staging_dir, "scaler_lstm.pkl"))
            with open(os.path.join(staging_dir, "lstm_threshold.txt"), "w") as f:
                f.write(str(lstm_threshold))

        import json
        # files the live app reads so scoring matches training exactly
        with open(os.path.join(staging_dir, "isolation_forest_threshold.txt"), "w") as f:
            f.write(str(fit_result["if_threshold"]))
        with open(os.path.join(staging_dir, "elliptic_envelope_threshold.txt"), "w") as f:
            f.write(str(fit_result["ee_threshold"]))
        with open(os.path.join(staging_dir, "ee_input_spec.json"), "w") as f:
            json.dump({
                "feature_order": list(main_features),
                "skewed_features": fit_result["skewed_features"],
                "normal_features": fit_result["normal_features"],
            }, f, indent=2)
        with open(os.path.join(staging_dir, "model_config.json"), "w") as f:
            json.dump({
                "features_list": main_features,
                "if_threshold": fit_result["if_threshold"],
                "ee_threshold": fit_result["ee_threshold"],
                "trained_at": started_at,
                "training_rows": len(df_clean),
            }, f, indent=2)

        # contract check before swap
        if fit_result["scaler"].n_features_in_ != len(main_features):
            raise ValueError(
                f"Contract check failed: scaler has {fit_result['scaler'].n_features_in_} features, "
                f"expected {len(main_features)}"
            )

        # keep XGBoost, VAE, model_config.pkl etc. from the current folder
        _copy_untouched_files(live_dir, staging_dir, include_lstm)

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        had_old = os.path.exists(live_dir)
        if had_old:
            shutil.move(live_dir, backup_dir)
        shutil.move(staging_dir, live_dir)

        # force=True makes the app really reload the new files
        if not model_state.load_ml_models(force=True):
            # new files could not be loaded: put the previous models back
            failed_dir = live_dir + "_failed"
            if os.path.exists(failed_dir):
                shutil.rmtree(failed_dir)
            shutil.move(live_dir, failed_dir)
            if had_old:
                shutil.move(backup_dir, live_dir)
            model_state.load_ml_models(force=True)
            raise ValueError("New models could not be loaded; the previous models were restored")

        if row_count_at_trigger is not None:
            mark_trained(row_count_at_trigger)

        record_training_run(
            started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
            status="success", trigger=trigger, row_count=row_count_at_trigger,
            rows_used=len(df_clean), error=None,
        )
        print("[train_runner] training SUCCESS, new models are live")

    except Exception as e:
        traceback.print_exc()
        record_training_run(
            started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
            status="failed", trigger=trigger, row_count=row_count_at_trigger, error=str(e),
        )
        print(f"[train_runner] training FAILED: {e} — live models left untouched")