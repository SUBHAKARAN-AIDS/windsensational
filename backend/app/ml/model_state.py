# app/ml/model_state.py
#
# LIVE ML state and model loading.
#
# Restructuring note (MOVE FIRST, REFACTOR LATER):
#   The VAEAnomalyDetector class, the shared ML_MODELS dict, and load_ml_models()
#   are moved here VERBATIM from app/main.py. Bodies are unchanged EXCEPT for the
#   saved_models path resolution, which is corrected for this file's new location
#   (an explicitly-permitted "fix paths required by the move"). No ML algorithm,
#   threshold, feature, or scientific logic is changed.
#
#   ML_MODELS is the single shared dict; load_ml_models() mutates it in place, so
#   every importer sees the same loaded models (single load, no duplication).

import os
import logging

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from tensorflow.keras.models import load_model

# ── Compatibility shim: these .pkl models were pickled under numpy 2.x,
# which renamed numpy.core to numpy._core. This venv runs numpy 1.23.5
# (pinned for TensorFlow compatibility), so alias the old name so old-numpy's
# unpickler can still find what the pickle references. Must run before any
# joblib.load() call below.
import sys
import numpy.core as _np_core
sys.modules['numpy._core'] = _np_core
sys.modules['numpy._core.multiarray'] = _np_core.multiarray
sys.modules['numpy._core.numeric'] = _np_core.numeric

logger = logging.getLogger(__name__)

# ============================================================
# MISSING VAE CLASS DEFINITION (FIXED)
# ============================================================
class VAEAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int = 18, latent_dim: int = 8):
        super(VAEAnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)


# ============================================================
# GLOBAL ML MODELS (5 artifacts: 3 voters + LSTM score-only + VAE excluded)
# ============================================================

ML_MODELS = {
    # Unsupervised Models
    "isolation_forest": None,
    "elliptic_envelope": None,
    
    # Supervised Model
    "xgboost": None,
    
    # Deep Learning Models
    "lstm_autoencoder": None,
    "vae_autoencoder": None,
    
    # Scalers
    "scaler": None,
    "scaler_ae": None,

    # Extra scoring files (so each model scores the way it was trained)
    "if_threshold": None,
    "ee_threshold": None,
    "power_transformer": None,
    "scaler_normal_ee": None,
    "ee_input_spec": None,
    "xgb_classes": None,

    # Config
    "loaded": False,
    "feature_columns": None,
    "ae_feature_columns": None,
    "vae_threshold": None,
    "vae_input_dim": None,
    "validation": {},
    
    # Model names for display
    "model_names": {
        "isolation_forest": "Isolation Forest",
        "elliptic_envelope": "Elliptic Envelope",
        "xgboost": "XGBoost Classifier",
        "lstm_autoencoder": "LSTM Autoencoder",
        "vae_autoencoder": "VAE Autoencoder"
    }
}

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

def _load_ml_models_impl():
    """Load model artifacts from saved_models/. Ensemble voters: Isolation
    Forest, Elliptic Envelope, XGBoost. LSTM AE is score-only. VAE is loaded for
    validation but excluded from ensemble scoring (3-feature mapping unknown)."""
    
    try:
        # PATH FIX (restructuring move): this file now lives at app/ml/model_state.py,
        # but saved_models/ remains under the app package root (app/saved_models/).
        # Resolve one level up from this file's directory (app/ml -> app).
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../app/ml
        app_dir = os.path.dirname(current_dir)                    # .../app
        backend_dir = os.path.dirname(app_dir)                    # .../backend
        
        candidate_paths = [
            os.path.join(backend_dir, "saved_models"),
            os.path.join(app_dir, "saved_models"),
            os.path.join(current_dir, "saved_models"),
            os.path.abspath("saved_models")
        ]
        
        model_path = None
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isdir(p):
                model_path = p
                logger.info(f"Found saved models directory at: {model_path}")
                break
                
        if not model_path:
            logger.error("Models directory not found in any candidate locations")
            return False
        
        files = os.listdir(model_path)
        logger.info(f"Files in saved_models: {files}")
        loaded_count = 0
        
        # ============================================================
        # 1. ISOLATION FOREST
        # ============================================================
        if "isolation_forest.pkl" in files:
            ML_MODELS["isolation_forest"] = joblib.load(os.path.join(model_path, "isolation_forest.pkl"))
            logger.info("Isolation Forest loaded")
            loaded_count += 1
        else:
            logger.warning("Isolation Forest not found")
        
        # ============================================================
        # 2. ELLIPTIC ENVELOPE
        # ============================================================
        if "elliptic_envelope.pkl" in files:
            ML_MODELS["elliptic_envelope"] = joblib.load(os.path.join(model_path, "elliptic_envelope.pkl"))
            logger.info("Elliptic Envelope loaded")
            loaded_count += 1
        else:
            logger.warning("Elliptic Envelope not found")
        
        # ============================================================
        # 3. XGBOOST CLASSIFIER
        # ============================================================
        if "xgb_classifier.pkl" in files:
            ML_MODELS["xgboost"] = joblib.load(os.path.join(model_path, "xgb_classifier.pkl"))
            logger.info("XGBoost Classifier loaded")
            loaded_count += 1
        else:
            logger.warning("XGBoost Classifier not found")
        
        # ============================================================
        # 4. LSTM AUTOENCODER
        # ============================================================
        # Try formats in order of preference:
        # 1. Keras 3 (.keras v3)
        # 2. Original .keras
        # 3. .h5 (legacy)
        # 4. Folder (SavedModel)

        lstm_formats = [
            "lstm_autoencoder_v3.keras",  
            "lstm_autoencoder.keras",       
            "lstm_autoencoder.h5",          
            "lstm_autoencoder",             
        ]

        lstm_loaded = False

        # Preferred: the plain-numpy copy of the LSTM (needs no TensorFlow, so it loads
        # on any backend). Made by the notebook export cell / saved after every retrain.
        npz_path = os.path.join(model_path, "lstm_autoencoder_weights.npz")
        if os.path.exists(npz_path):
            try:
                from app.ml.lstm_numpy import NumpyLSTM
                from app.core.config import LSTM_SEQ_LEN as _SEQ_LEN
                ML_MODELS["lstm_autoencoder"] = NumpyLSTM.from_npz(npz_path, seq_len=_SEQ_LEN)
                logger.info("LSTM Autoencoder loaded from lstm_autoencoder_weights.npz (numpy, no TensorFlow needed)")
                loaded_count += 1
                lstm_loaded = True
            except Exception as e:
                logger.warning(f"Could not load LSTM numpy weights: {e}")

        for fmt in ([] if lstm_loaded else lstm_formats):
            fmt_path = os.path.join(model_path, fmt)
            if os.path.exists(fmt_path):
                try:
                    ML_MODELS["lstm_autoencoder"] = load_model(fmt_path)
                    logger.info(f"LSTM Autoencoder loaded from {fmt}")
                    loaded_count += 1
                    lstm_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load LSTM from {fmt}: {e}")
                    continue

        if not lstm_loaded:
            # Try any file starting with lstm_autoencoder
            lstm_files = [f for f in files if f.startswith("lstm_autoencoder")]
            if lstm_files:
                lstm_path = os.path.join(model_path, lstm_files[0])
                try:
                    ML_MODELS["lstm_autoencoder"] = load_model(lstm_path)
                    logger.info(f"LSTM Autoencoder loaded from {lstm_files[0]} (fallback)")
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Could not load LSTM Autoencoder: {e}")
            else:
                logger.warning("LSTM Autoencoder not found")

        
        # ============================================================
        # 5. VAE AUTOENCODER (PyTorch) - Load from .pth file
        # ============================================================
        # Load the .pth file directly (not a folder)
        vae_pth_path = os.path.join(model_path, "vae_autoencoder.pth")
        if os.path.exists(vae_pth_path):
            try:
                state = torch.load(vae_pth_path, map_location='cpu')
                
                # Infer dimensions from checkpoint
                enc_w = state.get('encoder.0.weight')
                mu_w = state.get('fc_mu.weight')
                inferred_input_dim = int(enc_w.shape[1]) if enc_w is not None else None
                inferred_latent_dim = int(mu_w.shape[0]) if mu_w is not None else 4

                if inferred_input_dim is None:
                    raise ValueError("Could not infer VAE input_dim from checkpoint")

                vae_model = VAEAnomalyDetector(
                    input_dim=inferred_input_dim, latent_dim=inferred_latent_dim
                )
                vae_model.load_state_dict(state)
                vae_model.eval()

                ML_MODELS["vae_autoencoder"] = vae_model
                ML_MODELS["vae_input_dim"] = inferred_input_dim

                logger.info(
                    f"VAE Autoencoder loaded from vae_autoencoder.pth "
                    f"(input_dim={inferred_input_dim}, latent_dim={inferred_latent_dim})"
                )
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not load VAE Autoencoder: {e}")
        else:
            logger.warning("VAE Autoencoder .pth file not found")




        # ============================================================
        # 6. SCALERS
        # ============================================================
        scaler_file = "scaler_if.pkl" if "scaler_if.pkl" in files else ("scaler.pkl" if "scaler.pkl" in files else None)
        if scaler_file:
            ML_MODELS["scaler"] = joblib.load(os.path.join(model_path, scaler_file))
            logger.info(f"Main Scaler loaded from {scaler_file}")
        else:
            logger.warning("Main Scaler not found")

        ae_scaler_file = "scaler_lstm.pkl" if "scaler_lstm.pkl" in files else ("scaler_ae.pkl" if "scaler_ae.pkl" in files else None)
        if ae_scaler_file:
            ML_MODELS["scaler_ae"] = joblib.load(os.path.join(model_path, ae_scaler_file))
            logger.info(f"Autoencoder Scaler loaded from {ae_scaler_file}")
        else:
            logger.warning("Autoencoder Scaler not found")


        

        # ============================================================
        # 6b. EXTRA SCORING FILES (optional - the app falls back if missing)
        # ============================================================
        import json as _json

        def _read_float(fname):
            if fname in files:
                try:
                    with open(os.path.join(model_path, fname)) as fh:
                        return float(fh.read().strip())
                except Exception as e:
                    logger.warning(f"Could not read {fname}: {e}")
            return None

        ML_MODELS["if_threshold"] = _read_float("isolation_forest_threshold.txt")
        ML_MODELS["ee_threshold"] = _read_float("elliptic_envelope_threshold.txt")
        if "model_config.json" in files and (ML_MODELS["if_threshold"] is None or ML_MODELS["ee_threshold"] is None):
            try:
                with open(os.path.join(model_path, "model_config.json")) as fh:
                    _cfg = _json.load(fh)
                if ML_MODELS["if_threshold"] is None and _cfg.get("if_threshold") is not None:
                    ML_MODELS["if_threshold"] = float(_cfg["if_threshold"])
                if ML_MODELS["ee_threshold"] is None and _cfg.get("ee_threshold") is not None:
                    ML_MODELS["ee_threshold"] = float(_cfg["ee_threshold"])
            except Exception as e:
                logger.warning(f"Could not read thresholds from model_config.json: {e}")

        for _key, _fname in (("power_transformer", "power_transformer.pkl"),
                             ("scaler_normal_ee", "scaler_normal_ee.pkl")):
            if _fname in files:
                try:
                    ML_MODELS[_key] = joblib.load(os.path.join(model_path, _fname))
                except Exception as e:
                    logger.warning(f"Could not load {_fname}: {e}")
        for _key, _fname in (("ee_input_spec", "ee_input_spec.json"), ("xgb_classes", "xgb_classes.json")):
            if _fname in files:
                try:
                    with open(os.path.join(model_path, _fname)) as fh:
                        ML_MODELS[_key] = _json.load(fh)
                except Exception as e:
                    logger.warning(f"Could not read {_fname}: {e}")
        logger.info(
            f"Scoring files: if_threshold={ML_MODELS['if_threshold']}, ee_threshold={ML_MODELS['ee_threshold']}, "
            f"ee_transform={'yes' if ML_MODELS['power_transformer'] is not None and ML_MODELS['ee_input_spec'] else 'no'}, "
            f"xgb_classes={'yes' if ML_MODELS['xgb_classes'] else 'no'}"
        )

        # ============================================================
        # 7. FEATURE CONFIGURATION
        # ============================================================
        if "model_config.pkl" in files:
            config = joblib.load(os.path.join(model_path, "model_config.pkl"))
            # CORRECTNESS FIX: the artifact stores the 26-feature list under the
            # key 'features_list' (plus 'vae_threshold'); the old code read the
            # non-existent key 'feature_columns', which is why it logged
            # "0 features". Read the real key (with a legacy fallback).
            ML_MODELS["feature_columns"] = config.get('features_list', config.get('feature_columns', []))
            # No AE feature list is stored in the config; the 18 AE features are
            # defined by scaler_ae.feature_names_in_ / the ae_feature_cols list.
            ML_MODELS["ae_feature_columns"] = config.get('ae_feature_columns', [])
            ML_MODELS["vae_threshold"] = config.get('vae_threshold')
            logger.info(
                f"Feature config loaded: {len(ML_MODELS['feature_columns'])} features"
                f"; vae_threshold={ML_MODELS['vae_threshold']}"
            )
            # CONTRACT VALIDATION: the canonical inference order lives in
            # app.core.config.MAIN_FEATURES. If the artifact's features_list ever
            # diverges from it, inference would silently scale the wrong columns,
            # so we flag it loudly rather than let the two drift apart.
            try:
                from app.core.config import MAIN_FEATURES
                artifact_feats = list(ML_MODELS["feature_columns"] or [])
                if artifact_feats and artifact_feats != list(MAIN_FEATURES):
                    logger.error(
                        "FEATURE CONTRACT MISMATCH: model_config.features_list "
                        "does not equal config.MAIN_FEATURES. Inference order is "
                        "unreliable until reconciled. artifact=%s", artifact_feats,
                    )
                else:
                    logger.info("Feature contract validated against MAIN_FEATURES (match).")
            except Exception as e:
                logger.warning(f"Could not validate feature contract: {e}")

                
        
        # ============================================================
        # 8. DIMENSION / CONTRACT VALIDATION (rules 7, 8, 9)
        # ============================================================
        # Every model's ACTUAL input contract is verified against the canonical
        # config here. Results are stored so the API can (a) refuse to let a
        # dimensionally-invalid model into inference and (b) report participation
        # truthfully instead of silently mis-scoring.
        from app.core.config import (
            MAIN_FEATURES, AE_FEATURES, MAIN_FEATURE_COUNT,
            LSTM_SEQ_LEN, LSTM_INPUT_WIDTH, VAE_INPUT_DIM,
        )
        validation = {}

        def _check(name, ok, detail):
            validation[name] = {"valid": bool(ok), "detail": detail}
            (logger.info if ok else logger.error)(
                f"Contract {'OK' if ok else 'FAIL'} [{name}]: {detail}"
            )

        sc = ML_MODELS.get("scaler")
        if sc is not None:
            nf = getattr(sc, "n_features_in_", None)
            names_in = list(getattr(sc, "feature_names_in_", []))
            names_ok = len(names_in) in (18, MAIN_FEATURE_COUNT) or not names_in
            _check("scaler", nf in (18, MAIN_FEATURE_COUNT) and names_ok,
                   f"n_features_in_={nf} (expected 18 or {MAIN_FEATURE_COUNT})")
        scae = ML_MODELS.get("scaler_ae")
        if scae is not None:
            nf = getattr(scae, "n_features_in_", None)
            names_in = list(getattr(scae, "feature_names_in_", []))
            names_ok = len(names_in) in (18, LSTM_INPUT_WIDTH) or not names_in
            _check("scaler_ae", nf in (18, LSTM_INPUT_WIDTH) and names_ok,
                   f"n_features_in_={nf} (expected {LSTM_INPUT_WIDTH})")
        for mname in ("isolation_forest", "elliptic_envelope", "xgboost"):
            m = ML_MODELS.get(mname)
            if m is not None:
                nf = getattr(m, "n_features_in_", None)
                _check(mname, nf in (18, MAIN_FEATURE_COUNT), f"n_features_in_={nf} (expected 18 or {MAIN_FEATURE_COUNT})")
        lstm = ML_MODELS.get("lstm_autoencoder")
        if lstm is not None:
            shape = None
            try:
                shape = tuple(getattr(lstm, "input_shape", None) or lstm.inputs[0].shape)
            except Exception:
                shape = None
            ok = bool(shape) and tuple(shape)[-2:] == (LSTM_SEQ_LEN, LSTM_INPUT_WIDTH)
            _check("lstm_autoencoder", ok, f"input_shape={shape} (want (_, {LSTM_SEQ_LEN}, {LSTM_INPUT_WIDTH}))")
        vdim = ML_MODELS.get("vae_input_dim")
        if ML_MODELS.get("vae_autoencoder") is not None:
            _check("vae_autoencoder", vdim in (18, VAE_INPUT_DIM),
                   f"input_dim={vdim} (checkpoint input_dim={vdim})")

        ML_MODELS["validation"] = validation

        # Mark as loaded if at least 3 models are available
        if loaded_count >= 3:
            ML_MODELS["loaded"] = True
            logger.info(f"SUCCESS: {loaded_count} model artifacts loaded (VAE excluded from ensemble)")
            return True
        else:
            logger.warning(f"Only {loaded_count} model artifacts loaded")
            return False
            
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MODELS FOLDER LOOKUP + RELOAD WRAPPER
# ============================================================

def find_models_dir():
    """Return the folder the live app loads models from (same search order as
    the loader above), or None if no models folder exists."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../app/ml
    app_dir = os.path.dirname(current_dir)                    # .../app
    backend_dir = os.path.dirname(app_dir)                    # .../backend
    for p in (
        os.path.join(backend_dir, "saved_models"),
        os.path.join(app_dir, "saved_models"),
        os.path.join(current_dir, "saved_models"),
        os.path.abspath("saved_models"),
    ):
        if os.path.isdir(p):
            return p
    return None


_MODEL_SLOTS = (
    "isolation_forest", "elliptic_envelope", "xgboost", "lstm_autoencoder",
    "vae_autoencoder", "scaler", "scaler_ae", "feature_columns",
    "ae_feature_columns", "vae_threshold", "vae_input_dim",
    "if_threshold", "ee_threshold", "power_transformer", "scaler_normal_ee",
    "ee_input_spec", "xgb_classes",
)


def load_ml_models(force: bool = False):
    """Load models from disk.

    force=False (default): do nothing if models are already loaded.
    force=True: really reload from disk (used after retraining). If the reload
    fails, the models that were loaded before are put back, so the app keeps
    working."""
    if ML_MODELS["loaded"] and not force:
        return True

    snapshot = dict(ML_MODELS) if force else None
    if force:
        for key in _MODEL_SLOTS:
            ML_MODELS[key] = None
        ML_MODELS["loaded"] = False
        ML_MODELS["validation"] = {}

    ok = _load_ml_models_impl()

    if force and not ok and snapshot is not None:
        ML_MODELS.update(snapshot)
        logger.error("Forced reload failed - kept the previously loaded models")
    return ok