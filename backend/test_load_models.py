# test_load_models.py
import sys
sys.path.insert(0, '.')

from app.ml.model_state import load_ml_models, ML_MODELS

print("=" * 60)
print("🧪 TESTING MODEL LOADING")
print("=" * 60)

success = load_ml_models()

print(f"\n✅ Models loaded: {success}")
print(f"📊 ML_MODELS status:")
print(f"  loaded: {ML_MODELS['loaded']}")
print(f"  isolation_forest: {ML_MODELS['isolation_forest'] is not None}")
print(f"  elliptic_envelope: {ML_MODELS['elliptic_envelope'] is not None}")
print(f"  xgboost: {ML_MODELS['xgboost'] is not None}")
print(f"  lstm_autoencoder: {ML_MODELS['lstm_autoencoder'] is not None}")
print(f"  vae_autoencoder: {ML_MODELS['vae_autoencoder'] is not None}")
print(f"  scaler: {ML_MODELS['scaler'] is not None}")
print(f"  scaler_ae: {ML_MODELS['scaler_ae'] is not None}")