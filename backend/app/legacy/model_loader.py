# backend/app/services/model_loader.py

import pickle
import joblib
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import torch
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage all 7 trained models"""
    
    def __init__(self, model_dir="app/saved_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.config = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all 7 models from saved_models directory"""
        
        logger.info("="*60)
        logger.info("🚀 LOADING ALL 7 MODELS")
        logger.info("="*60)
        
        # 1. ISOLATION FOREST
        try:
            with open(f"{self.model_dir}/isolation_forest.pkl", 'rb') as f:
                self.models['isolation_forest'] = pickle.load(f)
            logger.info("✅ Isolation Forest loaded")
        except Exception as e:
            logger.error(f"❌ Isolation Forest failed: {e}")
            self.models['isolation_forest'] = None
        
        # 2. ELLIPTIC ENVELOPE
        try:
            with open(f"{self.model_dir}/elliptic_envelope.pkl", 'rb') as f:
                self.models['elliptic_envelope'] = pickle.load(f)
            logger.info("✅ Elliptic Envelope loaded")
        except Exception as e:
            logger.error(f"❌ Elliptic Envelope failed: {e}")
            self.models['elliptic_envelope'] = None
        
        # 3. XGBOOST CLASSIFIER
        try:
            with open(f"{self.model_dir}/xgb_classifier.pkl", 'rb') as f:
                self.models['xgboost'] = pickle.load(f)
            logger.info("✅ XGBoost Classifier loaded")
        except Exception as e:
            logger.error(f"❌ XGBoost failed: {e}")
            self.models['xgboost'] = None
        
        # 4. LSTM AUTOENCODER (Keras)
        try:
            # Check for .keras or .h5
            lstm_files = ['lstm_autoencoder.keras', 'lstm_autoencoder.h5']
            for fname in lstm_files:
                if os.path.exists(f"{self.model_dir}/{fname}"):
                    self.models['lstm_autoencoder'] = load_model(f"{self.model_dir}/{fname}")
                    logger.info(f"✅ LSTM Autoencoder loaded from {fname}")
                    break
            if 'lstm_autoencoder' not in self.models:
                self.models['lstm_autoencoder'] = None
                logger.warning("⚠️ LSTM Autoencoder not found")
        except Exception as e:
            logger.error(f"❌ LSTM Autoencoder failed: {e}")
            self.models['lstm_autoencoder'] = None
        
        # 5. VAE AUTOENCODER (PyTorch)
        try:
            if os.path.exists(f"{self.model_dir}/vae_autoencoder.pth"):
                self.models['vae_autoencoder'] = torch.load(
                    f"{self.model_dir}/vae_autoencoder.pth",
                    map_location='cpu'
                )
                logger.info("✅ VAE Autoencoder loaded")
            else:
                self.models['vae_autoencoder'] = None
                logger.warning("⚠️ VAE Autoencoder not found")
        except Exception as e:
            logger.error(f"❌ VAE Autoencoder failed: {e}")
            self.models['vae_autoencoder'] = None
        
        # 6. LOAD SCALERS
        try:
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scalers['main'] = pickle.load(f)
            logger.info("✅ Main Scaler (26 features) loaded")
        except Exception as e:
            logger.error(f"❌ Main Scaler failed: {e}")
            self.scalers['main'] = None
        
        try:
            with open(f"{self.model_dir}/scaler_ae.pkl", 'rb') as f:
                self.scalers['autoencoder'] = pickle.load(f)
            logger.info("✅ Autoencoder Scaler (18 features) loaded")
        except Exception as e:
            logger.error(f"❌ Autoencoder Scaler failed: {e}")
            self.scalers['autoencoder'] = None
        
        # 7. LOAD CONFIG
        try:
            with open(f"{self.model_dir}/model_config.pkl", 'rb') as f:
                self.config = pickle.load(f)
            logger.info(f"✅ Model Config loaded: {len(self.config.get('feature_columns', []))} features")
        except Exception as e:
            logger.error(f"❌ Model Config failed: {e}")
            self.config = {}
        
        # Count loaded models
        loaded_count = sum(1 for m in self.models.values() if m is not None)
        logger.info(f"✅ {loaded_count}/7 models loaded successfully")
        logger.info("="*60)
        
        return loaded_count
    
    def get_model(self, name):
        """Get a specific model by name"""
        return self.models.get(name)
    
    def get_scaler(self, name):
        """Get a specific scaler by name"""
        return self.scalers.get(name)
    
    def is_model_loaded(self, name):
        """Check if a specific model is loaded"""
        return self.models.get(name) is not None