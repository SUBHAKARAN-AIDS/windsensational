# backend/app/services/anomaly_service.py

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    def __init__(self, model_dir: str = "app/saved_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and scalers"""
        try:
            # Load XGBoost Classifier
            with open(f"{self.model_dir}/xgb_classifier.pkl", 'rb') as f:
                self.models['xgb'] = pickle.load(f)
            logger.info("✅ XGBoost model loaded")
            
            # Load Isolation Forest
            with open(f"{self.model_dir}/isolation_forest.pkl", 'rb') as f:
                self.models['isolation_forest'] = pickle.load(f)
            logger.info("✅ Isolation Forest model loaded")
            
            # Load Elliptic Envelope
            with open(f"{self.model_dir}/elliptic_envelope.pkl", 'rb') as f:
                self.models['elliptic_envelope'] = pickle.load(f)
            logger.info("✅ Elliptic Envelope model loaded")
            
            # Load LSTM Autoencoder
            self.models['lstm_ae'] = keras.models.load_model(
                f"{self.model_dir}/lstm_autoencoder.keras"
            )
            logger.info("✅ LSTM Autoencoder loaded")
            
            # Load scalers
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scalers['standard'] = pickle.load(f)
            logger.info("✅ Standard scaler loaded")
            
            with open(f"{self.model_dir}/scaler_ae.pkl", 'rb') as f:
                self.scalers['autoencoder'] = pickle.load(f)
            logger.info("✅ Autoencoder scaler loaded")
            
            # Load feature configuration
            with open(f"{self.model_dir}/model_config.pkl", 'rb') as f:
                config = pickle.load(f)
                self.feature_columns = config.get('feature_columns', [])
            logger.info(f"✅ Feature columns loaded: {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data using the trained scaler"""
        if self.feature_columns:
            # Use only features the model was trained on
            available_features = [f for f in self.feature_columns if f in df.columns]
            X = df[available_features].fillna(0).values
        else:
            X = df.fillna(0).values
        
        # Scale the data
        X_scaled = self.scalers['standard'].transform(X)
        return X_scaled, X  # Return both scaled and unscaled
    
    def detect_anomalies_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Ensemble anomaly detection using multiple models
        Returns anomaly scores and consensus
        """
        try:
            X_scaled, X_raw = self.preprocess_data(df)
            
            # Get predictions from each model
            results = {}
            
            # 1. XGBoost Classifier
            if 'xgb' in self.models:
                xgb_pred = self.models['xgb'].predict(X_scaled)
                xgb_proba = self.models['xgb'].predict_proba(X_scaled)
                results['xgb'] = {
                    'predictions': xgb_pred,
                    'scores': xgb_proba[:, 1] if xgb_proba.shape[1] > 1 else xgb_proba[:, 0],
                    'anomaly': xgb_pred == 1
                }
            
            # 2. Isolation Forest
            if 'isolation_forest' in self.models:
                iforest_scores = self.models['isolation_forest'].score_samples(X_scaled)
                # Convert to anomaly probability (lower score = more anomalous)
                iforest_proba = 1 - (iforest_scores - iforest_scores.min()) / (iforest_scores.max() - iforest_scores.min())
                results['isolation_forest'] = {
                    'scores': iforest_scores,
                    'probabilities': iforest_proba,
                    'anomaly': iforest_proba > 0.7  # Threshold
                }
            
            # 3. Elliptic Envelope
            if 'elliptic_envelope' in self.models:
                ee_pred = self.models['elliptic_envelope'].predict(X_scaled)
                ee_scores = self.models['elliptic_envelope'].score_samples(X_scaled)
                ee_proba = 1 - (ee_scores - ee_scores.min()) / (ee_scores.max() - ee_scores.min())
                results['elliptic_envelope'] = {
                    'scores': ee_scores,
                    'probabilities': ee_proba,
                    'anomaly': ee_pred == -1
                }
            
            # 4. LSTM Autoencoder
            if 'lstm_ae' in self.models:
                # Reshape for LSTM (samples, timesteps, features)
                timesteps = 10
                if len(X_scaled) >= timesteps:
                    X_lstm = self._create_sequences(X_scaled, timesteps)
                    reconstructed = self.models['lstm_ae'].predict(X_lstm, verbose=0)
                    
                    # Calculate reconstruction error
                    mse = np.mean(np.square(X_lstm - reconstructed), axis=(1, 2))
                    threshold = np.percentile(mse, 95)  # Top 5% as anomalies
                    results['lstm_autoencoder'] = {
                        'scores': mse,
                        'probabilities': (mse - mse.min()) / (mse.max() - mse.min() + 1e-8),
                        'anomaly': mse > threshold
                    }
            
            # Calculate ensemble anomaly score (consensus)
            anomaly_votes = []
            model_scores = []
            
            for model_name, result in results.items():
                if 'anomaly' in result:
                    anomaly_votes.append(result['anomaly'].astype(int))
                if 'probabilities' in result:
                    model_scores.append(result['probabilities'])
            
            # Ensemble decision: at least 2 models agree
            if anomaly_votes:
                consensus = np.sum(np.array(anomaly_votes), axis=0) >= 2
                avg_score = np.mean(model_scores, axis=0) if model_scores else np.zeros(len(df))
            else:
                consensus = np.zeros(len(df), dtype=bool)
                avg_score = np.zeros(len(df))
            
            return {
                'ensemble_predictions': consensus.astype(int),
                'ensemble_scores': avg_score,
                'model_results': results,
                'num_models': len(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble anomaly detection failed: {e}")
            return {
                'ensemble_predictions': np.zeros(len(df)),
                'ensemble_scores': np.zeros(len(df)),
                'error': str(e)
            }
    
    def _create_sequences(self, data: np.ndarray, timesteps: int) -> np.ndarray:
        """Create sequences for LSTM input"""
        X = []
        for i in range(len(data) - timesteps + 1):
            X.append(data[i:i + timesteps])
        return np.array(X)
    
    def get_turbine_status(self, turbine_id: str, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive status for a single turbine
        """
        try:
            # Run ensemble detection
            result = self.detect_anomalies_ensemble(df)
            
            # Calculate overall health
            anomaly_count = np.sum(result['ensemble_predictions'])
            total_samples = len(df)
            anomaly_ratio = anomaly_count / total_samples if total_samples > 0 else 0
            
            # Determine status
            if anomaly_ratio > 0.5:
                status = "Critical"
                health_score = max(0, 100 - (anomaly_ratio * 100))
            elif anomaly_ratio > 0.2:
                status = "Warning"
                health_score = max(0, 100 - (anomaly_ratio * 80))
            else:
                status = "Healthy"
                health_score = 100 - (anomaly_ratio * 30)
            
            # Get top anomalies with details
            anomalies = []
            if anomaly_count > 0:
                # Find indices of anomalies
                anomaly_indices = np.where(result['ensemble_predictions'] == 1)[0]
                
                for idx in anomaly_indices[:50]:  # Limit to 50
                    anomaly = {
                        'turbine_id': turbine_id,
                        'timestamp': df.iloc[idx].get('time', datetime.now()).isoformat(),
                        'score': float(result['ensemble_scores'][idx]),
                        'severity': 'high' if result['ensemble_scores'][idx] > 0.8 else 'medium',
                        'model': 'Ensemble (XGBoost + IsolationForest + Autoencoder)',
                        'sensor_values': df.iloc[idx].to_dict()
                    }
                    anomalies.append(anomaly)
            
            return {
                'turbine_id': turbine_id,
                'status': status,
                'health_score': int(health_score),
                'anomaly_ratio': float(anomaly_ratio),
                'anomaly_count': int(anomaly_count),
                'total_samples': total_samples,
                'has_anomaly': anomaly_count > 0,
                'anomalies': anomalies,
                'last_checked': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting turbine status: {e}")
            return {
                'turbine_id': turbine_id,
                'status': 'Unknown',
                'health_score': 0,
                'has_anomaly': False,
                'error': str(e)
            }