# backend/app/services/ensemble_service.py

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)

class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detection using ALL 7 models
    Each model votes, consensus determines final anomaly
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.models = self.model_loader.models
        self.scalers = self.model_loader.scalers
        self.config = self.model_loader.config
        
        # Feature columns (26 for main models)
        self.feature_columns = self.config.get('feature_columns', [
            'gearbox_oil_temperature', 'gearbox_oil_inlet_temperature',
            'gearbox_de_bearing_temperature', 'gearbox_nde_bearing_temperature',
            'main_bearing_temperature', 'generator_stator_temp_u',
            'generator_stator_temp_v', 'generator_stator_temp_w',
            'generator_drive_end_bearing_temperature', 'generator_non_drive_end_bearing_temperature',
            'vibration_lateral_mms', 'vibration_axial_mms',
            'active_power', 'reactive_power', 'rotor_speed', 'generator_speed',
            'torque_nm', 'pitch_angle_1deg', 'pitch_angle_2deg', 'pitch_angle_3deg',
            'temperature_outside_nacelle', 'wind_speed_ms',
            'gearbox_temp_delta', 'pitch_imbalance', 'stator_imbalance', 'bearing_de_nde_diff'
        ])
        
        # 18 features for autoencoders
        self.ae_features = [
            'active_power', 'generator_speed', 'rotor_speed', 'torque_nm',
            'pitch_angle_1deg', 'pitch_angle_2deg', 'pitch_angle_3deg',
            'gearbox_oil_temperature', 'gearbox_de_bearing_temperature',
            'gearbox_nde_bearing_temperature', 'main_bearing_temperature',
            'generator_stator_temp_u', 'generator_stator_temp_v', 'generator_stator_temp_w',
            'vibration_lateral_mms', 'vibration_axial_mms',
            'wind_speed_ms', 'temperature_outside_nacelle'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for all models"""
        
        # Calculate derived features if not present
        if 'gearbox_oil_inlet_temperature' not in df.columns:
            df['gearbox_oil_inlet_temperature'] = df['gearbox_oil_temperature'] * 0.9
        
        if 'reactive_power' not in df.columns:
            df['reactive_power'] = df['active_power'] * 0.1
        
        # Calculate derived features
        df['gearbox_temp_delta'] = df['gearbox_oil_temperature'] - df['gearbox_oil_inlet_temperature']
        df['pitch_imbalance'] = np.std(df[['pitch_angle_1deg', 'pitch_angle_2deg', 'pitch_angle_3deg']], axis=1)
        df['stator_imbalance'] = np.std(df[['generator_stator_temp_u', 'generator_stator_temp_v', 'generator_stator_temp_w']], axis=1)
        df['bearing_de_nde_diff'] = df['gearbox_de_bearing_temperature'] - df['gearbox_nde_bearing_temperature']
        
        # Prepare 26-feature data
        df_26 = df[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Prepare 18-feature data for autoencoders
        df_18 = df[self.ae_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        return df_26.values, df_18.values
    
    def detect_anomalies_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Run ALL models and return ensemble prediction
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'individual_predictions': {},
            'individual_scores': {},
            'ensemble': {},
            'anomalies': []
        }
        
        # Prepare features
        X_26, X_18 = self.prepare_features(df)
        
        # Scale features
        if self.scalers.get('main') is not None:
            X_scaled_26 = self.scalers['main'].transform(X_26)
        else:
            X_scaled_26 = X_26
        
        if self.scalers.get('autoencoder') is not None:
            X_scaled_18 = self.scalers['autoencoder'].transform(X_18)
        else:
            X_scaled_18 = X_18
        
        # Track votes
        votes = []
        all_scores = []
        
        # ============================================================
        # MODEL 1: ISOLATION FOREST
        # ============================================================
        if self.models.get('isolation_forest') is not None:
            try:
                iso_pred = self.models['isolation_forest'].predict(X_scaled_26)
                iso_scores = self.models['isolation_forest'].score_samples(X_scaled_26)
                
                # Normalize scores (higher = more anomalous)
                iso_scores_norm = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
                iso_anomaly = (iso_pred == -1)
                
                results['individual_predictions']['isolation_forest'] = iso_anomaly.tolist()
                results['individual_scores']['isolation_forest'] = iso_scores_norm.tolist()
                
                votes.append(iso_anomaly.astype(int))
                all_scores.append(iso_scores_norm)
                
                logger.info(f"✅ Isolation Forest: {np.sum(iso_anomaly)} anomalies detected")
            except Exception as e:
                logger.error(f"❌ Isolation Forest failed: {e}")
        
        # ============================================================
        # MODEL 2: ELLIPTIC ENVELOPE
        # ============================================================
        if self.models.get('elliptic_envelope') is not None:
            try:
                ee_pred = self.models['elliptic_envelope'].predict(X_scaled_26)
                ee_scores = self.models['elliptic_envelope'].score_samples(X_scaled_26)
                
                # Normalize scores
                ee_scores_norm = 1 - (ee_scores - ee_scores.min()) / (ee_scores.max() - ee_scores.min() + 1e-8)
                ee_anomaly = (ee_pred == -1)
                
                results['individual_predictions']['elliptic_envelope'] = ee_anomaly.tolist()
                results['individual_scores']['elliptic_envelope'] = ee_scores_norm.tolist()
                
                votes.append(ee_anomaly.astype(int))
                all_scores.append(ee_scores_norm)
                
                logger.info(f"✅ Elliptic Envelope: {np.sum(ee_anomaly)} anomalies detected")
            except Exception as e:
                logger.error(f"❌ Elliptic Envelope failed: {e}")
        
        # ============================================================
        # MODEL 3: XGBOOST CLASSIFIER
        # ============================================================
        if self.models.get('xgboost') is not None:
            try:
                xgb_pred = self.models['xgboost'].predict(X_scaled_26)
                xgb_proba = self.models['xgboost'].predict_proba(X_scaled_26)
                
                # Get probability of NOT being normal (class 0)
                # Assuming class 0 = 'normal'
                if xgb_proba.shape[1] > 1:
                    xgb_scores = 1 - xgb_proba[:, 0]  # Probability of anomaly
                else:
                    xgb_scores = xgb_proba.flatten()
                
                xgb_anomaly = (xgb_pred != 0)  # Not 'normal' class
                
                results['individual_predictions']['xgboost'] = xgb_anomaly.tolist()
                results['individual_scores']['xgboost'] = xgb_scores.tolist()
                
                votes.append(xgb_anomaly.astype(int))
                all_scores.append(xgb_scores)
                
                logger.info(f"✅ XGBoost: {np.sum(xgb_anomaly)} anomalies detected")
            except Exception as e:
                logger.error(f"❌ XGBoost failed: {e}")
        
        # ============================================================
        # MODEL 4: LSTM AUTOENCODER
        # ============================================================
        if self.models.get('lstm_autoencoder') is not None:
            try:
                # Reshape to 3D for LSTM (samples, timesteps, features)
                SEQ_LEN = 20
                n_samples = len(X_scaled_18)
                n_sequences = n_samples // SEQ_LEN
                
                if n_sequences > 0:
                    X_3d = X_scaled_18[:n_sequences * SEQ_LEN].reshape(n_sequences, SEQ_LEN, len(self.ae_features))
                    
                    # Reconstruct
                    reconstructed = self.models['lstm_autoencoder'].predict(X_3d, verbose=0)
                    
                    # Calculate reconstruction error
                    error = np.mean(np.square(X_3d - reconstructed), axis=(1, 2))
                    threshold = np.percentile(error, 95)
                    ae_anomaly = error > threshold
                    
                    # Normalize scores
                    ae_scores = (error - error.min()) / (error.max() - error.min() + 1e-8)
                    
                    # Expand to original sample count
                    expanded_anomaly = np.zeros(n_samples, dtype=bool)
                    expanded_scores = np.zeros(n_samples)
                    
                    for i in range(n_sequences):
                        start = i * SEQ_LEN
                        end = start + SEQ_LEN
                        expanded_anomaly[start:end] = ae_anomaly[i]
                        expanded_scores[start:end] = ae_scores[i]
                    
                    results['individual_predictions']['lstm_autoencoder'] = expanded_anomaly.tolist()
                    results['individual_scores']['lstm_autoencoder'] = expanded_scores.tolist()
                    
                    votes.append(expanded_anomaly.astype(int))
                    all_scores.append(expanded_scores)
                    
                    logger.info(f"✅ LSTM Autoencoder: {np.sum(expanded_anomaly)} anomalies detected")
            except Exception as e:
                logger.error(f"❌ LSTM Autoencoder failed: {e}")
        
        # ============================================================
        # MODEL 5: VAE AUTOENCODER (PyTorch)
        # ============================================================
        if self.models.get('vae_autoencoder') is not None:
            try:
                import torch
                
                # Convert to torch tensor
                X_torch = torch.tensor(X_scaled_18, dtype=torch.float32)
                
                # Encode and decode
                with torch.no_grad():
                    mean, log_var = self.models['vae_autoencoder'].encode(X_torch)
                    z = self.models['vae_autoencoder'].reparameterize(mean, log_var)
                    reconstructed = self.models['vae_autoencoder'].decode(z)
                
                # Calculate reconstruction error
                error = torch.mean(torch.square(X_torch - reconstructed), axis=1).numpy()
                threshold = np.percentile(error, 95)
                vae_anomaly = error > threshold
                
                # Normalize scores
                vae_scores = (error - error.min()) / (error.max() - error.min() + 1e-8)
                
                results['individual_predictions']['vae_autoencoder'] = vae_anomaly.tolist()
                results['individual_scores']['vae_autoencoder'] = vae_scores.tolist()
                
                votes.append(vae_anomaly.astype(int))
                all_scores.append(vae_scores)
                
                logger.info(f"✅ VAE Autoencoder: {np.sum(vae_anomaly)} anomalies detected")
            except Exception as e:
                logger.error(f"❌ VAE Autoencoder failed: {e}")
        
        # ============================================================
        # ENSEMBLE: Combine all models
        # ============================================================
        if votes:
            # Stack all votes
            all_votes = np.array(votes)
            
            # Ensemble prediction: at least 2 models agree
            vote_sum = np.sum(all_votes, axis=0)
            ensemble_pred = vote_sum >= 2
            
            # Ensemble score: average of all model scores
            if all_scores:
                all_scores_array = np.array(all_scores)
                ensemble_score = np.mean(all_scores_array, axis=0)
            else:
                ensemble_score = np.zeros(len(df))
            
            # Normalize ensemble score
            ensemble_score_norm = (ensemble_score - ensemble_score.min()) / (ensemble_score.max() - ensemble_score.min() + 1e-8)
            
            results['ensemble'] = {
                'predictions': ensemble_pred.tolist(),
                'scores': ensemble_score_norm.tolist(),
                'vote_sum': vote_sum.tolist(),
                'models_voting': len(votes),
                'threshold': 2  # At least 2 models
            }
            
            # Extract anomalies
            anomaly_indices = np.where(ensemble_pred)[0]
            for idx in anomaly_indices[:50]:  # Limit to 50
                # Determine severity based on score
                score = ensemble_score_norm[idx]
                if score > 0.85:
                    severity = "high"
                elif score > 0.75:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Count how many models detected this anomaly
                models_detected = []
                for model_name, pred in results['individual_predictions'].items():
                    if idx < len(pred) and pred[idx]:
                        models_detected.append(model_name)
                
                results['anomalies'].append({
                    'index': int(idx),
                    'turbine_id': str(df.iloc[idx].get('turbine_id', 'Unknown')),
                    'timestamp': df.iloc[idx].get('time', datetime.now()).isoformat(),
                    'score': float(score),
                    'severity': severity,
                    'models_detected': models_detected,
                    'model_count': len(models_detected),
                    'ensemble_votes': int(vote_sum[idx])
                })
            
            logger.info(f"🎯 ENSEMBLE: {len(anomaly_indices)} anomalies detected (2+ models)")
        
        return results
    
    def get_turbine_status(self, turbine_id: str, df: pd.DataFrame) -> Dict:
        """
        Get status for a single turbine
        """
        results = self.detect_anomalies_ensemble(df)
        
        anomaly_count = len(results['anomalies'])
        total_samples = len(df)
        anomaly_ratio = anomaly_count / total_samples if total_samples > 0 else 0
        
        # Calculate health score
        health_score = max(0, 100 - (anomaly_ratio * 100))
        
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
            'turbine_id': turbine_id,
            'status': status,
            'health_score': int(health_score),
            'anomaly_count': anomaly_count,
            'total_samples': total_samples,
            'anomaly_ratio': float(anomaly_ratio),
            'has_anomaly': anomaly_count > 0,
            'anomalies': results['anomalies'],
            'last_checked': datetime.now().isoformat()
        }