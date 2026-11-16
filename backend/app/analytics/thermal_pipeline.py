# app/pipeline.py - FULLY FIXED VERSION

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def run_thermal_inference(turbine_id: str, scada_records: list) -> Dict[str, Any]:
    """
    Complete thermal analysis for a turbine
    
    Analyzes:
    1. All temperature sensors
    2. Temperature patterns and trends
    3. Correlations between sensors
    4. Anomaly detection
    5. Health scoring
    """
    
    if not scada_records or len(scada_records) == 0:
        return {
            "turbine_id": turbine_id,
            "status": "No data available",
            "thermal_metrics": {},
            "alerts": [],
            "patterns": [],
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(scada_records)
        
        # ✅ FIX 1: Convert all numeric columns from strings to actual numbers
        numeric_columns = [
            'gearbox_oil_temp', 'gearbox_de_bearing', 'gearbox_nde_bearing',
            'main_bearing_temp', 'stator_temp_u', 'stator_temp_v', 'stator_temp_w',
            'drive_end_bearing', 'non_drive_end_bearing', 'inside_temp', 
            'outside_temp', 'active_power', 'wind_speed'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, forcing errors to NaN (so they don't crash the mean)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        #  FIX 2: Filter out garbage wind speed data (trillions values)
        if 'wind_speed' in df.columns:
            # Remove anything below 0 or above 40 m/s (realistic max wind speed)
            df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= 40)]
        
        # If filtering removed all data, return early
        if len(df) == 0:
            return {
                "turbine_id": turbine_id,
                "status": "No valid data",
                "message": "All data was corrupted (wind speed out of range)",
                "thermal_metrics": {},
                "alerts": [],
                "patterns": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Define ALL thermal sensors (from your database)
        thermal_sensors = {
            # Gearbox sensors
            'gearbox_oil_temp': {
                'column': 'gearbox_oil_temp',
                'category': 'gearbox',
                'warning': 80,
                'critical': 95,
                'description': 'Gearbox Oil Temperature'
            },
            'gearbox_de_bearing': {
                'column': 'gearbox_de_bearing',
                'category': 'gearbox',
                'warning': 75,
                'critical': 90,
                'description': 'Gearbox Drive End Bearing'
            },
            'gearbox_nde_bearing': {
                'column': 'gearbox_nde_bearing',
                'category': 'gearbox',
                'warning': 75,
                'critical': 90,
                'description': 'Gearbox Non-Drive End Bearing'
            },
            
            # Main bearing
            'main_bearing_temp': {
                'column': 'main_bearing_temp',
                'category': 'bearing',
                'warning': 70,
                'critical': 85,
                'description': 'Main Bearing Temperature'
            },
            
            # Generator stator
            'stator_temp_u': {
                'column': 'stator_temp_u',
                'category': 'generator',
                'warning': 90,
                'critical': 110,
                'description': 'Generator Stator Phase U'
            },
            'stator_temp_v': {
                'column': 'stator_temp_v',
                'category': 'generator',
                'warning': 90,
                'critical': 110,
                'description': 'Generator Stator Phase V'
            },
            'stator_temp_w': {
                'column': 'stator_temp_w',
                'category': 'generator',
                'warning': 90,
                'critical': 110,
                'description': 'Generator Stator Phase W'
            },
            
            # Generator bearings
            'drive_end_bearing': {
                'column': 'drive_end_bearing',
                'category': 'generator',
                'warning': 75,
                'critical': 95,
                'description': 'Generator Drive End Bearing'
            },
            'non_drive_end_bearing': {
                'column': 'non_drive_end_bearing',
                'category': 'generator',
                'warning': 75,
                'critical': 95,
                'description': 'Generator Non-Drive End Bearing'
            },
            
            # Nacelle temperatures
            'inside_temp': {
                'column': 'inside_temp',
                'category': 'nacelle',
                'warning': 50,
                'critical': 60,
                'description': 'Nacelle Inside Temperature'
            },
            'outside_temp': {
                'column': 'outside_temp',
                'category': 'nacelle',
                'warning': 40,
                'critical': 50,
                'description': 'Nacelle Outside Temperature'
            }
        }
        
        # Check which sensors are available
        available_sensors = {}
        for name, config in thermal_sensors.items():
            if config['column'] in df.columns:
                available_sensors[name] = config
        
        if not available_sensors:
            return {
                "turbine_id": turbine_id,
                "status": "No thermal sensors available",
                "thermal_metrics": {},
                "alerts": [],
                "patterns": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # 1. Calculate metrics for each sensor
        thermal_metrics = {}
        alerts = []
        all_temperatures = []
        
        for name, config in available_sensors.items():
            col = config['column']
            values = df[col].dropna()
            
            if len(values) == 0:
                continue
            
            current_temp = float(values.iloc[-1])
            avg_temp = float(values.mean())
            max_temp = float(values.max())
            min_temp = float(values.min())
            std_temp = float(values.std())
            
            # Calculate rate of change (trend)
            if len(values) > 5:
                recent = values.iloc[-10:].mean()
                older = values.iloc[:10].mean() if len(values) > 10 else values.mean()
                rate_of_change = float(recent - older)
            else:
                rate_of_change = 0.0
            
            # Check for anomalies (2 standard deviations from mean)
            is_anomaly = abs(current_temp - avg_temp) > 2 * std_temp if std_temp > 0 else False
            
            thermal_metrics[name] = {
                "current": current_temp,
                "average": avg_temp,
                "max": max_temp,
                "min": min_temp,
                "std": std_temp,
                "rate_of_change": rate_of_change,
                "is_anomaly": is_anomaly,
                "sample_count": len(values),
                "category": config['category'],
                "description": config['description']
            }
            
            all_temperatures.append(current_temp)
            
            # Check for alerts
            if current_temp > config['critical']:
                alerts.append({
                    "sensor": name,
                    "description": config['description'],
                    "value": current_temp,
                    "threshold": config['critical'],
                    "severity": "critical",
                    "message": f"{config['description']} at {current_temp:.1f}°C exceeds critical limit ({config['critical']}°C)"
                })
            elif current_temp > config['warning']:
                alerts.append({
                    "sensor": name,
                    "description": config['description'],
                    "value": current_temp,
                    "threshold": config['warning'],
                    "severity": "warning",
                    "message": f"{config['description']} at {current_temp:.1f}°C exceeds warning limit ({config['warning']}°C)"
                })
            
            # Check for rapid temperature rise
            if rate_of_change > 5.0:
                alerts.append({
                    "sensor": name,
                    "description": config['description'],
                    "value": rate_of_change,
                    "threshold": 5.0,
                    "severity": "warning",
                    "message": f"{config['description']} rising rapidly ({rate_of_change:.1f}°C per period)"
                })

        # 2. Detect temperature patterns
        patterns = detect_thermal_patterns(df, available_sensors)
        
        # 3. Calculate overall health score
        health_score = calculate_health_score(thermal_metrics, alerts)
        
        # 4. Generate summary statistics
        if all_temperatures:
            avg_all_temp = sum(all_temperatures) / len(all_temperatures)
            max_all_temp = max(all_temperatures)
            min_all_temp = min(all_temperatures)
        else:
            avg_all_temp = 0
            max_all_temp = 0
            min_all_temp = 0
        
        # 5. Determine status
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        
        if critical_alerts:
            status = "Critical"
        elif warning_alerts:
            status = "Warning"
        elif health_score > 85:
            status = "Healthy"
        elif health_score > 70:
            status = "Monitor"
        else:
            status = "Attention"
        
        return {
            "turbine_id": turbine_id,
            "status": status,
            "health_score": health_score,
            "summary": {
                "average_temperature": avg_all_temp,
                "max_temperature": max_all_temp,
                "min_temperature": min_all_temp,
                "sensors_monitored": len(thermal_metrics),
                "active_alerts": len(alerts),
                "critical_alerts": len(critical_alerts),
                "warning_alerts": len(warning_alerts),
                "patterns_detected": len(patterns)
            },
            "thermal_metrics": thermal_metrics,
            "alerts": alerts,
            "patterns": patterns,
            "data_points": len(scada_records),
            "data_period_days": None,
            "records_found": len(scada_records),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in thermal analysis for {turbine_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "turbine_id": turbine_id,
            "status": "Error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def detect_thermal_patterns(df: pd.DataFrame, sensors: Dict) -> List[Dict]:
    """
    Detect patterns in thermal data
    """
    patterns = []
    
    for name, config in sensors.items():
        col = config['column']
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        if len(values) < 10:
            continue
        
        # Pattern 1: Increasing trend
        if len(values) > 20:
            first_10 = values.iloc[:10].mean()
            last_10 = values.iloc[-10:].mean()
            if last_10 > first_10 * 1.1:  # 10% increase
                patterns.append({
                    "sensor": name,
                    "description": config['description'],
                    "pattern": "increasing_trend",
                    "severity": "warning",
                    "message": f"{config['description']} showing increasing trend over time"
                })
        
        # Pattern 2: High volatility
        if len(values) > 10:
            std = values.std()
            mean = values.mean()
            if std > mean * 0.1:  # 10% volatility
                patterns.append({
                    "sensor": name,
                    "description": config['description'],
                    "pattern": "high_volatility",
                    "severity": "info",
                    "message": f"{config['description']} showing unstable temperature pattern"
                })
        
        # Pattern 3: Spikes
        if len(values) > 10:
            mean = values.mean()
            std = values.std()
            spikes = values[values > mean + 3 * std]
            if len(spikes) > 0:
                patterns.append({
                    "sensor": name,
                    "description": config['description'],
                    "pattern": "temperature_spikes",
                    "severity": "warning",
                    "message": f"{config['description']} detected {len(spikes)} temperature spikes"
                })
    
    return patterns


def calculate_health_score(metrics: Dict, alerts: List) -> int:
    """
    Calculate overall health score (0-100)
    """
    if not metrics:
        return 0
    
    # Start with perfect score
    score = 100
    
    # Deduct for critical alerts
    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
    score -= len(critical_alerts) * 25
    
    # Deduct for warning alerts
    warning_alerts = [a for a in alerts if a['severity'] == 'warning']
    score -= len(warning_alerts) * 10
    
    # Deduct for anomalies
    for name, data in metrics.items():
        if data.get('is_anomaly', False):
            score -= 5
    
    # Deduct for high temperatures
    for name, data in metrics.items():
        if data['current'] > 90:
            score -= 15
        elif data['current'] > 80:
            score -= 8
        elif data['current'] > 70:
            score -= 3
    
    # Ensure score stays between 0 and 100
    return max(0, min(100, score))