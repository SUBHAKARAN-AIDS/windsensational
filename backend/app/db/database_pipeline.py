import os
import psycopg2 # type: ignore
from psycopg2.extras import RealDictCursor # type: ignore
from dotenv import load_dotenv # type: ignore
import logging
from datetime import datetime, timedelta

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.core.config import DATA_HISTORICAL_START, DATA_HISTORICAL_END
from app.db.connection import get_db_connection

# ============================================================
# 1. SCADA HISTORY FUNCTIONS
# ============================================================

def fetch_scada_history(turbine_id: str, days: int = 30):
    """
    Fetch SCADA history for a specific turbine from historical data
    """
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT MAX(time) as latest_time
                FROM site_dzh.scada_raw
                WHERE name = %s
            """, (turbine_id,))
            result = cursor.fetchone()
            
            if not result or result['latest_time'] is None:
                logger.warning(f"No data found for turbine {turbine_id}")
                return []
            
            latest_date = result['latest_time']
            
            query = """
                SELECT 
                    time,
                    name as turbine_id,
                    generator_active_power_kw as active_power,
                    generator_speed_r_min as generator_speed,
                    rotor_speed_r_min as rotor_speed,
                    torque_nm as torque,
                    pitch_angle_1 as pitch_angle_1,
                    pitch_angle_2 as pitch_angle_2,
                    pitch_angle_3 as pitch_angle_3,
                    gearbox_oil_temperature as gearbox_oil_temp,
                    gearbox_de_bearing_temperature as gearbox_de_bearing,
                    gearbox_nde_bearing_temperature as gearbox_nde_bearing,
                    main_bearing_temperature as main_bearing_temp,
                    generator_stator_temp_u as stator_temp_u,
                    generator_stator_temp_v as stator_temp_v,
                    generator_stator_temp_w as stator_temp_w,
                    generator_drive_end_bearing_temperature as drive_end_bearing,
                    generator_non_drive_end_bearing_temperature as non_drive_end_bearing,
                    temperature_outside_nacelle as outside_temp,
                    temperature_inside_nacelle as inside_temp,
                    wind_speed_m_s as wind_speed,
                    wind_direction as wind_direction
                FROM site_dzh.scada_raw
                WHERE name = %s
                AND time >= %s - INTERVAL '%s days'
                ORDER BY time DESC
                LIMIT 1000
            """
            cursor.execute(query, (turbine_id, latest_date, days))
            records = cursor.fetchall()
            logger.info(f"Fetched {len(records)} records for turbine {turbine_id} (last {days} days from {latest_date})")
            
            return records[::-1] if records else []
    except Exception as e:
        logger.error(f"Error executing SCADA query for {turbine_id}: {e}")
        return None
    finally:
        conn.close()


# ============================================================
# 2. THERMAL & TURBINE DATA FUNCTIONS
# ============================================================

def get_all_turbine_ids():
    """
    Get all unique turbine IDs from the database
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed")
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT DISTINCT name as turbine_id
                FROM site_dzh.scada_raw
                WHERE name IS NOT NULL 
                AND name != ''
                ORDER BY name
            """)
            records = cursor.fetchall()
            turbine_ids = [record['turbine_id'] for record in records]
            logger.info(f"Found {len(turbine_ids)} turbines: {turbine_ids[:10]}...")
            return turbine_ids
    except Exception as e:
        logger.error(f"Error fetching turbine IDs: {e}")
        return []
    finally:
        conn.close()


def get_turbine_latest_data(turbine_id: str):
    """
    Get the latest SCADA reading for a specific turbine
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    time,
                    name as turbine_id,
                    gearbox_oil_temperature as gearbox_oil_temperature,
                    gearbox_oil_inlet_temperature as gearbox_oil_inlet_temperature,
                    gearbox_de_bearing_temperature as gearbox_de_bearing_temperature,
                    gearbox_nde_bearing_temperature as gearbox_nde_bearing_temperature,
                    main_bearing_temperature as main_bearing_temperature,
                    generator_stator_temp_u as generator_stator_temp_u,
                    generator_stator_temp_v as generator_stator_temp_v,
                    generator_stator_temp_w as generator_stator_temp_w,
                    generator_drive_end_bearing_temperature as generator_drive_end_bearing_temperature,
                    generator_non_drive_end_bearing_temperature as generator_non_drive_end_bearing_temperature,
                    vibration_lateral_mm_s as vibration_lateral_mms,
                    vibration_axial_mm_s as vibration_axial_mms,
                    generator_active_power_kw as active_power,
                    q as reactive_power,
                    rotor_speed_r_min as rotor_speed,
                    generator_speed_r_min as generator_speed,
                    torque_nm as torque_nm,
                    pitch_angle_1 as pitch_angle_1deg,
                    pitch_angle_2 as pitch_angle_2deg,
                    pitch_angle_3 as pitch_angle_3deg,
                    temperature_outside_nacelle as temperature_outside_nacelle,
                    wind_speed_m_s as wind_speed_ms
                FROM site_dzh.scada_raw
                WHERE name = %s
                ORDER BY time DESC
                LIMIT %s
            """
            cursor.execute(query, (turbine_id, 1))
            record = cursor.fetchone()
            return record
    except Exception as e:
        logger.error(f"Error fetching latest data for {turbine_id}: {e}")
        return None
    finally:
        conn.close()


def fetch_rul_physical_history(turbine_id: str):
    """Fetch one turbine's physical RUL inputs, anchored to its latest data."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                WITH latest AS (
                    SELECT MAX(time) AS latest_time
                    FROM site_dzh.scada_raw
                    WHERE name = %s
                )
                SELECT
                    time,
                    name AS turbine_id,
                    vibration_lateral_mm_s AS vibration_lateral,
                    vibration_axial_mm_s AS vibration_axial,
                    main_bearing_temperature AS main_bearing_temperature
                FROM site_dzh.scada_raw, latest
                WHERE name = %s
                  AND time <= latest.latest_time
                ORDER BY time ASC
            """
            cursor.execute(query, (turbine_id, turbine_id))
            return cursor.fetchall()
    except Exception as exc:
        logger.error("Error fetching physical RUL history for %s: %s", turbine_id, exc)
        return []
    finally:
        conn.close()


# ============================================================
# 3. POWER CURVE FUNCTIONS
# ============================================================

def fetch_power_curve_data():
    """
    Fetches actual wind speed vs active power relationship directly from PostgreSQL
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("Power Curve: Database connection failed.")
        return _generate_fallback_data()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN wind_speed_m_s IS NOT NULL AND TRIM(wind_speed_m_s) != '' THEN 1 END) as wind_speed_count,
                    COUNT(CASE WHEN generator_active_power_kw IS NOT NULL AND TRIM(generator_active_power_kw) != '' THEN 1 END) as power_count
                FROM site_dzh.scada_raw
                WHERE time >= %s AND time <= %s
            """, (DATA_HISTORICAL_START, DATA_HISTORICAL_END))
            stats = cursor.fetchone()
            
            if not stats or stats['power_count'] == 0:
                logger.warning("No active power data found in database - using fallback data")
                return _generate_fallback_data()
            
            query = """
                WITH binned_data AS (
                    SELECT 
                        ROUND(NULLIF(TRIM(wind_speed_m_s), '')::numeric * 2, 0) / 2.0 as wind_bin,
                        NULLIF(TRIM(generator_active_power_kw), '')::numeric as actual_power_value,
                        CASE 
                            WHEN NULLIF(TRIM(wind_speed_m_s), '')::numeric < 3 THEN 0
                            WHEN NULLIF(TRIM(wind_speed_m_s), '')::numeric >= 15 THEN 6500
                            ELSE 6500 * POWER((NULLIF(TRIM(wind_speed_m_s), '')::numeric - 3) / 12.0, 3)
                        END as theoretical_value
                    FROM site_dzh.scada_raw
                    WHERE time >= %s 
                        AND time <= %s
                        AND wind_speed_m_s IS NOT NULL 
                        AND TRIM(wind_speed_m_s) != ''
                        AND NULLIF(TRIM(wind_speed_m_s), '')::numeric BETWEEN 0 AND 30
                        AND generator_active_power_kw IS NOT NULL 
                        AND TRIM(generator_active_power_kw) != ''
                        AND NULLIF(TRIM(generator_active_power_kw), '')::numeric >= 0
                )
                SELECT 
                    wind_bin as "windSpeed",
                    ROUND(AVG(actual_power_value), 1) as "actualPower",
                    ROUND(AVG(theoretical_value), 1) as "theoreticalPower",
                    COUNT(*) as sample_count
                FROM binned_data
                WHERE wind_bin BETWEEN 1 AND 25
                GROUP BY wind_bin
                ORDER BY wind_bin ASC
            """
            
            cursor.execute(query, (DATA_HISTORICAL_START, DATA_HISTORICAL_END))
            records = cursor.fetchall()
            
            if len(records) == 0:
                logger.warning("No records returned - using fallback data")
                return _generate_fallback_data()
            
            results = []
            for row in records:
                if row["windSpeed"] is not None:
                    actual_power = float(row["actualPower"]) if row["actualPower"] is not None else 0.0
                    theoretical_power = float(row["theoreticalPower"]) if row["theoreticalPower"] is not None else 0.0
                    
                    results.append({
                        "windSpeed": float(row["windSpeed"]),
                        "actualPower": actual_power,
                        "theoreticalPower": theoretical_power
                    })
            
            return results
            
    except Exception as e:
        logger.error(f"Error fetching power curve from DB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return _generate_fallback_data()
    finally:
        conn.close()


def _generate_fallback_data():
    """Generates realistic power curve data for testing"""
    logger.info("Generating fallback power curve data for testing")
    
    import random
    random.seed(42)
    
    results = []
    for ws in range(1, 26):
        if ws < 3:
            theo = 0
            actual = 0
        elif ws >= 15:
            theo = 6500
            actual = 6500 * (0.92 + 0.08 * (ws - 15) / 10)
        else:
            theo = 6500 * ((ws - 3) / 12.0) ** 3
            variation = 0.85 + (random.random() * 0.15)
            actual = theo * variation
        
        results.append({
            "windSpeed": float(ws),
            "actualPower": round(actual, 1),
            "theoreticalPower": round(theo, 1)
        })
    
    return results


# ============================================================
# 4. THERMAL DATA FUNCTION
# ============================================================

def get_turbine_thermal_data(turbine_id: str, days: int = 7, limit: int = 300):
    """
    Fetch thermal data for a specific turbine
    """
    conn = get_db_connection()
    if not conn:
        logger.error(f"Database connection failed for turbine {turbine_id}")
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT MAX(time) as latest_time
                FROM site_dzh.scada_raw
                WHERE name = %s
            """, (turbine_id,))
            result = cursor.fetchone()
            
            if not result or result['latest_time'] is None:
                logger.warning(f"No data found for turbine {turbine_id}")
                return []
            
            latest_date = result['latest_time']
            
            query = """
                SELECT 
                    time,
                    name as turbine_id,
                    gearbox_oil_temperature as gearbox_oil_temp,
                    gearbox_de_bearing_temperature as gearbox_de_bearing,
                    gearbox_nde_bearing_temperature as gearbox_nde_bearing,
                    main_bearing_temperature as main_bearing_temp,
                    generator_stator_temp_u as stator_temp_u,
                    generator_stator_temp_v as stator_temp_v,
                    generator_stator_temp_w as stator_temp_w,
                    generator_drive_end_bearing_temperature as drive_end_bearing,
                    generator_non_drive_end_bearing_temperature as non_drive_end_bearing,
                    temperature_outside_nacelle as outside_temp,
                    temperature_inside_nacelle as inside_temp,
                    generator_active_power_kw as active_power,
                    wind_speed_m_s as wind_speed
                FROM site_dzh.scada_raw
                WHERE name = %s
                AND time >= %s - INTERVAL '%s days'
                ORDER BY time DESC
                LIMIT %s
            """
            cursor.execute(query, (turbine_id, latest_date, days, limit))
            records = cursor.fetchall()
            logger.info(f"Fetched {len(records)} thermal records for turbine {turbine_id}")
            
            return records[::-1] if records else []
    except Exception as e:
        logger.error(f"Error fetching thermal data for {turbine_id}: {e}")
        return None
    finally:
        conn.close()


def fetch_fleet_trend(days: int = 90):
    """
    Daily fleet-wide power trend, anchored to the fleet's own latest
    timestamp (not wall-clock NOW()). generator_active_power_kw is
    stored as text in this schema, so it must be cast before aggregation.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed for fleet trend fetch")
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT MAX(time) AS latest_time FROM site_dzh.scada_raw")
            result = cursor.fetchone()
            if not result or result['latest_time'] is None:
                return []
            latest_time = result['latest_time']

            query = """
                SELECT
                    date_trunc('day', time) AS day,
                    AVG(NULLIF(TRIM(generator_active_power_kw), '')::numeric) AS avg_power,
                    SUM(NULLIF(TRIM(generator_active_power_kw), '')::numeric) AS total_power,
                    COUNT(DISTINCT name) AS turbine_count
                FROM site_dzh.scada_raw
                WHERE time >= %s - INTERVAL '%s days'
                  AND generator_active_power_kw IS NOT NULL
                GROUP BY day
                ORDER BY day ASC
            """
            cursor.execute(query, (latest_time, days))
            records = cursor.fetchall()
            logger.info(f"Fetched {len(records)} days of fleet trend data")
            return records
    except Exception as e:
        logger.error(f"Error fetching fleet trend: {e}")
        return []
    finally:
        conn.close()


def fetch_fleet_power_curve_points(days: int = 90, limit: int = 50000):
    """
    Fetch raw (wind_speed, active_power) pairs across the whole fleet for
    GPR power curve fitting. Both source columns are stored as text in
    this schema, so they must be cast before any numeric comparison.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed for fleet power curve fetch")
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT MAX(time) AS latest_time FROM site_dzh.scada_raw")
            result = cursor.fetchone()
            if not result or result['latest_time'] is None:
                return []
            latest_time = result['latest_time']

            query = """
                SELECT
                    name AS turbine_id,
                    NULLIF(TRIM(wind_speed_m_s), '')::numeric AS wind_speed,
                    NULLIF(TRIM(generator_active_power_kw), '')::numeric AS active_power
                FROM site_dzh.scada_raw
                WHERE time >= %s - INTERVAL '%s days'
                  AND wind_speed_m_s IS NOT NULL
                  AND generator_active_power_kw IS NOT NULL
                ORDER BY time DESC
                LIMIT %s
            """
            cursor.execute(query, (latest_time, days, limit))
            records = cursor.fetchall()
            # Range filtering happens here in Python, now that values are
            # actual numerics rather than text — avoids a second SQL cast.
            records = [
                r for r in records
                if r['wind_speed'] is not None and r['active_power'] is not None
                and 0 <= float(r['wind_speed']) <= 30
                and float(r['active_power']) >= -100
            ]
            logger.info(f"Fetched {len(records)} fleet-wide power curve points over {days}d")
            return records
    except Exception as e:
        logger.error(f"Error fetching fleet power curve points: {e}")
        return []
    finally:
        conn.close()