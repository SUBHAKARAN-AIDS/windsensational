import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from app.db.connection import get_db_connection

def get_live_fleet_telemetry():
    """
    Fetches live telemetry for each turbine from PostgreSQL
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            
            query = """
                SELECT DISTINCT ON (name) 
                    name as "turbineId",
                    generator_active_power_kw as "activePower",
                    wind_speed_m_s as "windSpeed",
                    generator_drive_end_bearing_temperature as "bearingTemp",
                    time as "timestamp"
                FROM site_dzh.scada_raw
                WHERE name IS NOT NULL
                ORDER BY name, time DESC;
            """
            cursor.execute(query)
            records = cursor.fetchall()
            
            turbines = []
            for row in records:
                # Handle None values
                active_power = float(row["activePower"]) if row["activePower"] is not None else 0.0
                wind_speed = float(row["windSpeed"]) if row["windSpeed"] is not None else 0.0
                bearing_temp = float(row["bearingTemp"]) if row["bearingTemp"] is not None else 0.0
                
                turbines.append({
                    "id": row["turbineId"],
                    "activePower": active_power,
                    "windSpeed": wind_speed,
                    "bearingTemp": bearing_temp,
                    "status": "Operational" if active_power > 0 else "Idle"
                })
            
            logger.info(f"✅ Fetched {len(turbines)} turbines with live telemetry")
            return turbines
    except Exception as e:
        logger.error(f"❌ Error fetching live fleet telemetry: {e}")
        return []
    finally:
        conn.close()