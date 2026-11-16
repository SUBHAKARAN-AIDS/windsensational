# app/db/event_queries.py
"""
Real database queries for event/fault data.
NO MOCK DATA - only actual database queries.
"""

import pandas as pd
from app.db.connection import get_db_connection
import logging

logger = logging.getLogger(__name__)


def get_real_fault_events(
    turbine_id: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 10000
) -> pd.DataFrame:
    """
    Fetch REAL fault events from the database.
    Returns ONLY actual fault records - no mock data.
    
    Args:
        turbine_id: Optional turbine filter
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        limit: Max records to return
    
    Returns:
        DataFrame with real fault events
    """
    conn = get_db_connection()
    if not conn:
        logger.error("❌ Database connection failed")
        return pd.DataFrame()
    
    try:
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT 
                id,
                turbine_id,
                event_time,
                event_type,
                severity,
                description,
                code,
                created_at
            FROM event_raw
            WHERE 1=1
        """
        params = []
        
        # Filter by turbine
        if turbine_id:
            query += " AND turbine_id = %s"
            params.append(turbine_id)
        
        # Filter by date range
        if start_date:
            query += " AND event_time >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND event_time <= %s"
            params.append(end_date)
        
        # Exclude non-fault events (operational logs)
        query += """
            AND (
                severity IN ('critical', 'high', 'fault', 'error')
                OR event_type IN ('fault', 'alarm', 'failure')
                OR code LIKE '%FAULT%'
                OR code LIKE '%ERROR%'
                OR code LIKE '%FAIL%'
            )
        """
        
        # Also exclude routine events
        query += """
            AND event_type NOT IN (
                'startup', 'shutdown', 'maintenance', 
                'curtailment', 'communication', 'status'
            )
        """
        
        query += " ORDER BY event_time DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("⚠️ No fault events found in database")
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = ['id', 'turbine_id', 'event_time', 'event_type', 
                   'severity', 'description', 'code', 'created_at']
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert to datetime
        df['event_time'] = pd.to_datetime(df['event_time'])
        
        logger.info(f"✅ Found {len(df)} real fault events")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error fetching fault events: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_real_fault_event_summary(
    turbine_id: str = None,
    days: int = 30
) -> dict:
    """
    Get summary of REAL fault events for the performance page.
    NO MOCK DATA - only actual database queries.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        
        # Get total fault count
        query = """
            SELECT COUNT(*) 
            FROM event_raw
            WHERE 1=1
        """
        params = []
        
        if turbine_id:
            query += " AND turbine_id = %s"
            params.append(turbine_id)
        
        # Filter for faults only
        query += """
            AND (
                severity IN ('critical', 'high', 'fault', 'error')
                OR event_type IN ('fault', 'alarm', 'failure')
            )
            AND event_type NOT IN (
                'startup', 'shutdown', 'maintenance', 
                'curtailment', 'communication', 'status'
            )
        """
        
        cursor.execute(query, params)
        total_faults = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        # Get faults by severity
        query2 = """
            SELECT severity, COUNT(*) as count
            FROM event_raw
            WHERE 1=1
        """
        if turbine_id:
            query2 += " AND turbine_id = %s"
        
        query2 += """
            AND severity IN ('critical', 'high', 'medium', 'low', 'fault', 'error')
            AND event_type NOT IN (
                'startup', 'shutdown', 'maintenance', 
                'curtailment', 'communication', 'status'
            )
            GROUP BY severity
            ORDER BY count DESC
        """
        
        cursor.execute(query2, params if turbine_id else [])
        severity_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get faults by turbine
        query3 = """
            SELECT turbine_id, COUNT(*) as count
            FROM event_raw
            WHERE 1=1
        """
        if turbine_id:
            query3 += " AND turbine_id = %s"
        
        query3 += """
            AND (
                severity IN ('critical', 'high', 'fault', 'error')
                OR event_type IN ('fault', 'alarm', 'failure')
            )
            AND event_type NOT IN (
                'startup', 'shutdown', 'maintenance', 
                'curtailment', 'communication', 'status'
            )
            GROUP BY turbine_id
            ORDER BY count DESC
            LIMIT 10
        """
        
        cursor.execute(query3, params if turbine_id else [])
        turbine_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "total_faults": total_faults,
            "by_severity": severity_counts,
            "by_turbine": turbine_counts,
            "has_data": total_faults > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting fault summary: {e}")
        return {"error": str(e)}
    finally:
        conn.close()