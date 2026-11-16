# app/db/data_mode.py
#
# Data-freshness / date-window helpers moved VERBATIM from app/main.py during
# restructuring. Behavior unchanged.
#
# Correctness pass: the historical window is now sourced from core.config
# (DATA_HISTORICAL_START / DATA_HISTORICAL_END, env-overridable) instead of being
# hard-coded here.

import logging

from app.db.connection import get_db_connection
from app.core.config import SCADA_TABLE

logger = logging.getLogger(__name__)


def check_data_mode():
    """Check if database has recent data (last 24 hours)"""
    try:
        conn = get_db_connection()
        if not conn:
            return 'historical'
        
        cursor = conn.cursor()
        query = f"""
            SELECT COUNT(*) as count
            FROM {SCADA_TABLE}
            WHERE "time" >= NOW() - INTERVAL '24 hours'
        """
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            logger.info(f"Data mode: LIVE ({result[0]} records in last 24 hours)")
            return 'live'
        else:
            logger.info("Data mode: HISTORICAL")
            return 'historical'
            
    except Exception as e:
        logger.error(f"Error checking data mode: {e}")
        return 'historical'


def get_date_condition(mode: str, days: int = 7):
    """Date window for queries.

    `days` is interpreted RELATIVE TO THE LATEST AVAILABLE DATA TIMESTAMP
    (MAX("time")) — NOT wall-clock NOW(). Rationale: the DZH dataset is
    historical (it ends months before the server clock), so a NOW()-relative
    window returns zero rows and makes `days` a silent no-op. Anchoring to
    MAX("time") makes `days` meaningful and IDENTICAL for both live and
    historical data, and consistent across every endpoint that passes `days`.

    `mode` is retained for signature/logging compatibility but no longer changes
    the window (previously 'historical' ignored `days` and used a fixed
    start/end window, which is the inconsistency this fixes).
    """
    return (
        f'"time" >= (SELECT MAX("time") FROM {SCADA_TABLE}) '
        f"- INTERVAL '{days} days'"
    )
