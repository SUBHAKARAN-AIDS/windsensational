# db/connection.py
import os
import sys
import psycopg2
import logging
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection using psycopg2"""
    try:
        host = os.getenv('WINDSENTINEL_DB_HOST')
        port = os.getenv('WINDSENTINEL_DB_PORT')
        dbname = os.getenv('WINDSENTINEL_DB_NAME')
        user = os.getenv('WINDSENTINEL_DB_USER')
        password = os.getenv('WINDSENTINEL_DB_PASSWORD')

        missing = [name for name, val in [
            ('WINDSENTINEL_DB_HOST', host),
            ('WINDSENTINEL_DB_PORT', port),
            ('WINDSENTINEL_DB_NAME', dbname),
            ('WINDSENTINEL_DB_USER', user),
            ('WINDSENTINEL_DB_PASSWORD', password),
        ] if not val]

        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            return None

        port = int(port)

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None