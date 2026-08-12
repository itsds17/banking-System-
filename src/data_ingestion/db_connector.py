"""
src/data_ingestion/db_connector.py
Database connection management for PostgreSQL.

Provides SQLAlchemy engine creation, raw psycopg2 connection handling,
and SQL script execution with retry logic.
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import contextmanager
from typing import Generator, Any
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, Engine

load_dotenv()

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Construct PostgreSQL JDBC/SQLAlchemy connection URL from environment."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB", "banking_db")
    user = os.getenv("POSTGRES_USER", "banking_user")
    password = os.getenv("POSTGRES_PASSWORD", "banking_pass")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_db_engine() -> Engine:
    """
    Create SQLAlchemy Engine tuned for memory efficiency (8 GB RAM budget).
    """
    db_url = get_db_url()
    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager yielding raw psycopg2 connection with auto-commit/rollback."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB", "banking_db")
    user = os.getenv("POSTGRES_USER", "banking_user")
    password = os.getenv("POSTGRES_PASSWORD", "banking_pass")

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password,
    )
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database error occurred: %s", str(e))
        raise e
    finally:
        conn.close()


def wait_for_postgres(timeout_seconds: int = 30) -> bool:
    """Poll PostgreSQL until it is ready to accept connections."""
    start_time = time.time()
    db_url = get_db_url()
    logger.info("Waiting for PostgreSQL connection at %s...", db_url)

    while time.time() - start_time < timeout_seconds:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    logger.info("PostgreSQL is ready and responding.")
                    return True
        except Exception:
            time.sleep(2)

    logger.error("Timed out waiting for PostgreSQL (%d seconds).", timeout_seconds)
    return False


def execute_sql_file(file_path: str | Path) -> None:
    """Read and execute a .sql file against PostgreSQL."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")

    sql_content = path.read_text(encoding="utf-8")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_content)
            logger.info("Successfully executed SQL script: %s", path.name)
