"""
scripts/ingest_data.py
CLI entry point for Phase 2 Data Ingestion and SQL Feature Engineering.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --data-dir data/synthetic --skip-truncate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.db_connector import (
    wait_for_postgres,
    execute_sql_file,
    get_db_connection,
)
from src.data_ingestion.loader import PostgreSQLDataLoader
from src.data_ingestion.validation import DataQualityValidator
from psycopg2.extras import RealDictCursor


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Ingest synthetic banking data into PostgreSQL and build SQL feature views.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Directory containing generated Parquet datasets.",
    )
    parser.add_argument(
        "--skip-truncate",
        action="store_true",
        help="Skip truncating existing table data before ingestion.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def demonstrate_new_device_detection() -> None:
    """Run a query demonstrating new-device detection logic on ingested transactions."""
    logger = logging.getLogger("ingest_data")
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: New-Device Detection Logic in SQL")
    logger.info("=" * 60)

    query = """
        SELECT
            transaction_id,
            customer_id,
            device_id,
            is_new_device,
            customer_device_use_count,
            device_risk_score,
            is_fraud,
            fraud_scenario
        FROM banking.v_new_device_detection
        WHERE is_new_device = TRUE
        LIMIT 10;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        logger.warning("No transactions flagged with is_new_device = TRUE.")
        return

    logger.info("Found %d sample transactions flagged as NEW / UNRECOGNIZED devices:", len(rows))
    logger.info("%-14s | %-12s | %-12s | %-10s | %-8s | %-8s",
                "Transaction ID", "Customer ID", "Device (10c)", "New Device", "Risk Sc", "Is Fraud")
    logger.info("-" * 75)
    for r in rows:
        dev_short = str(r['device_id'])[:10]
        logger.info("%-14s | %-12s | %-12s | %-10s | %-8.2f | %-8s",
                    r['transaction_id'], r['customer_id'], dev_short,
                    str(r['is_new_device']), float(r['device_risk_score']), str(r['is_fraud']))

    # Summary count
    summary_query = """
        SELECT
            COUNT(*)                                             AS total_transactions,
            SUM(CASE WHEN is_new_device THEN 1 ELSE 0 END)       AS new_device_count,
            ROUND(AVG(CASE WHEN is_new_device THEN 1.0 ELSE 0.0 END) * 100, 2) AS new_device_pct,
            SUM(CASE WHEN is_new_device AND is_fraud THEN 1 ELSE 0 END) AS new_device_fraud_count
        FROM banking.v_new_device_detection;
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(summary_query)
            stats = cur.fetchone()

    logger.info("")
    logger.info("New-Device Aggregates across dataset:")
    logger.info("  Total transactions        : %d", stats['total_transactions'])
    logger.info("  New-device transactions   : %d (%.2f%%)", stats['new_device_count'], float(stats['new_device_pct']))
    logger.info("  New-device fraud cases    : %d", stats['new_device_fraud_count'])
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("ingest_data")

    logger.info("Starting Phase 2: PostgreSQL Schema & Ingestion Pipeline")

    # 1. Wait for Postgres readiness
    if not wait_for_postgres(timeout_seconds=30):
        logger.error("PostgreSQL server is not reachable. Aborting Phase 2.")
        sys.exit(1)

    # 2. Ensure base schema tables exist
    init_sql_path = Path("docker/postgres/init.sql")
    if init_sql_path.exists():
        logger.info("Ensuring base schema and tables exist...")
        execute_sql_file(init_sql_path)

    # 3. Load Parquet datasets into PostgreSQL
    loader = PostgreSQLDataLoader(data_dir=args.data_dir)
    loaded_counts = loader.load_all(truncate_first=not args.skip_truncate)

    # 4. Execute SQL Feature Engineering script
    feature_sql_path = Path("docker/postgres/feature_views.sql")
    if feature_sql_path.exists():
        logger.info("Applying SQL Feature Engineering views and indexes...")
        execute_sql_file(feature_sql_path)

    # 5. Execute Data Quality & Referential Integrity Validation
    validator = DataQualityValidator()
    val_report = validator.run_all_checks()

    # 6. Demonstrate New-Device Detection
    demonstrate_new_device_detection()

    if not val_report["passed"]:
        logger.error("Phase 2 pipeline completed WITH VALIDATION ERRORS.")
        sys.exit(1)

    logger.info("Phase 2 Data Ingestion & SQL Feature Engineering COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
