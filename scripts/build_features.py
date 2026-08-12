"""
scripts/build_features.py
CLI entry point for Phase 3 PySpark Feature Engineering Pipeline.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --input-dir data/synthetic --output-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.spark_session import get_spark_session, stop_spark_session
from src.feature_engineering.pipeline import PySparkFeaturePipeline


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Run PySpark Feature Engineering Pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/synthetic",
        help="Directory containing synthetic raw Parquet datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save engineered PySpark Parquet feature datasets.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def demonstrate_device_ip_features(dfs: dict) -> None:
    """Print sample PySpark Device & IP features created."""
    logger = logging.getLogger("build_features")
    df_fraud = dfs["fraud_features"]

    logger.info("=" * 60)
    logger.info("DEMONSTRATION: PySpark Device & IP Features Generated")
    logger.info("=" * 60)

    # Select sample device/IP features
    sample_cols = [
        "transaction_id", "customer_id", "device_id", "ip_address",
        "is_new_device", "device_risk_score", "is_new_ip", "ip_risk_score",
        "txns_last_1h", "spend_last_24h", "amount_z_score", "is_fraud"
    ]

    sample_rows = (
        df_fraud.filter("is_new_device = true OR is_new_ip = true")
        .select(sample_cols)
        .limit(8)
        .collect()
    )

    logger.info("Sample Transactions with PySpark Device & IP Risk Signals:")
    logger.info("%-14s | %-12s | %-10s | %-8s | %-8s | %-8s | %-8s | %-8s",
                "Transaction ID", "Customer ID", "New Device", "Dev Risk", "New IP", "IP Risk", "Txns 1h", "Z-Score")
    logger.info("-" * 88)
    for r in sample_rows:
        logger.info("%-14s | %-12s | %-10s | %-8.2f | %-8s | %-8.2f | %-8d | %-8.2f",
                    r["transaction_id"], r["customer_id"], str(r["is_new_device"]),
                    float(r["device_risk_score"]), str(r["is_new_ip"]), float(r["ip_risk_score"]),
                    int(r["txns_last_1h"]), float(r["amount_z_score"]))

    # Summary counts
    total_txns = df_fraud.count()
    new_device_cnt = df_fraud.filter("is_new_device = true").count()
    new_ip_cnt = df_fraud.filter("is_new_ip = true").count()
    fraud_cnt = df_fraud.filter("is_fraud = true").count()

    logger.info("")
    logger.info("PySpark Feature Dataset Summary Metrics:")
    logger.info("  Total transactions processed     : %d", total_txns)
    logger.info("  Transactions with new_device=True: %d (%.2f%%)", new_device_cnt, (new_device_cnt / total_txns) * 100)
    logger.info("  Transactions with new_ip=True    : %d (%.2f%%)", new_ip_cnt, (new_ip_cnt / total_txns) * 100)
    logger.info("  Total fraud cases in dataset     : %d (%.2f%%)", fraud_cnt, (fraud_cnt / total_txns) * 100)
    logger.info("  Total feature columns generated  : %d", len(df_fraud.columns))
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("build_features")

    logger.info("Starting Phase 3: PySpark Feature Engineering")

    spark = get_spark_session()

    try:
        pipeline = PySparkFeaturePipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            spark=spark,
        )
        dfs = pipeline.run()

        demonstrate_device_ip_features(dfs)

        logger.info("Phase 3 PySpark Feature Engineering COMPLETED SUCCESSFULLY.")

    finally:
        stop_spark_session()


if __name__ == "__main__":
    main()
