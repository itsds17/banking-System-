"""
src/feature_engineering/pipeline.py
PySpark Feature Engineering Pipeline Orchestrator.

Combines velocity, device intelligence, IP intelligence, and behavioral anomaly
feature modules into a scalable end-to-end PySpark batch pipeline.

Outputs saved to data/processed/:
- fraud_features.parquet (transaction-level dataset for Phase 4 ML)
- credit_features.parquet (loan-level dataset for Phase 5 ML)
- customer_features.parquet (customer-level dataset for Phase 6 Segmentation)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.data_processing.spark_session import get_spark_session
from src.data_processing.cleaners import (
    clean_transactions,
    clean_customers,
    clean_devices,
    clean_ip_addresses,
    clean_loans,
)
from src.feature_engineering.velocity_features import compute_velocity_features
from src.feature_engineering.device_features import compute_device_features
from src.feature_engineering.ip_features import compute_ip_features
from src.feature_engineering.behavioral_features import compute_behavioral_features

logger = logging.getLogger(__name__)


class PySparkFeaturePipeline:
    """
    Orchestrates the PySpark Feature Engineering Pipeline.
    """

    def __init__(
        self,
        input_dir: str | Path = "data/synthetic",
        output_dir: str | Path = "data/processed",
        spark: Optional[SparkSession] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spark = spark or get_spark_session()

    def run(self) -> Dict[str, DataFrame]:
        """
        Execute full PySpark Feature Engineering pipeline.

        Returns
        -------
        Dict[str, DataFrame]
            Engineered DataFrames keyed by dataset name.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PySpark Feature Engineering Pipeline — Starting")
        logger.info("Input dir : %s", self.input_dir.resolve())
        logger.info("Output dir: %s", self.output_dir.resolve())
        logger.info("=" * 60)

        # ── 1. Read Raw Datasets into PySpark ────────────────────────────────
        logger.info("[1/5] Loading raw Parquet datasets into PySpark...")
        df_txns_raw = self.spark.read.parquet(str(self.input_dir / "transactions.parquet"))
        df_cust_raw = self.spark.read.parquet(str(self.input_dir / "customers.parquet"))
        df_dev_raw = self.spark.read.parquet(str(self.input_dir / "devices.parquet"))
        df_ip_raw = self.spark.read.parquet(str(self.input_dir / "ip_addresses.parquet"))
        df_cust_dev_raw = self.spark.read.parquet(str(self.input_dir / "customer_devices.parquet"))
        df_cust_ip_raw = self.spark.read.parquet(str(self.input_dir / "customer_ips.parquet"))
        df_merch_raw = self.spark.read.parquet(str(self.input_dir / "merchants.parquet"))
        df_loans_raw = self.spark.read.parquet(str(self.input_dir / "loans.parquet"))

        # ── 2. Data Cleaning & Type Enforcement ─────────────────────────────
        logger.info("[2/5] Cleaning and casting DataFrames...")
        df_txns = clean_transactions(df_txns_raw)
        df_cust = clean_customers(df_cust_raw)
        df_dev = clean_devices(df_dev_raw)
        df_ip = clean_ip_addresses(df_ip_raw)
        df_loans = clean_loans(df_loans_raw)

        # ── 3. Build Fraud Model Features (Transaction-level) ─────────────
        logger.info("[3/5] Building Fraud Detection Features (velocity, device, IP, behavioral)...")
        # Step A: Velocity
        df_fraud_features = compute_velocity_features(df_txns)
        # Step B: Device Intelligence
        df_fraud_features = compute_device_features(df_fraud_features, df_dev, df_cust_dev_raw)
        # Step C: IP Intelligence
        df_fraud_features = compute_ip_features(df_fraud_features, df_ip, df_cust_ip_raw)
        # Step D: Behavioral & Anomaly
        df_fraud_features = compute_behavioral_features(df_fraud_features, df_cust, df_merch_raw)

        # Cache fraud features DataFrame for downstream counts & writes
        df_fraud_features.cache()
        fraud_txn_count = df_fraud_features.count()
        logger.info("      Fraud feature dataset built — %d records, %d columns.",
                    fraud_txn_count, len(df_fraud_features.columns))

        # ── 4. Build Credit Risk Features (Loan-level) ──────────────────────
        logger.info("[4/5] Building Credit Risk Features...")
        cust_txn_summary = (
            df_txns.groupBy("customer_id")
            .agg(
                F.count("transaction_id").alias("hist_total_txns"),
                F.avg("amount").alias("hist_avg_txn_amount"),
                F.sum("amount").alias("hist_total_spend"),
                F.sum(F.when(F.col("is_fraud"), 1).otherwise(0)).alias("hist_fraud_count"),
            )
        )

        df_cust_for_loans = df_cust.drop("employment_status")

        df_credit_features = (
            df_loans.join(df_cust_for_loans, "customer_id", "left")
            .join(cust_txn_summary, "customer_id", "left")
            .withColumn(
                "installment_to_income",
                F.when(F.col("monthly_income") > 0, (F.col("loan_amount") / F.col("tenure_months")) / F.col("monthly_income"))
                .otherwise(0.0),
            )
            .withColumn(
                "delinquency_risk_flag",
                F.when((F.col("delinquency_history") > 2) | (F.col("credit_score") < 580), True).otherwise(False),
            )
        )
        df_credit_features.cache()
        credit_loan_count = df_credit_features.count()
        logger.info("      Credit risk feature dataset built — %d records, %d columns.",
                    credit_loan_count, len(df_credit_features.columns))

        # ── 5. Build Customer Segmentation Features (Customer-level) ────────
        logger.info("[5/5] Building Customer Profile Features...")
        df_customer_features = (
            df_cust.join(cust_txn_summary, "customer_id", "left")
            .withColumn("hist_total_txns", F.coalesce(F.col("hist_total_txns"), F.lit(0)))
            .withColumn("hist_total_spend", F.coalesce(F.col("hist_total_spend"), F.lit(0.0)))
            .withColumn("hist_fraud_count", F.coalesce(F.col("hist_fraud_count"), F.lit(0)))
        )
        df_customer_features.cache()
        customer_count = df_customer_features.count()
        logger.info("      Customer profile feature dataset built — %d records, %d columns.",
                    customer_count, len(df_customer_features.columns))

        # ── Write Output Parquet Datasets ────────────────────────────────────
        logger.info("Saving engineered feature datasets to %s ...", self.output_dir)

        path_fraud = self.output_dir / "fraud_features.parquet"
        path_credit = self.output_dir / "credit_features.parquet"
        path_customer = self.output_dir / "customer_features.parquet"

        # Use PyArrow via toPandas for cross-platform compatibility on Windows (avoids Hadoop winutils requirement)
        import shutil
        for path, df in [(path_fraud, df_fraud_features), (path_credit, df_credit_features), (path_customer, df_customer_features)]:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            df.toPandas().to_parquet(str(path), index=False, engine="pyarrow")

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("PYSPARK FEATURE ENGINEERING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("  fraud_features.parquet    : %8d rows (%s)", fraud_txn_count, path_fraud)
        logger.info("  credit_features.parquet   : %8d rows (%s)", credit_loan_count, path_credit)
        logger.info("  customer_features.parquet : %8d rows (%s)", customer_count, path_customer)
        logger.info("  Total Runtime             : %.2f seconds", elapsed)
        logger.info("=" * 60)

        return {
            "fraud_features": df_fraud_features,
            "credit_features": df_credit_features,
            "customer_features": df_customer_features,
        }
