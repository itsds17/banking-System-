"""
src/feature_engineering/device_features.py
PySpark Device Intelligence & New-Device Risk Feature Generator.

Calculates:
- is_new_device: boolean flag for unrecognized device for customer
- device_txn_count: global transaction volume per device
- device_customer_count: shared customer count per device
- device_fraud_rate: historical fraud rate on device
- device_risk_score: composite device risk signal
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def compute_device_features(
    df_transactions: DataFrame,
    df_devices: DataFrame,
    df_customer_devices: DataFrame,
) -> DataFrame:
    """
    Compute Device Intelligence features and join onto Transactions DataFrame.
    """
    logger.info("Computing PySpark Device Intelligence features...")

    # 1. Device summary features from df_devices & df_customer_devices
    cust_per_device = (
        df_customer_devices.groupBy("device_id")
        .agg(F.countDistinct("customer_id").alias("device_customer_count"))
    )

    device_stats = (
        df_devices.join(cust_per_device, "device_id", "left")
        .withColumn("device_customer_count", F.coalesce(F.col("device_customer_count"), F.lit(1)))
        .withColumn("device_txn_count", F.coalesce(F.col("transaction_count"), F.lit(0)))
        .withColumn("device_fraud_count", F.coalesce(F.col("fraud_count"), F.lit(0)))
        .withColumn(
            "device_fraud_rate",
            F.when(F.col("device_txn_count") > 0, F.col("device_fraud_count") / F.col("device_txn_count"))
            .otherwise(0.0),
        )
        .withColumn(
            "device_risk_score",
            F.when(F.col("device_customer_count") > 3, 0.85)
            .when(F.col("device_customer_count") > 1, 0.40)
            .otherwise(0.05),
        )
        .select(
            "device_id",
            "device_type",
            "operating_system",
            "device_customer_count",
            "device_txn_count",
            "device_fraud_rate",
            "device_risk_score",
        )
    )

    # 2. Historical customer-device mapping for new-device detection
    # A customer-device record is known if present in df_customer_devices
    known_cust_devices = (
        df_customer_devices.select("customer_id", "device_id")
        .distinct()
        .withColumn("known_device_flag", F.lit(True))
    )

    # 3. Join with transactions
    enriched = (
        df_transactions.join(device_stats, "device_id", "left")
        .join(known_cust_devices, ["customer_id", "device_id"], "left")
        .withColumn(
            "is_new_device",
            F.when(F.col("known_device_flag").isNull(), True).otherwise(False),
        )
        .drop("known_device_flag")
    )

    # Fill defaults for unmapped devices
    enriched = (
        enriched.fillna(0.0, subset=["device_fraud_rate", "device_risk_score"])
        .fillna(1, subset=["device_customer_count", "device_txn_count"])
    )

    logger.info("Device Intelligence features successfully computed.")
    return enriched
