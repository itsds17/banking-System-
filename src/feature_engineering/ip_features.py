"""
src/feature_engineering/ip_features.py
PySpark IP Intelligence & Network Risk Feature Generator.

Calculates:
- is_new_ip: boolean flag for unrecognized IP address for customer
- ip_txn_count: global transaction volume per IP
- ip_customer_count: shared customer count per IP
- ip_fraud_rate: historical fraud rate on IP
- ip_risk_score: composite IP risk signal
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def compute_ip_features(
    df_transactions: DataFrame,
    df_ip_addresses: DataFrame,
    df_customer_ips: DataFrame,
) -> DataFrame:
    """
    Compute IP Intelligence features and join onto Transactions DataFrame.
    """
    logger.info("Computing PySpark IP Intelligence features...")

    # 1. IP summary features
    cust_per_ip = (
        df_customer_ips.groupBy("ip_address")
        .agg(F.countDistinct("customer_id").alias("ip_customer_count"))
    )

    ip_stats = (
        df_ip_addresses.join(cust_per_ip, "ip_address", "left")
        .withColumn("ip_customer_count", F.coalesce(F.col("ip_customer_count"), F.lit(1)))
        .withColumn("ip_txn_count", F.coalesce(F.col("transaction_count"), F.lit(0)))
        .withColumn("ip_fraud_count", F.coalesce(F.col("fraud_count"), F.lit(0)))
        .withColumn(
            "ip_fraud_rate",
            F.when(F.col("ip_txn_count") > 0, F.col("ip_fraud_count") / F.col("ip_txn_count"))
            .otherwise(0.0),
        )
        .withColumn(
            "ip_risk_score",
            F.when(F.col("ip_customer_count") > 5, 0.90)
            .when(F.col("ip_customer_count") > 2, 0.50)
            .otherwise(0.05),
        )
        .select(
            "ip_address",
            "ip_customer_count",
            "ip_txn_count",
            "ip_fraud_rate",
            "ip_risk_score",
        )
    )

    # 2. Historical customer-IP mapping for new-IP detection
    known_cust_ips = (
        df_customer_ips.select("customer_id", "ip_address")
        .distinct()
        .withColumn("known_ip_flag", F.lit(True))
    )

    # 3. Join with transactions
    enriched = (
        df_transactions.join(ip_stats, "ip_address", "left")
        .join(known_cust_ips, ["customer_id", "ip_address"], "left")
        .withColumn(
            "is_new_ip",
            F.when(F.col("known_ip_flag").isNull(), True).otherwise(False),
        )
        .drop("known_ip_flag")
    )

    # Fill defaults
    enriched = (
        enriched.fillna(0.0, subset=["ip_fraud_rate", "ip_risk_score"])
        .fillna(1, subset=["ip_customer_count", "ip_txn_count"])
    )

    logger.info("IP Intelligence features successfully computed.")
    return enriched
