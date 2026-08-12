"""
src/feature_engineering/velocity_features.py
PySpark Transaction Velocity Feature Generator.

Computes rolling window aggregations (1-hour, 24-hour, 7-day counts and amounts)
and time-since-last-transaction deltas using PySpark Window functions.
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def compute_velocity_features(df_transactions: DataFrame) -> DataFrame:
    """
    Compute velocity features using PySpark Window functions over epoch timestamps.

    Parameters
    ----------
    df_transactions : PySpark DataFrame
        Transactions DataFrame with 'customer_id', 'timestamp_sec', 'amount'.

    Returns
    -------
    PySpark DataFrame enriched with velocity feature columns.
    """
    logger.info("Computing PySpark Transaction Velocity features...")

    # Ensure timestamp_sec column exists
    if "timestamp_sec" not in df_transactions.columns:
        df = df_transactions.withColumn("timestamp_sec", F.col("timestamp").cast("long"))
    else:
        df = df_transactions

    # Windows for range-based time intervals (in seconds)
    # Note: rangeBetween includes the current row. Subtracting 1 or subtracting current row amount
    # gives historical window prior to or including current transaction.
    window_1h = (
        Window.partitionBy("customer_id")
        .orderBy("timestamp_sec")
        .rangeBetween(-3600, 0)
    )
    window_24h = (
        Window.partitionBy("customer_id")
        .orderBy("timestamp_sec")
        .rangeBetween(-86400, 0)
    )
    window_7d = (
        Window.partitionBy("customer_id")
        .orderBy("timestamp_sec")
        .rangeBetween(-604800, 0)
    )

    # Window for lag calculation (sequential order)
    window_customer_seq = (
        Window.partitionBy("customer_id").orderBy("timestamp_sec")
    )

    enriched_df = (
        df
        # 1-hour velocity
        .withColumn("txns_last_1h", F.count("transaction_id").over(window_1h) - 1)
        .withColumn("spend_last_1h", F.coalesce(F.sum("amount").over(window_1h), F.lit(0.0)))
        # 24-hour velocity
        .withColumn("txns_last_24h", F.count("transaction_id").over(window_24h) - 1)
        .withColumn("spend_last_24h", F.coalesce(F.sum("amount").over(window_24h), F.lit(0.0)))
        # 7-day velocity
        .withColumn("txns_last_7d", F.count("transaction_id").over(window_7d) - 1)
        .withColumn("spend_last_7d", F.coalesce(F.sum("amount").over(window_7d), F.lit(0.0)))
        # Time since last transaction (in seconds)
        .withColumn("prev_txn_sec", F.lag("timestamp_sec", 1).over(window_customer_seq))
        .withColumn(
            "seconds_since_last_txn",
            F.when(F.col("prev_txn_sec").isNull(), F.lit(999999))
            .otherwise(F.col("timestamp_sec") - F.col("prev_txn_sec")),
        )
        .drop("prev_txn_sec")
    )

    logger.info("Transaction Velocity features successfully computed.")
    return enriched_df
