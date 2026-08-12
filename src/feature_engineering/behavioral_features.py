"""
src/feature_engineering/behavioral_features.py
PySpark Behavioral Anomaly, Amount Deviation & Time-Based Feature Generator.

Calculates:
- amount_vs_cust_avg: ratio of transaction amount to customer historical mean
- amount_z_score: z-score of transaction amount against customer historical distribution
- is_unusual_hour: flag for transactions between 2 AM and 4 AM
- is_weekend: weekend transaction flag
- is_different_city: flag if transaction city differs from customer home city
- customer aggregate profiles (income, credit_score, employment)
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def compute_behavioral_features(
    df_transactions: DataFrame,
    df_customers: DataFrame,
    df_merchants: DataFrame,
) -> DataFrame:
    """
    Compute behavioral anomaly, amount deviation, and location/time features.
    """
    logger.info("Computing PySpark Behavioral & Anomaly features...")

    # 1. Customer spending aggregates
    cust_aggregates = (
        df_transactions.groupBy("customer_id")
        .agg(
            F.avg("amount").alias("cust_avg_amount"),
            F.stddev("amount").alias("cust_std_amount"),
            F.max("amount").alias("cust_max_amount"),
            F.count("transaction_id").alias("cust_total_txns"),
        )
        .withColumn("cust_std_amount", F.coalesce(F.col("cust_std_amount"), F.lit(1.0)))
    )

    # 2. Join customer profile & merchant profile
    cust_slim = df_customers.select(
        "customer_id",
        "age",
        "income",
        "credit_score",
        "risk_profile",
        F.col("city").alias("home_city"),
    )

    merch_slim = df_merchants.select(
        "merchant_id",
        "merchant_name",
        F.col("merchant_risk_score").alias("merchant_risk_score_lookup"),
    )

    enriched = (
        df_transactions.join(cust_aggregates, "customer_id", "left")
        .join(cust_slim, "customer_id", "left")
        .join(merch_slim, "merchant_id", "left")
    )

    # 3. Compute derived features
    enriched = (
        enriched
        # Amount deviation
        .withColumn(
            "amount_vs_cust_avg",
            F.when(F.col("cust_avg_amount") > 0, F.col("amount") / F.col("cust_avg_amount"))
            .otherwise(1.0),
        )
        .withColumn(
            "amount_z_score",
            F.when(
                F.col("cust_std_amount") > 0,
                (F.col("amount") - F.col("cust_avg_amount")) / F.col("cust_std_amount"),
            ).otherwise(0.0),
        )
        # Time-based & unusual hour flags
        .withColumn("txn_hour", F.hour("timestamp"))
        .withColumn("txn_day_of_week", F.dayofweek("timestamp"))
        .withColumn(
            "is_weekend",
            F.when(F.col("txn_day_of_week").isin(1, 7), True).otherwise(False),
        )
        .withColumn(
            "is_unusual_hour",
            F.when(F.col("txn_hour").isin(2, 3, 4), True).otherwise(False),
        )
        .withColumn(
            "is_off_peak",
            F.when(F.col("txn_hour").isin(0, 1, 2, 3, 4, 5, 6, 22, 23), True).otherwise(False),
        )
        # Location anomaly
        .withColumn(
            "is_different_city",
            F.when(
                (F.col("home_city").isNotNull()) & (F.col("city").isNotNull()) & (F.col("city") != F.col("home_city")),
                True,
            ).otherwise(False),
        )
        # Merchant risk
        .withColumn(
            "merchant_risk_score",
            F.coalesce(F.col("merchant_risk_score_lookup"), F.col("merchant_risk_score"), F.lit(0.1)),
        )
        .drop("merchant_risk_score_lookup")
    )

    logger.info("Behavioral & Anomaly features successfully computed.")
    return enriched
