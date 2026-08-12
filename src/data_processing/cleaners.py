"""
src/data_processing/cleaners.py
PySpark Data Processing & Data Cleaning Pipeline.

Provides clean, modular PySpark DataFrame transformations:
- Timestamp parsing and epoch conversion
- Null handling & default fill
- Data type enforcement
- Duplicate removal
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType

logger = logging.getLogger(__name__)


def clean_transactions(df: DataFrame) -> DataFrame:
    """
    Clean and standardize PySpark transactions DataFrame.
    Casts amounts, parses timestamps into TimestampType and epoch seconds for windowing.
    """
    logger.info("Cleaning transactions DataFrame (%d columns)...", len(df.columns))

    cleaned_df = (
        df.withColumn("amount", F.col("amount").cast(DoubleType()))
        .withColumn("merchant_risk_score", F.col("merchant_risk_score").cast(DoubleType()))
        .withColumn("latitude", F.col("latitude").cast(DoubleType()))
        .withColumn("longitude", F.col("longitude").cast(DoubleType()))
        .withColumn("is_fraud", F.col("is_fraud").cast("boolean"))
    )

    # Convert timestamp to TimestampType if string/ISO
    if "timestamp" in df.columns:
        cleaned_df = cleaned_df.withColumn(
            "timestamp", F.to_timestamp(F.col("timestamp"))
        )
        # Epoch seconds column for window rangeBetween operations
        cleaned_df = cleaned_df.withColumn(
            "timestamp_sec", F.col("timestamp").cast("long")
        )

    # Deduplicate transaction IDs if any
    cleaned_df = cleaned_df.dropDuplicates(["transaction_id"])
    return cleaned_df


def clean_customers(df: DataFrame) -> DataFrame:
    """Clean and cast customer fields."""
    return (
        df.withColumn("age", F.col("age").cast(IntegerType()))
        .withColumn("income", F.col("income").cast(DoubleType()))
        .withColumn("credit_score", F.col("credit_score").cast(IntegerType()))
        .dropDuplicates(["customer_id"])
    )


def clean_devices(df: DataFrame) -> DataFrame:
    """Clean devices DataFrame."""
    return (
        df.withColumn("transaction_count", F.col("transaction_count").cast(IntegerType()))
        .withColumn("fraud_count", F.col("fraud_count").cast(IntegerType()))
        .dropDuplicates(["device_id"])
    )


def clean_ip_addresses(df: DataFrame) -> DataFrame:
    """Clean IP addresses DataFrame."""
    return (
        df.withColumn("transaction_count", F.col("transaction_count").cast(IntegerType()))
        .withColumn("fraud_count", F.col("fraud_count").cast(IntegerType()))
        .withColumn("customer_count", F.col("customer_count").cast(IntegerType()))
        .dropDuplicates(["ip_address"])
    )


def clean_loans(df: DataFrame) -> DataFrame:
    """Clean loans DataFrame."""
    return (
        df.withColumn("loan_amount", F.col("loan_amount").cast(DoubleType()))
        .withColumn("interest_rate", F.col("interest_rate").cast(DoubleType()))
        .withColumn("tenure_months", F.col("tenure_months").cast(IntegerType()))
        .withColumn("debt_to_income", F.col("debt_to_income").cast(DoubleType()))
        .withColumn("default_flag", F.col("default_flag").cast("boolean"))
        .dropDuplicates(["loan_id"])
    )
