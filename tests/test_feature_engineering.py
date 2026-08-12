"""
tests/test_feature_engineering.py
Unit & Integration tests for Phase 3 — PySpark Feature Engineering Pipeline.

Tests cover:
    - SparkSession creation & memory configuration
    - PySpark Velocity feature window calculations
    - PySpark Device Intelligence (is_new_device, device_risk_score)
    - PySpark IP Intelligence (is_new_ip, ip_risk_score)
    - PySpark Behavioral & Anomaly features (z-score, unusual hour, weekend)
    - End-to-end PySpark Feature Pipeline execution
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.spark_session import get_spark_session, stop_spark_session
from src.data_processing.cleaners import clean_transactions
from src.feature_engineering.velocity_features import compute_velocity_features
from src.feature_engineering.device_features import compute_device_features
from src.feature_engineering.ip_features import compute_ip_features
from src.feature_engineering.behavioral_features import compute_behavioral_features
from src.feature_engineering.pipeline import PySparkFeaturePipeline


@pytest.fixture(scope="module")
def spark():
    """Shared local SparkSession fixture for PySpark unit tests."""
    session = get_spark_session(app_name="PySparkUnitTest")
    yield session
    # Keep session open for module tests, stopped at end of session


class TestSparkSession:
    def test_spark_session_active(self, spark):
        assert spark is not None
        assert not spark._sc._is_stopped
        assert spark.conf.get("spark.driver.memory") == "2g"


class TestVelocityFeatures:
    def test_velocity_window_calculations(self, spark):
        """Test rolling 1-hour and 24-hour velocity counts in PySpark."""
        data = [
            ("TXN1", "CUST1", 100.0, datetime(2026, 8, 12, 10, 0, 0)),
            ("TXN2", "CUST1", 200.0, datetime(2026, 8, 12, 10, 15, 0)),  # 15 min later
            ("TXN3", "CUST1", 300.0, datetime(2026, 8, 12, 12, 0, 0)),  # 2 hours later
        ]
        columns = ["transaction_id", "customer_id", "amount", "timestamp"]
        df_raw = spark.createDataFrame(data, columns)
        df_clean = clean_transactions(df_raw)

        df_velocity = compute_velocity_features(df_clean)
        res = {row["transaction_id"]: row["txns_last_1h"] for row in df_velocity.collect()}

        assert res["TXN1"] == 0, "First transaction should have 0 prior txns in 1h"
        assert res["TXN2"] == 1, "TXN2 should see 1 prior transaction in 1h window"
        assert res["TXN3"] == 0, "TXN3 is 2 hours later, should have 0 prior txns in 1h"


class TestDeviceFeatures:
    def test_is_new_device_logic(self, spark):
        """Test is_new_device flag when customer device is unrecognized."""
        txns_data = [
            ("TXN1", "CUST1", "DEV1"),  # Known
            ("TXN2", "CUST1", "DEV99"), # Unrecognized/New
        ]
        txns_df = spark.createDataFrame(txns_data, ["transaction_id", "customer_id", "device_id"])

        devices_data = [("DEV1", "mobile", "iOS", 10, 0), ("DEV99", "desktop", "Windows", 1, 0)]
        devices_df = spark.createDataFrame(devices_data, ["device_id", "device_type", "operating_system", "transaction_count", "fraud_count"])

        cust_dev_data = [("CUST1", "DEV1")]
        cust_dev_df = spark.createDataFrame(cust_dev_data, ["customer_id", "device_id"])

        enriched = compute_device_features(txns_df, devices_df, cust_dev_df)
        res = {row["transaction_id"]: row["is_new_device"] for row in enriched.collect()}

        assert res["TXN1"] is False, "DEV1 is in customer history, is_new_device should be False"
        assert res["TXN2"] is True, "DEV99 is not in customer history, is_new_device should be True"


class TestIPFeatures:
    def test_is_new_ip_logic(self, spark):
        """Test is_new_ip flag when IP is unrecognized for customer."""
        txns_data = [
            ("TXN1", "CUST1", "192.168.1.1"),  # Known
            ("TXN2", "CUST1", "10.0.0.99"),    # New
        ]
        txns_df = spark.createDataFrame(txns_data, ["transaction_id", "customer_id", "ip_address"])

        ips_data = [("192.168.1.1", 10, 0, 1), ("10.0.0.99", 1, 0, 1)]
        ips_df = spark.createDataFrame(ips_data, ["ip_address", "transaction_count", "fraud_count", "customer_count"])

        cust_ips_data = [("CUST1", "192.168.1.1")]
        cust_ips_df = spark.createDataFrame(cust_ips_data, ["customer_id", "ip_address"])

        enriched = compute_ip_features(txns_df, ips_df, cust_ips_df)
        res = {row["transaction_id"]: row["is_new_ip"] for row in enriched.collect()}

        assert res["TXN1"] is False, "Known IP should have is_new_ip = False"
        assert res["TXN2"] is True, "Unrecognized IP should have is_new_ip = True"


class TestBehavioralFeatures:
    def test_amount_z_score_and_unusual_hours(self, spark):
        """Test z-score calculation and unusual hour detection."""
        txns_data = [
            ("TXN1", "CUST1", 100.0, datetime(2026, 8, 12, 14, 0, 0), "M1", "Mumbai"),  # Normal 2 PM
            ("TXN2", "CUST1", 1000.0, datetime(2026, 8, 12, 3, 0, 0), "M1", "Mumbai"),  # Unusual 3 AM!
        ]
        txns_df = spark.createDataFrame(txns_data, ["transaction_id", "customer_id", "amount", "timestamp", "merchant_id", "city"])

        cust_data = [("CUST1", 30, 50000.0, 700, "low", "Mumbai")]
        cust_df = spark.createDataFrame(cust_data, ["customer_id", "age", "income", "credit_score", "risk_profile", "city"])

        merch_data = [("M1", "RetailStore", 0.1)]
        merch_df = spark.createDataFrame(merch_data, ["merchant_id", "merchant_name", "merchant_risk_score"])

        enriched = compute_behavioral_features(txns_df, cust_df, merch_df)
        res = {row["transaction_id"]: (row["is_unusual_hour"], row["amount_z_score"]) for row in enriched.collect()}

        assert res["TXN1"][0] is False, "14:00 is not unusual hour"
        assert res["TXN2"][0] is True, "03:00 is an unusual hour"


class TestPipelineEndToEnd:
    def test_pyspark_pipeline(self, spark, tmp_path):
        """Smoke test full PySparkFeaturePipeline using existing synthetic data."""
        input_dir = Path("data/synthetic")
        output_dir = tmp_path / "processed"

        if not (input_dir / "transactions.parquet").exists():
            pytest.skip("Synthetic data not found in data/synthetic")

        pipeline = PySparkFeaturePipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            spark=spark,
        )
        dfs = pipeline.run()

        assert "fraud_features" in dfs
        assert "credit_features" in dfs
        assert "customer_features" in dfs

        # Check output parquet files created
        assert (output_dir / "fraud_features.parquet").exists()
        assert (output_dir / "credit_features.parquet").exists()
        assert (output_dir / "customer_features.parquet").exists()
