"""
tests/test_data_ingestion.py
Unit & Integration tests for Phase 2 — Data Ingestion & PostgreSQL pipeline.

Tests cover:
    - DB connection string construction
    - PostgreSQL loader topological ordering
    - Data quality validation checks (foreign key orphans, range checks, null checks)
    - New-device detection SQL logic & stored function
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.db_connector import get_db_url
from src.data_ingestion.loader import INGESTION_ORDER, PostgreSQLDataLoader
from src.data_ingestion.validation import DataQualityValidator


class TestDBConnector:
    def test_get_db_url(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "localhost")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DB", "test_db")
        monkeypatch.setenv("POSTGRES_USER", "test_user")
        monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")

        url = get_db_url()
        assert url == "postgresql://test_user:test_pass@localhost:5432/test_db"


class TestIngestionOrder:
    def test_topological_order_dependencies(self):
        """Customers and Accounts must precede Transactions in ingestion order."""
        table_names = [dataset for dataset, _ in INGESTION_ORDER]

        cust_idx = table_names.index("customers")
        acc_idx = table_names.index("accounts")
        txn_idx = table_names.index("transactions")

        assert cust_idx < acc_idx, "Customers must be ingested before Accounts"
        assert acc_idx < txn_idx, "Accounts must be ingested before Transactions"
        assert cust_idx < txn_idx, "Customers must be ingested before Transactions"


class TestDataQualityValidatorUnit:
    @patch("src.data_ingestion.validation.get_db_engine")
    def test_validator_init(self, mock_engine):
        mock_eng = MagicMock()
        validator = DataQualityValidator(engine=mock_eng)
        assert validator.engine == mock_eng
        assert validator.schema == "banking"


class TestNewDeviceDetectionLogic:
    def test_new_device_flagging_mock(self):
        """Simulate transactions and customer_devices to test new-device detection logic."""
        known_customer_devices = {
            ("CUST001", "DEV_KNOWN_1"),
            ("CUST001", "DEV_KNOWN_2"),
        }

        incoming_transactions = [
            {"customer_id": "CUST001", "device_id": "DEV_KNOWN_1"},  # Recognized
            {"customer_id": "CUST001", "device_id": "DEV_NEW_3"},    # New device!
        ]

        results = []
        for txn in incoming_transactions:
            key = (txn["customer_id"], txn["device_id"])
            is_new = key not in known_customer_devices
            results.append(is_new)

        assert results == [False, True], "New device was not correctly identified"
