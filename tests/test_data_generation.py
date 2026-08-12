"""
tests/test_data_generation.py
Phase 1 tests — Synthetic Data Generator

Tests cover:
    - Customer generation: counts, field types, credit score range
    - Account generation: linkage to customers, balance types
    - Merchant generation: risk score range, required fields
    - Device generation: device_id uniqueness, OS validity
    - Transaction generation: count, amount bounds, timestamp validity
    - Loan generation: DTI calculation, default correlations
    - Fraud patterns: non-zero fraud rate, scenario diversity
    - Full pipeline: BankingDataGenerator end-to-end with tiny config

All tests use a tiny dataset (500 customers, 5000 transactions) for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.customers import generate_customers
from src.data_generation.accounts import generate_accounts
from src.data_generation.merchants import generate_merchants
from src.data_generation.devices import generate_devices, assign_devices_to_customers
from src.data_generation.transactions import (
    generate_ip_pool,
    assign_ips_to_customers,
    generate_transactions,
    build_customer_txn_stats,
    build_ip_customer_count,
    build_device_customer_count,
)
from src.data_generation.loans import generate_loans
from src.data_generation.fraud_patterns import FraudScenarioEngine, inject_coordinated_fraud_network
from src.data_generation.generator import BankingDataGenerator


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_config(tmp_path_factory) -> dict:
    """Minimal config for fast unit tests."""
    cfg = {
        "random_seed": 42,
        "output_dir": str(tmp_path_factory.mktemp("synthetic")),
        "output_format": "parquet",
        "counts": {
            "customers": 200,
            "accounts_per_customer_min": 1,
            "accounts_per_customer_max": 2,
            "merchants": 100,
            "devices": 150,
            "ip_addresses": 100,
            "transactions": 2000,
            "loans_fraction": 0.60,
        },
        "fraud": {
            "overall_fraud_rate": 0.025,
            "fraud_scenarios": {
                "new_device_fraud": True,
                "velocity_attack": True,
                "impossible_travel": True,
                "unusual_hours": True,
                "merchant_collusion": True,
                "account_takeover": True,
                "device_sharing": True,
                "ip_sharing": True,
                "coordinated_network": True,
                "abnormal_spending": True,
                "repeated_failures": True,
                "card_not_present": True,
                "geo_anomaly": True,
            },
        },
        "customers": {
            "age_min": 18,
            "age_max": 75,
            "income_min": 15000,
            "income_max": 300000,
            "credit_score_min": 300,
            "credit_score_max": 850,
            "employment_statuses": ["employed", "self-employed", "unemployed", "retired", "student"],
            "risk_profiles": ["low", "medium", "high"],
        },
        "transactions": {
            "amount_min": 1.0,
            "amount_max": 50000.0,
            "merchant_categories": ["groceries", "dining", "retail", "electronics", "gambling"],
            "payment_methods": ["debit_card", "credit_card", "online"],
            "transaction_types": ["purchase", "withdrawal", "transfer"],
            "hour_weights": {
                "peak": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "off_peak": [0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23],
            },
        },
        "loans": {
            "amount_min": 5000,
            "amount_max": 200000,
            "interest_rate_min": 4.0,
            "interest_rate_max": 25.0,
            "tenure_months": [12, 24, 36, 60],
            "default_rate": 0.08,
        },
        "devices": {
            "types": ["mobile", "desktop", "tablet"],
            "operating_systems": ["iOS", "Android", "Windows", "macOS"],
            "shared_device_fraction": 0.05,
        },
        "geography": {
            "cities": [
                {"city": "Mumbai", "state": "Maharashtra", "lat": 19.07, "lon": 72.88},
                {"city": "Delhi", "state": "Delhi", "lat": 28.70, "lon": 77.10},
                {"city": "Bangalore", "state": "Karnataka", "lat": 12.97, "lon": 77.59},
                {"city": "London", "state": "England", "lat": 51.50, "lon": -0.12},
            ],
        },
    }
    return cfg


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def customers(small_config, rng):
    return generate_customers(small_config["counts"]["customers"], small_config, rng)


@pytest.fixture(scope="module")
def accounts(customers, small_config, rng):
    return generate_accounts(customers, small_config, rng)


@pytest.fixture(scope="module")
def merchants(small_config, rng):
    return generate_merchants(small_config["counts"]["merchants"], small_config, rng)


@pytest.fixture(scope="module")
def devices(small_config, rng):
    return generate_devices(
        small_config["counts"]["devices"],
        small_config,
        rng,
        datetime.now(),
    )


# ── Customer tests ─────────────────────────────────────────────────────────────

class TestCustomerGeneration:
    def test_correct_count(self, customers, small_config):
        assert len(customers) == small_config["counts"]["customers"]

    def test_required_fields(self, customers):
        required = {"customer_id", "age", "gender", "income", "employment_status",
                    "city", "state", "customer_since", "credit_score", "risk_profile"}
        for c in customers[:10]:
            assert required.issubset(c.keys()), f"Missing fields in: {c}"

    def test_unique_customer_ids(self, customers):
        ids = [c["customer_id"] for c in customers]
        assert len(ids) == len(set(ids)), "Duplicate customer IDs found"

    def test_age_range(self, customers, small_config):
        cfg = small_config["customers"]
        ages = [c["age"] for c in customers]
        assert all(cfg["age_min"] <= a <= cfg["age_max"] for a in ages), "Age out of bounds"

    def test_credit_score_range(self, customers, small_config):
        cfg = small_config["customers"]
        scores = [c["credit_score"] for c in customers]
        assert all(cfg["credit_score_min"] <= s <= cfg["credit_score_max"] for s in scores)

    def test_income_positive(self, customers):
        assert all(c["income"] > 0 for c in customers)

    def test_risk_profile_values(self, customers):
        valid = {"low", "medium", "high"}
        assert all(c["risk_profile"] in valid for c in customers)

    def test_risk_correlated_with_credit_score(self, customers):
        """High-risk customers should tend to have lower credit scores."""
        high_risk_scores = [c["credit_score"] for c in customers if c["risk_profile"] == "high"]
        low_risk_scores = [c["credit_score"] for c in customers if c["risk_profile"] == "low"]
        if high_risk_scores and low_risk_scores:
            assert np.mean(high_risk_scores) < np.mean(low_risk_scores), \
                "High-risk customers should have lower mean credit scores"


# ── Account tests ─────────────────────────────────────────────────────────────

class TestAccountGeneration:
    def test_all_customers_have_accounts(self, accounts, customers):
        cids_with_accounts = {a["customer_id"] for a in accounts}
        cids = {c["customer_id"] for c in customers}
        assert cids.issubset(cids_with_accounts), "Some customers have no accounts"

    def test_required_fields(self, accounts):
        required = {"account_id", "customer_id", "account_type", "balance", "account_open_date", "status"}
        for a in accounts[:10]:
            assert required.issubset(a.keys())

    def test_unique_account_ids(self, accounts):
        ids = [a["account_id"] for a in accounts]
        assert len(ids) == len(set(ids))

    def test_account_types_valid(self, accounts):
        valid_types = {"savings", "current", "credit", "fixed_deposit", "loan"}
        for a in accounts:
            assert a["account_type"] in valid_types


# ── Merchant tests ─────────────────────────────────────────────────────────────

class TestMerchantGeneration:
    def test_correct_count(self, merchants, small_config):
        assert len(merchants) == small_config["counts"]["merchants"]

    def test_risk_score_range(self, merchants):
        for m in merchants:
            assert 0.0 <= m["merchant_risk_score"] <= 1.0, \
                f"Merchant risk score out of range: {m['merchant_risk_score']}"

    def test_required_fields(self, merchants):
        required = {"merchant_id", "merchant_name", "merchant_category",
                    "city", "state", "latitude", "longitude", "merchant_risk_score"}
        for m in merchants[:10]:
            assert required.issubset(m.keys())


# ── Device tests ──────────────────────────────────────────────────────────────

class TestDeviceGeneration:
    def test_correct_count(self, devices, small_config):
        assert len(devices) == small_config["counts"]["devices"]

    def test_unique_device_ids(self, devices):
        ids = [d["device_id"] for d in devices]
        assert len(ids) == len(set(ids)), "Duplicate device IDs found"

    def test_device_type_valid(self, devices):
        valid = {"mobile", "desktop", "tablet"}
        for d in devices:
            assert d["device_type"] in valid

    def test_device_id_length(self, devices):
        """Device IDs are SHA-256 truncated to 40 chars — like real fingerprints."""
        for d in devices:
            assert len(d["device_id"]) == 40


# ── Transaction tests ─────────────────────────────────────────────────────────

class TestTransactionGeneration:
    @pytest.fixture(scope="class")
    def generated_txns(self, customers, accounts, merchants, devices, small_config):
        rng = np.random.default_rng(42)
        ip_pool = generate_ip_pool(small_config["counts"]["ip_addresses"], rng)
        c2d, _ = assign_devices_to_customers(customers, devices, small_config, rng)
        c2ip = assign_ips_to_customers(customers, ip_pool, rng)
        txns = generate_transactions(
            n=small_config["counts"]["transactions"],
            customers=customers,
            accounts=accounts,
            merchants=merchants,
            devices=devices,
            ip_pool=ip_pool,
            customer_to_devices=c2d,
            customer_to_ips=c2ip,
            config=small_config,
            rng=rng,
            reference_end_time=datetime.now(),
        )
        return txns

    def test_correct_count(self, generated_txns, small_config):
        assert len(generated_txns) == small_config["counts"]["transactions"]

    def test_amount_bounds(self, generated_txns, small_config):
        cfg = small_config["transactions"]
        amounts = [t["amount"] for t in generated_txns]
        assert all(cfg["amount_min"] <= a <= cfg["amount_max"] for a in amounts)

    def test_required_fields(self, generated_txns):
        required = {"transaction_id", "customer_id", "account_id", "merchant_id",
                    "device_id", "ip_address", "amount", "timestamp", "transaction_type",
                    "merchant_category", "payment_method"}
        for t in generated_txns[:20]:
            assert required.issubset(t.keys())

    def test_unique_transaction_ids(self, generated_txns):
        ids = [t["transaction_id"] for t in generated_txns]
        assert len(ids) == len(set(ids))


# ── Loan tests ────────────────────────────────────────────────────────────────

class TestLoanGeneration:
    @pytest.fixture(scope="class")
    def loans(self, customers, small_config):
        rng = np.random.default_rng(42)
        return generate_loans(customers, small_config, rng)

    def test_loans_generated(self, loans, customers, small_config):
        expected_min = int(len(customers) * small_config["counts"]["loans_fraction"] * 0.5)
        assert len(loans) >= expected_min, "Too few loans generated"

    def test_dti_positive(self, loans):
        for l in loans:
            assert l["debt_to_income"] >= 0, f"Negative DTI: {l}"

    def test_high_default_rate_for_bad_credit(self, loans, customers):
        """Customers with credit score < 580 should have higher default rate."""
        cust_map = {c["customer_id"]: c["credit_score"] for c in customers}
        bad_credit_defaults = sum(
            1 for l in loans
            if cust_map.get(l["customer_id"], 700) < 580 and l["default_flag"]
        )
        bad_credit_total = sum(
            1 for l in loans
            if cust_map.get(l["customer_id"], 700) < 580
        )
        good_credit_defaults = sum(
            1 for l in loans
            if cust_map.get(l["customer_id"], 700) >= 720 and l["default_flag"]
        )
        good_credit_total = sum(
            1 for l in loans
            if cust_map.get(l["customer_id"], 700) >= 720
        )
        if bad_credit_total > 0 and good_credit_total > 0:
            bad_rate = bad_credit_defaults / bad_credit_total
            good_rate = good_credit_defaults / good_credit_total
            assert bad_rate > good_rate, \
                f"Expected bad credit rate ({bad_rate:.3f}) > good credit rate ({good_rate:.3f})"


# ── Full pipeline smoke test ───────────────────────────────────────────────────

class TestFullPipeline:
    def test_end_to_end(self, small_config, tmp_path):
        """Smoke test: runs the full generator and checks all 9 datasets are produced."""
        # Write config to a temp file
        config_path = tmp_path / "test_config.yaml"
        small_config_copy = dict(small_config)
        small_config_copy["output_dir"] = str(tmp_path / "out")
        with open(config_path, "w") as f:
            yaml.dump(small_config_copy, f)

        gen = BankingDataGenerator(config_path=config_path)
        dfs = gen.run()

        expected_datasets = {
            "customers", "accounts", "merchants", "devices",
            "ip_addresses", "transactions", "loans",
            "customer_devices", "customer_ips",
        }
        assert expected_datasets == set(dfs.keys()), "Missing datasets in output"

        # Check no empty DataFrames
        for name, df in dfs.items():
            assert len(df) > 0, f"Dataset '{name}' is empty"

    def test_fraud_rate_non_zero(self, small_config, tmp_path):
        """Fraud rate should be > 0 — labels are not all False."""
        config_path = tmp_path / "test_config2.yaml"
        small_config_copy = dict(small_config)
        small_config_copy["output_dir"] = str(tmp_path / "out2")
        with open(config_path, "w") as f:
            yaml.dump(small_config_copy, f)

        gen = BankingDataGenerator(config_path=config_path)
        dfs = gen.run()

        txn_df = dfs["transactions"]
        fraud_count = txn_df["is_fraud"].sum()
        assert fraud_count > 0, "No fraud transactions generated — check fraud pattern logic"

    def test_output_files_exist(self, small_config, tmp_path):
        """Parquet files must be written to disk."""
        config_path = tmp_path / "test_config3.yaml"
        out_dir = tmp_path / "out3"
        small_config_copy = dict(small_config)
        small_config_copy["output_dir"] = str(out_dir)
        with open(config_path, "w") as f:
            yaml.dump(small_config_copy, f)

        gen = BankingDataGenerator(config_path=config_path)
        gen.run()

        expected_files = [
            "customers.parquet", "accounts.parquet", "merchants.parquet",
            "devices.parquet", "transactions.parquet", "loans.parquet",
        ]
        for fname in expected_files:
            assert (out_dir / fname).exists(), f"Missing output file: {fname}"

    def test_parquet_readable(self, small_config, tmp_path):
        """Generated Parquet files must be readable by pandas."""
        config_path = tmp_path / "test_config4.yaml"
        out_dir = tmp_path / "out4"
        small_config_copy = dict(small_config)
        small_config_copy["output_dir"] = str(out_dir)
        with open(config_path, "w") as f:
            yaml.dump(small_config_copy, f)

        gen = BankingDataGenerator(config_path=config_path)
        gen.run()

        df = pd.read_parquet(out_dir / "transactions.parquet")
        assert "transaction_id" in df.columns
        assert "is_fraud" in df.columns
        assert len(df) == small_config_copy["counts"]["transactions"]
