"""
src/data_generation/generator.py
Main orchestrator for the synthetic banking data generator.

Orchestration order:
    1. Load config
    2. Generate customers
    3. Generate accounts (linked to customers)
    4. Generate merchants
    5. Generate devices
    6. Assign devices to customers (with shared-device fraud patterns)
    7. Generate IP pool
    8. Assign IPs to customers (with shared-IP fraud patterns)
    9. Generate transactions
    10. Compute per-customer transaction statistics (for fraud labelling)
    11. Apply fraud patterns (FraudScenarioEngine)
    12. Inject coordinated fraud networks
    13. Generate loans
    14. Produce customer-device and customer-IP relationship tables
    15. Save all datasets (Parquet + optional CSV)
    16. Print summary report

Usage:
    from src.data_generation import BankingDataGenerator
    gen = BankingDataGenerator("config/data_config.yaml")
    gen.run()
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.data_generation.accounts import generate_accounts
from src.data_generation.customers import generate_customers
from src.data_generation.devices import (
    assign_devices_to_customers,
    generate_devices,
)
from src.data_generation.fraud_patterns import (
    FraudScenarioEngine,
    inject_coordinated_fraud_network,
)
from src.data_generation.loans import generate_loans
from src.data_generation.merchants import generate_merchants
from src.data_generation.transactions import (
    assign_ips_to_customers,
    build_customer_txn_stats,
    build_device_customer_count,
    build_ip_customer_count,
    generate_ip_pool,
    generate_transactions,
)

logger = logging.getLogger(__name__)


class BankingDataGenerator:
    """
    Generates a complete synthetic banking dataset.

    Parameters
    ----------
    config_path : str | Path
        Path to data_config.yaml.
    override_counts : dict, optional
        Override any 'counts' section values (e.g., for small test runs).
    """

    def __init__(
        self,
        config_path: str | Path = "config/data_config.yaml",
        override_counts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        if override_counts:
            self.config.setdefault("counts", {}).update(override_counts)

        seed = self.config.get("random_seed", 42)
        self.rng = np.random.default_rng(seed)

        output_dir = self.config.get("output_dir", "data/synthetic")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fmt = self.config.get("output_format", "parquet")
        self.reference_time = datetime.now().replace(microsecond=0)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the full data generation pipeline.

        Returns
        -------
        Dict[str, pd.DataFrame]
            All generated datasets keyed by name.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Banking Data Generator — Starting")
        logger.info("Config: %s", self.config_path)
        logger.info("=" * 60)

        counts = self.config.get("counts", {})
        n_customers = counts.get("customers", 10000)
        n_merchants = counts.get("merchants", 2000)
        n_devices = counts.get("devices", 5000)
        n_ips = counts.get("ip_addresses", 3000)
        n_transactions = counts.get("transactions", 500000)

        # ── Step 1: Customers ─────────────────────────────────────────────────
        logger.info("[1/9] Generating %d customers ...", n_customers)
        customers = generate_customers(n_customers, self.config, self.rng)
        logger.info("      Done — %d customers", len(customers))

        # ── Step 2: Accounts ──────────────────────────────────────────────────
        logger.info("[2/9] Generating accounts ...")
        accounts = generate_accounts(customers, self.config, self.rng)
        logger.info("      Done — %d accounts", len(accounts))

        # ── Step 3: Merchants ─────────────────────────────────────────────────
        logger.info("[3/9] Generating %d merchants ...", n_merchants)
        merchants = generate_merchants(n_merchants, self.config, self.rng)
        logger.info("      Done — %d merchants", len(merchants))

        # ── Step 4: Devices ───────────────────────────────────────────────────
        logger.info("[4/9] Generating %d devices ...", n_devices)
        devices = generate_devices(n_devices, self.config, self.rng, self.reference_time)
        customer_to_devices, _ = assign_devices_to_customers(customers, devices, self.config, self.rng)
        device_customer_count = build_device_customer_count(customer_to_devices)
        logger.info("      Done — %d devices", len(devices))

        # ── Step 5: IP Addresses ──────────────────────────────────────────────
        logger.info("[5/9] Generating %d IP addresses ...", n_ips)
        ip_pool = generate_ip_pool(n_ips, self.rng)
        customer_to_ips = assign_ips_to_customers(customers, ip_pool, self.rng)
        ip_customer_count = build_ip_customer_count(customer_to_ips)
        logger.info("      Done — %d IPs", len(ip_pool))

        # ── Step 6: Transactions ──────────────────────────────────────────────
        logger.info("[6/9] Generating %d transactions ...", n_transactions)
        transactions = generate_transactions(
            n=n_transactions,
            customers=customers,
            accounts=accounts,
            merchants=merchants,
            devices=devices,
            ip_pool=ip_pool,
            customer_to_devices=customer_to_devices,
            customer_to_ips=customer_to_ips,
            config=self.config,
            rng=self.rng,
            reference_end_time=self.reference_time,
        )
        logger.info("      Done — %d transactions", len(transactions))

        # ── Step 7: Fraud Labelling ───────────────────────────────────────────
        logger.info("[7/9] Applying fraud patterns ...")
        customer_txn_stats = build_customer_txn_stats(transactions)

        customer_city_map = {c["customer_id"]: c["city"] for c in customers}
        customer_lat_lon = {}
        cities_cfg = self.config.get("geography", {}).get("cities", [])
        for c in customers:
            matching = [loc for loc in cities_cfg if loc["city"] == c["city"]]
            if matching:
                customer_lat_lon[c["customer_id"]] = (matching[0]["lat"], matching[0]["lon"])

        engine = FraudScenarioEngine(self.config, self.rng)
        customer_device_history: Dict[str, set] = {
            cid: set(devs) for cid, devs in customer_to_devices.items()
        }
        transactions = engine.apply(
            transactions=transactions,
            customer_device_history=customer_device_history,
            device_customer_count=device_customer_count,
            ip_customer_count=ip_customer_count,
            customer_city_map=customer_city_map,
            customer_lat_lon=customer_lat_lon,
            customer_txn_stats=customer_txn_stats,
        )
        transactions = inject_coordinated_fraud_network(
            transactions=transactions,
            customers=customers,
            rng=self.rng,
            network_size=8,
            n_networks=5,
        )
        fraud_count = sum(1 for t in transactions if t.get("is_fraud", False))
        fraud_rate = fraud_count / len(transactions) * 100
        logger.info("      Fraud rate: %.2f%% (%d / %d)", fraud_rate, fraud_count, len(transactions))

        # ── Step 8: Loans ─────────────────────────────────────────────────────
        logger.info("[8/9] Generating loans ...")
        loans = generate_loans(customers, self.config, self.rng)
        default_count = sum(1 for l in loans if l.get("default_flag", False))
        logger.info("      Done — %d loans, %d defaults (%.1f%%)", len(loans), default_count, default_count/max(len(loans),1)*100)

        # ── Step 9: Relationship tables ───────────────────────────────────────
        logger.info("[9/9] Building relationship tables ...")
        customer_devices_records = self._build_customer_devices_table(customer_to_devices)
        customer_ips_records = self._build_customer_ips_table(customer_to_ips)
        ip_records = self._build_ip_table(ip_pool, ip_customer_count)

        # ── Serialise timestamps ───────────────────────────────────────────────
        for txn in transactions:
            if isinstance(txn["timestamp"], datetime):
                txn["timestamp"] = txn["timestamp"].isoformat()

        # ── Convert to DataFrames ─────────────────────────────────────────────
        dfs: Dict[str, pd.DataFrame] = {
            "customers": pd.DataFrame(customers),
            "accounts": pd.DataFrame(accounts),
            "merchants": pd.DataFrame(merchants),
            "devices": pd.DataFrame(devices),
            "ip_addresses": pd.DataFrame(ip_records),
            "transactions": pd.DataFrame(transactions),
            "loans": pd.DataFrame(loans),
            "customer_devices": pd.DataFrame(customer_devices_records),
            "customer_ips": pd.DataFrame(customer_ips_records),
        }

        # ── Save to disk ──────────────────────────────────────────────────────
        self._save_datasets(dfs)

        elapsed = time.time() - t0
        self._print_summary(dfs, fraud_rate, elapsed)

        return dfs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_datasets(self, dfs: Dict[str, pd.DataFrame]) -> None:
        for name, df in dfs.items():
            if self.fmt in ("parquet", "both"):
                path = self.output_dir / f"{name}.parquet"
                df.to_parquet(path, index=False, engine="pyarrow")
                logger.debug("Saved %s -> %s", name, path)
            if self.fmt in ("csv", "both"):
                path = self.output_dir / f"{name}.csv"
                df.to_csv(path, index=False)
                logger.debug("Saved %s -> %s", name, path)

    def _build_customer_devices_table(
        self, customer_to_devices: Dict[str, list]
    ):
        records = []
        for cid, devs in customer_to_devices.items():
            for did in devs:
                records.append({
                    "customer_id": cid,
                    "device_id": did,
                    "first_used": self.reference_time.isoformat(),
                    "last_used": self.reference_time.isoformat(),
                    "use_count": 1,
                })
        return records

    def _build_customer_ips_table(
        self, customer_to_ips: Dict[str, list]
    ):
        records = []
        for cid, ips in customer_to_ips.items():
            for ip in ips:
                records.append({
                    "customer_id": cid,
                    "ip_address": ip,
                    "first_used": self.reference_time.isoformat(),
                    "last_used": self.reference_time.isoformat(),
                    "use_count": 1,
                })
        return records

    def _build_ip_table(
        self, ip_pool: list, ip_customer_count: Dict[str, int]
    ):
        return [
            {
                "ip_address": ip,
                "first_seen": self.reference_time.isoformat(),
                "last_seen": self.reference_time.isoformat(),
                "transaction_count": 0,
                "fraud_count": 0,
                "customer_count": ip_customer_count.get(ip, 1),
            }
            for ip in ip_pool
        ]

    def _print_summary(
        self,
        dfs: Dict[str, pd.DataFrame],
        fraud_rate: float,
        elapsed: float,
    ) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("DATA GENERATION COMPLETE")
        logger.info("=" * 60)
        for name, df in dfs.items():
            logger.info("  %-25s %10d rows", name, len(df))
        logger.info("  %-25s %9.2f%%", "Fraud rate", fraud_rate)
        logger.info("  %-25s %9.1fs", "Time elapsed", elapsed)
        logger.info("  Output: %s", self.output_dir.resolve())
        logger.info("=" * 60)
