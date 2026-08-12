"""
src/data_generation/transactions.py
Generates synthetic banking transactions with device and IP intelligence.

Business context:
    Transactions are the central entity for fraud detection.
    Each transaction carries: amount, timestamp, location, device, IP,
    merchant, and payment method. The fraud patterns module then applies
    correlated fraud labels based on behavioral signals in these fields.

Design notes:
    - Transactions are distributed across customers proportional to their
      spending tendency (derived from income).
    - Device and IP are assigned automatically from the pre-built pools —
      the user never enters these manually.
    - Timestamps follow realistic intra-day distributions (peak: 9am–8pm).
"""

from __future__ import annotations

import ipaddress
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set

import numpy as np


def generate_ip_pool(n: int, rng: np.random.Generator) -> List[str]:
    """Generate n realistic-looking IP addresses (private + public ranges)."""
    ips: List[str] = []
    for _ in range(n):
        # Mix of public-looking IPs
        ip = f"{rng.integers(10, 223)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
        ips.append(ip)
    return list(set(ips))  # deduplicate


def assign_ips_to_customers(
    customers: List[Dict[str, Any]],
    ip_pool: List[str],
    rng: np.random.Generator,
    shared_ip_fraction: float = 0.04,
) -> Dict[str, List[str]]:
    """
    Assign 1–3 primary IPs to each customer.
    A fraction of IPs are shared across multiple customers (fraud signal).

    Returns
    -------
    customer_to_ips : Dict[str, List[str]]
        Maps customer_id -> list of IPs they typically use.
    """
    n_ips = len(ip_pool)
    customer_to_ips: Dict[str, List[str]] = {}

    for customer in customers:
        cid = customer["customer_id"]
        n_assigned = int(rng.integers(1, 4))
        idxs = rng.choice(n_ips, size=n_assigned, replace=False)
        customer_to_ips[cid] = [ip_pool[int(i)] for i in idxs]

    # Inject shared IPs (ip_sharing fraud scenario)
    n_shared = max(1, int(n_ips * shared_ip_fraction))
    shared_ips = rng.choice(ip_pool, size=n_shared, replace=False).tolist()
    for ip in shared_ips:
        extra_customers = rng.choice(len(customers), size=int(rng.integers(5, 15)), replace=False)
        for ci in extra_customers:
            cid = customers[int(ci)]["customer_id"]
            if ip not in customer_to_ips[cid]:
                customer_to_ips[cid].append(ip)

    return customer_to_ips


def generate_transactions(
    n: int,
    customers: List[Dict[str, Any]],
    accounts: List[Dict[str, Any]],
    merchants: List[Dict[str, Any]],
    devices: List[Dict[str, Any]],
    ip_pool: List[str],
    customer_to_devices: Dict[str, List[str]],
    customer_to_ips: Dict[str, List[str]],
    config: Dict[str, Any],
    rng: np.random.Generator,
    reference_end_time: datetime,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Generate n transactions distributed across customers.

    Key design decisions:
    - Transaction count per customer weighted by income (higher income → more txns)
    - Timestamps follow hourly weights from config (peak vs off-peak)
    - Device and IP sampled from customer's known pools (with occasional new ones)
    - Location either matches customer's home city or a nearby/distant city
    - Amount drawn from a log-normal distribution per customer (realistic tail)

    Parameters
    ----------
    n : int
        Total number of transactions to generate.
    ... (other params documented via type hints)
    lookback_days : int
        How far back in time transactions span.
    """
    txn_cfg = config.get("transactions", {})
    amount_min = txn_cfg.get("amount_min", 0.50)
    amount_max = txn_cfg.get("amount_max", 50000.00)
    categories = txn_cfg.get("merchant_categories", ["retail"])
    payment_methods = txn_cfg.get("payment_methods", ["debit_card"])
    txn_types = txn_cfg.get("transaction_types", ["purchase"])
    peak_hours = txn_cfg.get("hour_weights", {}).get("peak", list(range(9, 21)))

    cities_cfg = config.get("geography", {}).get("cities", [])
    city_list = [c["city"] for c in cities_cfg] if cities_cfg else ["Mumbai"]
    state_list = [c["state"] for c in cities_cfg] if cities_cfg else ["Maharashtra"]
    lat_list = [c["lat"] for c in cities_cfg] if cities_cfg else [19.07]
    lon_list = [c["lon"] for c in cities_cfg] if cities_cfg else [72.87]

    # Build lookup maps
    customer_map: Dict[str, Dict] = {c["customer_id"]: c for c in customers}
    merchant_map: Dict[str, Dict] = {m["merchant_id"]: m for m in merchants}
    device_ids = [d["device_id"] for d in devices]

    # Customer account map: cid -> list of account_ids
    cust_account_map: Dict[str, List[str]] = {}
    for acc in accounts:
        cust_account_map.setdefault(acc["customer_id"], []).append(acc["account_id"])

    # Weight customers by income (more income → more transactions)
    incomes = np.array([c.get("income", 30000) for c in customers])
    weights = incomes / incomes.sum()
    customer_indices = rng.choice(len(customers), size=n, p=weights)

    merchant_ids = [m["merchant_id"] for m in merchants]
    merchant_risk = {m["merchant_id"]: m.get("merchant_risk_score", 0.1) for m in merchants}

    transactions: List[Dict[str, Any]] = []

    # Start time = end_time - lookback_days
    start_time = reference_end_time - timedelta(days=lookback_days)
    total_seconds = lookback_days * 86400

    for i in range(n):
        ci = int(customer_indices[i])
        customer = customers[ci]
        cid = customer["customer_id"]
        income = customer.get("income", 30000)

        # Timestamp: weighted toward peak hours
        raw_seconds = int(rng.integers(0, total_seconds))
        txn_time = start_time + timedelta(seconds=raw_seconds)

        # Shift ~70% of txns toward peak hours
        if rng.random() < 0.70:
            target_hour = int(rng.choice(peak_hours))
            txn_time = txn_time.replace(hour=target_hour, minute=int(rng.integers(0, 60)))

        # Amount: log-normal, calibrated to customer income
        mean_amount = income / 100
        amount = float(np.clip(
            rng.lognormal(np.log(max(mean_amount, 1)), 1.2),
            amount_min, amount_max
        ))

        # Account
        cust_accs = cust_account_map.get(cid, [])
        account_id = str(rng.choice(cust_accs)) if cust_accs else "ACC_UNKNOWN"

        # Merchant
        merchant_id = str(rng.choice(merchant_ids))
        merch = merchant_map[merchant_id]

        # Device — sample from customer's known devices (90%) or a new one (10%)
        known_devices = customer_to_devices.get(cid, [])
        if known_devices and rng.random() < 0.90:
            device_id = str(rng.choice(known_devices))
        else:
            device_id = str(rng.choice(device_ids))

        # IP address — sample from customer's known IPs (88%) or a new one (12%)
        known_ips = customer_to_ips.get(cid, [])
        if known_ips and rng.random() < 0.88:
            ip_address = str(rng.choice(known_ips))
        else:
            ip_address = str(rng.choice(ip_pool))

        # Location — usually customer's home city, occasionally another
        home_city_name = customer.get("city", "Mumbai")
        if rng.random() < 0.85:
            # Home city location
            try:
                idx = city_list.index(home_city_name)
            except ValueError:
                idx = 0
            lat = float(lat_list[idx]) + rng.normal(0, 0.1)
            lon = float(lon_list[idx]) + rng.normal(0, 0.1)
            city = city_list[idx]
            state = state_list[idx]
        else:
            # Different city (travel or anomaly)
            idx = int(rng.integers(0, len(city_list)))
            lat = float(lat_list[idx]) + rng.normal(0, 0.1)
            lon = float(lon_list[idx]) + rng.normal(0, 0.1)
            city = city_list[idx]
            state = state_list[idx]

        transactions.append({
            "transaction_id": f"TXN{i+1:09d}",
            "customer_id": cid,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "ip_address": ip_address,
            "amount": round(amount, 2),
            "timestamp": txn_time,  # datetime object; serialised in generator
            "transaction_type": str(rng.choice(txn_types)),
            "merchant_category": merch.get("merchant_category", "retail"),
            "merchant_risk_score": merchant_risk.get(merchant_id, 0.1),
            "city": city,
            "state": state,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "payment_method": str(rng.choice(payment_methods)),
            "is_fraud": False,           # overwritten by fraud engine
            "fraud_scenario": None,
        })

    return transactions


def build_customer_txn_stats(
    transactions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Pre-compute per-customer transaction statistics for the fraud engine.
    Returns dict: customer_id -> {mean, std}
    """
    from collections import defaultdict
    amounts_by_customer: Dict[str, List[float]] = defaultdict(list)
    for txn in transactions:
        amounts_by_customer[txn["customer_id"]].append(txn["amount"])

    stats: Dict[str, Dict[str, float]] = {}
    for cid, amounts in amounts_by_customer.items():
        arr = np.array(amounts)
        stats[cid] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 1.0,
        }
    return stats


def build_ip_customer_count(customer_to_ips: Dict[str, List[str]]) -> Dict[str, int]:
    """Count how many customers use each IP (for ip_sharing detection)."""
    from collections import Counter
    counter: Counter = Counter()
    for ips in customer_to_ips.values():
        for ip in ips:
            counter[ip] += 1
    return dict(counter)


def build_device_customer_count(customer_to_devices: Dict[str, List[str]]) -> Dict[str, int]:
    """Count how many customers use each device (for device_sharing detection)."""
    from collections import Counter
    counter: Counter = Counter()
    for devices in customer_to_devices.values():
        for did in devices:
            counter[did] += 1
    return dict(counter)
