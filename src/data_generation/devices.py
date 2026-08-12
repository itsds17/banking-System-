"""
src/data_generation/devices.py
Generates synthetic device records and customer-device relationships.

Business context:
    In real banking, the upstream bank system captures device fingerprint
    information (device type, OS, device ID) for each mobile/web transaction.
    This module simulates that upstream data. The device intelligence layer
    then uses this to detect new devices, shared devices, and device risk.

    IMPORTANT: This is a SIMULATION. We are not detecting real physical devices.
    The transaction simulator provides device information automatically, just
    as a real banking system would capture it without manual user input.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np


_DEVICE_TYPES = ["mobile", "desktop", "tablet"]
_DEVICE_TYPE_PROBS = [0.65, 0.25, 0.10]

_OS_BY_TYPE = {
    "mobile": [("iOS", 0.45), ("Android", 0.54), ("Other", 0.01)],
    "desktop": [("Windows", 0.70), ("macOS", 0.22), ("Linux", 0.08)],
    "tablet": [("iOS", 0.55), ("Android", 0.44), ("Other", 0.01)],
}


def generate_devices(
    n: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    reference_time: datetime,
) -> List[Dict[str, Any]]:
    """
    Generate n synthetic device records.

    Parameters
    ----------
    n : int
        Number of unique devices.
    config : dict
        Full data config.
    rng : np.random.Generator
        Seeded random generator.
    reference_time : datetime
        The end-of-dataset timestamp (used to compute first/last seen).

    Returns
    -------
    List[Dict[str, Any]]
        List of device dictionaries.
    """
    devices: List[Dict[str, Any]] = []

    for i in range(n):
        device_type = str(rng.choice(_DEVICE_TYPES, p=_DEVICE_TYPE_PROBS))
        os_options = _OS_BY_TYPE[device_type]
        os_names = [o[0] for o in os_options]
        os_probs = [o[1] for o in os_options]
        operating_system = str(rng.choice(os_names, p=os_probs))

        # Generate a deterministic-looking device ID (like a real fingerprint hash)
        device_id = _make_device_id(device_type, operating_system, i, rng)

        # Device first seen: up to 3 years ago
        days_ago = int(rng.integers(1, 1095))
        first_seen = reference_time - timedelta(days=days_ago)
        last_seen = first_seen + timedelta(days=int(rng.integers(0, days_ago)))

        devices.append({
            "device_id": device_id,
            "device_type": device_type,
            "operating_system": operating_system,
            "first_seen": first_seen.isoformat(),
            "last_seen": last_seen.isoformat(),
            "transaction_count": 0,  # updated during transaction generation
            "fraud_count": 0,
        })

    return devices


def assign_devices_to_customers(
    customers: List[Dict[str, Any]],
    devices: List[Dict[str, Any]],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Assign 1–3 primary devices to each customer.
    A small fraction of devices are shared across multiple customers
    to simulate device-sharing fraud scenarios.

    Returns
    -------
    customer_to_devices : Dict[str, List[str]]
        Maps customer_id -> list of device_ids they use.
    device_primary_customer : Dict[str, str]
        Maps device_id -> its primary customer_id.
    """
    dev_cfg = config.get("devices", {})
    shared_frac = dev_cfg.get("shared_device_fraction", 0.05)

    n_devices = len(devices)
    n_customers = len(customers)

    # Mark ~shared_frac of devices as shared (used by 2–4 customers)
    n_shared = max(1, int(n_devices * shared_frac))
    shared_indices = set(rng.choice(n_devices, size=n_shared, replace=False).tolist())

    device_ids = [d["device_id"] for d in devices]
    customer_to_devices: Dict[str, List[str]] = {}
    device_primary_customer: Dict[str, str] = {}

    # First pass: assign primary device(s) to each customer
    for i, customer in enumerate(customers):
        cid = customer["customer_id"]
        n_dev = int(rng.integers(1, 4))  # 1–3 devices per customer
        assigned: List[str] = []

        for _ in range(n_dev):
            idx = int(rng.integers(0, n_devices))
            did = device_ids[idx]
            if did not in assigned:
                assigned.append(did)
                if did not in device_primary_customer:
                    device_primary_customer[did] = cid

        customer_to_devices[cid] = assigned

    # Second pass: add shared devices to multiple customers
    for idx in shared_indices:
        did = device_ids[idx]
        # Pick 1–3 additional random customers to share this device
        extra = int(rng.integers(1, 4))
        extra_customers = rng.choice(n_customers, size=extra, replace=False)
        for ci in extra_customers:
            cid = customers[int(ci)]["customer_id"]
            if did not in customer_to_devices[cid]:
                customer_to_devices[cid].append(did)

    return customer_to_devices, device_primary_customer


def _make_device_id(device_type: str, os: str, index: int, rng: np.random.Generator) -> str:
    """Generate a realistic-looking device fingerprint ID."""
    raw = f"{device_type}-{os}-{index}-{rng.integers(0, 999999)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]
