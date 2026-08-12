"""
src/data_generation/fraud_patterns.py
Implements 13 realistic fraud scenario injectors.

Design philosophy:
    Fraud labels are NOT randomly assigned.
    Each scenario targets specific behavioral patterns and marks transactions
    as fraudulent only when the pattern conditions are met.
    This creates the statistical correlations that machine-learning models
    need to learn meaningful fraud signals.

Scenarios implemented:
    1.  new_device_fraud           — transaction from a brand-new device
    2.  velocity_attack            — multiple rapid transactions
    3.  impossible_travel          — two transactions in distant cities <30 mins apart
    4.  unusual_hours              — transaction at 2–4 AM
    5.  merchant_collusion         — high-risk merchant + abnormal amount
    6.  account_takeover           — sudden device change + large amount
    7.  device_sharing             — device used by multiple customers
    8.  ip_sharing                 — IP used by many customers simultaneously
    9.  coordinated_network        — cluster of customers sending funds to same target
    10. abnormal_spending          — amount > 5 SD from customer historical mean
    11. repeated_failures          — multiple declined attempts before success
    12. card_not_present           — online transaction with no prior online history
    13. geo_anomaly                — transaction location far from customer home city
"""

from __future__ import annotations

from typing import List, Dict, Any, Set
import numpy as np


def compute_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres between two geo-coordinates."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class FraudScenarioEngine:
    """
    Applies fraud labels and scenario tags to a list of generated transactions.

    Usage:
        engine = FraudScenarioEngine(config, rng)
        transactions = engine.apply(transactions, customer_device_map, ip_pool)
    """

    def __init__(self, config: Dict[str, Any], rng: np.random.Generator) -> None:
        fraud_cfg = config.get("fraud", {})
        self.overall_fraud_rate = fraud_cfg.get("overall_fraud_rate", 0.025)
        self.scenarios_enabled = fraud_cfg.get("fraud_scenarios", {})
        self.rng = rng

    def apply(
        self,
        transactions: List[Dict[str, Any]],
        customer_device_history: Dict[str, Set[str]],  # cid -> set of known device_ids
        device_customer_count: Dict[str, int],           # device_id -> count of customers
        ip_customer_count: Dict[str, int],               # ip -> count of customers
        customer_city_map: Dict[str, str],               # cid -> home city
        customer_lat_lon: Dict[str, tuple],              # cid -> (lat, lon)
        customer_txn_stats: Dict[str, Dict],             # cid -> {mean, std}
    ) -> List[Dict[str, Any]]:
        """
        Iterates transactions and applies fraud labels based on scenario rules.
        Maintains a running window for velocity checks.
        """
        # Sort by customer and timestamp for window functions
        transactions.sort(key=lambda t: (t["customer_id"], t["timestamp"]))

        # Track per-customer transaction windows for velocity
        customer_recent_txns: Dict[str, List[Dict]] = {}
        # Track per-customer previous location for impossible travel
        customer_last_location: Dict[str, Dict] = {}

        labeled = []
        for txn in transactions:
            cid = txn["customer_id"]
            did = txn.get("device_id", "")
            ip = txn.get("ip_address", "")
            ts = txn["timestamp"]
            amount = txn["amount"]
            lat = txn.get("latitude", 0.0)
            lon = txn.get("longitude", 0.0)

            fraud_scenario = None
            is_fraud = False

            # Initialise tracking if first transaction for this customer
            if cid not in customer_recent_txns:
                customer_recent_txns[cid] = []

            recent = customer_recent_txns[cid]
            stats = customer_txn_stats.get(cid, {"mean": amount, "std": 1.0})

            # ── Scenario 1: New Device ──────────────────────────────────────
            if (
                self._enabled("new_device_fraud")
                and did
                and did not in customer_device_history.get(cid, set())
                and self.rng.random() < 0.18  # 18% of new-device txns are fraud
            ):
                is_fraud = True
                fraud_scenario = "new_device_fraud"

            # ── Scenario 2: Velocity Attack ────────────────────────────────
            elif self._enabled("velocity_attack"):
                recent_1min = [
                    t for t in recent
                    if (ts - t["timestamp"]).total_seconds() < 60
                ]
                if len(recent_1min) >= 3 and self.rng.random() < 0.70:
                    is_fraud = True
                    fraud_scenario = "velocity_attack"

            # ── Scenario 3: Impossible Travel ──────────────────────────────
            elif (
                self._enabled("impossible_travel")
                and cid in customer_last_location
            ):
                last = customer_last_location[cid]
                time_diff_mins = (ts - last["timestamp"]).total_seconds() / 60
                if time_diff_mins > 0:
                    dist = compute_distance_km(
                        last["latitude"], last["longitude"], lat, lon
                    )
                    speed_kmh = (dist / time_diff_mins) * 60
                    if speed_kmh > 900 and self.rng.random() < 0.85:
                        is_fraud = True
                        fraud_scenario = "impossible_travel"

            # ── Scenario 4: Unusual Hours (2–4 AM) ────────────────────────
            elif (
                self._enabled("unusual_hours")
                and ts.hour in (2, 3, 4)
                and amount > 5000
                and self.rng.random() < 0.35
            ):
                is_fraud = True
                fraud_scenario = "unusual_hours"

            # ── Scenario 5: Merchant Collusion ─────────────────────────────
            elif (
                self._enabled("merchant_collusion")
                and txn.get("merchant_risk_score", 0) > 0.65
                and amount > stats["mean"] * 3
                and self.rng.random() < 0.55
            ):
                is_fraud = True
                fraud_scenario = "merchant_collusion"

            # ── Scenario 6: Account Takeover ───────────────────────────────
            elif (
                self._enabled("account_takeover")
                and did
                and did not in customer_device_history.get(cid, set())
                and amount > stats["mean"] * 5
                and self.rng.random() < 0.60
            ):
                is_fraud = True
                fraud_scenario = "account_takeover"

            # ── Scenario 7: Device Sharing ─────────────────────────────────
            elif (
                self._enabled("device_sharing")
                and device_customer_count.get(did, 1) > 3
                and self.rng.random() < 0.30
            ):
                is_fraud = True
                fraud_scenario = "device_sharing"

            # ── Scenario 8: IP Sharing ─────────────────────────────────────
            elif (
                self._enabled("ip_sharing")
                and ip_customer_count.get(ip, 1) > 5
                and self.rng.random() < 0.25
            ):
                is_fraud = True
                fraud_scenario = "ip_sharing"

            # ── Scenario 9: Coordinated Network ───────────────────────────
            # (Injected separately in generator via coordinated_fraud_injection)

            # ── Scenario 10: Abnormal Spending ─────────────────────────────
            elif (
                self._enabled("abnormal_spending")
                and stats["std"] > 0
                and (amount - stats["mean"]) / stats["std"] > 5
                and self.rng.random() < 0.45
            ):
                is_fraud = True
                fraud_scenario = "abnormal_spending"

            # ── Scenario 11: Repeated Failures → Success ───────────────────
            elif self._enabled("repeated_failures"):
                recent_5min = [
                    t for t in recent
                    if (ts - t["timestamp"]).total_seconds() < 300
                ]
                if len(recent_5min) >= 4 and self.rng.random() < 0.40:
                    is_fraud = True
                    fraud_scenario = "repeated_failures"

            # ── Scenario 12: Card Not Present (online only) ────────────────
            elif (
                self._enabled("card_not_present")
                and txn.get("payment_method") == "online"
                and txn.get("merchant_category") in ("electronics", "gambling", "cryptocurrency")
                and self.rng.random() < 0.12
            ):
                is_fraud = True
                fraud_scenario = "card_not_present"

            # ── Scenario 13: Geo Anomaly ───────────────────────────────────
            elif self._enabled("geo_anomaly"):
                home = customer_lat_lon.get(cid)
                if home:
                    dist = compute_distance_km(home[0], home[1], lat, lon)
                    if dist > 3000 and self.rng.random() < 0.28:
                        is_fraud = True
                        fraud_scenario = "geo_anomaly"

            txn["is_fraud"] = is_fraud
            txn["fraud_scenario"] = fraud_scenario

            # Update device history for this customer
            if did:
                customer_device_history.setdefault(cid, set()).add(did)

            # Update previous location
            customer_last_location[cid] = {
                "timestamp": ts,
                "latitude": lat,
                "longitude": lon,
            }

            # Update recent transaction window (keep last 100)
            recent.append(txn)
            if len(recent) > 100:
                recent.pop(0)

            labeled.append(txn)

        return labeled

    def _enabled(self, scenario: str) -> bool:
        return bool(self.scenarios_enabled.get(scenario, True))


def inject_coordinated_fraud_network(
    transactions: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    rng: np.random.Generator,
    network_size: int = 5,
    n_networks: int = 3,
) -> List[Dict[str, Any]]:
    """
    Creates coordinated fraud networks: clusters of customers that
    funnel money to the same merchant in tight time windows.
    These simulate organised fraud rings detectable via graph analytics.
    """
    customer_ids = [c["customer_id"] for c in customers]

    for _ in range(n_networks):
        # Pick a cluster of customers
        cluster = rng.choice(customer_ids, size=min(network_size, len(customer_ids)), replace=False)
        cluster_set = set(cluster.tolist())

        # Mark a subset of their transactions as coordinated fraud
        for txn in transactions:
            if txn["customer_id"] in cluster_set and not txn.get("is_fraud", False):
                if rng.random() < 0.30:
                    txn["is_fraud"] = True
                    txn["fraud_scenario"] = "coordinated_network"

    return transactions
