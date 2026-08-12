"""
src/data_generation/customers.py
Generates synthetic customer records with realistic demographics.

Business context:
    Customers are the primary entity in the banking system.
    Their demographics (age, income, credit score, risk profile) drive
    lending decisions, fraud propensity, and segmentation downstream.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, timedelta
from typing import List, Dict, Any

import numpy as np
from faker import Faker

fake = Faker("en_IN")  # Indian locale for realistic names/addresses


def generate_customers(
    n: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generate n synthetic customer records.

    Parameters
    ----------
    n : int
        Number of customers to generate.
    config : dict
        Customer configuration block from data_config.yaml.
    rng : np.random.Generator
        Seeded numpy random generator for reproducibility.

    Returns
    -------
    List[Dict[str, Any]]
        List of customer dictionaries.
    """
    cities_cfg = config.get("geography", {}).get("cities", [])
    # Flatten to simple lists for sampling
    cities = [c["city"] for c in cities_cfg] if cities_cfg else ["Mumbai", "Delhi"]
    states = [c["state"] for c in cities_cfg] if cities_cfg else ["Maharashtra", "Delhi"]

    cust_cfg = config.get("customers", {})
    age_min = cust_cfg.get("age_min", 18)
    age_max = cust_cfg.get("age_max", 80)
    inc_min = cust_cfg.get("income_min", 15000)
    inc_max = cust_cfg.get("income_max", 500000)
    cs_min = cust_cfg.get("credit_score_min", 300)
    cs_max = cust_cfg.get("credit_score_max", 850)
    emp_statuses = cust_cfg.get("employment_statuses", ["employed"])
    risk_profiles = cust_cfg.get("risk_profiles", ["low", "medium", "high"])

    customers = []
    today = date.today()
    city_idx = rng.integers(0, len(cities), size=n)

    for i in range(n):
        customer_id = f"CUST{i+1:07d}"
        age = int(rng.integers(age_min, age_max + 1))

        # Income correlated with age (career progression) + noise
        income_factor = min((age - age_min) / (55 - age_min), 1.0)
        base_income = inc_min + income_factor * (inc_max - inc_min)
        income = float(np.clip(rng.normal(base_income, base_income * 0.3), inc_min, inc_max))

        # Credit score correlated with income and age — older/richer = better
        credit_score = int(np.clip(
            rng.normal(
                cs_min + ((income - inc_min) / (inc_max - inc_min)) * (cs_max - cs_min),
                60
            ),
            cs_min, cs_max
        ))

        # Risk profile based on credit score
        if credit_score >= 720:
            risk_profile = "low"
        elif credit_score >= 600:
            risk_profile = "medium"
        else:
            risk_profile = "high"

        # Customer since date (weighted toward recent years)
        years_back = int(rng.integers(0, min(age - 18 + 1, 20)))
        customer_since = today - timedelta(days=years_back * 365 + int(rng.integers(0, 365)))

        idx = int(city_idx[i])
        emp_probs = _employment_probs(age)
        employment_status = rng.choice(emp_statuses, p=_pad_probs(emp_probs, len(emp_statuses)))

        customers.append({
            "customer_id": customer_id,
            "age": age,
            "gender": rng.choice(["M", "F", "Other"], p=[0.49, 0.49, 0.02]),
            "income": round(income, 2),
            "employment_status": str(employment_status),
            "city": cities[idx],
            "state": states[idx],
            "customer_since": customer_since.isoformat(),
            "credit_score": credit_score,
            "risk_profile": risk_profile,
        })

    return customers


def _employment_probs(age: int) -> List[float]:
    """Return employment probability distribution based on age."""
    if age < 25:
        # young: mostly student/employed
        return [0.50, 0.10, 0.10, 0.02, 0.28]  # emp, self, unemp, retired, student
    elif age < 55:
        return [0.65, 0.18, 0.10, 0.02, 0.05]
    else:
        return [0.30, 0.12, 0.05, 0.50, 0.03]


def _pad_probs(probs: List[float], n: int) -> List[float]:
    """Pad or truncate probability list to length n and normalise."""
    if len(probs) >= n:
        p = probs[:n]
    else:
        p = probs + [0.0] * (n - len(probs))
    total = sum(p)
    return [x / total for x in p]
