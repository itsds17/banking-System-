"""
src/data_generation/merchants.py
Generates synthetic merchant records.

Business context:
    Merchants vary in category and risk profile. Some categories
    (gambling, cryptocurrency, money_transfer) carry inherently higher
    fraud risk and are weighted accordingly in the fraud model.
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from faker import Faker

fake = Faker("en_IN")


# High-risk categories have elevated base risk scores
_CATEGORY_RISK: Dict[str, float] = {
    "gambling": 0.75,
    "cryptocurrency": 0.70,
    "money_transfer": 0.60,
    "online_marketplace": 0.35,
    "electronics": 0.30,
    "atm_withdrawal": 0.25,
    "fuel": 0.15,
    "retail": 0.12,
    "dining": 0.10,
    "groceries": 0.08,
    "utilities": 0.05,
    "healthcare": 0.05,
    "entertainment": 0.12,
    "travel": 0.20,
}


def generate_merchants(
    n: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generate n synthetic merchant records.

    Parameters
    ----------
    n : int
        Number of merchants to generate.
    config : dict
        Full data configuration dict.
    rng : np.random.Generator
        Seeded numpy random generator.

    Returns
    -------
    List[Dict[str, Any]]
        List of merchant dictionaries.
    """
    categories = list(_CATEGORY_RISK.keys())
    cities_cfg = config.get("geography", {}).get("cities", [])

    merchants: List[Dict[str, Any]] = []

    for i in range(n):
        merchant_id = f"MERCH{i+1:06d}"
        category = str(rng.choice(categories))

        # Base risk score from category + random noise
        base_risk = _CATEGORY_RISK.get(category, 0.15)
        risk_score = float(np.clip(rng.normal(base_risk, 0.08), 0.01, 0.99))

        # Pick a random city from config
        if cities_cfg:
            city_data = cities_cfg[int(rng.integers(0, len(cities_cfg)))]
            city = city_data["city"]
            state = city_data["state"]
            lat = city_data["lat"] + rng.normal(0, 0.05)
            lon = city_data["lon"] + rng.normal(0, 0.05)
        else:
            city, state = "Mumbai", "Maharashtra"
            lat, lon = 19.0760, 72.8777

        merchants.append({
            "merchant_id": merchant_id,
            "merchant_name": fake.company()[:100],
            "merchant_category": category,
            "city": city,
            "state": state,
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "merchant_risk_score": round(risk_score, 4),
        })

    return merchants
