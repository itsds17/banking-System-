"""
src/data_generation/accounts.py
Generates bank accounts linked to customers.

Business context:
    Each customer may hold multiple accounts (current, savings, credit, etc.).
    Account balance and status feed into the decision engine and credit-risk model.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict, Any

import numpy as np


_ACCOUNT_TYPES = ["savings", "current", "credit", "fixed_deposit", "loan"]
_ACCOUNT_TYPE_PROBS = [0.40, 0.30, 0.20, 0.07, 0.03]
_STATUSES = ["active", "active", "active", "active", "inactive", "frozen", "closed"]


def generate_accounts(
    customers: List[Dict[str, Any]],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generate bank accounts for all customers.

    Each customer gets between min and max accounts (from config).
    Account balances are correlated with customer income.
    """
    counts_cfg = config.get("counts", {})
    min_acc = counts_cfg.get("accounts_per_customer_min", 1)
    max_acc = counts_cfg.get("accounts_per_customer_max", 3)

    accounts: List[Dict[str, Any]] = []
    account_counter = 1

    for customer in customers:
        n_accounts = int(rng.integers(min_acc, max_acc + 1))
        income = customer.get("income", 30000)
        customer_since = date.fromisoformat(customer["customer_since"])

        for _ in range(n_accounts):
            account_id = f"ACC{account_counter:08d}"
            account_counter += 1

            # Account open date: on or after customer_since
            days_since = (date.today() - customer_since).days
            if days_since > 0:
                days_offset = int(rng.integers(0, days_since))
            else:
                days_offset = 0
            open_date = customer_since + timedelta(days=days_offset)

            account_type = rng.choice(_ACCOUNT_TYPES, p=_ACCOUNT_TYPE_PROBS)

            # Balance correlated with income and account type
            if account_type == "savings":
                balance = float(np.clip(rng.normal(income * 0.5, income * 0.3), 0, income * 5))
            elif account_type == "current":
                balance = float(np.clip(rng.normal(income * 0.3, income * 0.2), 0, income * 3))
            elif account_type == "credit":
                # Credit accounts can have negative balance (outstanding)
                balance = float(rng.normal(-income * 0.1, income * 0.15))
            elif account_type == "fixed_deposit":
                balance = float(np.clip(rng.normal(income * 2, income * 0.5), 10000, income * 20))
            else:
                balance = float(np.clip(rng.normal(income * 0.2, income * 0.1), 0, income * 2))

            # Mostly active accounts
            status = str(rng.choice(_STATUSES))

            accounts.append({
                "account_id": account_id,
                "customer_id": customer["customer_id"],
                "account_type": str(account_type),
                "balance": round(balance, 2),
                "account_open_date": open_date.isoformat(),
                "status": status,
            })

    return accounts
