"""
src/data_generation/loans.py
Generates synthetic loan records for a subset of customers.

Business context:
    Loan data powers the credit-risk model (Phase 5).
    Features like debt-to-income ratio, delinquency history, and
    employment status predict probability of default (PD).
    Expected Loss = PD × LGD × EAD is explained in the README.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict, Any

import numpy as np


_LOAN_STATUSES = ["active", "paid_off", "defaulted", "in_arrears"]


def generate_loans(
    customers: List[Dict[str, Any]],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generate loan records for a fraction of customers.

    Loan default is correlated with:
    - High debt-to-income ratio
    - Low credit score
    - Unemployment
    - High interest rate
    - Previous delinquency history

    Parameters
    ----------
    customers : list
        Customer records (must include income, credit_score, employment_status).
    config : dict
        Full data config.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    List[Dict[str, Any]]
        Loan records.
    """
    counts_cfg = config.get("counts", {})
    loan_fraction = counts_cfg.get("loans_fraction", 0.60)

    loan_cfg = config.get("loans", {})
    amount_min = loan_cfg.get("amount_min", 1000)
    amount_max = loan_cfg.get("amount_max", 500000)
    rate_min = loan_cfg.get("interest_rate_min", 3.5)
    rate_max = loan_cfg.get("interest_rate_max", 28.0)
    tenures = loan_cfg.get("tenure_months", [12, 24, 36, 48, 60, 84, 120])
    default_rate = loan_cfg.get("default_rate", 0.08)

    loans: List[Dict[str, Any]] = []
    loan_counter = 1

    for customer in customers:
        if rng.random() > loan_fraction:
            continue

        cid = customer["customer_id"]
        income = customer.get("income", 30000)
        credit_score = customer.get("credit_score", 600)
        employment = customer.get("employment_status", "employed")
        customer_since = date.fromisoformat(customer["customer_since"])

        # Loan amount: bounded by income
        max_sensible = min(income * 5, amount_max)
        loan_amount = float(np.clip(
            rng.lognormal(np.log(income * 0.8), 0.8),
            amount_min,
            max_sensible
        ))

        # Interest rate inversely correlated with credit score
        credit_factor = 1 - (credit_score - 300) / (850 - 300)
        interest_rate = float(np.clip(
            rng.normal(rate_min + credit_factor * (rate_max - rate_min), 2.0),
            rate_min, rate_max
        ))

        tenure = int(rng.choice(tenures))
        monthly_income = round(income / 12, 2)
        monthly_payment = _emi(loan_amount, interest_rate / 12 / 100, tenure)
        dti = round(monthly_payment / max(monthly_income, 1), 4)

        # Delinquency history: more likely for high-risk customers
        if credit_score < 580:
            delinquency_history = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]))
        elif credit_score < 670:
            delinquency_history = int(rng.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.15, 0.05]))
        else:
            delinquency_history = int(rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))

        # Default probability — driven by risk factors
        p_default = _compute_default_probability(
            credit_score=credit_score,
            dti=dti,
            employment=employment,
            delinquency_history=delinquency_history,
            base_default_rate=default_rate,
            rng=rng,
        )

        default_flag = bool(rng.random() < p_default)

        if default_flag:
            loan_status = "defaulted"
        elif delinquency_history > 2:
            loan_status = str(rng.choice(["in_arrears", "active"], p=[0.6, 0.4]))
        else:
            loan_status = str(rng.choice(["active", "paid_off"], p=[0.7, 0.3]))

        # Loan start date: sometime after customer joined
        days_since = max(1, (date.today() - customer_since).days)
        days_offset = int(rng.integers(0, days_since))
        loan_start = customer_since + timedelta(days=days_offset)

        loans.append({
            "loan_id": f"LOAN{loan_counter:07d}",
            "customer_id": cid,
            "loan_amount": round(loan_amount, 2),
            "interest_rate": round(interest_rate, 2),
            "tenure_months": tenure,
            "monthly_income": monthly_income,
            "debt_to_income": dti,
            "employment_status": employment,
            "delinquency_history": delinquency_history,
            "loan_status": loan_status,
            "default_flag": default_flag,
            "loan_start_date": loan_start.isoformat(),
        })
        loan_counter += 1

    return loans


def _emi(principal: float, monthly_rate: float, months: int) -> float:
    """Calculate equated monthly instalment."""
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)


def _compute_default_probability(
    credit_score: int,
    dti: float,
    employment: str,
    delinquency_history: int,
    base_default_rate: float,
    rng: np.random.Generator,
) -> float:
    """
    Compute realistic default probability from risk factors.
    This is a rule-based approximation to create correlated labels.
    In production, the model learns these relationships from data.
    """
    p = base_default_rate

    # Credit score: strongest predictor
    if credit_score < 500:
        p *= 4.0
    elif credit_score < 580:
        p *= 2.5
    elif credit_score < 670:
        p *= 1.5
    elif credit_score >= 750:
        p *= 0.4

    # Debt-to-income ratio
    if dti > 0.5:
        p *= 3.0
    elif dti > 0.35:
        p *= 1.8
    elif dti < 0.15:
        p *= 0.6

    # Employment status
    employment_multipliers = {
        "unemployed": 3.5,
        "self-employed": 1.4,
        "student": 2.0,
        "employed": 1.0,
        "retired": 0.8,
    }
    p *= employment_multipliers.get(employment, 1.0)

    # Delinquency history
    p *= (1 + delinquency_history * 0.5)

    return float(np.clip(p, 0.005, 0.90))
