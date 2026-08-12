"""
src/data_ingestion/validation.py
Data Quality & Referential Integrity Validation Suite for PostgreSQL.

Validates row counts, primary key uniqueness, foreign key relationships,
non-null constraints, and numeric business bounds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any
from sqlalchemy import text, Engine

from src.data_ingestion.db_connector import get_db_engine

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Validates data integrity in the PostgreSQL `banking` schema.
    """

    def __init__(self, engine: Optional[Engine] = None, schema: str = "banking") -> None:
        self.engine = engine or get_db_engine()
        self.schema = schema

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Execute full validation suite and return report dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting Data Quality & Referential Integrity Validation")
        logger.info("=" * 60)

        results: Dict[str, Any] = {
            "passed": True,
            "row_counts": self.check_row_counts(),
            "orphan_checks": self.check_referential_integrity(),
            "null_checks": self.check_null_constraints(),
            "range_checks": self.check_value_ranges(),
            "unique_checks": self.check_primary_keys(),
        }

        # Determine overall status
        failed_orphans = any(cnt > 0 for cnt in results["orphan_checks"].values())
        failed_nulls = any(cnt > 0 for cnt in results["null_checks"].values())
        failed_ranges = any(cnt > 0 for cnt in results["range_checks"].values())
        failed_uniques = any(cnt > 0 for cnt in results["unique_checks"].values())

        if failed_orphans or failed_nulls or failed_ranges or failed_uniques:
            results["passed"] = False
            logger.error("Validation FAILED! Issues found in dataset.")
        else:
            logger.info("All Data Quality & Referential Integrity Checks PASSED!")

        return results

    def check_row_counts(self) -> Dict[str, int]:
        """Query row count for all tables in schema."""
        counts = {}
        tables = [
            "customers", "accounts", "merchants", "devices",
            "ip_addresses", "transactions", "customer_devices",
            "customer_ips", "loans"
        ]
        with self.engine.connect() as conn:
            for tbl in tables:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {self.schema}.{tbl};")).scalar()
                counts[tbl] = int(res or 0)
                logger.info("  %-25s : %d rows", tbl, counts[tbl])
        return counts

    def check_referential_integrity(self) -> Dict[str, int]:
        """
        Check for orphaned foreign key records. Returns count of orphaned records.
        """
        logger.info("Checking Foreign Key Referential Integrity...")
        queries = {
            "orphaned_accounts_customer": f"""
                SELECT COUNT(*) FROM {self.schema}.accounts a
                LEFT JOIN {self.schema}.customers c ON a.customer_id = c.customer_id
                WHERE a.customer_id IS NOT NULL AND c.customer_id IS NULL;
            """,
            "orphaned_txns_customer": f"""
                SELECT COUNT(*) FROM {self.schema}.transactions t
                LEFT JOIN {self.schema}.customers c ON t.customer_id = c.customer_id
                WHERE t.customer_id IS NOT NULL AND c.customer_id IS NULL;
            """,
            "orphaned_txns_account": f"""
                SELECT COUNT(*) FROM {self.schema}.transactions t
                LEFT JOIN {self.schema}.accounts a ON t.account_id = a.account_id
                WHERE t.account_id IS NOT NULL AND a.account_id IS NULL;
            """,
            "orphaned_txns_merchant": f"""
                SELECT COUNT(*) FROM {self.schema}.transactions t
                LEFT JOIN {self.schema}.merchants m ON t.merchant_id = m.merchant_id
                WHERE t.merchant_id IS NOT NULL AND m.merchant_id IS NULL;
            """,
            "orphaned_txns_device": f"""
                SELECT COUNT(*) FROM {self.schema}.transactions t
                LEFT JOIN {self.schema}.devices d ON t.device_id = d.device_id
                WHERE t.device_id IS NOT NULL AND d.device_id IS NULL;
            """,
            "orphaned_loans_customer": f"""
                SELECT COUNT(*) FROM {self.schema}.loans l
                LEFT JOIN {self.schema}.customers c ON l.customer_id = c.customer_id
                WHERE l.customer_id IS NOT NULL AND c.customer_id IS NULL;
            """,
            "orphaned_customer_devices": f"""
                SELECT COUNT(*) FROM {self.schema}.customer_devices cd
                LEFT JOIN {self.schema}.customers c ON cd.customer_id = c.customer_id
                WHERE cd.customer_id IS NOT NULL AND c.customer_id IS NULL;
            """,
        }

        orphan_counts = {}
        with self.engine.connect() as conn:
            for check_name, query in queries.items():
                cnt = int(conn.execute(text(query)).scalar() or 0)
                orphan_counts[check_name] = cnt
                if cnt > 0:
                    logger.error("  [FAIL] %s : %d orphans found!", check_name, cnt)
                else:
                    logger.info("  [PASS] %s : 0 orphans", check_name)
        return orphan_counts

    def check_null_constraints(self) -> Dict[str, int]:
        """Check for unexpected NULL values in key business columns."""
        logger.info("Checking NULL constraints...")
        queries = {
            "null_customer_ids": f"SELECT COUNT(*) FROM {self.schema}.customers WHERE customer_id IS NULL;",
            "null_transaction_ids": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE transaction_id IS NULL;",
            "null_transaction_amounts": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE amount IS NULL;",
            "null_transaction_timestamps": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE timestamp IS NULL;",
            "null_device_ids": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE device_id IS NULL;",
            "null_ip_addresses": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE ip_address IS NULL;",
        }
        null_counts = {}
        with self.engine.connect() as conn:
            for check_name, query in queries.items():
                cnt = int(conn.execute(text(query)).scalar() or 0)
                null_counts[check_name] = cnt
                if cnt > 0:
                    logger.error("  [FAIL] %s : %d nulls found!", check_name, cnt)
                else:
                    logger.info("  [PASS] %s : 0 nulls", check_name)
        return null_counts

    def check_value_ranges(self) -> Dict[str, int]:
        """Check numeric bounds and business range constraints."""
        logger.info("Checking Value Range Constraints...")
        queries = {
            "invalid_credit_scores": f"SELECT COUNT(*) FROM {self.schema}.customers WHERE credit_score < 300 OR credit_score > 850;",
            "negative_txn_amounts": f"SELECT COUNT(*) FROM {self.schema}.transactions WHERE amount <= 0;",
            "invalid_merchant_risk": f"SELECT COUNT(*) FROM {self.schema}.merchants WHERE merchant_risk_score < 0 OR merchant_risk_score > 1;",
            "negative_loan_amounts": f"SELECT COUNT(*) FROM {self.schema}.loans WHERE loan_amount <= 0;",
        }
        range_violations = {}
        with self.engine.connect() as conn:
            for check_name, query in queries.items():
                cnt = int(conn.execute(text(query)).scalar() or 0)
                range_violations[check_name] = cnt
                if cnt > 0:
                    logger.error("  [FAIL] %s : %d violations!", check_name, cnt)
                else:
                    logger.info("  [PASS] %s : 0 violations", check_name)
        return range_violations

    def check_primary_keys(self) -> Dict[str, int]:
        """Check primary key uniqueness across tables."""
        logger.info("Checking Primary Key Uniqueness...")
        queries = {
            "duplicate_customers": f"SELECT COUNT(customer_id) - COUNT(DISTINCT customer_id) FROM {self.schema}.customers;",
            "duplicate_accounts": f"SELECT COUNT(account_id) - COUNT(DISTINCT account_id) FROM {self.schema}.accounts;",
            "duplicate_transactions": f"SELECT COUNT(transaction_id) - COUNT(DISTINCT transaction_id) FROM {self.schema}.transactions;",
            "duplicate_merchants": f"SELECT COUNT(merchant_id) - COUNT(DISTINCT merchant_id) FROM {self.schema}.merchants;",
            "duplicate_devices": f"SELECT COUNT(device_id) - COUNT(DISTINCT device_id) FROM {self.schema}.devices;",
        }
        dup_counts = {}
        with self.engine.connect() as conn:
            for check_name, query in queries.items():
                cnt = int(conn.execute(text(query)).scalar() or 0)
                dup_counts[check_name] = cnt
                if cnt > 0:
                    logger.error("  [FAIL] %s : %d duplicates!", check_name, cnt)
                else:
                    logger.info("  [PASS] %s : 0 duplicates", check_name)
        return dup_counts
