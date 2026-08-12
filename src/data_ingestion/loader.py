"""
src/data_ingestion/loader.py
Data Ingestion Pipeline for PostgreSQL.

Loads generated Parquet datasets into the `banking` schema tables in strict
foreign key dependency order with batch processing and error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from sqlalchemy import text

from src.data_ingestion.db_connector import get_db_engine

logger = logging.getLogger(__name__)

# Topological loading order respecting Foreign Key constraints
INGESTION_ORDER = [
    ("customers", "banking.customers"),
    ("accounts", "banking.accounts"),
    ("merchants", "banking.merchants"),
    ("devices", "banking.devices"),
    ("ip_addresses", "banking.ip_addresses"),
    ("transactions", "banking.transactions"),
    ("customer_devices", "banking.customer_devices"),
    ("customer_ips", "banking.customer_ips"),
    ("loans", "banking.loans"),
]


class PostgreSQLDataLoader:
    """
    Ingests Parquet files from data/synthetic/ into PostgreSQL schema `banking`.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/synthetic",
        schema: str = "banking",
        batch_size: int = 5000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.schema = schema
        self.batch_size = batch_size
        self.engine = get_db_engine()

    def clear_schema_tables(self) -> None:
        """Truncate all banking tables in reverse foreign-key order."""
        logger.info("Clearing existing data from schema '%s'...", self.schema)
        reverse_order = [table for _, table in reversed(INGESTION_ORDER)]
        with self.engine.begin() as conn:
            for table_name in reverse_order:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))
        logger.info("All tables truncated successfully.")

    def load_table(self, dataset_name: str, target_table: str) -> int:
        """
        Load a single dataset Parquet file into target PostgreSQL table.
        """
        file_path = self.data_dir / f"{dataset_name}.parquet"
        if not file_path.exists():
            # Fallback to CSV if Parquet not found
            file_path = self.data_dir / f"{dataset_name}.csv"
            if not file_path.exists():
                logger.warning("Dataset file not found for '%s' in %s", dataset_name, self.data_dir)
                return 0
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        if df.empty:
            logger.warning("Dataset '%s' is empty. Skipping.", dataset_name)
            return 0

        # Type adjustments for Postgres compatibility
        if "created_at" not in df.columns and "id" not in df.columns:
            # Exclude id auto-increment column if present in dataframe but let DB auto-gen
            pass

        # Data cleanliness fixes
        if dataset_name == "transactions":
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif dataset_name in ("customers", "accounts", "loans"):
            date_cols = [c for c in df.columns if "date" in c or "since" in c]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col]).dt.date

        if dataset_name in ("customer_devices", "customer_ips"):
            # Exclude auto-increment 'id' column if dataframe generated it
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            for date_col in ("first_used", "last_used"):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

        if dataset_name == "ip_addresses":
            if "ip_id" in df.columns:
                df = df.drop(columns=["ip_id"])
            for date_col in ("first_seen", "last_seen"):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

        if dataset_name == "devices":
            for date_col in ("first_seen", "last_seen"):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

        # Write to PostgreSQL table
        table_basename = target_table.split(".")[-1]
        logger.info("Ingesting %d rows into %s ...", len(df), target_table)
        
        df.to_sql(
            name=table_basename,
            con=self.engine,
            schema=self.schema,
            if_exists="append",
            index=False,
            chunksize=self.batch_size,
            method="multi",
        )
        logger.info("Successfully loaded %d rows into %s.", len(df), target_table)
        return len(df)

    def load_all(self, truncate_first: bool = True) -> Dict[str, int]:
        """
        Execute ingestion for all datasets in topological order.
        """
        if truncate_first:
            self.clear_schema_tables()

        loaded_counts: Dict[str, int] = {}
        for dataset_name, target_table in INGESTION_ORDER:
            count = self.load_table(dataset_name, target_table)
            loaded_counts[dataset_name] = count

        logger.info("Full ingestion complete. Row summary: %s", loaded_counts)
        return loaded_counts
