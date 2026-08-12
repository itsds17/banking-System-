"""
scripts/generate_data.py
CLI entry point for the synthetic banking data generator.

Usage examples:
    # Default run (uses config/data_config.yaml):
    python scripts/generate_data.py

    # Override counts on the command line:
    python scripts/generate_data.py --customers 1000 --transactions 50000

    # Custom config file:
    python scripts/generate_data.py --config config/data_config.yaml

    # Change output format:
    python scripts/generate_data.py --format csv

    # Fully custom small run (good for CI tests):
    python scripts/generate_data.py --customers 500 --transactions 5000 --format parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.generator import BankingDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic banking data for the Decision Intelligence Platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py
  python scripts/generate_data.py --customers 1000 --transactions 50000
  python scripts/generate_data.py --config config/data_config.yaml --format csv
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_config.yaml",
        help="Path to data configuration YAML file.",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=None,
        help="Override number of customers (default: from config).",
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=None,
        help="Override number of transactions (default: from config).",
    )
    parser.add_argument(
        "--merchants",
        type=int,
        default=None,
        help="Override number of merchants.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override number of devices.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv", "both"],
        default=None,
        help="Output format (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("generate_data")

    # Build override dict for any command-line overrides
    override_counts = {}
    if args.customers is not None:
        override_counts["customers"] = args.customers
    if args.transactions is not None:
        override_counts["transactions"] = args.transactions
    if args.merchants is not None:
        override_counts["merchants"] = args.merchants
    if args.devices is not None:
        override_counts["devices"] = args.devices

    logger.info("Starting data generation...")
    logger.info("Config: %s", args.config)
    if override_counts:
        logger.info("Overriding counts: %s", override_counts)

    generator = BankingDataGenerator(
        config_path=args.config,
        override_counts=override_counts if override_counts else None,
    )

    # Apply additional overrides
    if args.format is not None:
        generator.fmt = args.format
    if args.output_dir is not None:
        from pathlib import Path
        generator.output_dir = Path(args.output_dir)
        generator.output_dir.mkdir(parents=True, exist_ok=True)

    dfs = generator.run()
    logger.info("Data generation complete. %d datasets produced.", len(dfs))
    logger.info("Files saved to: %s", generator.output_dir.resolve())


if __name__ == "__main__":
    main()
