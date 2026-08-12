"""
src/data_processing/spark_session.py
PySpark Session Manager tuned for 8 GB RAM local deployment.

Configures driver memory (2 GB), executor memory (1 GB), PyArrow optimization,
and local multi-threading execution.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from pyspark.sql import SparkSession

# Ensure PySpark worker processes on Windows use the current active Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger = logging.getLogger(__name__)

_spark_session: Optional[SparkSession] = None


def get_spark_session(
    app_name: str = "BankingDecisionIntelligence",
    master: str = "local[*]",
    driver_memory: str = "2g",
    executor_memory: str = "1g",
) -> SparkSession:
    """
    Get or create a memory-optimized SparkSession.
    """
    global _spark_session

    if _spark_session is not None:
        try:
            if not _spark_session.sparkContext._jsc.sc().isStopped():
                return _spark_session
        except Exception:
            pass

    logger.info("Initializing PySpark Session [App: %s, Memory: %s driver / %s executor]...",
                app_name, driver_memory, executor_memory)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")  # Disable web UI to save memory & ports
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC")
    )

    _spark_session = builder.getOrCreate()
    _spark_session.sparkContext.setLogLevel("WARN")
    logger.info("PySpark Session successfully initialized. Version: %s", _spark_session.version)
    return _spark_session


def stop_spark_session() -> None:
    """Stop the active SparkSession if running."""
    global _spark_session
    if _spark_session is not None:
        logger.info("Stopping active PySpark Session...")
        try:
            _spark_session.stop()
        except Exception as e:
            logger.debug("Spark session stop info: %s", e)
        _spark_session = None
        logger.info("PySpark Session stopped.")
