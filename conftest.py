"""
conftest.py — pytest root configuration
Adds the project root to sys.path so all `src.*` imports work
without needing the package to be installed in editable mode.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for all test modules
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
