"""Shared paths used across the test suite."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENTIC_ROOT = PROJECT_ROOT / "agentic_detection"
HYPOTHESES_ROOT = AGENTIC_ROOT / "hypotheses"
CONFIG_PATH = AGENTIC_ROOT / "config" / "hypothesis.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
