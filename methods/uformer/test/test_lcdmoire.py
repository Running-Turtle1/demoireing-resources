"""Evaluate Uformer-B on the paired LCDMoire validation split."""

import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_uhdm import main


if __name__ == "__main__":
    main("LCDMoire")
