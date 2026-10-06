"""Train Uformer-B on FHDMi using the shared full-resolution DDP pipeline."""

import sys
from pathlib import Path


TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from train_demoire_uhdm import main


if __name__ == "__main__":
    main("FHDMi")
