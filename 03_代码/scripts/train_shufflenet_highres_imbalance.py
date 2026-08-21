#!/usr/bin/env python3
"""Train a frozen S1N imbalance strategy for WM-811K."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.imbalance_training import main


if __name__ == "__main__":
    main()
