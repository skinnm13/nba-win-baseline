#!/usr/bin/env python3
"""Train baseline model (run from project root after fetch_data.py)."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    from src.train_baseline import main

    main()
