#!/usr/bin/env python3
"""Download NBA game data into ./data (run from project root)."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    from src.fetch_games import _cli

    _cli()
