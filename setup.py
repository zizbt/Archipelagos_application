"""
setup.py
========
One-click launcher for non-technical users.

What it does:
  1. Checks Python version
  2. Installs dependencies from requirements.txt (if needed)
  3. Runs preprocess.py if precomputed data is missing
  4. Launches app.py

Usage : python setup.py
(or double-click run.bat on Windows / run.command on Mac)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def step(msg):
    print(f"\n{'=' * 50}")
    print(f"  {msg}")
    print(f"{'=' * 50}")


def check_python():
    step("Checking Python version")
    v = sys.version_info
    if v < (3, 10):
        print(f"  WARNING: Python {v.major}.{v.minor} detected. Python 3.10+ recommended.")
    else:
        print(f"  OK: Python {v.major}.{v.minor}.{v.micro}")


def install_requirements():
    step("Installing dependencies (this may take a few minutes the first time)")
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("  WARNING: requirements.txt not found, skipping.")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"])
    print("  OK: dependencies installed.")


def precomputed_data_exists():
    heatmap_dir = ROOT / "data" / "precomputed" / "heatmaps"
    if not heatmap_dir.exists():
        return False
    return any(heatmap_dir.glob("*.parquet"))


def run_preprocess():
    step("Precomputing data (first run only -- this can take 20-60 min)")
    print("  Progress will be shown below. Please be patient.\n")
    subprocess.run([sys.executable, str(ROOT / "preprocess.py")])


def run_app():
    step("Starting Aegean Vessel Tracker")
    print("  Open your browser at: http://127.0.0.1:8050")
    print("  Press CTRL+C to stop the app.\n")
    subprocess.run([sys.executable, str(ROOT / "app.py")])


if __name__ == "__main__":
    check_python()
    install_requirements()

    if not precomputed_data_exists():
        run_preprocess()
    else:
        print("\nPrecomputed data found -- skipping preprocessing.")
        print("(Delete the data/precomputed folder to force a full recompute.)")

    run_app()
