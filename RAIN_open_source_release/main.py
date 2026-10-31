"""Run the regression and classification experiments reported in the paper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_experiment(name: str) -> None:
    folder = ROOT / name
    subprocess.run([sys.executable, "main.py"], cwd=folder, check=True)


if __name__ == "__main__":
    run_experiment("regression")
    run_experiment("classification")
    print("\nRAIN regression and classification experiments completed successfully.")
