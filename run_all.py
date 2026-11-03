"""Poori pipeline ek command se:  python run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    "inspect_data.py",
    "clean_data.py",
    "eda_analysis.py",
    "feature_engineering.py",
    "leakage_check.py",
    "train_model.py",
    "evaluate_model.py",
]

for script in STEPS:
    print(f"\n{'#' * 70}\n# {script}\n{'#' * 70}")
    subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=SRC)

print("\nPipeline complete. Charts: reports/figures | Tables: reports/tables")