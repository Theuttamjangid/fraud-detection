"""
config.py
---------
Project ke saare paths aur settings ek jagah.
Baaki sab scripts yahin se import karenge:  from config import RAW_FILE
Isse path badalna ho to sirf yahan badalna padega.
"""
from pathlib import Path

# Is file (src/config.py) se do level upar = project root (FRAUD-DETECTION/)
ROOT = Path(__file__).resolve().parent.parent

# ---------- Data paths ----------
RAW_FILE = ROOT / "data" / "raw" / "Fraud_Detection.xlsx"
CLEAN_FILE = ROOT / "data" / "processed" / "clean_transactions.csv"
FEATURE_FILE = ROOT / "data" / "processed" / "features.csv"

# ---------- Output folders ----------
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

# ---------- Settings ----------
TARGET = "isFraud"          # raw file mein target column ka naam
RANDOM_STATE = 42           # har baar same result aaye
N_SPLITS = 5                # cross-validation folds (fraud rows kam hain)

# Output folders agar exist nahi karte to bana do
for folder in (CLEAN_FILE.parent, FIG_DIR, TABLE_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)