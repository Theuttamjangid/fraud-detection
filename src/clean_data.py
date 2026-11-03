"""
clean_data.py  
"""
import pandas as pd
from config import RAW_FILE, CLEAN_FILE, TABLE_DIR, TARGET

df = pd.read_excel(RAW_FILE)
log = []   # (step, rows_removed, rows_left)


def record(step, before):
    """Har cleaning step ke baad log mein entry daalo."""
    log.append({"step": step, "rows_removed": before - len(df), "rows_left": len(df)})


log.append({"step": "0. raw data", "rows_removed": 0, "rows_left": len(df)})

# ---------------------------------------------------------------
# 1) Column names clean kro
# ---------------------------------------------------------------
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Acct type": "Acct_type",
    "Date of transaction": "Transaction_Date",
    "Time of day": "Time_of_day",
})

# ---------------------------------------------------------------
# 2) Target label: 'Not reviewed' wali rows hatao, baaki ko 0/1 banao
#    Kyun? In 2 rows ka sach pata nahi hai (fraud hai ya nahi),
#    to model ko inpe train/test nahi kar sakte.
# ---------------------------------------------------------------
before = len(df)
df[TARGET] = df[TARGET].astype(str).str.strip()
df = df[df[TARGET].isin(["Safe", "Fraud"])]
df[TARGET] = df[TARGET].map({"Safe": 0, "Fraud": 1}).astype(int)
record("1. 'Not reviewed' labels removed", before)

# ---------------------------------------------------------------
# 3) Text columns ke extra spaces hatao
# ---------------------------------------------------------------
for col in ["type", "branch", "nameOrig", "nameDest", "Acct_type", "Time_of_day"]:
    df[col] = df[col].astype("string").str.strip()

# ---------------------------------------------------------------
# 4) Missing values
#    inspect_data.py se pata chala: NaN wali koi bhi row fraud nahi hai,
#    isliye critical columns ki NaN rows hatana safe hai (fraud rows nahi khoyenge).
#    - Critical columns (type, amount, balances, names, date): row drop
#    - Acct_type, Time_of_day: 'Unknown' category bana do
# ---------------------------------------------------------------
before = len(df)
critical = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest",
            "Transaction_Date"]
df = df.dropna(subset=critical)
record("2. rows with NaN in critical columns removed", before)

df["Acct_type"] = df["Acct_type"].fillna("Unknown")
df["Time_of_day"] = df["Time_of_day"].fillna("Unknown")

# ---------------------------------------------------------------
# 5) Useless / leakage-risk / duplicate columns drop karo
#    isFraud - Copy : target ki hi copy (leakage!)
#    isFlaggedFraud : har row mein 0, koi information nahi
#    Column1        : sirf row-id
#    DayOfWeek, DayOfWeek(new) : Transaction_Date se dobara ban jayenge
# ---------------------------------------------------------------
drop_cols = ["isFraud - Copy", "isFlaggedFraud", "Column1",
             "DayOfWeek", "DayOfWeek(new)"]
df = df.drop(columns=drop_cols)

# ---------------------------------------------------------------
# 6) Sanity checks (galat values to nahi?)
# ---------------------------------------------------------------
before = len(df)
df = df[(df["amount"] > 0)]
for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
    df = df[df[col] >= 0]
record("3. invalid (negative/zero) values removed", before)

before = len(df)
df = df.drop_duplicates()
record("4. duplicate rows removed", before)

# ---------------------------------------------------------------
# 7) Data types theek karo aur save karo
# ---------------------------------------------------------------
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
df = df.reset_index(drop=True)

df.to_csv(CLEAN_FILE, index=False)
pd.DataFrame(log).to_csv(TABLE_DIR / "cleaning_log.csv", index=False)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("CLEANING LOG")
print(pd.DataFrame(log).to_string(index=False))
print("\nFinal shape:", df.shape)
print("Missing values left:", int(df.isna().sum().sum()))
print("\nColumns:", list(df.columns))
print("\nFraud count (0/1):")
print(df[TARGET].value_counts())
print(f"Fraud rate: {df[TARGET].mean() * 100:.2f}%")
print("\nSaved:", CLEAN_FILE)