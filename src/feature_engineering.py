"""
feature_engineering.py  --  STEP 4
----------------------------------
Cleaned data se model ke liye naye features banana.

IMPORTANT RULE (leakage se bachne ke liye):
  Real life mein model transaction ke SAATH decision leta hai.
  Isliye features 2 group mein baante hain:
    - PRE  features : transaction se pehle/saath pata hote hain  (safe)
    - POST features : transaction ke BAAD pata chalte hain       (leakage risk)
  Saare POST features ka naam "post_" se shuru hota hai.
  STEP 5 (leakage_check.py) mein dono group ka model compare karenge.

Input : data/processed/clean_transactions.csv
Output: data/processed/features.csv
        reports/tables/feature_groups.csv
Run   : python src/feature_engineering.py
"""
import numpy as np
import pandas as pd
from config import CLEAN_FILE, FEATURE_FILE, TABLE_DIR, TARGET

df = pd.read_csv(CLEAN_FILE, parse_dates=["Transaction_Date"])

# ---------------------------------------------------------------
# A) PRE-transaction features (safe)
# ---------------------------------------------------------------

# 1) Amount ka log: amount bahut skewed hai (10 se 1 crore tak)
df["log_amount"] = np.log1p(df["amount"])

# 2) Risky type flag: fraud sirf TRANSFER / CASH_OUT mein mila
df["is_risky_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

# 3) Origin account ka balance pehle se zero tha?
df["origin_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)

# 4) Amount / purana balance  (0 se divide na ho, isliye np.where)
df["amount_to_oldbalance"] = np.where(
    df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0
)

# 5) Poora balance nikaal liya?  (amount == purana balance)
#    amount aur oldbalanceOrg dono transaction se PEHLE pata hote hain, isliye safe hai
df["amount_equals_old_balance"] = (
    (df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
).astype(int)

# 6) Destination account ka purana balance zero?
df["dest_old_balance_zero"] = (df["oldbalanceDest"] == 0).astype(int)

# 7) Date se time features
df["weekday"] = df["Transaction_Date"].dt.day_of_week      # 0 = Monday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["day_of_month"] = df["Transaction_Date"].dt.day

# ---------------------------------------------------------------
# B) POST-transaction features (leakage risk -> "post_" prefix)
# ---------------------------------------------------------------

# Raw newbalance columns: transaction hone ke baad ka balance
df = df.rename(columns={
    "newbalanceOrig": "post_newbalanceOrig",
    "newbalanceDest": "post_newbalanceDest",
})

# Origin account khaali ho gaya?
df["post_origin_emptied"] = (
    (df["post_newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
).astype(int)

# Balance errors: accounting ke hisaab se numbers match hote hain ya nahi
#   origin  : old - amount  should equal new
#   dest    : old + amount  should equal new
df["post_origin_error"] = df["oldbalanceOrg"] - df["amount"] - df["post_newbalanceOrig"]
df["post_dest_error"] = df["oldbalanceDest"] + df["amount"] - df["post_newbalanceDest"]

# ---------------------------------------------------------------
# C) Bekaar columns hatao
#    nameOrig : har row mein unique ID (koi pattern nahi)
#    nameDest : sirf 'M' = merchant, 'C' = customer. Ye type=PAYMENT se
#               exactly match karta hai, to naya information nahi deta
#    Transaction_Date : weekday / day_of_month mein convert ho chuka
# ---------------------------------------------------------------
df = df.drop(columns=["nameOrig", "nameDest", "Transaction_Date"])

# ---------------------------------------------------------------
# D) Save
# ---------------------------------------------------------------
features = [c for c in df.columns if c != TARGET]
groups = pd.DataFrame({
    "feature": features,
    "group": ["POST (leakage risk)" if c.startswith("post_") else "PRE (safe)" for c in features],
    "dtype": [str(df[c].dtype) for c in features],
})
groups.to_csv(TABLE_DIR / "feature_groups.csv", index=False)
df.to_csv(FEATURE_FILE, index=False)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("Final shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Infinite values:", int(np.isinf(df.select_dtypes('number')).sum().sum()))
print("\nFEATURE GROUPS")
print(groups.to_string(index=False))

print("\nNaye flags: fraud rate (TRANSFER + CASH_OUT rows mein)")
risky = df[df["is_risky_type"] == 1]
for flag in ["amount_equals_old_balance", "post_origin_emptied",
             "dest_old_balance_zero", "origin_balance_zero"]:
    rate = risky.groupby(flag)[TARGET].agg(["count", "sum", "mean"])
    rate["mean"] = (rate["mean"] * 100).round(2)
    rate.columns = ["rows", "frauds", "fraud_rate_%"]
    print(f"\n{flag}:")
    print(rate.to_string())

print("\nSaved:", FEATURE_FILE)