"""
inspect_data.py  --  STEP 1
---------------------------
Raw data ko SIRF dekhna hai. Is script mein kuch change/save nahi hota.
Maksad: pata karna ki data mein kya-kya problems hain,
taaki agla step (clean_data.py) sahi likh sakein.

Run (project root se):   python src/inspect_data.py
"""
import pandas as pd
from config import RAW_FILE, TARGET

pd.set_option("display.max_columns", None)   # saare columns dikhao
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# 1) Data load karo
# ---------------------------------------------------------------
df = pd.read_excel(RAW_FILE)

print("=" * 70)
print("1) SHAPE")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

# ---------------------------------------------------------------
# 2) Columns aur unke data types
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2) COLUMNS & DTYPES")
print("=" * 70)
print(df.dtypes)

# ---------------------------------------------------------------
# 3) Pehli 5 rows
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3) FIRST 5 ROWS")
print("=" * 70)
print(df.head())

# ---------------------------------------------------------------
# 4) Missing values (sirf wahi columns jinme NaN hai)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("4) MISSING VALUES")
print("=" * 70)
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if len(missing) else "Koi missing value nahi")
print(f"\nKam-se-kam ek NaN wali rows: {df.isna().any(axis=1).sum()}")

# ---------------------------------------------------------------
# 5) Duplicates
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("5) DUPLICATES")
print("=" * 70)
print("Duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 6) Target column ki value_counts (yahan problem milegi!)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print(f"6) TARGET COLUMN: '{TARGET}'")
print("=" * 70)
print(df[TARGET].value_counts(dropna=False))

# ---------------------------------------------------------------
# 7) Suspicious / extra columns ka check
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("7) SUSPICIOUS COLUMNS")
print("=" * 70)
print("isFraud - Copy values :", df["isFraud - Copy"].value_counts(dropna=False).to_dict())
print("isFlaggedFraud values :", df["isFlaggedFraud"].value_counts(dropna=False).to_dict())
print("Column1 unique values :", df["Column1"].nunique(), "(row-id jaisa lag raha hai)")
print("DayOfWeek values      :", sorted(df["DayOfWeek"].dropna().unique().tolist()))
print("DayOfWeek(new) values :", df["DayOfWeek(new)"].dropna().unique().tolist())

# ---------------------------------------------------------------
# 8) Categorical columns ki unique values
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("8) CATEGORICAL COLUMNS")
print("=" * 70)
for col in ["type", "Acct type", "Time of day"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
print("\nbranch unique:", df["branch"].nunique())

# ---------------------------------------------------------------
# 9) Fraud kis 'type' mein hai
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("9) FRAUD BY TRANSACTION TYPE")
print("=" * 70)
print(df.groupby("type")[TARGET].apply(lambda s: (s == "Fraud").sum()).rename("fraud_count"))

# ---------------------------------------------------------------
# 10) Numeric columns ka summary
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("10) NUMERIC SUMMARY")
print("=" * 70)
print(df.describe().T)

print("\n" + "=" * 70)
print("INSPECTION DONE -> ab clean_data.py likhenge")
print("=" * 70)