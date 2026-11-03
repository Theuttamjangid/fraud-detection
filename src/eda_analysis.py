"""
eda_analysis.py  --  STEP 3  (Data Analytics part)
--------------------------------------------------
Cleaned data se fraud ke patterns dhundna. Charts reports/figures/ mein
aur summary tables reports/tables/ mein save hote hain.

Input : data/processed/clean_transactions.csv
Run   : python src/eda_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # chart window nahi kholega, sirf file save karega
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLEAN_FILE, FIG_DIR, TABLE_DIR, TARGET

sns.set_theme(style="whitegrid")
COLORS = {0: "#4C72B0", 1: "#C44E52"}   # legit = neela, fraud = laal

df = pd.read_csv(CLEAN_FILE, parse_dates=["Transaction_Date"])


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


# Fraud sirf TRANSFER aur CASH_OUT mein hai, isliye kai analysis sirf inpe karenge
risky = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

# ---------------------------------------------------------------
# 1) Class balance
# ---------------------------------------------------------------
counts = df[TARGET].value_counts().sort_index()
ax = counts.plot(kind="bar", color=[COLORS[0], COLORS[1]], figsize=(5, 4))
ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
for i, v in enumerate(counts):
    ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.2f}%)", ha="center", va="bottom")
ax.set_title("Class balance"); ax.set_ylabel("Transactions")
save("01_class_balance.png")

# ---------------------------------------------------------------
# 2) Transaction type: kitne transactions vs kitne fraud
# ---------------------------------------------------------------
by_type = df.groupby("type")[TARGET].agg(transactions="count", frauds="sum")
by_type["fraud_rate_%"] = (by_type["frauds"] / by_type["transactions"] * 100).round(2)
by_type = by_type.sort_values("transactions", ascending=False)
by_type.to_csv(TABLE_DIR / "eda_fraud_by_type.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
by_type["transactions"].plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("Transactions by type"); ax[0].tick_params(axis="x", rotation=30)
by_type["frauds"].plot(kind="bar", ax=ax[1], color="#C44E52")
ax[1].set_title("Fraud count by type"); ax[1].tick_params(axis="x", rotation=30)
save("02_type_vs_fraud.png")

# ---------------------------------------------------------------
# 3) Amount distribution (TRANSFER + CASH_OUT, log scale)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 4))
for label, grp in risky.groupby(TARGET):
    sns.kdeplot(np.log10(grp["amount"]), label=["Legit", "Fraud"][label],
                color=COLORS[label], fill=True, alpha=0.3, common_norm=False)
plt.xlabel("log10(amount)"); plt.title("Amount distribution (TRANSFER + CASH_OUT)")
plt.legend()
save("03_amount_distribution.png")

# ---------------------------------------------------------------
# 4) Balance behaviour: fraud mein account khaali hota hai?
# ---------------------------------------------------------------
risky["origin_emptied"] = ((risky["newbalanceOrig"] == 0) & (risky["oldbalanceOrg"] > 0)).astype(int)
risky["amount_equals_old_balance"] = (risky["amount"] == risky["oldbalanceOrg"]).astype(int)
risky["dest_balance_zero"] = ((risky["oldbalanceDest"] == 0) & (risky["newbalanceDest"] == 0)).astype(int)

pattern = (risky.groupby(TARGET)[["origin_emptied", "amount_equals_old_balance", "dest_balance_zero"]]
           .mean().mul(100).round(1).rename(index={0: "Legit", 1: "Fraud"}))
pattern.to_csv(TABLE_DIR / "eda_balance_patterns.csv")

pattern.T.plot(kind="bar", figsize=(8, 4.5), color=[COLORS[0], COLORS[1]])
plt.title("Balance patterns (% of TRANSFER + CASH_OUT)"); plt.ylabel("%")
plt.xticks(rotation=15)
save("04_balance_patterns.png")

# ---------------------------------------------------------------
# 5) unusuallogin: fraud rate by value
# ---------------------------------------------------------------
login = df.groupby("unusuallogin")[TARGET].agg(["count", "sum", "mean"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
login["sum"].plot(kind="bar", ax=ax[0], color="#C44E52")
ax[0].set_title("Fraud count by unusuallogin")
(login["mean"] * 100).plot(kind="bar", ax=ax[1], color="#DD8452")
ax[1].set_title("Fraud rate % by unusuallogin")
save("05_unusuallogin.png")

# ---------------------------------------------------------------
# 6) Time / account / step based patterns
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, col in zip(ax, ["step", "Time_of_day", "Acct_type"]):
    (df.groupby(col)[TARGET].mean() * 100).plot(kind="bar", ax=a, color="#55A868")
    a.axhline(df[TARGET].mean() * 100, color="red", ls="--", lw=1)
    a.set_title(f"Fraud rate % by {col}"); a.tick_params(axis="x", rotation=0)
dow = df.groupby(df["Transaction_Date"].dt.day_name())[TARGET].mean() * 100
dow = dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
dow.plot(kind="bar", ax=ax[3], color="#55A868")
ax[3].axhline(df[TARGET].mean() * 100, color="red", ls="--", lw=1)
ax[3].set_title("Fraud rate % by weekday"); ax[3].tick_params(axis="x", rotation=45)
save("06_time_patterns.png")

# ---------------------------------------------------------------
# 7) Top branches by fraud count
# ---------------------------------------------------------------
top_branch = df.groupby("branch")[TARGET].sum().sort_values(ascending=False).head(10)
top_branch.sort_values().plot(kind="barh", figsize=(7, 5), color="#C44E52")
plt.title("Top 10 branches by fraud count")
save("07_top_branches.png")

# ---------------------------------------------------------------
# 8) Correlation heatmap
# ---------------------------------------------------------------
num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
            "newbalanceDest", "unusuallogin", "step", TARGET]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation (numeric columns)")
save("08_correlation.png")

# ---------------------------------------------------------------
# Text summary (report mein likhne layak baatein)
# ---------------------------------------------------------------
print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"Total transactions : {len(df):,}")
print(f"Fraud              : {df[TARGET].sum()} ({df[TARGET].mean() * 100:.2f}%)")
print("\nFraud by transaction type:")
print(by_type.to_string())
print("\nTRANSFER + CASH_OUT mein amount (median):")
print(risky.groupby(TARGET)["amount"].median().rename(index={0: "Legit", 1: "Fraud"}).round(0).to_string())
print("\nBalance patterns (% of TRANSFER + CASH_OUT):")
print(pattern.to_string())
print("\nunusuallogin ka average (fraud vs legit):")
print(df.groupby(TARGET)["unusuallogin"].mean().rename(index={0: "Legit", 1: "Fraud"}).round(2).to_string())
print("\nStep 7 mein fraud rate: %.2f%%  |  Step 1-6 mein: %.2f%%" % (
    df[df["step"] == 7][TARGET].mean() * 100, df[df["step"] < 7][TARGET].mean() * 100))
print("\nCharts saved in:", FIG_DIR)
print("Tables saved in:", TABLE_DIR)