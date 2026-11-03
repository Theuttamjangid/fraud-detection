"""
app.py  --  Streamlit dashboard for the Fraud Detection project.
Run locally:  streamlit run app.py
Tabs: Overview | Explore data | Model results | Try it yourself
"""
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "processed" / "clean_transactions.csv"
FEAT = ROOT / "data" / "processed" / "features.csv"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
TARGET = "isFraud"
RULE = ["amount_equals_old_balance", "amount_to_oldbalance"]

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------
# Data + model loading (cached)
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Preparing data (first run only)...")
def load_data():
    """Processed files na ho (jaise Streamlit Cloud pe) to pipeline ke 2 steps chala do."""
    if not (CLEAN.exists() and FEAT.exists()):
        for script in ["clean_data.py", "feature_engineering.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / script)],
                           check=True, cwd=ROOT / "src", capture_output=True)
    return pd.read_csv(CLEAN, parse_dates=["Transaction_Date"]), pd.read_csv(FEAT)


@st.cache_resource(show_spinner="Training model...")
def train(use_rule: bool):
    _, feats = load_data()
    drop = [c for c in feats.columns if c.startswith("post_")] + ([] if use_rule else RULE)
    X, y = feats.drop(columns=[TARGET] + drop), feats[TARGET]
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)],
        remainder="passthrough")
    model = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        class_weight="balanced", max_depth=4, learning_rate=0.05, max_iter=150, random_state=42))])
    model.fit(X, y)
    return model, list(X.columns)


clean, feats = load_data()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🛡️ Bank Transaction Fraud Detection")
st.caption("End-to-end analytics and machine learning project on an imbalanced fraud dataset.")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore data", "Model results", "Try it yourself"])

# ---------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------
with tab1:
    c = st.columns(4)
    c[0].metric("Transactions", f"{len(clean):,}")
    c[1].metric("Frauds", int(clean[TARGET].sum()), f"{clean[TARGET].mean() * 100:.2f}%", delta_color="off")
    c[2].metric("Fraud types", "2", "TRANSFER, CASH_OUT", delta_color="off")
    c[3].metric("Caught by one rule", "65 / 68")

    st.subheader("Key findings")
    st.markdown("""
- **Fraud is confined to `TRANSFER` (33) and `CASH_OUT` (35).** PAYMENT, CASH_IN and DEBIT contain none.
- **One pattern explains almost everything.** When the amount equals the sender's entire previous balance, the transaction is fraud in 65 of 65 cases. The data looks synthetic or rule-generated.
- **Post-transaction balances are not needed.** Removing the `post_*` features does not reduce performance.
- **ROC-AUC is misleading here.** Transaction type alone gives 0.89 ROC-AUC but only 0.03 PR-AUC, so PR-AUC is the main metric.
""")
    st.info("With only 68 frauds (14 in the test set), metrics are noisy and the dataset looks synthetic. "
            "Read scores as a demonstration of method, not real-world performance.")
    st.markdown("**Code:** [github.com/Theuttamjangid/fraud-detection](https://github.com/Theuttamjangid/fraud-detection)")

# ---------------------------------------------------------------
# Tab 2: Explore
# ---------------------------------------------------------------
with tab2:
    st.subheader("Fraud by transaction type")
    by_type = clean.groupby("type")[TARGET].agg(transactions="count", frauds="sum")
    by_type["fraud_rate_%"] = (by_type["frauds"] / by_type["transactions"] * 100).round(2)
    a, b = st.columns([1, 1])
    a.dataframe(by_type.sort_values("transactions", ascending=False), width="stretch")
    b.bar_chart(by_type["frauds"])

    st.subheader("Fraud rate by category")
    col = st.selectbox("Choose a column", ["type", "Acct_type", "Time_of_day", "step"])
    rate = clean.groupby(col)[TARGET].agg(rows="count", frauds="sum")
    rate["fraud_rate_%"] = (rate["frauds"] / rate["rows"] * 100).round(2)
    a, b = st.columns([1, 1])
    a.dataframe(rate, width="stretch")
    b.bar_chart(rate["fraud_rate_%"])

    st.subheader("Amount distribution (TRANSFER and CASH_OUT)")
    risky = clean[clean["type"].isin(["TRANSFER", "CASH_OUT"])]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, name, color in [(0, "Legit", "#4C72B0"), (1, "Fraud", "#C44E52")]:
        ax.hist(np.log10(risky.loc[risky[TARGET] == label, "amount"]), bins=40, alpha=.6,
                density=True, label=name, color=color)
    ax.set_xlabel("log10(amount)"); ax.set_ylabel("Density"); ax.legend()
    st.pyplot(fig)

    st.subheader("Balance behaviour")
    r = risky.copy()
    r["Sender emptied"] = ((r["newbalanceOrig"] == 0) & (r["oldbalanceOrg"] > 0))
    r["Amount equals old balance"] = (r["amount"] == r["oldbalanceOrg"])
    r["Destination balance zero"] = (r["oldbalanceDest"] == 0) & (r["newbalanceDest"] == 0)
    pat = (r.groupby(TARGET)[["Sender emptied", "Amount equals old balance", "Destination balance zero"]]
           .mean().mul(100).round(1).rename(index={0: "Legit (%)", 1: "Fraud (%)"}))
    st.dataframe(pat, width="stretch")

# ---------------------------------------------------------------
# Tab 3: Model results
# ---------------------------------------------------------------
def show_table(path, title):
    if path.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(path), width="stretch", hide_index=True)


def show_fig(name, caption):
    if (FIG / name).exists():
        st.image(str(FIG / name), caption=caption, width="stretch")


with tab3:
    st.markdown("Two versions: **with_rule** (all decision-time features) and **no_rule** "
                "(rule-revealing features removed, the more realistic estimate). "
                "Test set: 2,020 rows, 14 frauds.")
    show_table(TAB / "leakage_check.csv", "Leakage and signal check (5-fold CV x3)")
    show_fig("09_leakage_check.png", "PR-AUC drops sharply without the rule features; ROC-AUC barely moves.")
    v = st.radio("Version", ["no_rule", "with_rule"], horizontal=True)
    show_table(TAB / f"test_results_{v}.csv", f"Test results ({v})")
    show_table(TAB / f"threshold_report_{v}.csv", f"Threshold report ({v})")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(f"10_pr_curves_{v}.png", "Precision-recall curves (test set)")
    with c2:
        show_fig(f"12_importance_{v}.png", "Permutation importance")
    show_fig(f"11_confusion_{v}.png", "Confusion matrices at default and tuned threshold")

# ---------------------------------------------------------------
# Tab 4: Try it
# ---------------------------------------------------------------
with tab4:
    st.markdown("Enter a transaction and see the model's fraud probability. "
                "The demo model is trained on all rows of the dataset (the evaluation numbers on the "
                "previous tab come from a held-out test set).")
    version = st.radio("Model version", ["without rule features (realistic)", "with rule features"],
                       horizontal=True)
    model, cols = train(version.startswith("with rule"))

    with st.form("txn"):
        c1, c2, c3 = st.columns(3)
        ttype = c1.selectbox("Transaction type", sorted(clean["type"].unique()), index=4)
        amount = c1.number_input("Amount", min_value=1.0, value=150000.0, step=1000.0)
        old_org = c2.number_input("Sender old balance", min_value=0.0, value=150000.0, step=1000.0)
        old_dest = c2.number_input("Receiver old balance", min_value=0.0, value=0.0, step=1000.0)
        login = c3.number_input("Unusual login count", min_value=0, value=1, step=1)
        step = c3.selectbox("Step", sorted(clean["step"].unique()))
        acct = c1.selectbox("Account type", sorted(clean["Acct_type"].unique()))
        tod = c2.selectbox("Time of day", sorted(clean["Time_of_day"].unique()))
        branch = c3.selectbox("Branch", sorted(clean["branch"].unique()))
        date = c1.date_input("Date", value=pd.Timestamp("2018-01-10"))
        go = st.form_submit_button("Predict")

    if go:
        d = pd.Timestamp(date)
        row = {
            "step": step, "type": ttype, "branch": branch, "amount": amount,
            "oldbalanceOrg": old_org, "oldbalanceDest": old_dest, "unusuallogin": login,
            "Acct_type": acct, "Time_of_day": tod, "log_amount": np.log1p(amount),
            "is_risky_type": int(ttype in ["TRANSFER", "CASH_OUT"]),
            "origin_balance_zero": int(old_org == 0),
            "amount_to_oldbalance": amount / old_org if old_org > 0 else 0,
            "amount_equals_old_balance": int(amount == old_org and old_org > 0),
            "dest_old_balance_zero": int(old_dest == 0),
            "weekday": d.day_of_week, "is_weekend": int(d.day_of_week >= 5), "day_of_month": d.day,
        }
        X_new = pd.DataFrame([row])[cols]
        p = float(model.predict_proba(X_new)[0, 1])
        st.metric("Fraud probability", f"{p * 100:.1f}%")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= 0.5:
            st.error("Flagged as likely FRAUD (probability above 0.5).")
        else:
            st.success("Looks legitimate (probability below 0.5).")
        if row["amount_equals_old_balance"]:
            st.warning("Rule triggered: the amount equals the sender's entire balance.")
        if not row["is_risky_type"]:
            st.caption("Only TRANSFER and CASH_OUT transactions contain fraud in this dataset.")