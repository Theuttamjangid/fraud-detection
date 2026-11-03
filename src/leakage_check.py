"""
leakage_check.py  --  STEP 5
----------------------------
Sawal: model ka score asli hai ya "cheating" se aa raha hai?

Hum alag-alag feature sets ke saath same model chalate hain:
  1. ALL                : saare features (post_ wale bhi)
  2. PRE only           : post_ features hata ke (real-time mein jo pata hota hai)
  3. PRE - rule         : + amount_equals_old_balance / amount_to_oldbalance hata ke
  4. PRE - rule - bal   : + purane balance columns bhi hata ke
  5. Type only          : sirf is_risky_type (baseline)

Metrics (5-fold stratified CV, 3 baar repeat):
  PR-AUC  : imbalanced data ke liye asli metric
  ROC-AUC : dhyan rakho, ye is data mein bahut optimistic dikhata hai

Input : data/processed/features.csv
Output: reports/tables/leakage_check.csv, reports/figures/09_leakage_check.png
Run   : python src/leakage_check.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from config import FEATURE_FILE, FIG_DIR, TABLE_DIR, TARGET, RANDOM_STATE

df = pd.read_csv(FEATURE_FILE)
X, y = df.drop(columns=[TARGET]), df[TARGET]

# ---------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------
post = [c for c in X.columns if c.startswith("post_")]
rule = ["amount_equals_old_balance", "amount_to_oldbalance"]
balances = ["oldbalanceOrg", "oldbalanceDest", "origin_balance_zero", "dest_old_balance_zero"]

feature_sets = {
    "1. ALL features":              list(X.columns),
    "2. PRE only":                  [c for c in X.columns if c not in post],
    "3. PRE - rule features":       [c for c in X.columns if c not in post + rule],
    "4. PRE - rule - balances":     [c for c in X.columns if c not in post + rule + balances],
    "5. Type only (baseline)":      ["is_risky_type"],
}


def make_model(cols):
    """Preprocessing + model ek pipeline mein (CV ke andar leakage na ho)."""
    cat = X[cols].select_dtypes(include=["object", "string"]).columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                               sparse_output=False), cat)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", max_depth=4, learning_rate=0.05,
        max_iter=150, random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

rows = []
for name, cols in feature_sets.items():
    res = cross_validate(make_model(cols), X[cols], y, cv=cv,
                         scoring={"pr": "average_precision", "roc": "roc_auc"})
    rows.append({
        "feature_set": name,
        "n_features": len(cols),
        "PR_AUC_mean": res["test_pr"].mean(),
        "PR_AUC_std": res["test_pr"].std(),
        "ROC_AUC_mean": res["test_roc"].mean(),
        "ROC_AUC_std": res["test_roc"].std(),
    })
    print(f"{name:28s} PR-AUC {rows[-1]['PR_AUC_mean']:.3f} ± {rows[-1]['PR_AUC_std']:.3f}"
          f" | ROC-AUC {rows[-1]['ROC_AUC_mean']:.3f}")

out = pd.DataFrame(rows).round(4)
out.to_csv(TABLE_DIR / "leakage_check.csv", index=False)

# ---------------------------------------------------------------
# Chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(out))
w = 0.38
ax.bar(x - w / 2, out["PR_AUC_mean"], w, yerr=out["PR_AUC_std"], label="PR-AUC", color="#C44E52", capsize=3)
ax.bar(x + w / 2, out["ROC_AUC_mean"], w, yerr=out["ROC_AUC_std"], label="ROC-AUC", color="#4C72B0", capsize=3)
ax.axhline(y.mean(), color="gray", ls="--", lw=1, label=f"Random PR-AUC ({y.mean():.3f})")
ax.set_xticks(x)
ax.set_xticklabels([n.split(". ")[1] for n in out["feature_set"]], rotation=15)
ax.set_ylim(0, 1.05); ax.set_title("Leakage check: score by feature set (5-fold CV x3)")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(FIG_DIR / "09_leakage_check.png", dpi=130); plt.close()

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\nRandom guess PR-AUC (fraud rate): %.4f" % y.mean())
print("\nFull table saved:", TABLE_DIR / "leakage_check.csv")
print("Chart saved     :", FIG_DIR / "09_leakage_check.png")