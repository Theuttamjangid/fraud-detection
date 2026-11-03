"""
train_model.py  --  STEP 6
--------------------------
Models train karna aur compare karna. Ek hi run mein DO versions banate hain:

  with_rule : saare PRE features (rule features ke saath)  -> bahut easy, scores ~1.0
  no_rule   : amount_equals_old_balance / amount_to_oldbalance hata ke -> mushkil, realistic

Har version mein:
  1. Sirf PRE features (post_ columns real-time mein available nahi hote)
  2. 80% train / 20% test split (stratified, dono versions mein same rows)
  3. Train set pe 5-fold CV se 4 models compare (PR-AUC main metric)
  4. Models ko poore train set pe fit karke save karna
  5. Test set ko haath nahi lagate. Wo evaluate_model.py (STEP 7) mein use hoga

Input : data/processed/features.csv
Output: models/<version>/*.joblib (models + train/test sets)
        reports/tables/model_comparison_<version>.csv
Run   : python src/train_model.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import FEATURE_FILE, MODEL_DIR, TABLE_DIR, TARGET, RANDOM_STATE, N_SPLITS

RULE_FEATURES = ["amount_equals_old_balance", "amount_to_oldbalance"]
VERSIONS = {"with_rule": [], "no_rule": RULE_FEATURES}

df = pd.read_csv(FEATURE_FILE)
post_cols = [c for c in df.columns if c.startswith("post_")]
y_all = df[TARGET]

# zero_division=0: jab model koi fraud predict hi na kare (Dummy), warning na aaye
SCORING = {
    "pr": "average_precision",
    "roc": "roc_auc",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


def build_models(num_cols, cat_cols):
    def pre(scale):
        """Text columns -> one-hot. Linear model ke liye numbers scale bhi karo."""
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False), cat_cols),
        ])

    # class_weight="balanced": kam fraud rows ko zyada weight milta hai
    return {
        "Dummy": Pipeline([("pre", pre(False)),
                           ("clf", DummyClassifier(strategy="prior"))]),
        "LogisticRegression": Pipeline([("pre", pre(True)),
                                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "RandomForest": Pipeline([("pre", pre(False)),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=300, min_samples_leaf=3,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("pre", pre(False)),
                                          ("clf", HistGradientBoostingClassifier(
                                              class_weight="balanced", max_depth=4,
                                              learning_rate=0.05, max_iter=150,
                                              random_state=RANDOM_STATE))]),
    }


cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for version, extra_drop in VERSIONS.items():
    print("\n" + "=" * 70)
    print(f"VERSION: {version}")
    print("=" * 70)
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[TARGET] + post_cols + extra_drop)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"Features: {X.shape[1]} | Train: {len(X_train)} rows ({y_train.sum()} fraud)"
          f" | Test: {len(X_test)} rows ({y_test.sum()} fraud)")

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    rows = []
    for name, pipe in build_models(num_cols, cat_cols).items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            "CV_PR_AUC": res["test_pr"].mean(),
            "CV_PR_AUC_std": res["test_pr"].std(),
            "CV_ROC_AUC": res["test_roc"].mean(),
            "CV_recall@0.5": res["test_recall"].mean(),
            "CV_precision@0.5": res["test_precision"].mean(),
        })
        r = rows[-1]
        print(f"{name:22s} PR-AUC {r['CV_PR_AUC']:.3f} ± {r['CV_PR_AUC_std']:.3f}"
              f" | ROC-AUC {r['CV_ROC_AUC']:.3f}"
              f" | recall {r['CV_recall@0.5']:.2f} precision {r['CV_precision@0.5']:.2f}")
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out_dir / f"{name}.joblib")

    comparison = pd.DataFrame(rows).round(4).sort_values("CV_PR_AUC", ascending=False)
    comparison.to_csv(TABLE_DIR / f"model_comparison_{version}.csv", index=False)
    joblib.dump((X_train, y_train), out_dir / "train_set.joblib")
    joblib.dump((X_test, y_test), out_dir / "test_set.joblib")
    print(f"Best by CV PR-AUC: {comparison.iloc[0]['model']}")

print("\nSaved models in:", MODEL_DIR)