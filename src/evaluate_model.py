"""
evaluate_model.py  --  STEP 7  (final step)
-------------------------------------------
Test set pe models ko asli imtihaan dena. Dono versions (with_rule, no_rule) ke liye:

  1. Test PR-AUC / ROC-AUC (sab models)
  2. Threshold tuning: 0.5 imbalanced data mein kaam nahi karta.
     Threshold TRAIN set ki out-of-fold predictions se chunte hain (test se nahi),
     phir test pe lagate hain -> koi cheating nahi.
  3. Confusion matrix, precision, recall, F1
  4. Permutation feature importance (best model)

Input : models/<version>/  (train_model.py ne banaye)
Output: reports/tables/test_results_<version>.csv, threshold_report_<version>.csv
        reports/figures/10_pr_curves_<v>.png, 11_confusion_<v>.png, 12_importance_<v>.png
Run   : python src/evaluate_model.py
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from config import FIG_DIR, MODEL_DIR, TABLE_DIR, RANDOM_STATE, N_SPLITS

MODEL_NAMES = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]


def metrics_at(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, pred, zero_division=0), 3),
    }


for version in ["with_rule", "no_rule"]:
    print("\n" + "=" * 70)
    print(f"VERSION: {version}")
    print("=" * 70)
    vdir = MODEL_DIR / version
    X_train, y_train = joblib.load(vdir / "train_set.joblib")
    X_test, y_test = joblib.load(vdir / "test_set.joblib")
    models = {n: joblib.load(vdir / f"{n}.joblib") for n in MODEL_NAMES}
    print(f"Test set: {len(y_test)} rows, {y_test.sum()} fraud (kam hai, isliye results noisy)")

    # ---------------------------------------------------------------
    # 1) Test scores + PR curves
    # ---------------------------------------------------------------
    rows, probas = [], {}
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        probas[name] = p
        rows.append({"model": name,
                     "test_PR_AUC": round(average_precision_score(y_test, p), 4),
                     "test_ROC_AUC": round(roc_auc_score(y_test, p), 4)})
        pr, rc, _ = precision_recall_curve(y_test, p)
        plt.plot(rc, pr, label=f"{name} (AP={rows[-1]['test_PR_AUC']:.2f})")
    plt.axhline(y_test.mean(), color="gray", ls="--", label="Random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curves on test set ({version})")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / f"10_pr_curves_{version}.png", dpi=130); plt.close()

    results = pd.DataFrame(rows).sort_values("test_PR_AUC", ascending=False)
    results.to_csv(TABLE_DIR / f"test_results_{version}.csv", index=False)
    print("\nTEST RESULTS")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    best = models[best_name]
    print(f"\nBest model on test PR-AUC: {best_name}")

    # ---------------------------------------------------------------
    # 2) Threshold tuning (train OOF predictions se)
    # ---------------------------------------------------------------
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr = thr[np.argmax(f1)]

    report = pd.DataFrame([
        {"setting": "default 0.5", **metrics_at(y_test, probas[best_name], 0.5)},
        {"setting": "tuned (train OOF best F1)", **metrics_at(y_test, probas[best_name], best_thr)},
    ])
    report.to_csv(TABLE_DIR / f"threshold_report_{version}.csv", index=False)
    print("\nTHRESHOLD REPORT (test set)")
    print(report.to_string(index=False))

    # ---------------------------------------------------------------
    # 3) Confusion matrices
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, thr_val, title in zip(ax, [0.5, best_thr], ["Threshold 0.5", f"Tuned threshold {best_thr:.3f}"]):
        pred = (probas[best_name] >= thr_val).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=a, colorbar=False,
                                                display_labels=["Legit", "Fraud"])
        a.set_title(f"{best_name}\n{title}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"11_confusion_{version}.png", dpi=130); plt.close()

    # ---------------------------------------------------------------
    # 4) Permutation importance (feature hatane/shuffle karne se PR-AUC kitna girta hai)
    # ---------------------------------------------------------------
    imp = permutation_importance(best, X_test, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    imp_s.head(10).sort_values().plot(kind="barh", figsize=(7, 5), color="#55A868")
    plt.title(f"Permutation importance ({best_name}, {version})\nPR-AUC drop")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"12_importance_{version}.png", dpi=130); plt.close()
    imp_s.round(4).to_csv(TABLE_DIR / f"feature_importance_{version}.csv", header=["importance"])
    print("\nTOP 5 FEATURES")
    print(imp_s.head(5).round(4).to_string())

print("\nFigures:", FIG_DIR)
print("Tables :", TABLE_DIR)