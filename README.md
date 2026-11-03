# Bank Transaction Fraud Detection

Data analytics + data science project on a 10k-row bank transaction dataset
(`Fraud_Detection.xlsx`, 0.67% fraud).

## Project structure
```
data/raw/Fraud_Detection.xlsx      original data (never edited)
data/processed/                    clean_transactions.csv, features.csv
src/
  config.py                        paths + settings
  inspect_data.py                  1. look at raw data
  clean_data.py                    2. cleaning
  eda_analysis.py                  3. EDA (analytics)
  feature_engineering.py           4. PRE / POST (post_) feature groups
  leakage_check.py                 5. which features drive the score?
  train_model.py                   6. 4 models, CV, two versions
  evaluate_model.py                7. test set, threshold, importance
models/<version>/                  saved models
reports/figures, reports/tables    charts and summary tables
```

## How to run
```
pip install -r requirements.txt
python run_all.py            # or run each script in src/ in order
```

## Key findings
- Fraud occurs only in TRANSFER and CASH_OUT transactions.
- `amount == oldbalanceOrg` flags 65 of 68 frauds with zero false positives,
  so the dataset behaves like a rule-generated (synthetic) label.
- Post-transaction columns (`post_*`) are excluded from the models because they
  are not available at decision time.
- Two model versions are reported: `with_rule` (near-perfect scores) and
  `no_rule` (realistic, harder). PR-AUC is the main metric; ROC-AUC is
  misleading here (type alone gives 0.89).
- The test set has only 14 fraud rows, so test metrics are noisy.