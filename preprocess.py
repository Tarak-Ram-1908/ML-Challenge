"""
Step 1: Load and clean the competition data.

Usage:
    python preprocess.py            # expects train.csv and test.csv in the same folder

Outputs:
    train_clean.csv, test_clean.csv  (NaNs kept, for LightGBM / XGBoost)
    train_imputed.csv, test_imputed.csv  (median-imputed, for other models)
"""
import numpy as np
import pandas as pd

TARGET = "target_popularity"
ID = "id"

# ---------------------------------------------------------------- load
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("train:", train.shape, " test:", test.shape)

# ---------------------------------------------------------------- types
# Some columns can be read as text because of the "NA" entries.
# Force every feature column to numeric (anything unparseable -> NaN).
feature_cols = [c for c in train.columns if c not in (ID, TARGET)]
for df in (train, test):
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# ---------------------------------------------------------------- sanity checks
assert train[ID].is_unique and test[ID].is_unique, "duplicate ids"
assert list(test.columns) == [ID] + feature_cols, "test columns differ from train"
print("\nClass counts:\n", train[TARGET].value_counts().sort_index())

# ---------------------------------------------------------------- missing values
missing = train[feature_cols].isna().sum()
print("\nMissing values per column (train):")
print(missing[missing > 0].sort_values(ascending=False))
print("Total missing  train:", int(missing.sum()),
      " test:", int(test[feature_cols].isna().sum().sum()))

# ---------------------------------------------------------------- fix bad values
# The token-rate columns are ratios and should be <= 1.
# The source dataset has one corrupt article with values in the hundreds.
rate_cols = ["n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens"]
for df in (train, test):
    bad = (df[rate_cols] > 1).any(axis=1)
    if bad.any():
        print(f"Setting {bad.sum()} row(s) with token rates > 1 to NaN")
        df.loc[bad, rate_cols] = np.nan

# Counts and share statistics cannot be negative.
nonneg_cols = ["n_tokens_title", "n_tokens_content", "num_hrefs", "num_self_hrefs",
               "num_imgs", "num_videos", "num_keywords",
               "self_reference_min_shares", "self_reference_max_shares",
               "self_reference_avg_sharess"]
for df in (train, test):
    for c in nonneg_cols:
        df.loc[df[c] < 0, c] = np.nan
# Note: kw_min_min / kw_avg_min / kw_min_avg use -1 as "no data" in the
# original source, and negative sentiment polarities are valid; both are kept.

# ---------------------------------------------------------------- outputs
# Option A: keep NaNs (LightGBM / XGBoost handle them natively)
train.to_csv("train_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)

# Option B: median imputation, with medians from TRAIN only (no test leakage)
medians = train[feature_cols].median()
train_imp = train.copy()
test_imp = test.copy()
train_imp[feature_cols] = train_imp[feature_cols].fillna(medians)
test_imp[feature_cols] = test_imp[feature_cols].fillna(medians)
assert train_imp[feature_cols].isna().sum().sum() == 0
train_imp.to_csv("train_imputed.csv", index=False)
test_imp.to_csv("test_imputed.csv", index=False)

print("\nSaved: train_clean.csv, test_clean.csv, train_imputed.csv, test_imputed.csv")
