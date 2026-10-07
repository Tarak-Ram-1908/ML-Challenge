"""
Step 3a: Compare candidate models with 5-fold stratified cross-validation.

Usage:
    python features.py          # creates train_fe.csv
    python compare_models.py    # prints a Macro F1 table, saves model_comparison.csv

Every model is trained with balanced class weights, so the rare classes
(A, B) are not ignored. LightGBM / XGBoost / CatBoost are used only if
installed (pip install lightgbm xgboost catboost); otherwise they are skipped.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
SEED = 42
CLASSES = np.array(["A", "B", "C", "D", "E"])

df = pd.read_csv("train_fe.csv")
y = df["target_popularity"].map({c: i for i, c in enumerate(CLASSES)}).values
X = df.drop(columns=["id", "target_popularity"])


# ---------------------------------------------------------------- candidates
def candidates():
    m = {}
    m["Logistic Regression"] = lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    m["Random Forest"] = lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=SEED))
    m["Extra Trees"] = lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                             class_weight="balanced", n_jobs=-1, random_state=SEED))
    m["HistGradientBoosting"] = lambda: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=SEED)

    try:
        import lightgbm as lgb
        m["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
            class_weight="balanced", random_state=SEED, verbose=-1)
    except ImportError:
        print("LightGBM not installed - skipped")
    try:
        import xgboost as xgb
        m["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
            eval_metric="mlogloss", random_state=SEED)  # weights passed in fit
    except ImportError:
        print("XGBoost not installed - skipped")
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda: CatBoostClassifier(
            iterations=800, learning_rate=0.05, depth=6, auto_class_weights="Balanced",
            random_seed=SEED, verbose=0)
    except ImportError:
        print("CatBoost not installed - skipped")
    return m


# ---------------------------------------------------------------- CV loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
oof_store = {}

for name, make in candidates().items():
    t0 = time.time()
    oof = np.zeros((len(X), len(CLASSES)))
    fold_f1 = []
    for tr, va in skf.split(X, y):
        model = make()
        if name == "XGBoost":  # XGBoost has no class_weight option
            model.fit(X.iloc[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        else:
            model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])
        fold_f1.append(f1_score(y[va], oof[va].argmax(1), average="macro"))
    pred = oof.argmax(1)
    per_class = f1_score(y, pred, average=None)
    rows.append({
        "model": name,
        "macro_f1": f1_score(y, pred, average="macro"),
        "fold_std": np.std(fold_f1),
        "accuracy": accuracy_score(y, pred),
        **{f"F1_{c}": v for c, v in zip(CLASSES, per_class)},
        "minutes": (time.time() - t0) / 60,
    })
    oof_store[name] = oof
    print(f"{name:22s} macro F1 = {rows[-1]['macro_f1']:.4f} "
          f"(+/- {rows[-1]['fold_std']:.4f})  [{rows[-1]['minutes']:.1f} min]")

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
print("\n" + res.round(4).to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# ---------------------------------------------------------------- simple ensemble
# Average the probabilities of the top 3 models - often beats any single model.
top = list(res["model"].head(3))
avg = np.mean([oof_store[m] for m in top], axis=0)
ens_f1 = f1_score(y, avg.argmax(1), average="macro")
print(f"\nAverage of top 3 ({', '.join(top)}): macro F1 = {ens_f1:.4f}")
print("Saved model_comparison.csv")
