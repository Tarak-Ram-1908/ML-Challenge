"""
Step 3b: Final model - train, tune for Macro F1, predict, write submission.csv

Usage:
    python features.py        # creates train_fe.csv / test_fe.csv
    python train_predict.py   # creates submission.csv  (about 5-10 minutes)

Method
  1. Three models, each with balanced class weights:
     Random Forest, Extra Trees, HistGradientBoosting.
  2. 5-fold stratified CV gives out-of-fold (OOF) probabilities for every
     training row. The final probabilities are the average of the 3 models.
  3. Per-class weights are tuned on the OOF probabilities to maximise
     Macro F1 (prediction = argmax(probability * weight)).
  4. Each model is refit on the full training set, test probabilities are
     averaged, the tuned weights are applied, and submission.csv is written.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

SEED = 42
CLASSES = np.array(["A", "B", "C", "D", "E"])

train = pd.read_csv("train_fe.csv")
test = pd.read_csv("test_fe.csv")
y = train["target_popularity"].map({c: i for i, c in enumerate(CLASSES)}).values
X = train.drop(columns=["id", "target_popularity"])
X_test = test[X.columns]

MODELS = {
    "Random Forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=SEED)),
    "Extra Trees": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                             class_weight="balanced", n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=SEED),
}


def macro_f1(y_true, pred):
    return f1_score(y_true, pred, average="macro")


def tune_class_weights(proba, y_true, rounds=3):
    """Coordinate search for one multiplier per class that maximises Macro F1."""
    w = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(-1, 1, 41))  # multipliers from 0.37 to 2.7
    for _ in range(rounds):
        for c in range(len(w)):
            best_score, best_g = macro_f1(y_true, (proba * w).argmax(1)), w[c]
            for g in grid:
                w_try = w.copy()
                w_try[c] = g
                s = macro_f1(y_true, (proba * w_try).argmax(1))
                if s > best_score:
                    best_score, best_g = s, g
            w[c] = best_g
    return w


# ---------------------------------------------------------------- 1. CV / OOF
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {name: np.zeros((len(X), len(CLASSES))) for name in MODELS}
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    for name, make in MODELS.items():
        model = make()
        model.fit(X.iloc[tr], y[tr])
        oof[name][va] = model.predict_proba(X.iloc[va])
    print(f"fold {fold}/5 done")

for name in MODELS:
    print(f"  {name:22s} CV Macro F1 = {macro_f1(y, oof[name].argmax(1)):.4f}")
oof_ens = np.mean(list(oof.values()), axis=0)
print(f"  {'Ensemble (average)':22s} CV Macro F1 = {macro_f1(y, oof_ens.argmax(1)):.4f}")

# ---------------------------------------------------------------- 2. tune class weights
# Honest estimate: tune on 4/5 of the OOF rows, score on the other 1/5.
check = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
scores = []
for a, b in check.split(oof_ens, y):
    w = tune_class_weights(oof_ens[a], y[a])
    scores.append(macro_f1(y[b], (oof_ens[b] * w).argmax(1)))
print(f"  Ensemble + tuned class weights (nested CV) = {np.mean(scores):.4f}")

weights = tune_class_weights(oof_ens, y)  # final weights use all OOF rows
print("  class weights:", {c: round(float(v), 2) for c, v in zip(CLASSES, weights)})
print("\nOOF classification report (ensemble + weights):")
print(classification_report(y, (oof_ens * weights).argmax(1), target_names=CLASSES, digits=3))

# ---------------------------------------------------------------- 3. refit on all data + predict
test_proba = np.zeros((len(X_test), len(CLASSES)))
for name, make in MODELS.items():
    model = make()
    model.fit(X, y)
    test_proba += model.predict_proba(X_test) / len(MODELS)
    print(f"refit {name} on full training data")

pred = CLASSES[(test_proba * weights).argmax(1)]
sub = pd.DataFrame({"id": test["id"], "target_popularity": pred})

# ---------------------------------------------------------------- 4. checks + save
sample = pd.read_csv("sample_submission.csv")
assert len(sub) == len(sample) == 2500, "wrong number of rows"
assert set(sub["id"]) == set(sample["id"]), "ids do not match sample_submission"
assert sub["id"].is_unique and sub["target_popularity"].isin(CLASSES).all()
sub.to_csv("submission.csv", index=False)

print("\nPredicted class mix on test (%):")
print((sub["target_popularity"].value_counts(normalize=True).reindex(CLASSES) * 100).round(1))
print("\nSaved submission.csv")
