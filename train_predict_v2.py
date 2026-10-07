"""
Step 3c (v2): Stacked ensemble - train, tune for Macro F1, predict.

Usage:
    python features.py           # creates train_fe.csv / test_fe.csv
    python train_predict_v2.py   # creates submission_v2.csv  (about 5-10 minutes)

Difference from v1 (train_predict.py):
  v1 averaged the three models' probabilities.
  v2 "stacks" them: a logistic-regression meta-model learns how to combine
  the three models' out-of-fold probabilities, then per-class weights are
  tuned for Macro F1 on the meta-model's own out-of-fold predictions.

  Nested CV Macro F1:  v1 = 0.324   v2 = 0.329

Method
  1. Base models (balanced class weights): Random Forest, Extra Trees,
     HistGradientBoosting. 5-fold stratified CV gives out-of-fold (OOF)
     probabilities for train; test probabilities = average of the 5 fold models.
  2. Meta-model: logistic regression on the log-probabilities of the 3 models
     (15 inputs).
  3. Per-class weights tuned on the meta-model's OOF probabilities.
  4. Test prediction = argmax(meta probability * weight).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

BASE_MODELS = {
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


def make_meta():
    return LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")


def macro_f1(y_true, pred):
    return f1_score(y_true, pred, average="macro")


def tune_class_weights(proba, y_true, rounds=3):
    """Coordinate search for one multiplier per class that maximises Macro F1."""
    w = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(-1, 1, 41))
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


def to_meta_features(prob_list):
    return np.hstack([np.log(np.clip(p, 1e-6, 1.0)) for p in prob_list])


# ---------------------------------------------------------------- 1. base models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {n: np.zeros((len(X), len(CLASSES))) for n in BASE_MODELS}
test_prob = {n: np.zeros((len(X_test), len(CLASSES))) for n in BASE_MODELS}
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    for name, make in BASE_MODELS.items():
        model = make()
        model.fit(X.iloc[tr], y[tr])
        oof[name][va] = model.predict_proba(X.iloc[va])
        test_prob[name] += model.predict_proba(X_test) / skf.n_splits
    print(f"fold {fold}/5 done")

for name in BASE_MODELS:
    print(f"  {name:22s} CV Macro F1 = {macro_f1(y, oof[name].argmax(1)):.4f}")

# ---------------------------------------------------------------- 2. meta-model
F_train = to_meta_features([oof[n] for n in BASE_MODELS])
F_test = to_meta_features([test_prob[n] for n in BASE_MODELS])

# OOF predictions of the meta-model (needed to tune class weights honestly)
meta_oof = np.zeros((len(F_train), len(CLASSES)))
meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
for a, b in meta_cv.split(F_train, y):
    meta_oof[b] = make_meta().fit(F_train[a], y[a]).predict_proba(F_train[b])
print(f"  {'Stacked (meta argmax)':22s} CV Macro F1 = {macro_f1(y, meta_oof.argmax(1)):.4f}")

# Honest estimate of "stack + tuned weights": tune on 4/5, score on 1/5
scores = []
for a, b in StratifiedKFold(n_splits=5, shuffle=True, random_state=11).split(meta_oof, y):
    w = tune_class_weights(meta_oof[a], y[a])
    scores.append(macro_f1(y[b], (meta_oof[b] * w).argmax(1)))
print(f"  Stacked + tuned class weights (nested CV) = {np.mean(scores):.4f}")

weights = tune_class_weights(meta_oof, y)
print("  class weights:", {str(c): round(float(v), 2) for c, v in zip(CLASSES, weights)})
print("\nOOF classification report (stack + weights):")
print(classification_report(y, (meta_oof * weights).argmax(1), target_names=CLASSES, digits=3))

meta = make_meta().fit(F_train, y)
pred = CLASSES[(meta.predict_proba(F_test) * weights).argmax(1)]
sub = pd.DataFrame({"id": test["id"], "target_popularity": pred})

# ---------------------------------------------------------------- 3. checks + save
sample = pd.read_csv("sample_submission.csv")
assert len(sub) == len(sample) == 2500, "wrong number of rows"
assert set(sub["id"]) == set(sample["id"]), "ids do not match sample_submission"
assert sub["id"].is_unique and sub["target_popularity"].isin(CLASSES).all()
sub.to_csv("submission_v2.csv", index=False)

print("Predicted class mix on test (%):")
print((sub["target_popularity"].value_counts(normalize=True).reindex(CLASSES) * 100).round(1))
print("\nSaved submission_v2.csv")
