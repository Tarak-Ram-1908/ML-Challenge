"""
Step 3d (v3): v1 ensemble + class weights adjusted for the test set's class mix

Usage:
    python features.py           # creates train_fe.csv / test_fe.csv
    python train_predict_v3.py   # creates submission_v3.csv  (about 5-10 minutes)

Why: the test set's predicted probabilities indicate a different class mix
than train (more A/B, fewer D/E). v3 estimates the test mix without labels
(EM algorithm, Saerens et al. 2002, on calibrated probabilities), moves
halfway from the train mix toward that estimate (a hedge, since the
estimate is uncertain), and tunes the per-class weights for Macro F1 on
OOF rows re-weighted to that mix. Models and features are identical to v1.

Method
  1. Three models, each with balanced class weights:
     Random Forest, Extra Trees, HistGradientBoosting.
  2. 5-fold stratified CV gives out-of-fold (OOF) probabilities for every
     training row. The final probabilities are the average of the 3 models.
  3. Each model is refit on the full training set; test probabilities are
     averaged (identical to v1).
  4. The test class mix is estimated, and per-class weights are tuned to
     maximise Macro F1 at the halfway mix; submission_v3.csv is written.
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


def macro_f1(y_true, pred, sample_weight=None):
    return f1_score(y_true, pred, average="macro", sample_weight=sample_weight)


def tune_class_weights(proba, y_true, sample_weight=None, rounds=3):
    """Coordinate search for one multiplier per class that maximises Macro F1."""
    w = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(-1.5, 1.5, 41))
    for _ in range(rounds):
        for c in range(len(w)):
            best_score, best_g = macro_f1(y_true, (proba * w).argmax(1), sample_weight), w[c]
            for g in grid:
                w_try = w.copy()
                w_try[c] = g
                s = macro_f1(y_true, (proba * w_try).argmax(1), sample_weight)
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

# ---------------------------------------------------------------- 2. refit + test probabilities
test_proba = np.zeros((len(X_test), len(CLASSES)))
test_each = []
for name, make in MODELS.items():
    model = make()
    model.fit(X, y)
    p = model.predict_proba(X_test)
    test_each.append(p)
    test_proba += p / len(MODELS)
    print(f"refit {name} on full training data")

# ---------------------------------------------------------------- 3. estimate the test class mix
# Calibrate: unweighted logistic regression on the models' log-probabilities
# gives probabilities that reflect the TRAIN class mix.
def logp(ps):
    return np.hstack([np.log(np.clip(p, 1e-6, 1.0)) for p in ps])

calib = LogisticRegression(C=0.3, max_iter=3000).fit(logp(list(oof.values())), y)
cal_test = calib.predict_proba(logp(test_each))
pi_train = np.bincount(y) / len(y)

pi = pi_train.copy()  # EM for label shift (Saerens et al., 2002)
for _ in range(1000):
    q = cal_test * (pi / pi_train)
    q /= q.sum(axis=1, keepdims=True)
    new = q.mean(axis=0)
    if np.abs(new - pi).max() < 1e-8:
        break
    pi = new
pi_half = np.sqrt(pi_train * pi)  # halfway (geometric mean) - a hedge
pi_half /= pi_half.sum()
print("\nClass mix (%)   ", " ".join(f"{c:>6s}" for c in CLASSES))
print("  train         ", " ".join(f"{v*100:6.1f}" for v in pi_train))
print("  test (EM est.)", " ".join(f"{v*100:6.1f}" for v in pi))
print("  used (halfway)", " ".join(f"{v*100:6.1f}" for v in pi_half))

# ---------------------------------------------------------------- 4. tune weights for that mix
row_weight = (pi_half / pi_train)[y]  # re-weight OOF rows to the target mix
weights = tune_class_weights(oof_ens, y, sample_weight=row_weight)
print("  class weights:", {str(c): round(float(v), 2) for c, v in zip(CLASSES, weights)})
print(f"  OOF Macro F1 at train mix  = {macro_f1(y, (oof_ens * weights).argmax(1)):.4f}")
print(f"  OOF Macro F1 at target mix = "
      f"{macro_f1(y, (oof_ens * weights).argmax(1), row_weight):.4f}")

pred = CLASSES[(test_proba * weights).argmax(1)]
sub = pd.DataFrame({"id": test["id"], "target_popularity": pred})

# ---------------------------------------------------------------- 5. checks + save
sample = pd.read_csv("sample_submission.csv")
assert len(sub) == len(sample) == 2500, "wrong number of rows"
assert set(sub["id"]) == set(sample["id"]), "ids do not match sample_submission"
assert sub["id"].is_unique and sub["target_popularity"].isin(CLASSES).all()
sub.to_csv("submission_v3.csv", index=False)

print("\nPredicted class mix on test (%):")
print((sub["target_popularity"].value_counts(normalize=True).reindex(CLASSES) * 100).round(1))
print("\nSaved submission_v3.csv")
