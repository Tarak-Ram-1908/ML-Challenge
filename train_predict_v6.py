"""
v6: v1 ensemble + a 4th model (Random Forest with min_samples_leaf=10,
the best single model in CV). Reuses v1's saved OOF / test probabilities
(oof_probs.npz, test_probs.npy) and the RF-leaf10 OOF (oof_rf_l10.npy);
refits RF-leaf10 on all training data, re-tunes class weights, writes
submission_v6.csv.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
C = np.array(list("ABCDE"))
tr = pd.read_csv("train_fe.csv"); te = pd.read_csv("test_fe.csv")
X = tr.drop(columns=["id", "target_popularity"]); y = tr.target_popularity.map({c: i for i, c in enumerate(C)}).values
d = np.load("oof_probs.npz"); oof3, T3 = d["oof"], np.load("test_probs.npy")
oof_rf10 = np.load("oof_rf_l10.npy")
rf10 = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=1)).fit(X, y)
T_rf10 = rf10.predict_proba(te[X.columns])
oof = (3 * oof3 + oof_rf10) / 4; T = (3 * T3 + T_rf10) / 4      # equal weight per model
f = lambda yt, p: f1_score(yt, p, average="macro")
w = np.ones(5); g = np.exp(np.linspace(-1, 1, 41))
for _ in range(3):
    for c in range(5):
        bs, bg = f(y, (oof*w).argmax(1)), w[c]
        for v in g:
            w2 = w.copy(); w2[c] = v; s = f(y, (oof*w2).argmax(1))
            if s > bs: bs, bg = s, v
        w[c] = bg
print("weights", np.round(w, 2), " OOF Macro F1", round(f(y, (oof*w).argmax(1)), 4))
pd.DataFrame({"id": te.id, "target_popularity": C[(T*w).argmax(1)]}).to_csv("submission_v6.csv", index=False)
print("Saved submission_v6.csv")
