"""
Step 1b: Exploratory Data Analysis (EDA)

Usage:
    python preprocess.py   # run first (creates train_clean.csv / test_clean.csv)
    python eda.py

All plots are saved in the folder eda_plots/. Key numbers are printed to the console.
Needs: pandas, numpy, matplotlib, scikit-learn
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")
OUT = "eda_plots"
os.makedirs(OUT, exist_ok=True)

TARGET, ID = "target_popularity", "id"
CLASSES = ["A", "B", "C", "D", "E"]
COLORS = ["#2a9d8f", "#57b894", "#e9c46a", "#f4a261", "#e76f51"]

train = pd.read_csv("train_clean.csv")
test = pd.read_csv("test_clean.csv")
features = [c for c in train.columns if c not in (ID, TARGET)]
# Ordinal version of the target: A (most popular) = 4 ... E (least) = 0
train["target_ord"] = train[TARGET].map({c: 4 - i for i, c in enumerate(CLASSES)})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=120)
    plt.close()
    print(f"  saved {OUT}/{name}")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- 1. overview
section("1. Overview")
print("train:", train.shape, " test:", test.shape)
print(train[features].describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())

# ---------------------------------------------------------------- 2. target
section("2. Target distribution")
counts = train[TARGET].value_counts().reindex(CLASSES)
print(pd.DataFrame({"count": counts, "percent": (100 * counts / len(train)).round(1)}))
print(f"Imbalance ratio (largest / smallest class): {counts.max() / counts.min():.1f}x")

plt.figure(figsize=(6, 4))
bars = plt.bar(CLASSES, counts, color=COLORS)
for b, v in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100 * v / len(train):.0f}%)",
             ha="center", va="bottom", fontsize=9)
plt.title("Class distribution (A = most popular)")
plt.ylabel("articles")
plt.ylim(0, counts.max() * 1.2)
save("01_class_distribution.png")

# ---------------------------------------------------------------- 3. missing values
section("3. Missing values")
miss = train[features].isna().mean().mul(100).sort_values(ascending=False)
miss = miss[miss > 0]
print(miss.round(2).to_string())

plt.figure(figsize=(7, 5))
plt.barh(miss.index[::-1], miss.values[::-1], color="#6c757d")
plt.xlabel("% missing (train)")
plt.title("Missing values per column")
save("02_missing_values.png")

# Is "missing" itself informative? Compare class mix of rows with vs without NaN.
train["any_missing"] = train[features].isna().any(axis=1)
mix = pd.crosstab(train["any_missing"], train[TARGET], normalize="index").mul(100).round(1)
print("\nClass mix (%) for rows with / without any missing value:")
print(mix)
print("-> If these rows look the same, values are missing at random and simple "
      "imputation (or leaving NaN for trees) is safe.")

# ---------------------------------------------------------------- 4. distributions / skew
section("4. Skewness (heavy-tailed columns need log transform)")
skew = train[features].skew().sort_values(ascending=False)
print(skew.round(2).head(15).to_string())

skewed = ["n_tokens_content", "num_hrefs", "num_imgs", "num_videos",
          "kw_avg_avg", "kw_max_avg", "self_reference_avg_sharess", "kw_avg_max"]
fig, axes = plt.subplots(2, len(skewed) // 2, figsize=(16, 7))
for ax, col in zip(axes.ravel(), skewed):
    vals = train[col].dropna()
    ax.hist(np.log1p(vals.clip(lower=0)), bins=50, color="#457b9d")
    ax.set_title(f"log1p({col})\nraw skew = {skew[col]:.1f}", fontsize=9)
fig.suptitle("Heavy-tailed features after log1p transform")
save("03_log_distributions.png")

# ---------------------------------------------------------------- 5. feature vs target
section("5. Which features separate the classes?")
# (a) Spearman correlation with the ordinal target
spear = train[features].corrwith(train["target_ord"], method="spearman")
spear = spear.reindex(spear.abs().sort_values(ascending=False).index)
print("Top 20 by |Spearman correlation| with popularity (positive = more popular):")
print(spear.head(20).round(3).to_string())

# (b) Mutual information (captures non-linear relations too)
X_mi = train[features].fillna(train[features].median())
mi = pd.Series(mutual_info_classif(X_mi, train[TARGET], random_state=0),
               index=features).sort_values(ascending=False)
print("\nTop 20 by mutual information:")
print(mi.head(20).round(4).to_string())
print("\nWeakest 10 by mutual information (candidates to drop / ignore):")
print(mi.tail(10).round(4).to_string())

plt.figure(figsize=(7, 8))
top_mi = mi.head(25)
plt.barh(top_mi.index[::-1], top_mi.values[::-1], color="#2a9d8f")
plt.xlabel("mutual information with target")
plt.title("Top 25 features by mutual information")
save("04_mutual_information.png")

# (c) Boxplots of the top features by class (log scale where needed)
top_feats = list(mi.head(8).index)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), top_feats):
    data = [train.loc[train[TARGET] == c, col].dropna() for c in CLASSES]
    if train[col].min() >= 0 and skew[col] > 2:
        data = [np.log1p(d) for d in data]
        ax.set_title(f"log1p({col})", fontsize=10)
    else:
        ax.set_title(col, fontsize=10)
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticks(range(1, len(CLASSES) + 1))
    ax.set_xticklabels(CLASSES)  # works on old and new matplotlib
    for patch, colr in zip(bp["boxes"], COLORS):
        patch.set_facecolor(colr)
fig.suptitle("Top features by class (outliers hidden)")
save("05_boxplots_by_class.png")

# Median of top features per class -> a quick table to read
print("\nMedian of top features per class:")
print(train.groupby(TARGET)[top_feats].median().round(2).T.to_string())

# ---------------------------------------------------------------- 6. categorical: channel & weekday
section("6. Data channel and weekday vs popularity")
chan_cols = [c for c in features if c.startswith("data_channel_is_")]
train["channel"] = train[chan_cols].idxmax(axis=1).str.replace("data_channel_is_", "")
train.loc[train[chan_cols].sum(axis=1) == 0, "channel"] = "other"
day_cols = [c for c in features if c.startswith("weekday_is_")]
train["weekday"] = train[day_cols].idxmax(axis=1).str.replace("weekday_is_", "")

for name, order in [("channel", None),
                    ("weekday", ["monday", "tuesday", "wednesday", "thursday",
                                 "friday", "saturday", "sunday"])]:
    tab = pd.crosstab(train[name], train[TARGET], normalize="index").mul(100)
    if order:
        tab = tab.reindex(order)
    tab["n"] = train[name].value_counts()
    print(f"\nClass mix (%) by {name}:")
    print(tab.round(1).to_string())

    ax = tab[CLASSES].plot(kind="barh", stacked=True, color=COLORS, figsize=(9, 5))
    ax.set_xlabel("% of articles")
    ax.set_title(f"Popularity mix by {name}")
    ax.legend(title="class", bbox_to_anchor=(1.01, 1), loc="upper left")
    save(f"06_class_mix_by_{name}.png")

# ---------------------------------------------------------------- 7. correlation / redundancy
section("7. Highly correlated feature pairs (redundant information)")
corr = train[features].corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
         .stack().rename("corr").reset_index())
pairs = pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index)
print(pairs[pairs["corr"].abs() > 0.8].round(3).to_string(index=False))

plt.figure(figsize=(14, 12))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(fraction=0.046)
plt.xticks(range(len(features)), features, rotation=90, fontsize=6)
plt.yticks(range(len(features)), features, fontsize=6)
plt.title("Feature correlation matrix")
save("07_correlation_heatmap.png")

# ---------------------------------------------------------------- 8. train vs test drift
section("8. Train vs test: are they from the same distribution?")
drift = pd.DataFrame({
    "train_median": train[features].median(),
    "test_median": test[features].median(),
})
drift["pct_diff"] = ((drift["test_median"] - drift["train_median"])
                     / drift["train_median"].abs().replace(0, np.nan) * 100)
big = drift[drift["pct_diff"].abs() > 10].round(3)
print("Columns whose median differs by >10% between train and test:")
print(big.to_string() if len(big) else "  none - train and test look alike")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, ["timedelta", "kw_avg_avg", "self_reference_avg_sharess",
                          "n_tokens_content"]):
    for df, lab, colr in [(train, "train", "#457b9d"), (test, "test", "#e76f51")]:
        ax.hist(np.log1p(df[col].dropna().clip(lower=0)), bins=40, density=True,
                alpha=0.5, label=lab, color=colr)
    ax.set_title(f"log1p({col})", fontsize=9)
    ax.legend()
save("08_train_vs_test.png")

# ---------------------------------------------------------------- summary
section("Summary")
print(f"- Strongest single features (MI): {', '.join(mi.head(5).index)}")
print(f"- Weakest features (MI): {', '.join(mi.tail(5).index)}")
print(f"- Most skewed: {', '.join(skew.head(5).index)} -> use log1p")
print(f"- Class imbalance: {counts.max() / counts.min():.1f}x -> use class weights / "
      "threshold tuning for Macro F1")
print(f"\nAll plots are in ./{OUT}/")
