# ML Challenge CS_CLUB – Online News Popularity

Solution for the Kaggle competition [ML Challenge CS_CLUB](https://www.kaggle.com/competitions/ml-challenge-cs-club).

**Task:** predict the popularity class (`A`–`E`, A = most popular) of Mashable news articles from 58 content, keyword, topic, sentiment and publication features.
**Metric:** Macro F1 (every class counts equally, so the rare classes A and B matter as much as the common ones).

No external datasets and no pretrained models are used. Only the competition files are used.

---

## Repository structure

| File | Purpose |
|---|---|
| `train.csv`, `test.csv`, `sample_submission.csv` | Original competition data |
| `preprocess.py` | Step 1 – load, type-fix and clean the data |
| `eda.py` | Exploratory data analysis (plots saved to `eda_plots/`) |
| `features.py` | Step 2 – feature engineering |
| `train_clean.csv`, `test_clean.csv` | Cleaned data, NaNs kept (for gradient-boosted trees) |
| `train_imputed.csv`, `test_imputed.csv` | Cleaned data, median-imputed (for other models) |
| `train_fe.csv`, `test_fe.csv` | Cleaned data + engineered features |
| `eda_plots/` | EDA figures |
| `requirements.txt` | Python dependencies |

All generated CSVs can be recreated from the original data by running the scripts below.

---

## How to reproduce

Requires Python 3.9+.

```bash
pip install -r requirements.txt

python preprocess.py   # -> train_clean.csv, test_clean.csv, train_imputed.csv, test_imputed.csv
python eda.py          # -> eda_plots/*.png and a printed summary (optional)
python features.py     # -> train_fe.csv, test_fe.csv
```

Run the scripts from the repository root, in this order.

---

## Step 1 – Data cleaning (`preprocess.py`)

- Converts every feature column to numeric (some were read as text because of `NA` entries).
- Checks for duplicate ids/rows and that train and test have identical feature columns (none found / they match).
- **Missing values:** 20,066 NaNs across 15 columns in train (≈2–6% per column); test has none.
  EDA shows rows with and without NaNs have the same class mix, so values are missing at random.
  - `*_clean.csv`: NaNs kept – gradient-boosted trees handle them natively.
  - `*_imputed.csv`: NaNs filled with the **train** median (no information from test is used).
- **Invalid values:** one article had token-rate ratios in the hundreds (must be ≤ 1); these were set to NaN. Negative counts would also be set to NaN (none found).
- Kept as-is: the `-1` "no data" markers in `kw_min_min`, `kw_avg_min`, `kw_min_avg`, and negative sentiment polarities (valid values).

## Exploratory data analysis (`eda.py`)

Key findings:

- **Class imbalance:** D 36.4%, E 28.9%, C 19.5%, B 10.6%, A 4.6% (7.9× between largest and smallest class).
- **Strongest signals:** keyword popularity (`kw_avg_avg`, `kw_max_avg`) and shares of self-referenced articles (`self_reference_*`). Their medians fall steadily from A to E.
- **Topic matters:** World and Entertainment articles are ~40% class E; Social Media articles only ~10% E. Articles with no channel have the most A's (~11%).
- **Weekends:** Saturday/Sunday articles are ~10–12% class E vs ~32% on weekdays.
- **Skew:** keyword and share statistics are heavily right-skewed (skew up to 37) → log transform.
- **Redundancy:** several pairs correlate above 0.8 (e.g. `kw_max_min`/`kw_avg_min`, `data_channel_is_world`/`LDA_02`).
- **Weak individual signal:** best Spearman correlation with the target is only 0.24, so feature engineering and model tuning matter.
- **Train vs test:** share-related features have 10–17% higher medians in test, so test may contain slightly more popular articles.

## Step 2 – Feature engineering (`features.py`)

24 new features (59 → 83), each computed only from the article's own columns (no target information, no leakage):

1. **Categorical summaries** – `channel` (from the 6 channel flags), `weekday` (from the 7 day flags).
2. **Topics** – dominant LDA topic, its weight, topic entropy.
3. **Keyword popularity** – spread and ratio of keyword averages, "no data" flags for `-1` values, keyword average × keyword count.
4. **Self-references** – has-self-reference flag, share spread, self-reference shares × keyword popularity.
5. **Content structure** – links and images per word, internal-link ratio, total media, empty-article flag, unique word count.
6. **Sentiment** – polarity gap, positive/negative word ratio, title sentiment strength, title-has-sentiment flag.
7. **Time** – `timedelta` grouped into weeks.
8. **Log transform** (`log1p`) of 19 heavy-tailed columns.

**Validation (5-fold stratified CV, gradient boosting with balanced class weights):**

| Data | CV Macro F1 |
|---|---|
| Cleaned only (59 features) | 0.302 |
| + engineered features (83) | **0.310** |

---

## Step 3 – Model training and prediction

_To be added: model training with cross-validation, class-threshold tuning for Macro F1, and generation of `submission.csv`._
