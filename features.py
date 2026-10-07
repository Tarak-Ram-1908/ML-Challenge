"""
Step 2: Feature engineering

Usage:
    python preprocess.py   # creates train_clean.csv / test_clean.csv
    python features.py     # creates train_fe.csv / test_fe.csv

Every feature is built row by row from that article's own columns,
so nothing leaks between train and test (no target information is used).
"""
import numpy as np
import pandas as pd

TARGET, ID = "target_popularity", "id"
EPS = 1.0  # added to denominators to avoid division by zero


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------ 1. categorical summaries
    # Six one-hot channel flags -> one integer column (0 = "other", no channel)
    chan_cols = ["data_channel_is_lifestyle", "data_channel_is_entertainment",
                 "data_channel_is_bus", "data_channel_is_socmed",
                 "data_channel_is_tech", "data_channel_is_world"]
    df["channel"] = (df[chan_cols].values * np.arange(1, 7)).sum(axis=1)

    # Seven weekday flags -> one integer column (0 = Monday ... 6 = Sunday)
    day_cols = ["weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
                "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
                "weekday_is_sunday"]
    df["weekday"] = (df[day_cols].values * np.arange(7)).sum(axis=1)

    # ------------------------------------------------ 2. LDA topic summaries
    lda_cols = [f"LDA_0{i}" for i in range(5)]
    lda = df[lda_cols]
    df["lda_dominant"] = lda.fillna(-1).values.argmax(axis=1)
    df["lda_max"] = lda.max(axis=1)
    p = lda.clip(lower=1e-6)
    df["lda_entropy"] = -(p * np.log(p)).sum(axis=1, min_count=5)  # topic "focus"

    # ------------------------------------------------ 3. keyword popularity
    # EDA: kw_avg_avg / kw_max_avg are the strongest single features
    df["kw_avg_spread"] = df["kw_max_avg"] - df["kw_min_avg"]
    df["kw_avg_ratio"] = df["kw_avg_avg"] / (df["kw_max_avg"] + EPS)
    df["kw_min_avg_missing"] = (df["kw_min_avg"] <= 0).astype(int)  # -1/0 = no data
    df["kw_min_min_missing"] = (df["kw_min_min"] < 0).astype(int)
    df["kw_avg_x_count"] = df["kw_avg_avg"] * df["num_keywords"]

    # ------------------------------------------------ 4. self-reference shares
    df["has_self_ref"] = (df["self_reference_max_shares"] > 0).astype(int)
    df["self_ref_spread"] = df["self_reference_max_shares"] - df["self_reference_min_shares"]
    df["self_ref_x_kw"] = np.log1p(df["self_reference_avg_sharess"].clip(lower=0)) * \
        np.log1p(df["kw_avg_avg"].clip(lower=0))

    # ------------------------------------------------ 5. content structure
    df["empty_article"] = (df["n_tokens_content"] == 0).astype(int)
    tokens = df["n_tokens_content"] + EPS
    df["hrefs_per_token"] = df["num_hrefs"] / tokens
    df["imgs_per_token"] = df["num_imgs"] / tokens
    df["self_href_ratio"] = df["num_self_hrefs"] / (df["num_hrefs"] + EPS)
    df["media_count"] = df["num_imgs"] + df["num_videos"]
    df["unique_words"] = df["n_tokens_content"] * df["n_unique_tokens"]

    # ------------------------------------------------ 6. sentiment
    df["polarity_gap"] = df["avg_positive_polarity"] + df["avg_negative_polarity"]
    df["pos_neg_ratio"] = df["global_rate_positive_words"] / \
        (df["global_rate_negative_words"] + 1e-3)
    df["title_sent_strength"] = df["title_subjectivity"] * df["abs_title_sentiment_polarity"]
    df["title_has_sentiment"] = (df["title_subjectivity"] > 0).astype(int)

    # ------------------------------------------------ 7. time
    # timedelta = days before the dataset was collected; weekly bucket
    df["week_index"] = df["timedelta"] // 7

    # ------------------------------------------------ 8. log transforms
    # EDA: these are heavily right-skewed (skew 4-37). Trees don't need this,
    # but it helps linear / distance-based models and keeps scales sane.
    skewed = ["n_tokens_content", "num_hrefs", "num_self_hrefs", "num_imgs",
              "num_videos", "kw_max_min", "kw_avg_min", "kw_min_max", "kw_avg_max",
              "kw_max_avg", "kw_avg_avg", "self_reference_min_shares",
              "self_reference_max_shares", "self_reference_avg_sharess",
              "kw_avg_spread", "self_ref_spread", "media_count", "unique_words",
              "kw_avg_x_count"]
    for c in skewed:
        df[c] = np.log1p(df[c].clip(lower=0))

    return df


if __name__ == "__main__":
    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")

    train_fe = add_features(train)
    test_fe = add_features(test)

    # same columns in the same order (target last in train)
    feat_cols = [c for c in train_fe.columns if c not in (ID, TARGET)]
    assert feat_cols == [c for c in test_fe.columns if c != ID]
    train_fe = train_fe[[ID] + feat_cols + [TARGET]]
    test_fe = test_fe[[ID] + feat_cols]

    n_new = len(feat_cols) - (train.shape[1] - 2)
    print(f"Features: {train.shape[1] - 2} original + {n_new} new = {len(feat_cols)}")
    train_fe.to_csv("train_fe.csv", index=False)
    test_fe.to_csv("test_fe.csv", index=False)
    print("Saved: train_fe.csv, test_fe.csv")
