"""
Step 3f: Majority vote of three submissions (v1, v2, v4).

Usage:
    python vote.py      # needs submission.csv, submission_v2.csv, submission_v4.csv
                        # creates submission_vote.csv

For each article, the class predicted by at least 2 of the 3 versions wins;
if all three disagree, v1's prediction is kept. v3 is excluded because it
was built on a different assumption (shifted test class mix) that the
public leaderboard rejected.
"""
import pandas as pd

files = {"v1": "submission.csv", "v2": "submission_v2.csv", "v4": "submission_v4.csv"}
subs = {k: pd.read_csv(f).set_index("id")["target_popularity"] for k, f in files.items()}
ids = subs["v1"].index
df = pd.DataFrame({k: s.reindex(ids) for k, s in subs.items()})
assert not df.isna().any().any(), "submission files do not contain the same ids"


def vote(row):
    counts = row.value_counts()
    return counts.index[0] if counts.iloc[0] >= 2 else row["v1"]


out = pd.DataFrame({"id": ids, "target_popularity": df.apply(vote, axis=1).values})
assert len(out) == 2500 and out["id"].is_unique
out.to_csv("submission_vote.csv", index=False)

print(f"Same as v1 for {(out['target_popularity'].values == df['v1'].values).mean():.1%} of rows")
print("Class mix (%):")
print((out["target_popularity"].value_counts(normalize=True).sort_index() * 100).round(1))
print("Saved submission_vote.csv")
