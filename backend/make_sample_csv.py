
"""
Creates a small sample of creditcard.csv for the live demo feed.
Run this once, in the same place you have the full creditcard.csv,
then copy the resulting creditcard_sample.csv into backend/.
"""

import pandas as pd

df = pd.read_csv("creditcard.csv")

fraud = df[df["Class"] == 1]                       # keep ALL fraud cases (there aren't many)
normal = df[df["Class"] == 0].sample(n=4000, random_state=42)  # a manageable slice of normal ones

sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42)  # shuffle so it's not grouped
sample.to_csv("creditcard_sample.csv", index=False)

print(f"Saved creditcard_sample.csv with {len(sample)} rows "
      f"({len(fraud)} fraud, {len(normal)} normal) — should be a few MB, GitHub-friendly.")
