from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

# -----------------------------
# Step 1: Load and combine all CSV files
# -----------------------------
data_folder = Path(__file__).resolve().parent.parent / "data"
files = sorted(data_folder.glob("epl_*.csv"))

print("Files found:")
for file in files:
    print(file)

dfs = []

for file in files:
    df = pd.read_csv(file)

    # Add season column from filename
    season = file.stem.replace("epl_", "")
    df["Season"] = season

    dfs.append(df)

matches = pd.concat(dfs, ignore_index=True)

print("\nCombined data shape:", matches.shape)

print("\nColumns:")
print(matches.columns)

print("\nFirst few rows:")
print(matches.head())

# -----------------------------
# Step 2: Compute descriptive statistics
# -----------------------------
season_summary = (
    matches.groupby("Season")["FTR"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

season_summary = season_summary.rename(
    columns={
        "H": "HomeWinRate",
        "A": "AwayWinRate",
        "D": "DrawRate",
    }
).reset_index()

# Flag COVID seasons
season_summary["COVID"] = season_summary["Season"].isin(["2019-20", "2020-21"])

print("\nSeason Win Rates:")
print(season_summary)

# -----------------------------
# Step 3: Line plot of home/away/draw rates over time
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    season_summary["Season"],
    season_summary["HomeWinRate"],
    marker="o",
    label="Home Win Rate",
)
plt.plot(
    season_summary["Season"],
    season_summary["AwayWinRate"],
    marker="o",
    label="Away Win Rate",
)
plt.plot(
    season_summary["Season"],
    season_summary["DrawRate"],
    marker="o",
    label="Draw Rate",
)

# Highlight COVID seasons
for i, row in season_summary.iterrows():
    if row["COVID"]:
        plt.axvline(x=i, color="red", linestyle="--", alpha=0.3)

plt.title("EPL Match Outcome Rates by Season")
plt.xlabel("Season")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Step 4: Overall bar plot
# -----------------------------
overall_rates = matches["FTR"].value_counts(normalize=True).reindex(["H", "A", "D"])

overall_rates.index = ["Home Wins", "Away Wins", "Draws"]

plt.figure(figsize=(8, 5))
overall_rates.plot(kind="bar")

plt.title("Overall EPL Match Outcome Distribution")
plt.xlabel("Result Type")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 5: Chi-square test
# -----------------------------
observed = matches["FTR"].value_counts().reindex(["H", "A", "D"], fill_value=0)
expected = [len(matches) / 3] * 3

chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

print("\nChi-Square Test Results:")
print("Observed counts:")
print(observed)
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.6f}")

print("\nSeason summary with COVID flag:")
print(season_summary)