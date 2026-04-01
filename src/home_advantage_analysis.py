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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 6: Binary Logistic Regression
# Predict whether the home team wins
# -----------------------------
binary_data = matches[["HTHG", "HTAG", "FTR"]].dropna().copy()
binary_data["HomeWin"] = (binary_data["FTR"] == "H").astype(int)

X = binary_data[["HTHG", "HTAG"]]
y = binary_data["HomeWin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nBinary Logistic Regression Results (Home Win vs Not Home Win):")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))