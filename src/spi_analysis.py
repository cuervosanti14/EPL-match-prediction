import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load data
df = pd.read_csv("data/spi_matches.csv")

# Filter to EPL & 2016-2023 seasons
df = df[df["league"] == "Barclays Premier League"]
print(df["season"].unique())

# Select Relevant Columns
df = df[["season", "date", "team1", "team2", 
         "spi1", "spi2", "prob1", "prob2", 
         "probtie", "score1", "score2"]]

# Handle missing values
print("Missing values per column:")
print(df.isnull().sum())

df = df.dropna()

# Confirm data looks right
print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Feature engineering - SPI Differental
df["spi_diff"] = df["spi1"] - df["spi2"]

# Derive actual match result
def get_result(row):
    if row["score1"] > row["score2"]:
        return "team1_win"
    elif row["score1"] < row["score2"]:
        return "team2_win"
    else:
        return "draw"
    
df["actual_result"] = df.apply(get_result, axis=1)

# Favored team win rate
df["favored_won"] = (
    ((df["spi_diff"] > 0) & (df["actual_result"] == "team1_win")) |
    ((df["spi_diff"] < 0) & (df["actual_result"] == "team2_win"))
)

# Bin matches by SPI gap
bins = [0, 5, 10, 15, 20, float("inf")]
labels = ["0-5", "5-10", "10-15", "15-20", "20+"]
df["spi_gap_bin"] = pd.cut(df["spi_diff"].abs(), bins=bins, labels=labels)

# Descriptive statistics
print("Overrall descriptive statistics:")
print(df[["spi_diff", "prob1", "prob2", "probtie"]].describe())

print("\nResult distribution (counts):")
print(df["actual_result"].value_counts())

print("\nResult distribution (percentages):")
print(df["actual_result"].value_counts(normalize=True))

print("\nMean prob1 by SPI gap bin:")
print(df.groupby("spi_gap_bin")["prob1"].mean())

# Visualization 1 - Bar plot: win rates by SPI gap bin
win_rates = df.groupby("spi_gap_bin")["favored_won"].mean()

plt.figure(figsize=(8, 5))
win_rates.plot(kind="bar", color="steelblue", edgecolor="black")
for i, v in enumerate(win_rates):
    plt.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10,fontweight="bold")
plt.title("Favored Team Win Rate by SPI Gap Bin")
plt.xlabel("SPI Gap Bin")
plt.ylabel("Win Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visuals/win_rate_by_spi_bin.png")
plt.close()

# Visualization 2 - Scatter plot: SPI gap vs prob1
colors = df["actual_result"].map({
    "team1_win": "green",
    "team2_win": "red",
    "draw": "gray"
})

plt.figure(figsize=(8, 5))
plt.scatter(df["spi_diff"], df["prob1"], alpha=0.3, c=colors)
plt.title("SPI Gap vs. Predicted Win Probability (prob1)")
plt.xlabel("SPI Differential (team1 - team2)")
plt.ylabel("Predicted Win Probability (prob1)")

# Add legend
legend = [
    Patch(color="green", label="Team1 Win"),
    Patch(color="red", label="Team1 Loss"),
    Patch(color="gray", label="Draw")
]

plt.legend(handles=legend, title="Actual Result")
plt.tight_layout()
plt.savefig("visuals/spi_gap_vs_prob1.png")
plt.close()

# Visualization 3- Boxplot: SPI gap by actual result
df.boxplot(column="spi_diff", by="actual_result", figsize=(8,5),grid=False)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.title("SPI Gap Distribution by Match Result")
plt.suptitle("")
plt.xlabel("Actual Result")
plt.ylabel("SPI Differential")
plt.tight_layout()
plt.savefig("visuals/spi_gap_by_result.png")
plt.close()

print("All visualizations saved to visuals/ folder.")