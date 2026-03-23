import pandas as pd

# Load data
df = pd.read_csv("data/spi_matches.csv")

# Filter to EPL & 2016-2023 seasons
df = df[df["league"] == "Barclays Premier League"]
print(df["season"].unique())
# df = df[df["season"].between(2016, 2023)]

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