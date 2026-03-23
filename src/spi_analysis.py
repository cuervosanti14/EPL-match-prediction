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