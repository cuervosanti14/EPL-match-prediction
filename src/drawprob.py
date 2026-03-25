import pandas as pd

df = pd.read_csv("../data/spi_matches.csv")

df = df[df["league"] == "Barclays Premier League"]

df = df[[
    "season",
    "date",
    "league",
    "team1",
    "team2",
    "spi1",
    "spi2",
    "prob1",
    "probtie",
    "prob2",
    "proj_score1",
    "proj_score2"
]]

df = df.rename(columns={
    "team1": "home_team",
    "team2": "away_team"
})

df = df.dropna()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df = df.reset_index(drop=True)

print(df.head())

df.to_csv("../data/prem_spi_cleaned.csv", index=False)

print("cleaned file saved")