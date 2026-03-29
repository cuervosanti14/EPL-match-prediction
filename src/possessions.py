from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SEASONS = [
    "2024-2025",
    "2023-2024",
    "2022-2023",
    "2021-2022",
    "2020-2021",
    "2019-2020",
    "2018-2019",
    "2017-2018",
    "2016-2017",
]


def assign_points_group(points):
    """Assign a team to a final-points group."""
    if 0 <= points <= 20:
        return "0-20"
    elif 21 <= points <= 40:
        return "21-40"
    elif 41 <= points <= 60:
        return "41-60"
    elif 61 <= points <= 80:
        return "61-80"
    elif 81 <= points <= 100:
        return "81-100"
    else:
        return "Outside range"


def load_one_season(season):
    """Load one season's standard stats and points tables, then merge them."""
    base_dir = Path(__file__).resolve().parents[1]
    standard_path = base_dir / "data" / "possession" / f"{season}_standard.csv"
    points_path = base_dir / "data" / "possession" / f"{season}_points.csv"

    std_df = pd.read_csv(standard_path, header=1)
    pts_df = pd.read_csv(points_path)

    print(f"\nLoading season: {season}")
    print("STANDARD COLUMNS:")
    print(std_df.columns.tolist())

    print("\nPOINTS COLUMNS:")
    print(pts_df.columns.tolist())

    std_df = std_df.rename(columns={
        "Squad": "team",
        "Poss": "possession",
        "Gls": "goals"
    })

    pts_df = pts_df.rename(columns={
        "Squad": "team",
        "Pts": "points"
    })

    std_df = std_df[["team", "possession", "goals"]].copy()
    pts_df = pts_df[["team", "points"]].copy()

    std_df["possession"] = (
        std_df["possession"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    std_df["possession"] = pd.to_numeric(std_df["possession"], errors="coerce")

    std_df["goals"] = pd.to_numeric(std_df["goals"], errors="coerce")
    pts_df["points"] = pd.to_numeric(pts_df["points"], errors="coerce")

    df = pd.merge(std_df, pts_df, on="team", how="inner")
    df["season"] = season
    df["points_group"] = df["points"].apply(assign_points_group)

    df = df.dropna(subset=["team", "possession", "goals", "points"]).copy()

    return df


def descriptive_stats_by_points_group(df):
    """Compute mean, median, std, and count of possession by points group."""
    return (
        df.groupby("points_group")["possession"]
        .agg(["mean", "median", "std", "count"])
        .reindex(["0-20", "21-40", "41-60", "61-80", "81-100"])
    )


def plot_boxplot_possession_by_points_group(df):
    """Boxplot of possession by final points group."""
    groups = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    data = [df[df["points_group"] == g]["possession"].dropna() for g in groups]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=groups)
    plt.title("Average Possession by Final Points Group")
    plt.xlabel("Final Points Group")
    plt.ylabel("Average Possession (%)")
    plt.tight_layout()
    plt.show()


def plot_mean_possession_by_points_group(df):
    """Bar chart of mean possession by final points group."""
    groups = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    means = (
        df.groupby("points_group")["possession"]
        .mean()
        .reindex(groups)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values)
    plt.title("Mean Possession by Final Points Group")
    plt.xlabel("Final Points Group")
    plt.ylabel("Mean Possession (%)")
    plt.tight_layout()
    plt.show()


def plot_scatter_possession_vs_goals(df):
    """Scatterplot of possession vs goals scored."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df["possession"], df["goals"])
    plt.title("Season Goals vs Average Possession")
    plt.xlabel("Average Possession (%)")
    plt.ylabel("Goals Scored")
    plt.tight_layout()
    plt.show()


def main():
    frames = []

    for season in SEASONS:
        season_df = load_one_season(season)
        frames.append(season_df)

    df = pd.concat(frames, ignore_index=True)

    print("\nMerged data:")
    print(df.head())

    print("\nDescriptive stats by points group:")
    print(descriptive_stats_by_points_group(df))

    print("\nCorrelation between possession and goals:")
    print(df["possession"].corr(df["goals"]))

    plot_boxplot_possession_by_points_group(df)
    plot_mean_possession_by_points_group(df)
    plot_scatter_possession_vs_goals(df)

    df.to_csv("epl_possession_points_goals_2016_2017_to_2024_2025.csv", index=False)
    print("\nSaved dataset to epl_possession_points_goals_2016_2017_to_2024_2025.csv")


if __name__ == "__main__":
    main()