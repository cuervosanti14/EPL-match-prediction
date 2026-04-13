from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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


def read_standard_table(path):
    """
    Read the squad standard stats table.
    FBref standard tables usually need header=1 because of grouped headers.
    """
    return pd.read_csv(path, header=1)


def read_points_table(path):
    """
    Read the league table and handle files with or without headers.
    """
    df = pd.read_csv(path)

    # Normal case: file already has headers
    if "Squad" in df.columns and "Pts" in df.columns:
        return df

    # Fallback: file was saved without headers
    df = pd.read_csv(path, header=None)

    expected_cols = [
        "Rk", "Squad", "MP", "W", "D", "L", "GF", "GA",
        "GD", "Pts", "Pts/MP", "Attendance",
        "Top Team Scorer", "Goalkeeper", "Notes"
    ]

    if df.shape[1] >= len(expected_cols):
        df = df.iloc[:, :len(expected_cols)].copy()
        df.columns = expected_cols

    return df


def load_one_season(season):
    """Load one season's standard stats and points tables, then merge them."""
    base_dir = Path(__file__).resolve().parents[1]
    standard_path = base_dir / "data" / "possession" / f"{season}_standard.csv"
    points_path = base_dir / "data" / "possession" / f"{season}_points.csv"

    std_df = read_standard_table(standard_path)
    pts_df = read_points_table(points_path)

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

    std_df["team"] = std_df["team"].astype(str).str.strip()
    pts_df["team"] = pts_df["team"].astype(str).str.strip()

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


def load_all_seasons():
    """Load and combine all seasons."""
    frames = []

    for season in SEASONS:
        season_df = load_one_season(season)
        frames.append(season_df)

    return pd.concat(frames, ignore_index=True)


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


def run_logistic_regression_50_points(df):
    """
    Logistic regression:
    1 = team earned 50 or more points
    0 = team earned 49 or fewer points
    """
    df = df.copy()
    df["high_points"] = (df["points"] >= 50).astype(int)

    X = df[["possession"]]
    y = df["high_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2500, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Logistic Regression: 50+ Points ===")
    print("Coefficient for possession:", model.coef_[0][0])
    print("Intercept:", model.intercept_[0])

    print("\nAccuracy Score:")
    print(round(acc, 4))

    print("\nConfusion Matrix:")
    print(cm)
    print("Rows = actual class [0, 1]")
    print("Columns = predicted class [0, 1]")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc, cm


def main():
    df = load_all_seasons()

    print("\nMerged data:")
    print(df.head())

    print("\nDescriptive stats by points group:")
    print(descriptive_stats_by_points_group(df))

    print("\nCorrelation between possession and goals:")
    print(df["possession"].corr(df["goals"]))

    plot_boxplot_possession_by_points_group(df)
    plot_mean_possession_by_points_group(df)
    plot_scatter_possession_vs_goals(df)

    run_logistic_regression_50_points(df)

    df.to_csv("epl_possession_points_goals_2016_2017_to_2024_2025.csv", index=False)
    print("\nSaved dataset to epl_possession_points_goals_2016_2017_to_2024_2025.csv")


if __name__ == "__main__":
    main()