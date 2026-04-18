import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_and_clean_data(filepath):
    """
    Load the SPI match data, filter to EPL matches, and create cleaned features.
    """
    matches = pd.read_csv(filepath)

    matches = matches[matches["league"] == "Barclays Premier League"].copy()

    matches = matches[[
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
        "score1",
        "score2"
    ]]

    matches = matches.rename(columns={
        "date": "match_date",
        "team1": "home_team",
        "team2": "away_team",
        "spi1": "home_team_spi",
        "spi2": "away_team_spi",
        "prob1": "home_win_prob",
        "probtie": "draw_prob",
        "prob2": "away_win_prob",
        "proj_score1": "home_expected_goals",
        "proj_score2": "away_expected_goals",
        "score1": "home_goals",
        "score2": "away_goals"
    })

    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna().copy()

    matches["actual_draw"] = (matches["home_goals"] == matches["away_goals"]).astype(int)
    matches["spi_gap"] = (matches["home_team_spi"] - matches["away_team_spi"]).abs()

    matches = matches.sort_values("match_date").reset_index(drop=True)

    return matches


def split_data(df):
    """
    Split the dataset into training, validation, and test sets.
    """
    X = df[["spi_gap", "draw_prob"]]
    y = df["actual_draw"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2500, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, random_state=2500, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def feature_selection(X_train, X_val, X_test, features):
    """
    Select only the specified features from each feature set.
    """
    X_train_selected = X_train[features].copy()
    X_val_selected = X_val[features].copy()
    X_test_selected = X_test[features].copy()

    return X_train_selected, X_val_selected, X_test_selected


def scale_features(X_train, X_val, X_test):
    """
    Scale the features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, k=5):
    """
    Train a KNN classifier on the training data.
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """
    Evaluate a trained model and return accuracy, F1 score, and confusion matrix.
    """
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    matrix = confusion_matrix(y, predictions)

    return accuracy, f1, matrix


def plot_draw_rate_by_season(matches):
    """
    Plot draw rate by season.
    """
    draw_by_season = matches.groupby("season")["actual_draw"].mean().sort_index()

    plt.figure(figsize=(10, 5))
    plt.plot(draw_by_season.index, draw_by_season.values, marker="o")
    plt.title("Premier League Draw Rate by Season")
    plt.xlabel("Season")
    plt.ylabel("Draw Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, max(draw_by_season.values) + 0.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/draw_prob/draw_rate_by_season.png")
    plt.close()

    return draw_by_season


def plot_draw_rate_by_spi_gap(matches):
    """
    Plot draw rate by SPI gap bin.
    """
    matches = matches.copy()

    matches["spi_bucket"] = pd.cut(
        matches["spi_gap"],
        bins=[0, 5, 10, 15, 20, 40],
        include_lowest=True
    )

    draw_rates_by_gap = matches.groupby("spi_bucket")["actual_draw"].mean()

    plt.figure(figsize=(9, 5))
    draw_rates_by_gap.plot(kind="bar")
    plt.title("Draw Rate by SPI Gap")
    plt.xlabel("SPI Gap Bin")
    plt.ylabel("Draw Rate")
    plt.xticks(rotation=30)
    plt.ylim(0, max(draw_rates_by_gap.values) + 0.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/draw_prob/draw_rate_by_spi_gap.png")
    plt.close()

    return draw_rates_by_gap


def plot_predicted_vs_actual_draw(matches):
    """
    Plot predicted draw probability against actual draw frequency.
    """
    matches = matches.copy()

    matches["draw_prob_bucket"] = pd.cut(
        matches["draw_prob"],
        bins=[0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        include_lowest=True
    )

    draw_prob_compare = matches.groupby("draw_prob_bucket").agg(
        predicted_draw_prob=("draw_prob", "mean"),
        actual_draw_rate=("actual_draw", "mean"),
        matches_in_bin=("actual_draw", "size")
    ).reset_index()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        draw_prob_compare["predicted_draw_prob"],
        draw_prob_compare["actual_draw_rate"],
        s=80
    )

    for i in range(len(draw_prob_compare)):
        plt.text(
            draw_prob_compare["predicted_draw_prob"].iloc[i],
            draw_prob_compare["actual_draw_rate"].iloc[i],
            str(draw_prob_compare["draw_prob_bucket"].iloc[i]),
            fontsize=8
        )

    min_val = min(draw_prob_compare["predicted_draw_prob"].min(),
                  draw_prob_compare["actual_draw_rate"].min())
    max_val = max(draw_prob_compare["predicted_draw_prob"].max(),
                  draw_prob_compare["actual_draw_rate"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title("Predicted Draw Probability vs Actual Draw Frequency")
    plt.xlabel("Average Predicted Draw Probability")
    plt.ylabel("Actual Draw Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/draw_prob/predicted_vs_actual_draw.png")
    plt.close()

    return draw_prob_compare


def main():
    matches = load_and_clean_data("data/SPI/spi_matches.csv")
    
    matches.head(1000).to_csv("data/draw_prob/spi_matches_reduced.csv", index=False)

    matches.to_csv("data/draw_prob/premier_league_matches_cleaned.csv", index=False)

    print("First 5 rows:")
    print(matches.head())
    print()

    print("Dataset shape:")
    print(matches.shape)
    print()

    print("Overall draw rate:")
    print(round(matches["actual_draw"].mean(), 3))
    print()

    draw_by_season = matches.groupby("season")["actual_draw"].mean().sort_index()
    print("Draw rate by season:")
    print(draw_by_season)
    print()

    matches["spi_bucket"] = pd.cut(
        matches["spi_gap"],
        bins=[0, 5, 10, 15, 20, 40],
        include_lowest=True
    )
    draw_rates_by_gap = matches.groupby("spi_bucket")["actual_draw"].mean()
    print("Draw rate by SPI gap:")
    print(draw_rates_by_gap)
    print()

    correlation = matches[["draw_prob", "actual_draw"]].corr()
    print("Correlation between draw_prob and actual_draw:")
    print(correlation)
    print()

    plot_draw_rate_by_season(matches)
    plot_draw_rate_by_spi_gap(matches)
    plot_predicted_vs_actual_draw(matches)

    print("KNN model:")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(matches)

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    features = ["spi_gap", "draw_prob"]
    X_train_sel, X_val_sel, X_test_sel = feature_selection(
        X_train, X_val, X_test, features
    )

    print(f"Selected features: {features}")
    print()

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_sel, X_val_sel, X_test_sel
    )

    model = train_model(X_train_scaled, y_train, k=5)

    train_acc, train_f1, train_cm = evaluate_model(model, X_train_scaled, y_train)
    val_acc, val_f1, val_cm = evaluate_model(model, X_val_scaled, y_val)
    test_acc, test_f1, test_cm = evaluate_model(model, X_test_scaled, y_test)

    print(f"Training Set - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print("Training Confusion Matrix:")
    print(train_cm)
    print()

    print(f"Validation Set - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    print("Validation Confusion Matrix:")
    print(val_cm)
    print()

    print(f"Test Set - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)
    print()

    print("EDA and KNN modeling complete.")

if __name__ == "__main__":
    main()