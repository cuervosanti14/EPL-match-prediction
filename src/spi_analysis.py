import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

# Load raw SPI dataset
df = pd.read_csv("data/SPI/spi_matches.csv")

# Filter to English Premier League only
df = df[df["league"] == "Barclays Premier League"]
print("Seasons available:",df["season"].unique())

# Select only the columns relevant to this analysis
df = df[["season", "date", "team1", "team2", 
         "spi1", "spi2", "prob1", "prob2", 
         "probtie", "score1", "score2"]]

# Handle missing values
print("\nMissing values per column:")
print(df.isnull().sum())

df = df.dropna()

# Confirm data was cleaned properly
print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Calculate SPI differential
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

# Flag whether the favored (higher SPI) team won
df["favored_won"] = (
    ((df["spi_diff"] > 0) & (df["actual_result"] == "team1_win")) |
    ((df["spi_diff"] < 0) & (df["actual_result"] == "team2_win"))
)

# Bin matches by SPI gap (absolute value)
bins = [0, 5, 10, 15, 20, float("inf")]
labels = ["0-5", "5-10", "10-15", "15-20", "20+"]
df["spi_gap_bin"] = pd.cut(df["spi_diff"].abs(), bins=bins, labels=labels)

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\nOverall descriptive statistics:")
print(df[["spi_diff", "prob1", "prob2", "probtie"]].describe())

# How often each outcome (win,draw,loss) occurs
print("\nResult distribution (counts):")
print(df["actual_result"].value_counts())

print("\nResult distribution (percentages):")
print(df["actual_result"].value_counts(normalize=True))

# Average predicted win probability for team1 within each SPI gap bin
print("\nMean prob1 by SPI gap bin:")
print(df.groupby("spi_gap_bin")["prob1"].mean())

# ============================================================
# 4. VISUALIZATIONS
# ============================================================

# --- Visualization 1: Favored team win rate by SPI gap bin ---
# Shows whether larger SPI gaps lead to more predictable outcomes
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
plt.savefig("visuals//SPI/win_rate_by_spi_bin.png")
plt.close()

# --- Visualization 2: SPI gap vs predicted win probability (prob1) ---
# Colored by actual result to show how well predictions align with outcomes
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

legend = [
    Patch(color="green", label="Team1 Win"),
    Patch(color="red", label="Team1 Loss"),
    Patch(color="gray", label="Draw")
]

plt.legend(handles=legend, title="Actual Result")
plt.tight_layout()
plt.savefig("visuals/SPI/spi_gap_vs_prob1.png")
plt.close()

# --- Visualization 3: SPI gap distribution by actual result ---
# Shows how SPI differential is distributed across wins, losses, and draws
df.boxplot(column="spi_diff", by="actual_result", figsize=(8,5),grid=False)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.title("SPI Gap Distribution by Match Result")
plt.suptitle("")
plt.xlabel("Actual Result")
plt.ylabel("SPI Differential")
plt.tight_layout()
plt.savefig("visuals/SPI/spi_gap_by_result.png")
plt.close()

print("All visualizations saved to visuals/ folder.")

# ============================================================
# 5. KNN MODEL
# ============================================================

# Normalize spi_diff using z-score
scaler = StandardScaler()
df["spi_diff_scaled"] = scaler.fit_transform(df[["spi_diff"]])

# Define features and target
X = df[["spi_diff_scaled"]]
y = df["actual_result"]

# Split into train (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# Find best K using validation set
val_accuracies = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    val_accuracies.append(acc)
    print(f"K={k}: Validation Accuracy = {acc:.4f}")

# Plot validation accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(k_values, val_accuracies, marker="o", color="steelblue")
plt.title("Validation Accuracy vs. K")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("visuals/SPI/knn_validation_accuracy.png")
plt.close()

best_k = k_values[val_accuracies.index(max(val_accuracies))]
print("\nBest K:", best_k)
print("Best Validation Accuracy:", max(val_accuracies))

# --- Final model: train with best K and evaluate on test set ---
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
test_pred = knn_final.predict(X_test)

# Test accuracy
knn_accuracy = accuracy_score(y_test, test_pred)
knn_f1 = f1_score(y_test, test_pred, average="weighted")
knn_precision = precision_score(y_test, test_pred, average="weighted", zero_division=0)
knn_recall = recall_score(y_test, test_pred, average="weighted")
print(f"KNN Test Accuracy: {knn_accuracy:.4f}")
print(f"KNN Test Precision: {knn_precision:.4f}")
print(f"KNN Test Recall: {knn_recall:.4f}")
print(f"KNN Test F1 Score: {knn_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=["team1_win", "team2_win", "draw"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=["team1_win", "team2_win", "draw"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"KNN Confusion Matrix (Test Set, K={best_k})")
plt.tight_layout()
plt.savefig("visuals/SPI/knn_confusion_matrix.png")
plt.close()

# ============================================================
# 6. LOGISTIC REGRESSION MODEL
# ============================================================

# Train logistic regression using same train/test split as KNN
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Evaluate on test set
test_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, test_pred_lr)
lr_precision = precision_score(y_test, test_pred_lr, average="weighted", zero_division=0)
lr_recall = recall_score(y_test, test_pred_lr, average="weighted")
lr_f1 = f1_score(y_test, test_pred_lr, average="weighted")
print(f"\nLR Test Accuracy: {lr_accuracy:.4f}")
print(f"LR Test Precision: {lr_precision:.4f}")
print(f"LR Test Recall: {lr_recall:.4f}")
print(f"LR Test F1 Score: {lr_f1:.4f}")

# Confusion matrix
cm_lr = confusion_matrix(y_test, test_pred_lr, labels=["team1_win", "team2_win", "draw"])
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                  display_labels=["team1_win", "team2_win", "draw"])
disp_lr.plot(cmap="Blues", values_format="d")
plt.title("Logistic Regression Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("visuals/SPI/lr_confusion_matrix.png")
plt.close()

# ============================================================
# 7. FIVETHIRTYEIGHT MODEL COMPARISON
# ============================================================

# Convert SPI dataset's probabilities into predicted outcomes
# by taking whichever probability is highest for each match
df["SPI_dataset_prediction"] = df[["prob1", "prob2", "probtie"]].idxmax(axis=1).map({
    "prob1": "team1_win",
    "prob2": "team2_win",
    "probtie": "draw"
})

# Evaluate SPI predictions against actual results
spi_dataset_acc = accuracy_score(df["actual_result"], df["SPI_dataset_prediction"])
spi_dataset_precision = precision_score(df["actual_result"], df["SPI_dataset_prediction"], average="weighted", zero_division=0)
spi_dataset_recall = recall_score(df["actual_result"], df["SPI_dataset_prediction"], average="weighted")
spi_dataset_f1 = f1_score(df["actual_result"], df["SPI_dataset_prediction"], average="weighted")

print(f"\nSPI Dataset Accuracy: {spi_dataset_acc:.4f}")
print(f"SPI Dataset Precision: {spi_dataset_precision:.4f}")
print(f"SPI Dataset Recall: {spi_dataset_recall:.4f}")
print(f"SPI Dataset F1 Score: {spi_dataset_f1:.4f}")

# Summary comparison table
print("\n--- Model Comparison ---")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
print(f"{'Random Baseline':<25} {'0.3333':>10} {'-':>10} {'-':>10} {'-':>10}")
print(f"{'Naive Baseline':<25} {'0.4501':>10} {'-':>10} {'-':>10} {'-':>10}")
print(f"{'KNN (K=20)':<25} {knn_accuracy:>10.4f} {knn_precision:>10.4f} {knn_recall:>10.4f} {knn_f1:>10.4f}")
print(f"{'Logistic Regression':<25} {lr_accuracy:>10.4f} {lr_precision:>10.4f} {lr_recall:>10.4f} {lr_f1:>10.4f}")
print(f"{'SPI':<25} {spi_dataset_acc:>10.4f} {spi_dataset_precision:>10.4f} {spi_dataset_recall:>10.4f} {spi_dataset_f1:>10.4f}")

df.head(1000).to_csv("data/SPI/spi_matches_reduced.csv", index=False)