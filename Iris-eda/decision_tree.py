"""Decision tree classifier for the Iris dataset."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

os.makedirs("images", exist_ok=True)

# ---------- 1. Load data ----------
df = pd.read_csv("Iris.csv")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
classes = ["setosa", "versicolor", "virginica"]

X = df[features]
y = df["Species"]

# ---------- 2. Train/test split (80/20, keeping species proportions equal) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train), "| Test samples:", len(X_test))

# ---------- 3. Train the decision tree ----------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nTree depth:", model.get_depth(), "| Leaves:", model.get_n_leaves())

# ---------- 4. Evaluate on the test set ----------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

print("\n--- Test set performance ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (macro average)")
print(f"Recall   : {recall:.4f}  (macro average)")

print("\nPer-class precision and recall:")
print(classification_report(y_test, y_pred, labels=classes, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=classes)
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=classes, columns=classes))

# ---------- 5. Cross-validation (checks the result is not just luck of one split) ----------
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5)
print("\n5-fold cross-validation accuracy:", np.round(cv_scores, 3))
print(f"Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")

# ---------- 6. Feature importance ----------
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:\n", importance.round(3))

# ---------- 7. Plots ----------
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 10,
})

# Confusion matrix
fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(cm, cmap="Blues", vmin=0)
ax.set_xticks(range(3), classes)
ax.set_yticks(range(3), classes)
ax.set_xlabel("Predicted species")
ax.set_ylabel("Actual species")
ax.set_title("Confusion matrix (test set)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
fig.colorbar(im, ax=ax, label="Number of flowers")
fig.tight_layout()
fig.savefig("images/9_confusion_matrix.png", dpi=130)
plt.close(fig)

# The tree itself
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(model, feature_names=features, class_names=classes, filled=True,
          rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision tree")
fig.tight_layout()
fig.savefig("images/10_decision_tree.png", dpi=130)
plt.close(fig)

# Feature importance
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(importance.index[::-1], importance.values[::-1], color="#2a78d6")
ax.set_xlabel("Importance")
ax.set_title("Feature importance")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig("images/11_feature_importance.png", dpi=130)
plt.close(fig)

print("\nSaved 3 plots to the images/ folder.")
