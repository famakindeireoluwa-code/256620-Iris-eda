import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("Iris.csv")
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"].str.replace("Iris-", "")

# Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train the decision tree and predict
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluate
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro"))
print("Recall   :", recall_score(y_test, pred, average="macro"))
print(confusion_matrix(y_test, pred))

# Plot the tree
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.savefig("tree.png")
