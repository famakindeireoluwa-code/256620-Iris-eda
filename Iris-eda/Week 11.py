import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")
df["Species"] = df["Species"].str.replace("Iris-", "")
cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

# Overview and summary statistics
print(df.shape)
print(df.isnull().sum())
print(df["Species"].value_counts())
print(df[cols].describe())
print(df.groupby("Species")[cols].mean())
print(df[cols].corr())

# Histograms
df[cols].hist(bins=15, figsize=(8, 6))
plt.savefig("histograms.png")

# Box plots by species
df.boxplot(column=cols, by="Species", figsize=(10, 7))
plt.savefig("boxplots.png")

# Scatter plot of petal length vs petal width
colors = {"setosa": "blue", "versicolor": "orange", "virginica": "green"}
plt.figure()
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"], c=df["Species"].map(colors))
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.savefig("scatter.png")

# Pair plot (scatter matrix)
pd.plotting.scatter_matrix(df[cols], c=df["Species"].map(colors), figsize=(8, 8))
plt.savefig("pairplot.png")
