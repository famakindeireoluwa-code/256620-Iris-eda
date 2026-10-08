"""Exploratory data analysis of the Iris dataset (pandas + NumPy + matplotlib only)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save to files without opening a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------- Setup ----------
os.makedirs("images", exist_ok=True)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
species_order = ["setosa", "versicolor", "virginica"]
colors = {"setosa": "#2a78d6", "versicolor": "#eb6834", "virginica": "#1baf7a"}
markers = {"setosa": "o", "versicolor": "s", "virginica": "^"}  # second cue besides colour

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "font.size": 10,
})


def species_legend(ax, loc="best"):
    handles = [Line2D([0], [0], marker=markers[s], color="none", markerfacecolor=colors[s],
                      markeredgecolor="#fcfcfb", markersize=8, label=s) for s in species_order]
    ax.legend(handles=handles, title="Species", frameon=False, loc=loc)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join("images", name), dpi=130)
    plt.close(fig)


# ---------- 1. Load ----------
df = pd.read_csv("Iris.csv")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)

# ---------- 2. Overview ----------
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows (excluding Id):", df.drop(columns="Id").duplicated().sum())
print("\nClass balance:\n", df["Species"].value_counts())

# ---------- 3. Summary statistics ----------
print("\nOverall summary statistics:\n", df[features].describe().round(3))
print("\nMean by species:\n", df.groupby("Species")[features].mean().round(3))
print("\nMedian by species:\n", df.groupby("Species")[features].median().round(3))
print("\nStd by species:\n", df.groupby("Species")[features].std().round(3))
print("\nSkewness:\n", df[features].skew().round(3))
print("\nKurtosis:\n", df[features].kurt().round(3))
corr = df[features].corr()
print("\nCorrelation matrix:\n", corr.round(3))

print("\nOutliers (IQR rule):")
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col}: {n}")

# ---------- 4. Plots ----------
groups = {s: df[df["Species"] == s] for s in species_order}

# 4.1 Histograms of each feature (all species)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), features):
    ax.hist(df[col], bins=20, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
    ax.set_title(col)
    ax.set_ylabel("Count")
fig.suptitle("Feature distributions")
save(fig, "1_histograms.png")

# 4.2 Histograms by species (overlaid, step outline)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), features):
    bins = np.linspace(df[col].min(), df[col].max(), 21)
    for s in species_order:
        ax.hist(groups[s][col], bins=bins, color=colors[s], alpha=0.35)
        ax.hist(groups[s][col], bins=bins, histtype="step", color=colors[s], linewidth=2)
    ax.set_title(col)
    ax.set_ylabel("Count")
species_legend(axes[0, 1], loc="upper right")
fig.suptitle("Distributions by species")
save(fig, "2_distributions_by_species.png")

# 4.3 Box plots
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), features):
    data = [groups[s][col] for s in species_order]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=species_order, widths=0.5,
                    medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none",
                                "markeredgecolor": "#52514e"})
    for patch, s in zip(bp["boxes"], species_order):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.6)
    ax.set_title(col)
    ax.grid(axis="x", visible=False)
fig.suptitle("Box plots by species")
save(fig, "3_boxplots.png")

# 4.4 Violin plots
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), features):
    data = [groups[s][col] for s in species_order]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, s in zip(parts["bodies"], species_order):
        body.set_facecolor(colors[s])
        body.set_edgecolor("#52514e")
        body.set_alpha(0.6)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_edgecolor("#52514e")
    ax.set_xticks([1, 2, 3], species_order)
    ax.set_title(col)
    ax.grid(axis="x", visible=False)
fig.suptitle("Violin plots by species")
save(fig, "4_violinplots.png")

# 4.5 Pair plot (scatter matrix, histograms on the diagonal)
n = len(features)
fig, axes = plt.subplots(n, n, figsize=(11, 10))
for i, yc in enumerate(features):
    for j, xc in enumerate(features):
        ax = axes[i, j]
        if i == j:
            bins = np.linspace(df[xc].min(), df[xc].max(), 16)
            for s in species_order:
                ax.hist(groups[s][xc], bins=bins, color=colors[s], alpha=0.5)
        else:
            for s in species_order:
                ax.scatter(groups[s][xc], groups[s][yc], s=18, color=colors[s],
                           marker=markers[s], edgecolor="#fcfcfb", linewidth=0.5)
        if i == n - 1:
            ax.set_xlabel(xc, fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if j == 0:
            ax.set_ylabel(yc, fontsize=9)
        else:
            ax.tick_params(labelleft=False)
axes[0, 0].tick_params(labelleft=False)  # counts, not SepalLengthCm values
fig.suptitle("Pair plot")
handles = [Line2D([0], [0], marker=markers[s], color="none", markerfacecolor=colors[s],
                  markeredgecolor="#fcfcfb", markersize=8, label=s) for s in species_order]
fig.tight_layout(rect=(0, 0, 0.88, 0.98))
fig.legend(handles=handles, title="Species", frameon=False, loc="center right")
fig.savefig(os.path.join("images", "5_pairplot.png"), dpi=130)
plt.close(fig)

# 4.6 Correlation heatmap (diverging: blue = negative, orange = positive, grey midpoint)
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(n), features, rotation=30, ha="right")
ax.set_yticks(range(n), features)
ax.grid(False)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="#0b0b0b", fontsize=10)
fig.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Correlation heatmap")
save(fig, "6_correlation_heatmap.png")

# 4.7 Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (xc, yc, title) in zip(axes, [
        ("PetalLengthCm", "PetalWidthCm", "Petal length vs width"),
        ("SepalLengthCm", "SepalWidthCm", "Sepal length vs width")]):
    for s in species_order:
        ax.scatter(groups[s][xc], groups[s][yc], s=36, color=colors[s], marker=markers[s],
                   edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_xlabel(xc)
    ax.set_ylabel(yc)
    ax.set_title(title)
species_legend(axes[0], loc="upper left")
save(fig, "7_scatter.png")

# 4.8 Species counts
counts = df["Species"].value_counts().reindex(species_order)
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(species_order, counts.values, color=[colors[s] for s in species_order], width=0.6)
ax.bar_label(bars, color="#0b0b0b")
ax.set_ylabel("Samples")
ax.set_title("Samples per species")
ax.grid(axis="x", visible=False)
save(fig, "8_species_count.png")

print("\nSaved 8 plots to the images/ folder.")
