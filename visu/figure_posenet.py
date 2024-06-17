import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def generate_colormap(base_color, lvl=0.2, alpha=0.5):
    # Convert base color to RGB values between 0 and 1
    base_color = np.array(base_color) / 255.0

    # Define the color at the extremes (0 and 1)
    lighter_color = base_color + lvl
    darker_color = base_color - lvl

    # Ensure colors are within valid range
    lighter_color = np.clip(lighter_color, 0, 1)
    darker_color = np.clip(darker_color, 0, 1)

    # Generate colormap
    colors = [lighter_color, base_color, darker_color]
    positions = [0, alpha, 1]
    return LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))


o = generate_colormap([224, 157, 52, 255])
b = generate_colormap([57, 84, 122, 255])


# Load data
df = pd.read_csv(os.path.join(HITCHHIKING_ROOT_DIR, "results", "dense_fusion", "dense_fusion_experiment.csv"))

plt.style.use(os.path.join(HITCHHIKING_ROOT_DIR, "assets", "prettyplots.mplstyle"))
sns.set_style("whitegrid")


# Define symbols for each method
method_symbols = {
    "$\mathbb{R}^9$+SVD": "D",  # square
    "$\mathbb{R}^6$+GSO": "h",  # diamond
    "Quat$^+$": "*",  # circle
    "Euler": "X",  # cross
}

# Define colors for each metric
metric_colors = {
    "AUC ADD-S": o,
    "<2cm": b,
}

# Set up colormap for gradient based on scores
# Get unique methods and assign x-values to them
unique_methods = df["method"].unique()
method_indices = {method: idx for idx, method in enumerate(unique_methods)}

i = 0
# Plotting
fig = plt.figure(figsize=(5, 3))
for method, symbol in method_symbols.items():
    for metric, color in metric_colors.items():
        sub_df = df[(df["method"] == method) & (df["metric"] == metric)]
        scores = sub_df["score"]

        min_v = df[(df["metric"] == metric)]["score"].min()
        max_v = df[(df["metric"] == metric)]["score"].max()
        # Normalize scores for gradient colormap
        normalized_scores = (scores - min_v.min()) / (max_v.max() - min_v.min())
        x_values = [method_indices[method]] * len(sub_df.index)
        x_values += np.random.uniform(-0.15, 0.15, len(x_values))  # Add jitter to x-values
        plt.scatter(x_values, scores, c=color(normalized_scores), marker=symbol, edgecolor="black", linewidth=0.5, s=70)

        # Markers for legend - put them on negative y-axis
        if i == 0:
            plt.scatter(
                x_values,
                scores * -1,
                c=o([0.5] * len(scores)),
                marker="s",
                label="AUC ADD-S",
                edgecolor="black",
                linewidth=0.5,
                s=70,
            )
            plt.scatter(
                x_values,
                scores * -1,
                c=b([0.5] * len(scores)),
                marker="s",
                label="<2cm",
                edgecolor="black",
                linewidth=0.5,
                s=70,
            )
            i = 1

# Limity y-axis to not see the markers on negative side
fig.axes[0].set_ylim([90.2, 95.8])

plt.legend(
    title="",
    bbox_to_anchor=(0.5, 1.15),
    loc="upper center",
    ncol=len(method_symbols),
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.5,
    markerscale=1.0,
)
plt.xticks(range(len(unique_methods)), unique_methods)  # Set ticks to method names
plt.ylabel("Score")
plt.grid(True, linestyle="--", color="gray", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(HITCHHIKING_ROOT_DIR, "results", "dense_fusion", "figure_posenet.pdf"))
plt.show()
