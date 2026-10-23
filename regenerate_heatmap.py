"""
Regenerate fig8 Jaccard heatmap: lower triangle only, top 20, larger text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

metrics = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))

# Top 20 conditions by degree
N = 20
top = metrics.nlargest(N, "degree_centrality")
top_ids = top["condition_concept_id"].tolist()
top_names = []
for name in top["condition_name"]:
    name = (name.replace("Electrocardiogram", "ECG")
                .replace("Essential hypertension", "Hypertension")
                .replace("Abnormal findings on diagnostic imaging", "Abn. diagnostic imaging")
                .replace("Acute renal failure syndrome", "Acute renal failure")
                .replace("Gastroesophageal reflux disease", "GERD"))
    if len(name) > 26:
        name = name[:24] + "..."
    top_names.append(name)

# Build Jaccard lookup
edge_lookup = {}
for _, row in edges.iterrows():
    edge_lookup[(row["condition_id_a"], row["condition_id_b"])] = row["jaccard"]
    edge_lookup[(row["condition_id_b"], row["condition_id_a"])] = row["jaccard"]

# Build matrix
jaccard_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            jaccard_matrix[i, j] = 1.0
        else:
            jaccard_matrix[i, j] = edge_lookup.get((top_ids[i], top_ids[j]), 0)

# Mask upper triangle
mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool), k=1)
jaccard_masked = np.ma.array(jaccard_matrix, mask=mask)

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(jaccard_masked, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(top_names, fontsize=9)

ax.set_title("Co-occurrence Heatmap (Jaccard Similarity)\nTop 20 Conditions by Degree — Lower Triangle",
             fontsize=14, fontweight="bold", pad=15)

cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Jaccard Similarity", fontsize=11)

# Annotate all visible cells (lower triangle including diagonal)
for i in range(N):
    for j in range(N):
        if not mask[i, j]:
            val = jaccard_matrix[i, j]
            if i == j:
                continue  # skip diagonal (all 1.0)
            color = "white" if val > 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

# Remove spines from masked area
for i in range(N):
    for j in range(N):
        if mask[i, j]:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=True, facecolor="white",
                                       edgecolor="white", linewidth=0))

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_jaccard_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: fig8_jaccard_heatmap.png")
