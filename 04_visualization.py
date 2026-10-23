"""
04_visualization.py
====================
Purpose:  Generate publication-quality figures for the T2D comorbidity network
          analysis. Produces figures for both the course project and WSDM paper.

Input:    data/analysis_node_metrics.csv
          data/analysis_communities.csv
          data/analysis_bridge_diagnoses.csv
          data/analysis_network_stats.csv
          data/analysis_subgroup_comparison.csv
          data/analysis_temporal_hubs.csv
          data/network_nodes.csv
          data/network_cooccurrence_edges.csv

Output:   figures/fig1_network_overview.png        — full network colored by community
          figures/fig2_degree_distribution.png      — degree distribution (log-log)
          figures/fig3_centrality_rankings.png      — top conditions by 4 centralities
          figures/fig4_community_composition.png    — community sizes and top members
          figures/fig5_bridge_diagnoses.png         — bridge conditions highlighted
          figures/fig6_temporal_gateways.png        — temporal in/out degree comparison
          figures/fig7_demographic_comparison.png   — subgroup network statistics
          figures/fig8_jaccard_heatmap.png          — top condition co-occurrence heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── Setup ────────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Load data ────────────────────────────────────────────────────────────
print("Loading data...")

metrics = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))
communities_csv = pd.read_csv(os.path.join(OUT_DIR, "analysis_communities.csv"))
bridges = pd.read_csv(os.path.join(OUT_DIR, "analysis_bridge_diagnoses.csv"))
net_stats = pd.read_csv(os.path.join(OUT_DIR, "analysis_network_stats.csv"))
subgroup_comp = pd.read_csv(os.path.join(OUT_DIR, "analysis_subgroup_comparison.csv"))
temporal_hubs = pd.read_csv(os.path.join(OUT_DIR, "analysis_temporal_hubs.csv"))
nodes = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))

# Build NetworkX graph
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row["condition_concept_id"], name=row["condition_name"],
               patient_count=row["patient_count"])
for _, row in edges.iterrows():
    G.add_edge(row["condition_id_a"], row["condition_id_b"],
               weight=row["jaccard"])

comm_map = dict(zip(metrics["condition_concept_id"], metrics["community"]))

# Community color palette (12 communities)
COMM_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324",
]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Network Overview (community-colored)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Network overview...")

fig, ax = plt.subplots(figsize=(14, 14))

# Use spring layout weighted by Jaccard
pos = nx.spring_layout(G, k=0.3, iterations=80, weight="weight", seed=42)

# Node colors by community, sizes by patient count
node_colors = [COMM_COLORS[comm_map.get(n, 0) % len(COMM_COLORS)] for n in G.nodes()]
node_sizes = [max(5, G.nodes[n].get("patient_count", 50) / 15) for n in G.nodes()]

# Draw edges (light gray, thin)
nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.3, edge_color="gray", ax=ax)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.8, linewidths=0.3, edgecolors="white", ax=ax)

# Label top 15 nodes by betweenness
top_nodes = metrics.nlargest(15, "betweenness_centrality")
labels = {}
for _, row in top_nodes.iterrows():
    cid = row["condition_concept_id"]
    name = row["condition_name"]
    # Shorten long names
    if len(name) > 25:
        name = name[:23] + "..."
    labels[cid] = name

nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", ax=ax)

# Legend for major communities (size >= 6)
major_comms = communities_csv[communities_csv["size"] >= 6].sort_values("size", ascending=False)
legend_patches = []
for _, crow in major_comms.iterrows():
    cid = crow["community_id"]
    top_cond = crow["top_conditions"].split(";")[0].strip()
    if len(top_cond) > 30:
        top_cond = top_cond[:28] + "..."
    legend_patches.append(
        mpatches.Patch(color=COMM_COLORS[cid % len(COMM_COLORS)],
                       label=f"C{cid} ({crow['size']}): {top_cond}")
    )

ax.legend(handles=legend_patches, loc="upper left", fontsize=7, framealpha=0.9,
          title="Communities", title_fontsize=8)

ax.set_title("T2D Comorbidity Network — Communities (Louvain)", fontsize=15)
ax.axis("off")
fig.savefig(os.path.join(FIG_DIR, "fig1_network_overview.png"))
plt.close(fig)
print("  Saved: figures/fig1_network_overview.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Degree Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 2: Degree distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

degrees = metrics["degree"].values

# Histogram
axes[0].hist(degrees, bins=40, color="#4363d8", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Number of Conditions")
axes[0].set_title("Degree Distribution")
axes[0].axvline(np.mean(degrees), color="red", linestyle="--", label=f"Mean={np.mean(degrees):.1f}")
axes[0].axvline(np.median(degrees), color="orange", linestyle="--", label=f"Median={np.median(degrees):.0f}")
axes[0].legend(fontsize=9)

# Log-log CCDF
sorted_deg = np.sort(degrees)[::-1]
ccdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
axes[1].loglog(sorted_deg, ccdf, "o", markersize=3, color="#4363d8", alpha=0.6)
axes[1].set_xlabel("Degree (log)")
axes[1].set_ylabel("P(X >= x) (log)")
axes[1].set_title("Degree CCDF (Log-Log)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("T2D Comorbidity Network — Degree Distribution", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_degree_distribution.png"))
plt.close(fig)
print("  Saved: figures/fig2_degree_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Centrality Rankings
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 3: Centrality rankings...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

centrality_measures = [
    ("degree_centrality", "Degree Centrality", "#e6194b"),
    ("betweenness_centrality", "Betweenness Centrality", "#4363d8"),
    ("closeness_centrality", "Closeness Centrality", "#3cb44b"),
    ("eigenvector_centrality", "Eigenvector Centrality", "#f58231"),
]

for ax, (col, title, color) in zip(axes.flat, centrality_measures):
    top = metrics.nlargest(15, col)
    names = [n[:30] if len(n) <= 30 else n[:28] + "..." for n in top["condition_name"]]
    values = top[col].values

    bars = ax.barh(range(len(names)), values, color=color, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_title(f"Top 15 by {title}")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=6)

fig.suptitle("T2D Comorbidity Network — Centrality Rankings", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_centrality_rankings.png"))
plt.close(fig)
print("  Saved: figures/fig3_centrality_rankings.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Community Composition
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 4: Community composition...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: community sizes bar chart
major = communities_csv[communities_csv["size"] >= 2].sort_values("size", ascending=True)
colors = [COMM_COLORS[int(cid) % len(COMM_COLORS)] for cid in major["community_id"]]
labels = [f"C{int(cid)}" for cid in major["community_id"]]

axes[0].barh(range(len(major)), major["size"].values, color=colors, edgecolor="white")
axes[0].set_yticks(range(len(major)))
axes[0].set_yticklabels(labels)
axes[0].set_xlabel("Number of Conditions")
axes[0].set_title("Community Sizes")

for i, (_, row) in enumerate(major.iterrows()):
    top_cond = row["top_conditions"].split(";")[0].strip()
    if len(top_cond) > 40:
        top_cond = top_cond[:38] + "..."
    axes[0].text(row["size"] + 1, i, top_cond, va="center", fontsize=6)

# Right: community density vs size
all_comms = communities_csv[communities_csv["size"] >= 2]
sizes = all_comms["size"].values
densities = all_comms["density"].values
comm_ids = all_comms["community_id"].values
c_colors = [COMM_COLORS[int(cid) % len(COMM_COLORS)] for cid in comm_ids]

axes[1].scatter(sizes, densities, c=c_colors, s=sizes * 3, alpha=0.8, edgecolors="black", linewidth=0.5)
for i, cid in enumerate(comm_ids):
    axes[1].annotate(f"C{int(cid)}", (sizes[i], densities[i]), fontsize=8,
                     ha="center", va="bottom")
axes[1].set_xlabel("Community Size")
axes[1].set_ylabel("Internal Density")
axes[1].set_title("Community Size vs Internal Density")

fig.suptitle("T2D Comorbidity Network — Community Structure", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_community_composition.png"))
plt.close(fig)
print("  Saved: figures/fig4_community_composition.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Bridge Diagnoses
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 5: Bridge diagnoses...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: betweenness vs participation coefficient for all nodes
ax = axes[0]
all_bt = metrics["betweenness_centrality"].values
all_pc = metrics["participation_coefficient"].values
all_colors = [COMM_COLORS[comm_map.get(cid, 0) % len(COMM_COLORS)]
              for cid in metrics["condition_concept_id"]]

ax.scatter(all_bt, all_pc, c=all_colors, alpha=0.4, s=15, edgecolors="none")

# Highlight bridges
bridge_bt = bridges["betweenness_centrality"].values
bridge_pc = bridges["participation_coefficient"].values
ax.scatter(bridge_bt, bridge_pc, c="red", s=40, edgecolors="black",
           linewidth=0.5, zorder=5, label=f"Bridge ({len(bridges)})")

# Label top 10 bridges
for _, row in bridges.head(10).iterrows():
    name = row["condition_name"]
    if len(name) > 22:
        name = name[:20] + "..."
    ax.annotate(name, (row["betweenness_centrality"], row["participation_coefficient"]),
                fontsize=5.5, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")

# Draw threshold lines
bt_thresh = metrics["betweenness_centrality"].quantile(0.75)
pc_thresh = metrics["participation_coefficient"].quantile(0.75)
ax.axvline(bt_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
ax.axhline(pc_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

ax.set_xlabel("Betweenness Centrality")
ax.set_ylabel("Participation Coefficient")
ax.set_title("Bridge Diagnosis Identification")
ax.legend(fontsize=8)

# Right: top 20 bridges bar chart
ax2 = axes[1]
top_bridges = bridges.head(20)
names = [n[:28] if len(n) <= 28 else n[:26] + "..." for n in top_bridges["condition_name"]]
bt_vals = top_bridges["betweenness_centrality"].values
b_colors = [COMM_COLORS[comm_map.get(cid, 0) % len(COMM_COLORS)]
            for cid in top_bridges["condition_concept_id"]]

ax2.barh(range(len(names)), bt_vals, color=b_colors, alpha=0.8, edgecolor="white")
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=7)
ax2.invert_yaxis()
ax2.set_xlabel("Betweenness Centrality")
ax2.set_title("Top 20 Bridge Diagnoses")

for i, (_, row) in enumerate(top_bridges.iterrows()):
    ax2.text(bt_vals[i] + 0.001, i,
             f"PC={row['participation_coefficient']:.2f}, C={int(row['n_communities_connected'])}",
             va="center", fontsize=5.5)

fig.suptitle("T2D Comorbidity Network — Bridge Diagnoses", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_bridge_diagnoses.png"))
plt.close(fig)
print("  Saved: figures/fig5_bridge_diagnoses.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Temporal Gateways
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 6: Temporal gateways...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: top 15 by outgoing strength
top_out = temporal_hubs.head(15)
names_out = [n[:30] if len(n) <= 30 else n[:28] + "..." for n in top_out["condition_name"]]

axes[0].barh(range(len(names_out)), top_out["temporal_out_strength"].values,
             color="#e6194b", alpha=0.8, edgecolor="white")
axes[0].set_yticks(range(len(names_out)))
axes[0].set_yticklabels(names_out, fontsize=7)
axes[0].invert_yaxis()
axes[0].set_xlabel("Outgoing Transition Strength")
axes[0].set_title("Top 15 Gateway Conditions\n(Precede most others)")

# Right: net direction (out - in) for top 30 conditions
top30 = temporal_hubs.head(30)
names30 = [n[:25] if len(n) <= 25 else n[:23] + "..." for n in top30["condition_name"]]
net_dir = top30["net_direction"].values
colors_dir = ["#3cb44b" if v >= 0 else "#e6194b" for v in net_dir]

axes[1].barh(range(len(names30)), net_dir, color=colors_dir, alpha=0.8, edgecolor="white")
axes[1].set_yticks(range(len(names30)))
axes[1].set_yticklabels(names30, fontsize=6)
axes[1].invert_yaxis()
axes[1].set_xlabel("Net Direction (Out-Degree - In-Degree)")
axes[1].set_title("Temporal Direction Bias\n(Green=precedes, Red=follows)")
axes[1].axvline(0, color="black", linewidth=0.5)

fig.suptitle("T2D Comorbidity Network — Temporal Gateway Analysis", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_temporal_gateways.png"))
plt.close(fig)
print("  Saved: figures/fig6_temporal_gateways.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Demographic Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 7: Demographic comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Gender comparison
gender_data = subgroup_comp[subgroup_comp["subgroup"].isin(["male", "female"])]
gender_names = gender_data["subgroup"].values

# Plot 1: Density comparison
all_groups = subgroup_comp["subgroup"].values
all_density = subgroup_comp["density"].values
bar_colors = ["#4363d8", "#e6194b", "#3cb44b", "#f58231", "#911eb4"]

axes[0, 0].bar(range(len(all_groups)), all_density, color=bar_colors, alpha=0.8, edgecolor="white")
axes[0, 0].set_xticks(range(len(all_groups)))
axes[0, 0].set_xticklabels(all_groups, fontsize=9)
axes[0, 0].set_ylabel("Network Density")
axes[0, 0].set_title("Network Density by Subgroup")
for i, v in enumerate(all_density):
    axes[0, 0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

# Plot 2: Number of communities
all_comms_n = subgroup_comp["n_communities"].values
axes[0, 1].bar(range(len(all_groups)), all_comms_n, color=bar_colors, alpha=0.8, edgecolor="white")
axes[0, 1].set_xticks(range(len(all_groups)))
axes[0, 1].set_xticklabels(all_groups, fontsize=9)
axes[0, 1].set_ylabel("Number of Communities")
axes[0, 1].set_title("Community Count by Subgroup")
for i, v in enumerate(all_comms_n):
    axes[0, 1].text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)

# Plot 3: Modularity
all_mod = subgroup_comp["modularity"].values
axes[1, 0].bar(range(len(all_groups)), all_mod, color=bar_colors, alpha=0.8, edgecolor="white")
axes[1, 0].set_xticks(range(len(all_groups)))
axes[1, 0].set_xticklabels(all_groups, fontsize=9)
axes[1, 0].set_ylabel("Modularity")
axes[1, 0].set_title("Modularity by Subgroup")
for i, v in enumerate(all_mod):
    axes[1, 0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)

# Plot 4: Edge count
all_edges_n = subgroup_comp["edges"].values
axes[1, 1].bar(range(len(all_groups)), all_edges_n, color=bar_colors, alpha=0.8, edgecolor="white")
axes[1, 1].set_xticks(range(len(all_groups)))
axes[1, 1].set_xticklabels(all_groups, fontsize=9)
axes[1, 1].set_ylabel("Number of Edges")
axes[1, 1].set_title("Edge Count by Subgroup")
for i, v in enumerate(all_edges_n):
    axes[1, 1].text(i, v + 500, f"{int(v):,}", ha="center", fontsize=8)

fig.suptitle("T2D Comorbidity Network — Demographic Subgroup Comparison", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_demographic_comparison.png"))
plt.close(fig)
print("  Saved: figures/fig7_demographic_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Top Condition Co-occurrence Heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 8: Co-occurrence heatmap...")

# Top 25 conditions by degree centrality
top25 = metrics.nlargest(25, "degree_centrality")
top25_ids = top25["condition_concept_id"].tolist()
top25_names = [n[:28] if len(n) <= 28 else n[:26] + "..." for n in top25["condition_name"]]

# Build Jaccard matrix for top 25
jaccard_matrix = np.zeros((25, 25))
edge_lookup = {}
for _, row in edges.iterrows():
    edge_lookup[(row["condition_id_a"], row["condition_id_b"])] = row["jaccard"]
    edge_lookup[(row["condition_id_b"], row["condition_id_a"])] = row["jaccard"]

for i in range(25):
    for j in range(25):
        if i == j:
            jaccard_matrix[i, j] = 1.0
        else:
            jaccard_matrix[i, j] = edge_lookup.get(
                (top25_ids[i], top25_ids[j]), 0
            )

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(jaccard_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(25))
ax.set_yticks(range(25))
ax.set_xticklabels(top25_names, rotation=45, ha="right", fontsize=6.5)
ax.set_yticklabels(top25_names, fontsize=6.5)
ax.set_title("Co-occurrence Heatmap (Jaccard) — Top 25 Conditions by Degree", fontsize=13)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Jaccard Similarity", fontsize=10)

# Annotate cells with values > 0.2
for i in range(25):
    for j in range(25):
        if i != j and jaccard_matrix[i, j] >= 0.2:
            ax.text(j, i, f"{jaccard_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=5, color="black")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_jaccard_heatmap.png"))
plt.close(fig)
print("  Saved: figures/fig8_jaccard_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("All 8 figures saved to figures/ folder.")
print("=" * 70)
