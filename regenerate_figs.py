"""
Regenerate network visualization figures for publication quality.
- fig1: Main network diagram (top 80 nodes, visible edges, community colors)
- fig1b: Community zoom-in (cardiometabolic core C4)
- fig1c: Subgroup comparison (male vs female side-by-side)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Load data
metrics = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))
communities_csv = pd.read_csv(os.path.join(OUT_DIR, "analysis_communities.csv"))
nodes = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))

# Build full graph
G_full = nx.Graph()
for _, row in nodes.iterrows():
    G_full.add_node(row["condition_concept_id"], name=row["condition_name"],
                    patient_count=row["patient_count"])
for _, row in edges.iterrows():
    G_full.add_edge(row["condition_id_a"], row["condition_id_b"],
                    weight=row["jaccard"])

comm_map = dict(zip(metrics["condition_concept_id"], metrics["community"]))
name_map = dict(zip(metrics["condition_concept_id"], metrics["condition_name"]))
count_map = dict(zip(metrics["condition_concept_id"], metrics["patient_count"]))

COMM_COLORS = {
    0: "#808080", 1: "#3cb44b", 2: "#4363d8", 3: "#f58231",
    4: "#911eb4", 5: "#808080", 6: "#42d4f4", 7: "#808080",
    8: "#808080", 9: "#808080", 10: "#e6194b", 11: "#808080",
}

COMM_NAMES = {
    3: "General/GI",
    1: "Hematologic/Infectious",
    10: "Acute/Critical Care",
    4: "Cardiometabolic Core",
    2: "Renal (CKD/Dialysis)",
    6: "Cerebrovascular",
}

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main Network Diagram (top 80 nodes by degree)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 1: Main network diagram (top 80 nodes)...")

# Select top 80 nodes by degree
top_n = 80
top_nodes = metrics.nlargest(top_n, "degree")
top_ids = set(top_nodes["condition_concept_id"])

# Build subgraph
G_sub = G_full.subgraph(top_ids).copy()

# Layout: use spring layout with moderate spacing
pos = nx.spring_layout(G_sub, k=2.5/np.sqrt(len(G_sub)), iterations=150,
                       weight="weight", seed=42)

fig, ax = plt.subplots(figsize=(16, 14))

# Node colors and sizes
node_colors = [COMM_COLORS.get(comm_map.get(n, 0), "#808080") for n in G_sub.nodes()]
node_sizes = [max(40, count_map.get(n, 50) / 3) for n in G_sub.nodes()]

# Draw edges — color by community if both nodes same community, gray otherwise
edge_colors = []
edge_widths = []
edge_alphas = []
for u, v in G_sub.edges():
    w = G_sub[u][v]["weight"]
    if comm_map.get(u) == comm_map.get(v):
        edge_colors.append(COMM_COLORS.get(comm_map.get(u, 0), "#808080"))
        edge_widths.append(0.3 + w * 8)
        edge_alphas.append(min(0.4, 0.1 + w * 3))
    else:
        edge_colors.append("#cccccc")
        edge_widths.append(0.2 + w * 4)
        edge_alphas.append(min(0.15, 0.03 + w))

# Draw edges one at a time for per-edge alpha
for i, (u, v) in enumerate(G_sub.edges()):
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=edge_colors[i], alpha=edge_alphas[i],
            linewidth=edge_widths[i])

# Draw nodes
nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, linewidths=0.8, edgecolors="white", ax=ax)

# Labels for top 30 nodes
top_labeled = metrics.nlargest(30, "degree")
labels = {}
for _, row in top_labeled.iterrows():
    cid = row["condition_concept_id"]
    if cid in G_sub.nodes():
        name = row["condition_name"]
        # Shorten names
        name = (name.replace("without complication", "w/o comp.")
                     .replace("Electrocardiogram", "ECG")
                     .replace("Gastroesophageal reflux disease", "GERD")
                     .replace("Essential hypertension", "Hypertension")
                     .replace("Abnormal findings on diagnostic", "Abn. diagnostic"))
        if len(name) > 25:
            name = name[:23] + "..."
        labels[cid] = name

nx.draw_networkx_labels(G_sub, pos, labels, font_size=7, font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

# Legend for major communities
legend_patches = []
for cid, cname in sorted(COMM_NAMES.items()):
    count = sum(1 for n in G_sub.nodes() if comm_map.get(n) == cid)
    if count > 0:
        legend_patches.append(
            mpatches.Patch(color=COMM_COLORS.get(cid, "#808080"),
                           label=f"{cname} ({count} nodes)")
        )

ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9,
          title="Communities", title_fontsize=10)
ax.set_title("T2D Comorbidity Network — Top 80 Conditions by Degree\n"
             "(edge width ∝ Jaccard similarity, node size ∝ patient count)",
             fontsize=14, fontweight="bold")
ax.axis("off")
fig.savefig(os.path.join(FIG_DIR, "fig1_network_overview.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: fig1_network_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1b: Community Zoom-In (Cardiometabolic Core C4)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 1b: Cardiometabolic community zoom-in...")

# Get top 30 C4 nodes by patient count for readability
c4_all = [n for n in G_full.nodes() if comm_map.get(n) == 4]
c4_metrics_sorted = metrics[metrics["community"] == 4].nlargest(30, "patient_count")
c4_top_ids = set(c4_metrics_sorted["condition_concept_id"])
G_c4 = G_full.subgraph(c4_top_ids).copy()

# Layout with more spacing
pos_c4 = nx.spring_layout(G_c4, k=3.0/np.sqrt(len(G_c4)), iterations=200,
                          weight="weight", seed=42)

fig, ax = plt.subplots(figsize=(14, 12))

# Node sizes
c4_sizes = [max(50, count_map.get(n, 50) / 2.5) for n in G_c4.nodes()]

# Draw edges with weight-proportional width
for u, v in G_c4.edges():
    w = G_c4[u][v]["weight"]
    ax.plot([pos_c4[u][0], pos_c4[v][0]], [pos_c4[u][1], pos_c4[v][1]],
            color="#911eb4", alpha=min(0.45, 0.05 + w * 3),
            linewidth=0.3 + w * 12)

nx.draw_networkx_nodes(G_c4, pos_c4, node_color="#911eb4", node_size=c4_sizes,
                       alpha=0.9, linewidths=0.8, edgecolors="white", ax=ax)

# Label all nodes in this view
labels_c4 = {}
for _, row in c4_metrics_sorted.iterrows():
    cid = row["condition_concept_id"]
    if cid in G_c4.nodes():
        name = row["condition_name"]
        name = (name.replace("without complication", "w/o comp.")
                     .replace("Electrocardiogram", "ECG")
                     .replace("Essential hypertension", "Hypertension")
                     .replace("Type 2 diabetes mellitus", "T2D")
                     .replace("Chronic kidney disease", "CKD")
                     .replace("Congestive heart failure", "CHF")
                     .replace("Aortocoronary bypass", "CABG"))
        if len(name) > 25:
            name = name[:23] + "..."
        labels_c4[cid] = name

nx.draw_networkx_labels(G_c4, pos_c4, labels_c4, font_size=8, font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.set_title(f"Community 4: Cardiometabolic Core — Top 30 Conditions\n"
             f"(edge width ∝ Jaccard similarity, {len(c4_all)} total in community)",
             fontsize=14, fontweight="bold")
ax.axis("off")
fig.savefig(os.path.join(FIG_DIR, "fig1b_community_zoomin.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: fig1b_community_zoomin.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1c: Male vs Female Subgroup Network Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 1c: Male vs Female subgroup comparison...")

# Load subgroup edges
edges_male = pd.read_csv(os.path.join(OUT_DIR, "network_subgroup_edges_male.csv"))
edges_female = pd.read_csv(os.path.join(OUT_DIR, "network_subgroup_edges_female.csv"))

# Use same top 50 nodes for both
top50 = metrics.nlargest(50, "degree")
top50_ids = set(top50["condition_concept_id"])

def build_subgraph(edge_df, node_set, jaccard_min=0.15):
    """Build subgraph with Jaccard threshold to show meaningful differences."""
    G = nx.Graph()
    for nid in node_set:
        G.add_node(nid)
    for _, row in edge_df.iterrows():
        if (row["condition_id_a"] in node_set and row["condition_id_b"] in node_set
                and row["jaccard"] >= jaccard_min):
            G.add_edge(row["condition_id_a"], row["condition_id_b"],
                       weight=row["jaccard"])
    return G

G_male = build_subgraph(edges_male, top50_ids, jaccard_min=0.25)
G_female = build_subgraph(edges_female, top50_ids, jaccard_min=0.25)

# Use same layout for both (based on full network positions)
pos_shared = nx.spring_layout(G_full.subgraph(top50_ids), k=2.0/np.sqrt(50),
                              iterations=150, weight="weight", seed=42)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for ax, G_sub, title, color in [(axes[0], G_male, f"Male (n=764)", "#4363d8"),
                                 (axes[1], G_female, f"Female (n=590)", "#e6194b")]:
    # Sizes
    sizes = [max(30, count_map.get(n, 50) / 4) for n in G_sub.nodes()]
    colors = [COMM_COLORS.get(comm_map.get(n, 0), "#808080") for n in G_sub.nodes()]

    # Edges
    for u, v in G_sub.edges():
        if G_sub[u][v].get("weight", 0) > 0:
            w = G_sub[u][v]["weight"]
            if comm_map.get(u) == comm_map.get(v):
                ec = COMM_COLORS.get(comm_map.get(u, 0), "#808080")
                ea = min(0.35, 0.05 + w * 2)
            else:
                ec = "#cccccc"
                ea = min(0.1, 0.02 + w)
            ax.plot([pos_shared[u][0], pos_shared[v][0]],
                    [pos_shared[u][1], pos_shared[v][1]],
                    color=ec, alpha=ea, linewidth=0.2 + w * 6)

    nx.draw_networkx_nodes(G_sub, pos_shared, node_color=colors, node_size=sizes,
                           alpha=0.9, linewidths=0.6, edgecolors="white", ax=ax)

    # Label top 15
    top15 = metrics[metrics["condition_concept_id"].isin(top50_ids)].nlargest(15, "degree")
    labels = {}
    for _, row in top15.iterrows():
        cid = row["condition_concept_id"]
        if cid in G_sub.nodes():
            name = row["condition_name"]
            name = (name.replace("without complication", "w/o comp.")
                         .replace("Electrocardiogram", "ECG")
                         .replace("Essential hypertension", "Hypertension")
                         .replace("Abnormal findings on diagnostic", "Abn. diagnostic"))
            if len(name) > 22:
                name = name[:20] + "..."
            labels[cid] = name

    nx.draw_networkx_labels(G_sub, pos_shared, labels, font_size=6.5, font_weight="bold",
                            ax=ax, bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                             ec="none", alpha=0.75))

    n_edges = G_sub.number_of_edges()
    density = nx.density(G_sub)
    ax.set_title(f"{title}\n{n_edges} edges, density={density:.3f}", fontsize=13, fontweight="bold")
    ax.axis("off")

fig.suptitle("Demographic Subgroup Network Comparison — Top 50 Conditions",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1c_subgroup_networks.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: fig1c_subgroup_networks.png")

print("\nAll network diagrams regenerated.")
