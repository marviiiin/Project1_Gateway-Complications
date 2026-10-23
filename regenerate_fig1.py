"""
Regenerate fig1 with a clearer community-aware layout.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Load data
metrics = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))
communities_csv = pd.read_csv(os.path.join(OUT_DIR, "analysis_communities.csv"))
nodes = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))

# Build graph
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row["condition_concept_id"], name=row["condition_name"],
               patient_count=row["patient_count"])
for _, row in edges.iterrows():
    G.add_edge(row["condition_id_a"], row["condition_id_b"],
               weight=row["jaccard"])

comm_map = dict(zip(metrics["condition_concept_id"], metrics["community"]))

COMM_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324",
]

# --- Community-aware layout ---
# Step 1: Create a coarse graph of communities and position them in a circle
unique_comms = sorted(set(comm_map.values()))
n_comms = len(unique_comms)

# Place communities in a circle with radius proportional to size
comm_sizes = {}
for c in unique_comms:
    comm_sizes[c] = sum(1 for v in comm_map.values() if v == c)

# Circular positions for community centers, with larger spacing
comm_center = {}
for i, c in enumerate(unique_comms):
    angle = 2 * np.pi * i / n_comms
    radius = 5.0
    comm_center[c] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

# Step 2: For each community, run spring layout on its subgraph, then offset
pos = {}
for c in unique_comms:
    comm_nodes = [n for n in G.nodes() if comm_map.get(n, 0) == c]
    subG = G.subgraph(comm_nodes)
    if len(comm_nodes) == 1:
        sub_pos = {comm_nodes[0]: np.array([0.0, 0.0])}
    else:
        sub_pos = nx.spring_layout(subG, k=1.5/np.sqrt(max(len(comm_nodes), 1)),
                                   iterations=100, weight="weight", seed=42)
    # Scale sub-layout by community size
    scale = 0.8 * np.sqrt(len(comm_nodes) / max(comm_sizes.values()))
    center = comm_center[c]
    for n, p in sub_pos.items():
        pos[n] = center + p * scale

# --- Separate intra- vs inter-community edges ---
intra_edges = [(u, v) for u, v in G.edges()
               if comm_map.get(u, -1) == comm_map.get(v, -2)]
inter_edges = [(u, v) for u, v in G.edges()
               if comm_map.get(u, -1) != comm_map.get(v, -2)]

# Only show top 30% of intra-community edges and top 5% of inter-community edges
intra_weights = [G[u][v]["weight"] for u, v in intra_edges]
inter_weights = [G[u][v]["weight"] for u, v in inter_edges]

intra_thresh = np.percentile(intra_weights, 70) if intra_weights else 0
inter_thresh = np.percentile(inter_weights, 95) if inter_weights else 0

visible_intra = [(u, v) for u, v in intra_edges if G[u][v]["weight"] >= intra_thresh]
visible_inter = [(u, v) for u, v in inter_edges if G[u][v]["weight"] >= inter_thresh]

# --- Draw ---
fig, ax = plt.subplots(figsize=(16, 16))

node_colors = [COMM_COLORS[comm_map.get(n, 0) % len(COMM_COLORS)] for n in G.nodes()]
node_sizes = [max(20, G.nodes[n].get("patient_count", 50) / 6) for n in G.nodes()]

# Draw intra-community edges (colored faintly by community)
for u, v in visible_intra:
    c = comm_map.get(u, 0)
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=COMM_COLORS[c % len(COMM_COLORS)], alpha=0.08, linewidth=0.3)

# Draw inter-community edges (light gray, very faint)
for u, v in visible_inter:
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color="#cccccc", alpha=0.02, linewidth=0.15)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, linewidths=0.5, edgecolors="white", ax=ax)

# Label top 20 nodes by betweenness
top_nodes = metrics.nlargest(20, "betweenness_centrality")
labels = {}
for _, row in top_nodes.iterrows():
    cid = row["condition_concept_id"]
    name = row["condition_name"]
    if len(name) > 28:
        name = name[:26] + "..."
    labels[cid] = name

nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

# Legend
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

ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9,
          title="Communities (Louvain)", title_fontsize=10)

ax.set_title("T2D Comorbidity Network — Community Structure", fontsize=16, fontweight="bold")
ax.axis("off")
fig.savefig(os.path.join(FIG_DIR, "fig1_network_overview.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/fig1_network_overview.png")
