"""
Spectral clustering vs Louvain comparison.
Implements spectral graph partitioning from Lecture 04 (graph Laplacian method)
and compares with our existing Louvain results.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ─── Load data ────────────────────────────────────────────────────────────
print("Loading network data...")
nodes = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))
metrics = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))

# Build graph
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row["condition_concept_id"], name=row["condition_name"],
               patient_count=row["patient_count"])
for _, row in edges.iterrows():
    G.add_edge(row["condition_id_a"], row["condition_id_b"], weight=row["jaccard"])

# Work with LCC for spectral methods (need connected graph)
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
node_list = sorted(G_lcc.nodes())
n = len(node_list)
node_idx = {node: i for i, node in enumerate(node_list)}

print(f"  LCC: {n} nodes, {G_lcc.number_of_edges()} edges")

# ─── Louvain results (already computed) ──────────────────────────────────
louvain_map = dict(zip(metrics["condition_concept_id"], metrics["community"]))
louvain_labels = np.array([louvain_map.get(n, -1) for n in node_list])
n_louvain = len(set(louvain_labels) - {-1})
louvain_modularity = nx.community.modularity(
    G_lcc,
    [{n for n in node_list if louvain_map.get(n) == c}
     for c in sorted(set(louvain_labels)) if c >= 0],
    weight="weight"
)

print(f"\nLouvain: {n_louvain} communities, modularity = {louvain_modularity:.4f}")

# ─── Spectral Partitioning (from Lecture 04) ─────────────────────────────
print("\n--- Spectral Graph Partitioning (Lecture 04) ---")

# Build adjacency matrix and graph Laplacian
A = nx.adjacency_matrix(G_lcc, nodelist=node_list, weight="weight").toarray()
D = np.diag(A.sum(axis=1))
L = D - A  # Graph Laplacian L = D - A

# Normalized Laplacian for better spectral clustering
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-10)))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt

# Compute eigenvalues and eigenvectors of L
print("Computing eigendecomposition of graph Laplacian...")
eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

# Sort by eigenvalue (ascending)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Print first 20 eigenvalues to find the spectral gap
print(f"  First 15 eigenvalues: {np.round(eigenvalues[:15], 6)}")

# Find optimal k using eigengap heuristic
# The gap between eigenvalue k and k+1 suggests k communities
eigengaps = np.diff(eigenvalues[1:21])  # skip first (always 0 for connected graph)
best_k_spectral = np.argmax(eigengaps) + 2  # +2 because we skipped first and 0-indexed
print(f"  Eigengap heuristic suggests k = {best_k_spectral}")

# Run spectral clustering for multiple k values
results = []

for k in range(2, 16):
    # Use the first k eigenvectors (skip the first trivial one)
    V_k = eigenvectors[:, 1:k+1]

    # K-means on the spectral embedding
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    spectral_labels = kmeans.fit_predict(V_k)

    # Compute modularity
    spectral_comms = [{node_list[i] for i in range(n) if spectral_labels[i] == c}
                      for c in range(k)]
    spectral_comms = [c for c in spectral_comms if len(c) > 0]  # remove empty
    spectral_mod = nx.community.modularity(G_lcc, spectral_comms, weight="weight")

    # Compare with Louvain
    ari = adjusted_rand_score(louvain_labels, spectral_labels)
    nmi = normalized_mutual_info_score(louvain_labels, spectral_labels)

    results.append({
        "k": k,
        "modularity": spectral_mod,
        "ari_vs_louvain": ari,
        "nmi_vs_louvain": nmi,
        "n_communities_actual": len(spectral_comms),
    })

    if k in [best_k_spectral, n_louvain, 6, 12]:
        print(f"  k={k}: modularity={spectral_mod:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")

results_df = pd.DataFrame(results)

# Find best spectral k by modularity
best_row = results_df.loc[results_df["modularity"].idxmax()]
best_k = int(best_row["k"])
best_mod = best_row["modularity"]
print(f"\n  Best spectral k by modularity: k={best_k}, Q={best_mod:.4f}")
print(f"  Louvain: k={n_louvain}, Q={louvain_modularity:.4f}")

# ─── Run final spectral clustering at best k ─────────────────────────────
V_best = eigenvectors[:, 1:best_k+1]
kmeans_best = KMeans(n_clusters=best_k, n_init=20, random_state=42)
spectral_best_labels = kmeans_best.fit_predict(V_best)

# Also run at k = n_louvain for direct comparison
V_louvain_k = eigenvectors[:, 1:n_louvain+1]
kmeans_louvain_k = KMeans(n_clusters=n_louvain, n_init=20, random_state=42)
spectral_louvain_k_labels = kmeans_louvain_k.fit_predict(V_louvain_k)

# ─── Save spectral community assignments ─────────────────────────────────
spectral_map = {node_list[i]: int(spectral_best_labels[i]) for i in range(n)}

# Add spectral community to metrics
metrics["spectral_community"] = metrics["condition_concept_id"].map(spectral_map)
metrics.to_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"), index=False)

# ─── Save comparison results ─────────────────────────────────────────────
results_df.to_csv(os.path.join(OUT_DIR, "analysis_spectral_comparison.csv"), index=False)

# ─── Build community summary for spectral (best k) ──────────────────────
spectral_comms_best = [{node_list[i] for i in range(n) if spectral_best_labels[i] == c}
                       for c in range(best_k)]

spectral_comm_summary = []
for c_id, c_nodes in enumerate(spectral_comms_best):
    if len(c_nodes) == 0:
        continue
    sub = G_lcc.subgraph(c_nodes)
    top_conds = sorted(c_nodes, key=lambda x: G_lcc.nodes[x].get("patient_count", 0),
                       reverse=True)[:5]
    top_names = "; ".join([G_lcc.nodes[n]["name"] for n in top_conds])
    spectral_comm_summary.append({
        "community_id": c_id,
        "size": len(c_nodes),
        "density": round(nx.density(sub), 4) if len(c_nodes) > 1 else 0,
        "top_conditions": top_names,
    })

spectral_comm_df = pd.DataFrame(spectral_comm_summary)
spectral_comm_df.to_csv(os.path.join(OUT_DIR, "analysis_spectral_communities.csv"), index=False)

# ─── Comparison Figure ───────────────────────────────────────────────────
print("\nGenerating comparison figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Modularity vs k
ax = axes[0, 0]
ax.plot(results_df["k"], results_df["modularity"], "o-", color="#4363d8", linewidth=2,
        markersize=6, label="Spectral")
ax.axhline(louvain_modularity, color="#e6194b", linewidth=2, linestyle="--",
           label=f"Louvain (k={n_louvain}, Q={louvain_modularity:.3f})")
ax.axvline(best_k, color="#3cb44b", linewidth=1, linestyle=":", alpha=0.7,
           label=f"Best spectral k={best_k}")
ax.set_xlabel("Number of Communities (k)")
ax.set_ylabel("Modularity Q")
ax.set_title("Modularity: Spectral vs Louvain")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2) Eigenvalue spectrum (spectral gap)
ax = axes[0, 1]
ax.plot(range(1, 21), eigenvalues[1:21], "o-", color="#911eb4", markersize=6)
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel("Eigenvalue λ")
ax.set_title("Laplacian Eigenvalue Spectrum\n(gap indicates natural number of communities)")
ax.axvline(best_k_spectral, color="#3cb44b", linewidth=1.5, linestyle="--",
           label=f"Eigengap at k={best_k_spectral}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3) ARI and NMI vs k
ax = axes[1, 0]
ax.plot(results_df["k"], results_df["ari_vs_louvain"], "o-", color="#f58231",
        linewidth=2, markersize=5, label="ARI")
ax.plot(results_df["k"], results_df["nmi_vs_louvain"], "s-", color="#42d4f4",
        linewidth=2, markersize=5, label="NMI")
ax.set_xlabel("Number of Communities (k)")
ax.set_ylabel("Agreement with Louvain")
ax.set_title("Spectral vs Louvain Agreement")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4) Community size comparison (Louvain vs Spectral at best k)
ax = axes[1, 1]
louvain_sizes = sorted([sum(1 for v in louvain_labels if v == c)
                        for c in set(louvain_labels)], reverse=True)
spectral_sizes = sorted([len(c) for c in spectral_comms_best if len(c) > 0], reverse=True)

x_l = range(len(louvain_sizes))
x_s = range(len(spectral_sizes))
ax.bar([x - 0.2 for x in x_l], louvain_sizes, width=0.35, color="#e6194b",
       alpha=0.8, label=f"Louvain (k={n_louvain})")
ax.bar([x + 0.2 for x in x_s], spectral_sizes, width=0.35, color="#4363d8",
       alpha=0.8, label=f"Spectral (k={best_k})")
ax.set_xlabel("Community Rank")
ax.set_ylabel("Community Size")
ax.set_title("Community Size Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Community Detection Comparison: Spectral Partitioning vs Louvain",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_spectral_vs_louvain.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: figures/fig_spectral_vs_louvain.png")

# ─── Print Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SPECTRAL VS LOUVAIN COMPARISON SUMMARY")
print("=" * 70)
print(f"\nLouvain:")
print(f"  Communities: {n_louvain}")
print(f"  Modularity: {louvain_modularity:.4f}")
print(f"  Auto-detects k: Yes")
print(f"  Handles weights: Yes (Jaccard)")

print(f"\nSpectral (best k={best_k}):")
print(f"  Modularity: {best_mod:.4f}")
print(f"  Eigengap suggests k: {best_k_spectral}")
print(f"  ARI agreement with Louvain: {results_df.loc[results_df['k']==best_k, 'ari_vs_louvain'].iloc[0]:.4f}")
print(f"  NMI agreement with Louvain: {results_df.loc[results_df['k']==best_k, 'nmi_vs_louvain'].iloc[0]:.4f}")

# Check if major communities agree
print(f"\nSpectral communities (k={best_k}):")
for _, row in spectral_comm_df.iterrows():
    top = row["top_conditions"].split(";")[0].strip()
    print(f"  C{row['community_id']}: {row['size']} nodes — {top}")

print(f"\nConclusion: {'Louvain achieves higher modularity' if louvain_modularity > best_mod else 'Spectral achieves higher modularity'}")
print(f"  Louvain Q={louvain_modularity:.4f} vs Spectral Q={best_mod:.4f}")
print("=" * 70)
