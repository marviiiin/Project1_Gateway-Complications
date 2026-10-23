"""
03_network_analysis.py
=======================
Purpose:  Perform Project 1 network analysis on the T2D comorbidity networks.
          Computes centrality, community detection, bridge diagnoses, and
          demographic subgroup comparisons.

Input:    data/network_nodes.csv
          data/network_cooccurrence_edges.csv
          data/network_temporal_edges.csv
          data/network_subgroup_edges_*.csv
          data/t2d_cohort_demographics.csv

Output:   data/analysis_node_metrics.csv        — centrality + community per node
          data/analysis_communities.csv          — community summary with top members
          data/analysis_bridge_diagnoses.csv     — bridge conditions between communities
          data/analysis_network_stats.csv        — global network statistics
          data/analysis_subgroup_comparison.csv  — demographic subgroup network stats
          data/analysis_temporal_hubs.csv        — top gateway conditions in temporal net
          data/analysis_summary.txt              — full analysis report

Analysis Plan (from course Lecture 05, Topic 5)
-------------------------------------------------

1. CENTRALITY MEASURES — Identify the most important conditions
   - Degree centrality: how many other conditions co-occur with it
   - Betweenness centrality: how often it lies on shortest paths between others
   - Closeness centrality: how close it is to all other conditions
   - Eigenvector centrality: importance based on connections to other important nodes
   Reference: Lectures 03-04 (Centrality), Hidalgo et al. (2009)

2. COMMUNITY DETECTION — Louvain algorithm
   - Groups conditions into comorbidity clusters
   - "Name each community by reading its member conditions" (Topic 5)
   Reference: Blondel et al., "Fast unfolding of communities in large
   networks," J. Statistical Mechanics, 2008

3. BRIDGE DIAGNOSES — Conditions connecting separate communities
   - High betweenness centrality + high participation coefficient
   - Participation coefficient = fraction of edges going to other communities
   - "A diagnosis that bridges two clusters is a candidate early marker for
     crossing into a new disease domain" (Topic 5)
   Reference: Guimera & Amaral, "Functional cartography of metabolic
   networks," Nature, 2005

4. DEMOGRAPHIC COMPARISON — Subgroup network structure
   - Compare community structure, centrality rankings, density across
     age/sex subgroups
   - "Does the community structure change by age or sex?" (Topic 5)

5. TEMPORAL GATEWAY ANALYSIS — Conditions that precede many others
   - In-degree vs out-degree in the temporal network
   - High out-degree = gateway condition (precedes many complications)
   Reference: Jensen et al. (2014)
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Step 1: Load data ────────────────────────────────────────────────────
print("Step 1: Loading network data...")

nodes = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))
temporal = pd.read_csv(os.path.join(OUT_DIR, "network_temporal_edges.csv"))
demo = pd.read_csv(os.path.join(OUT_DIR, "t2d_cohort_demographics.csv"))

cid_to_name = dict(zip(nodes["condition_concept_id"], nodes["condition_name"]))

print(f"  Nodes: {len(nodes)}, Co-occ edges: {len(edges):,}, "
      f"Temporal edges: {len(temporal):,}")

# ─── Step 2: Build NetworkX graphs ────────────────────────────────────────
print("\nStep 2: Building NetworkX graphs...")

# Co-occurrence graph (undirected, weighted)
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(
        row["condition_concept_id"],
        name=row["condition_name"],
        patient_count=row["patient_count"],
        patient_pct=row["patient_pct"],
    )

for _, row in edges.iterrows():
    G.add_edge(
        row["condition_id_a"],
        row["condition_id_b"],
        weight=row["jaccard"],
        patients_both=row["patients_both"],
    )

# Temporal graph (directed, weighted)
G_temp = nx.DiGraph()
for _, row in nodes.iterrows():
    G_temp.add_node(row["condition_concept_id"], name=row["condition_name"])

for _, row in temporal.iterrows():
    G_temp.add_edge(
        row["source_condition_id"],
        row["target_condition_id"],
        weight=row["patient_count"],
    )

print(f"  Co-occurrence: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Temporal: {G_temp.number_of_nodes()} nodes, {G_temp.number_of_edges()} edges")

# ─── Step 3: Global network statistics ────────────────────────────────────
print("\nStep 3: Computing global network statistics...")

# Work with the largest connected component for diameter/avg path
components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
G_lcc = G.subgraph(largest_cc).copy()

density = nx.density(G)
avg_clustering = nx.average_clustering(G, weight="weight")
avg_clustering_unw = nx.average_clustering(G)
transitivity = nx.transitivity(G)
n_components = len(components)

# Degree distribution stats
degrees = [d for _, d in G.degree()]
avg_degree = np.mean(degrees)
median_degree = np.median(degrees)
max_degree = np.max(degrees)

# LCC stats
lcc_size = len(largest_cc)
avg_path_lcc = nx.average_shortest_path_length(G_lcc)
diameter_lcc = nx.diameter(G_lcc)

# Weighted degree (strength)
strengths = [sum(d["weight"] for d in G[n].values()) for n in G.nodes()]
avg_strength = np.mean(strengths)

net_stats = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "density": round(density, 6),
    "connected_components": n_components,
    "largest_component_size": lcc_size,
    "largest_component_pct": round(lcc_size / G.number_of_nodes() * 100, 2),
    "avg_degree": round(avg_degree, 2),
    "median_degree": round(median_degree, 2),
    "max_degree": max_degree,
    "avg_strength": round(avg_strength, 4),
    "avg_clustering_weighted": round(avg_clustering, 4),
    "avg_clustering_unweighted": round(avg_clustering_unw, 4),
    "transitivity": round(transitivity, 4),
    "avg_path_length_lcc": round(avg_path_lcc, 4),
    "diameter_lcc": diameter_lcc,
}

stats_df = pd.DataFrame([net_stats])
stats_df.to_csv(os.path.join(OUT_DIR, "analysis_network_stats.csv"), index=False)

print(f"  Density: {density:.4f}")
print(f"  Connected components: {n_components} (LCC: {lcc_size} nodes, {lcc_size/G.number_of_nodes()*100:.1f}%)")
print(f"  Avg degree: {avg_degree:.1f}, Median: {median_degree:.0f}, Max: {max_degree}")
print(f"  Avg clustering (weighted): {avg_clustering:.4f}")
print(f"  Transitivity: {transitivity:.4f}")
print(f"  Avg path length (LCC): {avg_path_lcc:.3f}")
print(f"  Diameter (LCC): {diameter_lcc}")
print(f"  Saved: data/analysis_network_stats.csv")

# ─── Step 4: Centrality measures ──────────────────────────────────────────
print("\nStep 4: Computing centrality measures...")

degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight="weight")
closeness_cent = nx.closeness_centrality(G)
eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")

print("  Degree, betweenness, closeness, eigenvector — done")

# ─── Step 5: Community detection (Louvain) ────────────────────────────────
print("\nStep 5: Running Louvain community detection...")

communities = nx.community.louvain_communities(G, weight="weight", seed=42)

# Map node -> community ID
node_to_community = {}
for comm_id, comm_nodes in enumerate(communities):
    for node in comm_nodes:
        node_to_community[node] = comm_id

n_communities = len(communities)
comm_sizes = [len(c) for c in communities]

print(f"  Communities found: {n_communities}")
print(f"  Sizes: {sorted(comm_sizes, reverse=True)}")

modularity = nx.community.modularity(G, communities, weight="weight")
print(f"  Modularity: {modularity:.4f}")

# ─── Step 6: Participation coefficient ────────────────────────────────────
print("\nStep 6: Computing participation coefficients...")


def participation_coefficient(G, node_to_community):
    """
    P(i) = 1 - sum_s (k_is / k_i)^2
    where k_is = edges from node i to community s, k_i = total degree.
    High P = edges distributed across communities (bridge node).
    Low P = edges concentrated in one community.
    Reference: Guimera & Amaral, Nature, 2005.
    """
    pc = {}
    for node in G.nodes():
        ki = G.degree(node)
        if ki == 0:
            pc[node] = 0.0
            continue
        # Count edges to each community
        comm_edge_counts = Counter()
        for neighbor in G.neighbors(node):
            comm_edge_counts[node_to_community[neighbor]] += 1
        # Participation coefficient
        pc[node] = 1.0 - sum((kis / ki) ** 2 for kis in comm_edge_counts.values())
    return pc


part_coeff = participation_coefficient(G, node_to_community)

print("  Done")

# ─── Step 7: Assemble node metrics table ──────────────────────────────────
print("\nStep 7: Assembling node metrics table...")

node_metrics = []
for _, row in nodes.iterrows():
    cid = row["condition_concept_id"]
    node_metrics.append({
        "condition_concept_id": cid,
        "condition_name": row["condition_name"],
        "patient_count": row["patient_count"],
        "patient_pct": row["patient_pct"],
        "degree": G.degree(cid),
        "strength": round(sum(d["weight"] for d in G[cid].values()), 4) if cid in G else 0,
        "degree_centrality": round(degree_cent.get(cid, 0), 6),
        "betweenness_centrality": round(betweenness_cent.get(cid, 0), 6),
        "closeness_centrality": round(closeness_cent.get(cid, 0), 6),
        "eigenvector_centrality": round(eigenvector_cent.get(cid, 0), 6),
        "community": node_to_community.get(cid, -1),
        "participation_coefficient": round(part_coeff.get(cid, 0), 6),
        "temporal_out_degree": G_temp.out_degree(cid) if cid in G_temp else 0,
        "temporal_in_degree": G_temp.in_degree(cid) if cid in G_temp else 0,
        "temporal_out_strength": sum(
            d["weight"] for _, _, d in G_temp.out_edges(cid, data=True)
        ) if cid in G_temp else 0,
    })

metrics_df = pd.DataFrame(node_metrics)
metrics_df = metrics_df.sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)
metrics_df.to_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"), index=False)

print(f"  Saved: data/analysis_node_metrics.csv ({len(metrics_df)} rows)")

# Print top 10 by each centrality
for cent_name in ["degree_centrality", "betweenness_centrality",
                   "closeness_centrality", "eigenvector_centrality"]:
    top = metrics_df.nlargest(5, cent_name)
    print(f"\n  Top 5 by {cent_name}:")
    for _, r in top.iterrows():
        print(f"    {r[cent_name]:.4f}  {r['condition_name'][:50]}")

# ─── Step 8: Community summary ────────────────────────────────────────────
print("\nStep 8: Summarizing communities...")

community_summary = []
for comm_id, comm_nodes in enumerate(communities):
    comm_df = metrics_df[metrics_df["community"] == comm_id]
    # Top 5 members by patient count
    top_members = comm_df.nlargest(5, "patient_count")
    top_names = "; ".join(top_members["condition_name"].tolist())

    # Subgraph for community
    sub_g = G.subgraph(comm_nodes)

    community_summary.append({
        "community_id": comm_id,
        "size": len(comm_nodes),
        "density": round(nx.density(sub_g), 4) if len(comm_nodes) > 1 else 0,
        "avg_clustering": round(nx.average_clustering(sub_g), 4) if len(comm_nodes) > 1 else 0,
        "avg_patient_count": round(comm_df["patient_count"].mean(), 1),
        "top_conditions": top_names,
    })

comm_summary_df = pd.DataFrame(community_summary)
comm_summary_df = comm_summary_df.sort_values("size", ascending=False).reset_index(drop=True)
comm_summary_df.to_csv(os.path.join(OUT_DIR, "analysis_communities.csv"), index=False)

print(f"  Saved: data/analysis_communities.csv")
print(f"\n  Communities (sorted by size):")
for _, row in comm_summary_df.iterrows():
    print(f"    Community {row['community_id']:>2}: {row['size']:>3} conditions, "
          f"density={row['density']:.3f}")
    print(f"      Top: {row['top_conditions'][:80]}")

# ─── Step 9: Bridge diagnoses ─────────────────────────────────────────────
print("\nStep 9: Identifying bridge diagnoses...")

# Bridge = high betweenness AND high participation coefficient
# Use top quartile of both as threshold
bt_thresh = metrics_df["betweenness_centrality"].quantile(0.75)
pc_thresh = metrics_df["participation_coefficient"].quantile(0.75)

bridges = metrics_df[
    (metrics_df["betweenness_centrality"] >= bt_thresh)
    & (metrics_df["participation_coefficient"] >= pc_thresh)
].copy()

bridges = bridges.sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)

# Add which communities each bridge connects
bridge_communities = []
for _, row in bridges.iterrows():
    cid = row["condition_concept_id"]
    neighbor_comms = set()
    for neighbor in G.neighbors(cid):
        neighbor_comms.add(node_to_community[neighbor])
    bridge_communities.append(
        ", ".join(str(c) for c in sorted(neighbor_comms))
    )

bridges["connected_communities"] = bridge_communities
bridges["n_communities_connected"] = bridges["connected_communities"].apply(
    lambda x: len(x.split(", "))
)

bridges.to_csv(os.path.join(OUT_DIR, "analysis_bridge_diagnoses.csv"), index=False)

print(f"  Betweenness threshold (Q75): {bt_thresh:.6f}")
print(f"  Participation coefficient threshold (Q75): {pc_thresh:.4f}")
print(f"  Bridge diagnoses found: {len(bridges)}")
print(f"  Saved: data/analysis_bridge_diagnoses.csv")

print(f"\n  Top 15 bridge diagnoses:")
for _, r in bridges.head(15).iterrows():
    print(f"    BT={r['betweenness_centrality']:.4f}  "
          f"PC={r['participation_coefficient']:.3f}  "
          f"Comms={r['n_communities_connected']}  "
          f"{r['condition_name'][:45]}")

# ─── Step 10: Temporal gateway analysis ───────────────────────────────────
print("\nStep 10: Temporal gateway analysis...")

temporal_hubs = metrics_df[["condition_concept_id", "condition_name",
                             "temporal_out_degree", "temporal_in_degree",
                             "temporal_out_strength", "community"]].copy()
temporal_hubs["net_direction"] = temporal_hubs["temporal_out_degree"] - temporal_hubs["temporal_in_degree"]
temporal_hubs = temporal_hubs.sort_values("temporal_out_strength", ascending=False).reset_index(drop=True)
temporal_hubs.to_csv(os.path.join(OUT_DIR, "analysis_temporal_hubs.csv"), index=False)

print(f"  Saved: data/analysis_temporal_hubs.csv")
print(f"\n  Top 10 gateway conditions (by outgoing transition strength):")
for _, r in temporal_hubs.head(10).iterrows():
    print(f"    Out={r['temporal_out_strength']:>6.0f}  "
          f"In={r['temporal_in_degree']:>3}  "
          f"Net={r['net_direction']:>+4}  "
          f"{r['condition_name'][:45]}")

print(f"\n  Top 10 'endpoint' conditions (highest in-degree, low out-degree):")
endpoints = temporal_hubs.nlargest(10, "temporal_in_degree")
for _, r in endpoints.iterrows():
    print(f"    In={r['temporal_in_degree']:>3}  "
          f"Out={r['temporal_out_degree']:>3}  "
          f"Net={r['net_direction']:>+4}  "
          f"{r['condition_name'][:45]}")

# ─── Step 11: Demographic subgroup comparison ─────────────────────────────
print("\nStep 11: Comparing demographic subgroup networks...")

subgroup_stats = []

subgroup_files = {
    "male": "network_subgroup_edges_male.csv",
    "female": "network_subgroup_edges_female.csv",
    "under60": "network_subgroup_edges_under60.csv",
    "age60_74": "network_subgroup_edges_age60_74.csv",
    "age75plus": "network_subgroup_edges_age75plus.csv",
}

for sg_name, sg_file in subgroup_files.items():
    sg_edges = pd.read_csv(os.path.join(OUT_DIR, sg_file))

    sg_G = nx.Graph()
    for _, row in nodes.iterrows():
        sg_G.add_node(row["condition_concept_id"])

    for _, row in sg_edges.iterrows():
        sg_G.add_edge(row["condition_id_a"], row["condition_id_b"], weight=row["jaccard"])

    sg_degrees = [d for _, d in sg_G.degree()]
    sg_connected = sum(1 for d in sg_degrees if d > 0)

    sg_comms = nx.community.louvain_communities(sg_G, weight="weight", seed=42)
    sg_mod = nx.community.modularity(sg_G, sg_comms, weight="weight")

    # Top 5 by degree in this subgroup
    sg_degree_dict = dict(sg_G.degree())
    sg_top5 = sorted(sg_degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    sg_top5_names = "; ".join(
        f"{cid_to_name.get(cid, '?')[:30]}({deg})" for cid, deg in sg_top5
    )

    subgroup_stats.append({
        "subgroup": sg_name,
        "edges": sg_G.number_of_edges(),
        "connected_nodes": sg_connected,
        "density": round(nx.density(sg_G), 6),
        "avg_degree": round(np.mean(sg_degrees), 2),
        "max_degree": max(sg_degrees),
        "avg_clustering": round(nx.average_clustering(sg_G, weight="weight"), 4),
        "n_communities": len(sg_comms),
        "modularity": round(sg_mod, 4),
        "top5_by_degree": sg_top5_names,
    })

    print(f"\n  {sg_name}:")
    print(f"    Edges: {sg_G.number_of_edges():,}, Connected: {sg_connected}, "
          f"Density: {nx.density(sg_G):.4f}")
    print(f"    Communities: {len(sg_comms)}, Modularity: {sg_mod:.4f}")

sg_comp_df = pd.DataFrame(subgroup_stats)
sg_comp_df.to_csv(os.path.join(OUT_DIR, "analysis_subgroup_comparison.csv"), index=False)
print(f"\n  Saved: data/analysis_subgroup_comparison.csv")

# ─── Step 12: Summary report ─────────────────────────────────────────────
print("\n" + "=" * 70)

summary_lines = [
    "=" * 70,
    "T2D COMORBIDITY NETWORK ANALYSIS — PROJECT 1 RESULTS",
    "=" * 70,
    "",
    "--- Global Network Statistics ---",
]
for k, v in net_stats.items():
    summary_lines.append(f"  {k}: {v}")

summary_lines.extend([
    "",
    f"--- Community Detection (Louvain, modularity={modularity:.4f}) ---",
    f"  Communities: {n_communities}",
    f"  Sizes: {sorted(comm_sizes, reverse=True)}",
])

for _, row in comm_summary_df.iterrows():
    summary_lines.append(
        f"  Community {row['community_id']}: {row['size']} conditions, "
        f"density={row['density']:.3f}"
    )
    summary_lines.append(f"    Top: {row['top_conditions'][:80]}")

summary_lines.extend([
    "",
    f"--- Bridge Diagnoses ({len(bridges)} found) ---",
    f"  Criteria: betweenness >= {bt_thresh:.6f} (Q75) AND "
    f"participation_coeff >= {pc_thresh:.4f} (Q75)",
])

for _, r in bridges.head(10).iterrows():
    summary_lines.append(
        f"  BT={r['betweenness_centrality']:.4f} PC={r['participation_coefficient']:.3f} "
        f"Comms={r['n_communities_connected']}  {r['condition_name'][:50]}"
    )

summary_lines.extend([
    "",
    "--- Top 10 Gateway Conditions (Temporal) ---",
])
for _, r in temporal_hubs.head(10).iterrows():
    summary_lines.append(
        f"  OutStrength={r['temporal_out_strength']:>6.0f}  "
        f"{r['condition_name'][:50]}"
    )

summary_lines.extend([
    "",
    "--- Demographic Subgroup Comparison ---",
])
for _, r in sg_comp_df.iterrows():
    summary_lines.append(
        f"  {r['subgroup']:>10}: {r['edges']:>6,} edges, "
        f"density={r['density']:.4f}, "
        f"communities={r['n_communities']}, "
        f"modularity={r['modularity']:.4f}"
    )

summary_lines.extend(["", "=" * 70])

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUT_DIR, "analysis_summary.txt"), "w") as f:
    f.write(summary_text)

print(summary_text)
print("\nDone. All analysis outputs saved to data/ folder.")
