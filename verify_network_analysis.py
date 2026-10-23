"""
verify_network_analysis.py
===========================
Verifies the output of 03_network_analysis.py by independently recomputing
metrics from the network edge/node files.
Runs 30+ checks.
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

print("=" * 70)
print("VERIFICATION OF T2D NETWORK ANALYSIS")
print("=" * 70)

# ─── Load analysis outputs ────────────────────────────────────────────────
nodes = pd.read_csv("data/network_nodes.csv")
edges = pd.read_csv("data/network_cooccurrence_edges.csv")
temporal = pd.read_csv("data/network_temporal_edges.csv")
metrics = pd.read_csv("data/analysis_node_metrics.csv")
communities_csv = pd.read_csv("data/analysis_communities.csv")
bridges = pd.read_csv("data/analysis_bridge_diagnoses.csv")
net_stats = pd.read_csv("data/analysis_network_stats.csv")
subgroup_comp = pd.read_csv("data/analysis_subgroup_comparison.csv")
temporal_hubs = pd.read_csv("data/analysis_temporal_hubs.csv")

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# ─── Rebuild the graph independently ──────────────────────────────────────
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row["condition_concept_id"], name=row["condition_name"],
               patient_count=row["patient_count"])
for _, row in edges.iterrows():
    G.add_edge(row["condition_id_a"], row["condition_id_b"], weight=row["jaccard"],
               patients_both=row["patients_both"])

G_temp = nx.DiGraph()
for _, row in nodes.iterrows():
    G_temp.add_node(row["condition_concept_id"])
for _, row in temporal.iterrows():
    G_temp.add_edge(row["source_condition_id"], row["target_condition_id"],
                    weight=row["patient_count"])

# ─── CHECK 1: Global Network Statistics ───────────────────────────────────
print("\n--- CHECK 1: Global Network Statistics ---")

stats = net_stats.iloc[0]

check("Node count", int(stats["nodes"]) == G.number_of_nodes(),
      f"Reported: {int(stats['nodes'])}, Rebuilt: {G.number_of_nodes()}")

check("Edge count", int(stats["edges"]) == G.number_of_edges(),
      f"Reported: {int(stats['edges'])}, Rebuilt: {G.number_of_edges()}")

manual_density = nx.density(G)
check("Density", abs(stats["density"] - manual_density) < 1e-5,
      f"Reported: {stats['density']}, Manual: {round(manual_density, 6)}")

manual_components = nx.number_connected_components(G)
check("Connected components", int(stats["connected_components"]) == manual_components,
      f"Reported: {int(stats['connected_components'])}, Manual: {manual_components}")

largest_cc = max(nx.connected_components(G), key=len)
check("LCC size", int(stats["largest_component_size"]) == len(largest_cc),
      f"Reported: {int(stats['largest_component_size'])}, Manual: {len(largest_cc)}")

degrees = [d for _, d in G.degree()]
manual_avg_deg = np.mean(degrees)
check("Average degree", abs(stats["avg_degree"] - round(manual_avg_deg, 2)) < 0.1,
      f"Reported: {stats['avg_degree']}, Manual: {round(manual_avg_deg, 2)}")

check("Max degree", int(stats["max_degree"]) == max(degrees),
      f"Reported: {int(stats['max_degree'])}, Manual: {max(degrees)}")

manual_transitivity = nx.transitivity(G)
check("Transitivity", abs(stats["transitivity"] - round(manual_transitivity, 4)) < 1e-3,
      f"Reported: {stats['transitivity']}, Manual: {round(manual_transitivity, 4)}")

G_lcc = G.subgraph(largest_cc).copy()
manual_diameter = nx.diameter(G_lcc)
check("Diameter (LCC)", int(stats["diameter_lcc"]) == manual_diameter,
      f"Reported: {int(stats['diameter_lcc'])}, Manual: {manual_diameter}")

manual_avg_path = nx.average_shortest_path_length(G_lcc)
check("Avg path length (LCC)", abs(stats["avg_path_length_lcc"] - round(manual_avg_path, 4)) < 0.01,
      f"Reported: {stats['avg_path_length_lcc']}, Manual: {round(manual_avg_path, 4)}")

# ─── CHECK 2: Node Metrics Completeness ───────────────────────────────────
print("\n--- CHECK 2: Node Metrics Completeness ---")

check("Metrics cover all nodes", len(metrics) == len(nodes),
      f"Metrics rows: {len(metrics)}, Nodes: {len(nodes)}")

check("No null centrality values",
      metrics[["degree_centrality", "betweenness_centrality",
               "closeness_centrality", "eigenvector_centrality"]].isnull().sum().sum() == 0)

check("All nodes have community assignment", metrics["community"].isnull().sum() == 0)

check("No null participation coefficients",
      metrics["participation_coefficient"].isnull().sum() == 0)

check("Participation coefficients in [0, 1]",
      metrics["participation_coefficient"].min() >= 0
      and metrics["participation_coefficient"].max() <= 1,
      f"Range: [{metrics['participation_coefficient'].min():.4f}, "
      f"{metrics['participation_coefficient'].max():.4f}]")

# ─── CHECK 3: Centrality Spot-Checks ─────────────────────────────────────
print("\n--- CHECK 3: Centrality Spot-Checks ---")

manual_degree_cent = nx.degree_centrality(G)
manual_betweenness = nx.betweenness_centrality(G, weight="weight")
manual_closeness = nx.closeness_centrality(G)

# Spot-check 5 nodes
check_nodes = [
    metrics.iloc[0]["condition_concept_id"],
    metrics.iloc[len(metrics) // 4]["condition_concept_id"],
    metrics.iloc[len(metrics) // 2]["condition_concept_id"],
    metrics.iloc[3 * len(metrics) // 4]["condition_concept_id"],
    metrics.iloc[-1]["condition_concept_id"],
]

for cid in check_nodes:
    cid = int(cid)
    row = metrics[metrics["condition_concept_id"] == cid].iloc[0]
    name = row["condition_name"][:30]

    # Degree centrality
    dc_match = abs(row["degree_centrality"] - round(manual_degree_cent[cid], 6)) < 1e-4
    # Betweenness
    bt_match = abs(row["betweenness_centrality"] - round(manual_betweenness[cid], 6)) < 1e-4
    # Closeness
    cc_match = abs(row["closeness_centrality"] - round(manual_closeness[cid], 6)) < 1e-4

    check(f"Centralities for {name}",
          dc_match and bt_match and cc_match,
          f"DC: {row['degree_centrality']:.6f} vs {round(manual_degree_cent[cid], 6):.6f}, "
          f"BT: {row['betweenness_centrality']:.6f} vs {round(manual_betweenness[cid], 6):.6f}, "
          f"CC: {row['closeness_centrality']:.6f} vs {round(manual_closeness[cid], 6):.6f}")

# ─── CHECK 4: Degree Spot-Checks ─────────────────────────────────────────
print("\n--- CHECK 4: Degree Spot-Checks ---")

for cid in check_nodes:
    cid = int(cid)
    row = metrics[metrics["condition_concept_id"] == cid].iloc[0]
    manual_deg = G.degree(cid)
    check(f"Degree of {row['condition_name'][:30]}",
          int(row["degree"]) == manual_deg,
          f"Reported: {int(row['degree'])}, Manual: {manual_deg}")

# ─── CHECK 5: Community Detection ────────────────────────────────────────
print("\n--- CHECK 5: Community Detection ---")

# Rebuild communities from metrics
comm_mapping = dict(zip(metrics["condition_concept_id"], metrics["community"]))
comm_groups = {}
for cid, comm in comm_mapping.items():
    comm_groups.setdefault(comm, set()).add(cid)
partition = list(comm_groups.values())

n_communities_reported = len(communities_csv)
n_communities_from_metrics = len(comm_groups)
check("Community count matches summary",
      n_communities_reported == n_communities_from_metrics,
      f"Summary: {n_communities_reported}, From metrics: {n_communities_from_metrics}")

# All nodes assigned to exactly one community
node_assignments = metrics["condition_concept_id"].tolist()
check("Each node in exactly one community",
      len(node_assignments) == len(set(node_assignments)),
      f"Nodes: {len(node_assignments)}, Unique: {len(set(node_assignments))}")

total_in_communities = sum(len(c) for c in partition)
check("All nodes covered by communities",
      total_in_communities == len(nodes),
      f"In communities: {total_in_communities}, Total nodes: {len(nodes)}")

# Verify modularity is positive (well-separated communities)
manual_modularity = nx.community.modularity(G, partition, weight="weight")
check("Modularity is positive (meaningful communities)",
      manual_modularity > 0,
      f"Modularity: {manual_modularity:.4f}")

# Community sizes match
for _, crow in communities_csv.iterrows():
    comm_id = crow["community_id"]
    reported_size = crow["size"]
    actual_size = len(comm_groups.get(comm_id, set()))
    if reported_size != actual_size:
        check(f"Community {comm_id} size", False,
              f"Reported: {reported_size}, Actual: {actual_size}")
        break
else:
    check("All community sizes match", True)

# ─── CHECK 6: Participation Coefficient ──────────────────────────────────
print("\n--- CHECK 6: Participation Coefficient ---")


def manual_participation_coefficient(G, node, comm_map):
    ki = G.degree(node)
    if ki == 0:
        return 0.0
    comm_counts = Counter()
    for neighbor in G.neighbors(node):
        comm_counts[comm_map[neighbor]] += 1
    return 1.0 - sum((kis / ki) ** 2 for kis in comm_counts.values())


for cid in check_nodes:
    cid = int(cid)
    row = metrics[metrics["condition_concept_id"] == cid].iloc[0]
    manual_pc = manual_participation_coefficient(G, cid, comm_mapping)
    check(f"Participation coeff for {row['condition_name'][:30]}",
          abs(row["participation_coefficient"] - round(manual_pc, 6)) < 1e-4,
          f"Reported: {row['participation_coefficient']:.6f}, Manual: {round(manual_pc, 6):.6f}")

# ─── CHECK 7: Bridge Diagnoses ───────────────────────────────────────────
print("\n--- CHECK 7: Bridge Diagnoses ---")

bt_thresh = metrics["betweenness_centrality"].quantile(0.75)
pc_thresh = metrics["participation_coefficient"].quantile(0.75)

manual_bridges = metrics[
    (metrics["betweenness_centrality"] >= bt_thresh)
    & (metrics["participation_coefficient"] >= pc_thresh)
]

check("Bridge count matches Q75 filter",
      len(bridges) == len(manual_bridges),
      f"Reported: {len(bridges)}, Manual: {len(manual_bridges)}")

check("All bridges have betweenness >= Q75 threshold",
      bridges["betweenness_centrality"].min() >= bt_thresh,
      f"Min BT: {bridges['betweenness_centrality'].min():.6f}, Threshold: {bt_thresh:.6f}")

check("All bridges have participation_coeff >= Q75 threshold",
      bridges["participation_coefficient"].min() >= pc_thresh,
      f"Min PC: {bridges['participation_coefficient'].min():.6f}, Threshold: {pc_thresh:.6f}")

# Verify connected_communities for top bridge
top_bridge_cid = int(bridges.iloc[0]["condition_concept_id"])
neighbor_comms = set()
for neighbor in G.neighbors(top_bridge_cid):
    neighbor_comms.add(comm_mapping[neighbor])
reported_n_comms = int(bridges.iloc[0]["n_communities_connected"])
check(f"Top bridge ({bridges.iloc[0]['condition_name'][:25]}) communities connected",
      reported_n_comms == len(neighbor_comms),
      f"Reported: {reported_n_comms}, Manual: {len(neighbor_comms)}")

# ─── CHECK 8: Temporal Hub Metrics ───────────────────────────────────────
print("\n--- CHECK 8: Temporal Hub Metrics ---")

# Spot-check temporal out-degree and in-degree
for cid in check_nodes[:3]:
    cid = int(cid)
    row = metrics[metrics["condition_concept_id"] == cid].iloc[0]
    manual_out = G_temp.out_degree(cid) if cid in G_temp else 0
    manual_in = G_temp.in_degree(cid) if cid in G_temp else 0

    check(f"Temporal degrees for {row['condition_name'][:30]}",
          int(row["temporal_out_degree"]) == manual_out
          and int(row["temporal_in_degree"]) == manual_in,
          f"Out: {int(row['temporal_out_degree'])} vs {manual_out}, "
          f"In: {int(row['temporal_in_degree'])} vs {manual_in}")

# Spot-check temporal out-strength for top hub
top_hub = temporal_hubs.iloc[0]
top_cid = int(top_hub["condition_concept_id"])
manual_out_str = sum(d["weight"] for _, _, d in G_temp.out_edges(top_cid, data=True))
check(f"Top gateway ({top_hub['condition_name'][:25]}) out-strength",
      int(top_hub["temporal_out_strength"]) == int(manual_out_str),
      f"Reported: {int(top_hub['temporal_out_strength'])}, Manual: {int(manual_out_str)}")

# Verify net_direction = out_degree - in_degree
net_dir_correct = all(
    temporal_hubs.iloc[i]["net_direction"]
    == temporal_hubs.iloc[i]["temporal_out_degree"] - temporal_hubs.iloc[i]["temporal_in_degree"]
    for i in range(min(20, len(temporal_hubs)))
)
check("net_direction = out_degree - in_degree (top 20)", net_dir_correct)

# ─── CHECK 9: Subgroup Comparison ────────────────────────────────────────
print("\n--- CHECK 9: Subgroup Comparison ---")

# Spot-check male subgroup
sub_male_edges = pd.read_csv("data/network_subgroup_edges_male.csv")
sg_G = nx.Graph()
for _, row in nodes.iterrows():
    sg_G.add_node(row["condition_concept_id"])
for _, row in sub_male_edges.iterrows():
    sg_G.add_edge(row["condition_id_a"], row["condition_id_b"], weight=row["jaccard"])

male_row = subgroup_comp[subgroup_comp["subgroup"] == "male"].iloc[0]
manual_male_edges = sg_G.number_of_edges()
check("Male subgroup edge count",
      int(male_row["edges"]) == manual_male_edges,
      f"Reported: {int(male_row['edges'])}, Manual: {manual_male_edges}")

manual_male_density = nx.density(sg_G)
check("Male subgroup density",
      abs(male_row["density"] - round(manual_male_density, 6)) < 1e-4,
      f"Reported: {male_row['density']}, Manual: {round(manual_male_density, 6)}")

# Spot-check under60 subgroup
sub_u60_edges = pd.read_csv("data/network_subgroup_edges_under60.csv")
sg_G2 = nx.Graph()
for _, row in nodes.iterrows():
    sg_G2.add_node(row["condition_concept_id"])
for _, row in sub_u60_edges.iterrows():
    sg_G2.add_edge(row["condition_id_a"], row["condition_id_b"], weight=row["jaccard"])

u60_row = subgroup_comp[subgroup_comp["subgroup"] == "under60"].iloc[0]
check("Under60 subgroup edge count",
      int(u60_row["edges"]) == sg_G2.number_of_edges(),
      f"Reported: {int(u60_row['edges'])}, Manual: {sg_G2.number_of_edges()}")

# All 5 subgroups present
check("All 5 subgroups in comparison",
      len(subgroup_comp) == 5,
      f"Subgroups: {len(subgroup_comp)}")

expected_groups = {"male", "female", "under60", "age60_74", "age75plus"}
check("Correct subgroup names",
      set(subgroup_comp["subgroup"]) == expected_groups)

# ─── CHECK 10: Clinical Sanity ───────────────────────────────────────────
print("\n--- CHECK 10: Clinical Sanity ---")

# Top centrality node should be clinically meaningful (not a coding artifact)
top_bt = metrics.iloc[0]
check("Top betweenness node is clinically meaningful",
      top_bt["condition_name"] in [
          "Dyspnea", "Essential hypertension", "Electrocardiogram abnormal",
          "Pleural effusion", "Anemia", "Type 2 diabetes mellitus",
      ],
      f"Top: {top_bt['condition_name']}")

# Hypertension, T2D, hyperlipidemia should be in the same community
ht_comm = metrics[metrics["condition_name"] == "Essential hypertension"]["community"].iloc[0]
hl_comm = metrics[metrics["condition_name"] == "Hyperlipidemia"]["community"].iloc[0]
check("Hypertension and hyperlipidemia in same community",
      ht_comm == hl_comm,
      f"HT community: {ht_comm}, HL community: {hl_comm}")

# Renal conditions should cluster together
esrd = metrics[metrics["condition_name"].str.contains("End-stage renal", na=False)]
ckd = metrics[metrics["condition_name"].str.contains("chronic kidney disease", na=False, case=False)]
if len(esrd) > 0 and len(ckd) > 0:
    check("ESRD and CKD-related conditions in same community",
          esrd.iloc[0]["community"] == ckd.iloc[0]["community"],
          f"ESRD comm: {esrd.iloc[0]['community']}, CKD comm: {ckd.iloc[0]['community']}")

# Essential hypertension should be a top temporal gateway
ht_temporal = temporal_hubs[
    temporal_hubs["condition_name"] == "Essential hypertension"
]
if len(ht_temporal) > 0:
    ht_rank = ht_temporal.index[0]
    check("Essential hypertension is top temporal gateway",
          ht_rank == 0,
          f"Rank: {ht_rank} (0 = top)")

# ─── SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED out of {passed + failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED - Network analysis is verified correct.")
else:
    print(f"WARNING: {failed} check(s) failed - review above.")
print("=" * 70)
