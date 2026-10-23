"""
verify_network_construction.py
===============================
Verifies the output of 02_network_construction.py against raw extracted data.
Runs 25+ checks to ensure network integrity.
"""
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

print("=" * 70)
print("VERIFICATION OF T2D NETWORK CONSTRUCTION")
print("=" * 70)

# ─── Load network outputs ─────────────────────────────────────────────────
nodes = pd.read_csv("data/network_nodes.csv")
edges = pd.read_csv("data/network_cooccurrence_edges.csv")
temporal = pd.read_csv("data/network_temporal_edges.csv")
cohort = pd.read_csv("data/t2d_cohort_conditions.csv")
demo = pd.read_csv("data/t2d_cohort_demographics.csv")
prevalence = pd.read_csv("data/condition_prevalence.csv")

cohort["condition_start_DATE"] = pd.to_datetime(cohort["condition_start_DATE"])

# Load subgroup files
sub_male = pd.read_csv("data/network_subgroup_edges_male.csv")
sub_female = pd.read_csv("data/network_subgroup_edges_female.csv")
sub_under60 = pd.read_csv("data/network_subgroup_edges_under60.csv")
sub_age60_74 = pd.read_csv("data/network_subgroup_edges_age60_74.csv")
sub_age75plus = pd.read_csv("data/network_subgroup_edges_age75plus.csv")

PATIENT_THRESHOLD = 50
JACCARD_THRESHOLD = 0.05
FISHER_ALPHA = 0.01
TEMPORAL_MIN_PATIENTS = 5

n_patients = cohort["person_id"].nunique()
valid_cond_ids = set(nodes["condition_concept_id"])
n_conditions = len(nodes)
n_pairs = n_conditions * (n_conditions - 1) // 2
bonferroni_alpha = FISHER_ALPHA / n_pairs

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


# ─── CHECK 1: Node Filtering ──────────────────────────────────────────────
print("\n--- CHECK 1: Node Filtering ---")

expected_nodes = prevalence[prevalence["patient_count"] >= PATIENT_THRESHOLD]
check(
    f"Node count matches prevalence >= {PATIENT_THRESHOLD}",
    len(nodes) == len(expected_nodes),
    f"Nodes: {len(nodes)}, Expected: {len(expected_nodes)}",
)

check(
    "All nodes have patient_count >= threshold",
    nodes["patient_count"].min() >= PATIENT_THRESHOLD,
    f"Min patient_count: {nodes['patient_count'].min()}",
)

check(
    "Node concept IDs match prevalence table",
    set(nodes["condition_concept_id"]) == set(expected_nodes["condition_concept_id"]),
)

check("No duplicate nodes", nodes["condition_concept_id"].duplicated().sum() == 0)

check("All nodes have names", nodes["condition_name"].isna().sum() == 0)

# ─── CHECK 2: Co-occurrence Edge Structure ─────────────────────────────────
print("\n--- CHECK 2: Co-occurrence Edge Structure ---")

check("No null values in edges", edges.isnull().sum().sum() == 0)

check(
    "All edge condition_id_a are valid nodes",
    set(edges["condition_id_a"]).issubset(valid_cond_ids),
)

check(
    "All edge condition_id_b are valid nodes",
    set(edges["condition_id_b"]).issubset(valid_cond_ids),
)

# No self-loops
self_loops = edges[edges["condition_id_a"] == edges["condition_id_b"]]
check("No self-loops", len(self_loops) == 0, f"Self-loops found: {len(self_loops)}")

# No duplicate edges (undirected: check both orderings)
edge_pairs = set()
dup_count = 0
for _, row in edges.iterrows():
    pair = (min(row["condition_id_a"], row["condition_id_b"]),
            max(row["condition_id_a"], row["condition_id_b"]))
    if pair in edge_pairs:
        dup_count += 1
    edge_pairs.add(pair)
check("No duplicate edges", dup_count == 0, f"Duplicates: {dup_count}")

# All edges are ordered (a < b)
check(
    "Edge IDs are canonically ordered (a < b)",
    (edges["condition_id_a"] < edges["condition_id_b"]).all(),
)

# ─── CHECK 3: Jaccard Computation ──────────────────────────────────────────
print("\n--- CHECK 3: Jaccard Computation (spot-checks) ---")

# Build patient-condition lookup from raw data
cohort_filtered = cohort[cohort["condition_concept_id"].isin(valid_cond_ids)]
patient_conditions = cohort_filtered.groupby("person_id")["condition_concept_id"].apply(set)

# Spot-check 5 edges
for idx in [0, len(edges) // 4, len(edges) // 2, 3 * len(edges) // 4, len(edges) - 1]:
    row = edges.iloc[idx]
    cid_a = row["condition_id_a"]
    cid_b = row["condition_id_b"]

    patients_a = set(
        cohort_filtered[cohort_filtered["condition_concept_id"] == cid_a]["person_id"].unique()
    )
    patients_b = set(
        cohort_filtered[cohort_filtered["condition_concept_id"] == cid_b]["person_id"].unique()
    )
    manual_both = len(patients_a & patients_b)
    manual_a = len(patients_a)
    manual_b = len(patients_b)
    manual_jaccard = manual_both / (manual_a + manual_b - manual_both)

    check(
        f"Edge #{idx} Jaccard ({row['condition_name_a'][:25]} <-> {row['condition_name_b'][:25]})",
        abs(row["jaccard"] - round(manual_jaccard, 6)) < 1e-5
        and row["patients_both"] == manual_both
        and row["patients_a"] == manual_a
        and row["patients_b"] == manual_b,
        f"Table: J={row['jaccard']}, both={row['patients_both']}, a={row['patients_a']}, b={row['patients_b']} | "
        f"Manual: J={round(manual_jaccard, 6)}, both={manual_both}, a={manual_a}, b={manual_b}",
    )

# ─── CHECK 4: Edge Thresholds ─────────────────────────────────────────────
print("\n--- CHECK 4: Edge Thresholds ---")

check(
    f"All edges have Jaccard >= {JACCARD_THRESHOLD}",
    edges["jaccard"].min() >= JACCARD_THRESHOLD,
    f"Min Jaccard: {edges['jaccard'].min():.6f}",
)

check(
    f"All edges have p_value < Bonferroni alpha ({bonferroni_alpha:.2e})",
    edges["p_value"].max() < bonferroni_alpha,
    f"Max p-value: {edges['p_value'].max():.2e}",
)

check(
    "Jaccard values are in [0, 1]",
    edges["jaccard"].min() >= 0 and edges["jaccard"].max() <= 1,
    f"Range: [{edges['jaccard'].min():.6f}, {edges['jaccard'].max():.6f}]",
)

# Spot-check Fisher's exact test on one edge
row = edges.iloc[0]
table = [
    [row["patients_both"], row["patients_a"] - row["patients_both"]],
    [row["patients_b"] - row["patients_both"],
     n_patients - row["patients_a"] - row["patients_b"] + row["patients_both"]],
]
_, manual_p = fisher_exact(table, alternative="greater")
check(
    "Fisher's exact p-value spot-check (top edge)",
    abs(row["p_value"] - manual_p) < 1e-10,
    f"Table: {row['p_value']:.2e}, Manual: {manual_p:.2e}",
)

# ─── CHECK 5: Missing Edges ───────────────────────────────────────────────
print("\n--- CHECK 5: Missing Edges (spot-check that excluded pairs fail threshold) ---")

# Pick a pair that is NOT in the edge list and verify it should be excluded
all_cond_list = sorted(valid_cond_ids)
excluded_count = 0
checked_excluded = 0
for i in range(min(50, n_conditions)):
    for j in range(i + 1, min(50, n_conditions)):
        cid_a = all_cond_list[i]
        cid_b = all_cond_list[j]
        pair = (cid_a, cid_b)
        if pair not in edge_pairs:
            # Verify this pair should indeed be excluded
            pa = set(
                cohort_filtered[cohort_filtered["condition_concept_id"] == cid_a]["person_id"]
            )
            pb = set(
                cohort_filtered[cohort_filtered["condition_concept_id"] == cid_b]["person_id"]
            )
            both = len(pa & pb)
            if both == 0:
                excluded_count += 1
                checked_excluded += 1
                continue
            jac = both / (len(pa) + len(pb) - both)
            if jac < JACCARD_THRESHOLD:
                excluded_count += 1
                checked_excluded += 1
                continue
            # Has Jaccard >= threshold — check Fisher
            tbl = [
                [both, len(pa) - both],
                [len(pb) - both, n_patients - len(pa) - len(pb) + both],
            ]
            _, pv = fisher_exact(tbl, alternative="greater")
            if pv >= bonferroni_alpha:
                excluded_count += 1
            checked_excluded += 1
            if checked_excluded >= 20:
                break
    if checked_excluded >= 20:
        break

check(
    f"Excluded pairs correctly fail thresholds ({checked_excluded} checked)",
    excluded_count == checked_excluded,
    f"Correctly excluded: {excluded_count}/{checked_excluded}",
)

# ─── CHECK 6: Temporal Network ────────────────────────────────────────────
print("\n--- CHECK 6: Temporal Network ---")

check("No null values in temporal edges", temporal.isnull().sum().sum() == 0)

check(
    "All source conditions are valid nodes",
    set(temporal["source_condition_id"]).issubset(valid_cond_ids),
)

check(
    "All target conditions are valid nodes",
    set(temporal["target_condition_id"]).issubset(valid_cond_ids),
)

# No self-loops in temporal
temporal_self = temporal[temporal["source_condition_id"] == temporal["target_condition_id"]]
check(
    "No temporal self-loops",
    len(temporal_self) == 0,
    f"Self-loops: {len(temporal_self)}",
)

check(
    f"All temporal edges have patient_count >= {TEMPORAL_MIN_PATIENTS}",
    temporal["patient_count"].min() >= TEMPORAL_MIN_PATIENTS,
    f"Min: {temporal['patient_count'].min()}",
)

# Spot-check top temporal transition
top_t = temporal.iloc[0]
src_id = top_t["source_condition_id"]
tgt_id = top_t["target_condition_id"]

first_occ = (
    cohort_filtered.sort_values("condition_start_DATE")
    .groupby(["person_id", "condition_concept_id"])["condition_start_DATE"]
    .first()
    .reset_index()
)

manual_count = 0
for pid, grp in first_occ.groupby("person_id"):
    src_rows = grp[grp["condition_concept_id"] == src_id]
    tgt_rows = grp[grp["condition_concept_id"] == tgt_id]
    if len(src_rows) > 0 and len(tgt_rows) > 0:
        src_date = src_rows["condition_start_DATE"].iloc[0]
        tgt_date = tgt_rows["condition_start_DATE"].iloc[0]
        if src_date < tgt_date:
            manual_count += 1

check(
    f"Top temporal transition count ({top_t['source_name'][:25]} -> {top_t['target_name'][:25]})",
    int(top_t["patient_count"]) == manual_count,
    f"Table: {int(top_t['patient_count'])}, Manual: {manual_count}",
)

# Verify temporal direction: source should precede target in time for counted patients
mid_t = temporal.iloc[len(temporal) // 2]
src_id2 = mid_t["source_condition_id"]
tgt_id2 = mid_t["target_condition_id"]

manual_count2 = 0
for pid, grp in first_occ.groupby("person_id"):
    sr = grp[grp["condition_concept_id"] == src_id2]
    tr = grp[grp["condition_concept_id"] == tgt_id2]
    if len(sr) > 0 and len(tr) > 0:
        if sr["condition_start_DATE"].iloc[0] < tr["condition_start_DATE"].iloc[0]:
            manual_count2 += 1

check(
    f"Mid temporal transition count ({mid_t['source_name'][:25]} -> {mid_t['target_name'][:25]})",
    int(mid_t["patient_count"]) == manual_count2,
    f"Table: {int(mid_t['patient_count'])}, Manual: {manual_count2}",
)

# ─── CHECK 7: Demographic Subgroups ───────────────────────────────────────
print("\n--- CHECK 7: Demographic Subgroups ---")

# Check all subgroups use the same node set
for name, sub_df in [
    ("male", sub_male),
    ("female", sub_female),
    ("under60", sub_under60),
    ("age60_74", sub_age60_74),
    ("age75plus", sub_age75plus),
]:
    sub_cids = set(sub_df["condition_id_a"]) | set(sub_df["condition_id_b"])
    check(
        f"{name}: all edge nodes are in the master node set",
        sub_cids.issubset(valid_cond_ids),
    )

# Spot-check male subgroup: recompute one Jaccard
male_pids = set(demo[demo["gender"] == "Male"]["person_id"])
male_cohort = cohort_filtered[cohort_filtered["person_id"].isin(male_pids)]

if len(sub_male) > 0:
    top_m = sub_male.iloc[0]
    m_pa = set(
        male_cohort[male_cohort["condition_concept_id"] == top_m["condition_id_a"]]["person_id"]
    )
    m_pb = set(
        male_cohort[male_cohort["condition_concept_id"] == top_m["condition_id_b"]]["person_id"]
    )
    m_both = len(m_pa & m_pb)
    m_jac = m_both / (len(m_pa) + len(m_pb) - m_both) if (len(m_pa) + len(m_pb) - m_both) > 0 else 0

    check(
        "Male subgroup: top edge Jaccard spot-check",
        abs(top_m["jaccard"] - round(m_jac, 6)) < 1e-5,
        f"Table: {top_m['jaccard']}, Manual: {round(m_jac, 6)}",
    )

# Spot-check female subgroup
female_pids = set(demo[demo["gender"] == "Female"]["person_id"])
female_cohort = cohort_filtered[cohort_filtered["person_id"].isin(female_pids)]

if len(sub_female) > 0:
    top_f = sub_female.iloc[0]
    f_pa = set(
        female_cohort[female_cohort["condition_concept_id"] == top_f["condition_id_a"]]["person_id"]
    )
    f_pb = set(
        female_cohort[female_cohort["condition_concept_id"] == top_f["condition_id_b"]]["person_id"]
    )
    f_both = len(f_pa & f_pb)
    f_jac = f_both / (len(f_pa) + len(f_pb) - f_both) if (len(f_pa) + len(f_pb) - f_both) > 0 else 0

    check(
        "Female subgroup: top edge Jaccard spot-check",
        abs(top_f["jaccard"] - round(f_jac, 6)) < 1e-5,
        f"Table: {top_f['jaccard']}, Manual: {round(f_jac, 6)}",
    )

# Check subgroup Jaccard threshold
for name, sub_df in [
    ("male", sub_male),
    ("female", sub_female),
    ("under60", sub_under60),
    ("age60_74", sub_age60_74),
    ("age75plus", sub_age75plus),
]:
    if len(sub_df) > 0:
        check(
            f"{name}: all edges have Jaccard >= {JACCARD_THRESHOLD}",
            sub_df["jaccard"].min() >= JACCARD_THRESHOLD,
            f"Min: {sub_df['jaccard'].min():.6f}",
        )

# ─── CHECK 8: Clinical Sanity ─────────────────────────────────────────────
print("\n--- CHECK 8: Clinical Sanity ---")

# Verify known T2D comorbidities appear as strong edges
known_pairs = [
    ("Essential hypertension", "Hyperlipidemia"),
    ("Essential hypertension", "Type 2 diabetes mellitus"),
]

for name_a, name_b in known_pairs:
    found = edges[
        ((edges["condition_name_a"].str.contains(name_a, na=False))
         & (edges["condition_name_b"].str.contains(name_b, na=False)))
        | ((edges["condition_name_a"].str.contains(name_b, na=False))
           & (edges["condition_name_b"].str.contains(name_a, na=False)))
    ]
    check(
        f"Known comorbidity pair exists: {name_a[:25]} <-> {name_b[:25]}",
        len(found) > 0,
        f"Edges found: {len(found)}, top Jaccard: {found['jaccard'].max():.4f}" if len(found) > 0 else "",
    )

# Verify temporal: hypertension should precede many T2D complications
ht_as_source = temporal[temporal["source_name"].str.contains("Essential hypertension", na=False)]
check(
    "Hypertension is a major temporal source (gateway condition)",
    len(ht_as_source) > 50,
    f"Outgoing transitions from hypertension: {len(ht_as_source)}",
)

# ─── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED out of {passed + failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED - Network construction is verified correct.")
else:
    print(f"WARNING: {failed} check(s) failed - review above.")
print("=" * 70)
