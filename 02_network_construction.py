"""
02_network_construction.py
===========================
Purpose:  Build comorbidity networks from the T2D cohort extraction.
          Constructs two networks and demographic subgroup variants.

Input:    data/t2d_cohort_conditions.csv
          data/t2d_cohort_demographics.csv
          data/condition_prevalence.csv

Output:   data/network_nodes.csv              — filtered condition nodes (>= 50 patients)
          data/network_cooccurrence_edges.csv  — undirected Jaccard-weighted edges
          data/network_temporal_edges.csv      — directed first-occurrence transition edges
          data/network_subgroup_edges_*.csv    — demographic subgroup co-occurrence edges
          data/network_summary.txt             — summary statistics

Design Decisions & Justifications
----------------------------------

1. NODE THRESHOLD — >= 50 patients
   Each condition must appear in >= 50 patients (~3.7% of cohort) to become a
   node. Jensen et al., "Temporal Disease Trajectories Condensed from
   Population-Wide Registry Data," Nature Communications, 2014, required
   sufficient patient support per diagnosis for reliable transition
   probabilities. This retains key T2D complications (nephropathy, retinopathy,
   neuropathy) while filtering noise.

2. CO-OCCURRENCE WINDOW — Same-patient lifetime
   Two conditions co-occur if they ever appear in the same patient's record.
   Hidalgo et al., "A Dynamic Network Approach for the Study of Human
   Phenotypes," PLoS Computational Biology, 2009, used lifetime co-occurrence
   for chronic disease network construction.

3. EDGE WEIGHTING — Jaccard similarity
   J(A,B) = |patients with both| / |patients with A or B|
   Normalizes for prevalence, preventing high-prevalence conditions from
   becoming false hubs. Recommended by course slides (Lecture 05, Topic 5):
   "Jaccard similarity or lift, so common conditions do not dominate by
   volume alone." Bounded [0,1] and more stable than lift for rare pairs.

4. EDGE THRESHOLD — Fisher's exact test (p < 0.01, Bonferroni) AND Jaccard >= 0.05
   Fisher's exact test on the 2x2 contingency table ensures edges represent
   real associations, not chance (Hidalgo et al., 2009). Bonferroni correction
   controls family-wise error across all pair tests. Jaccard floor removes
   statistically significant but clinically trivial associations.

5. TEMPORAL NETWORK — First-occurrence directed edges
   For each patient, the first diagnosis date of each condition establishes
   temporal ordering. Directed edge A → B if A precedes B. Jensen et al.
   (2014) used exactly this "temporal disease trajectory" methodology.
   Minimum >= 5 patients per transition.

6. DEMOGRAPHIC SUBGROUPS — Same node set, recomputed weights
   Course slides (Lecture 05, Topic 5): "Whether subgroup graphs use the same
   node set, which is required for a fair comparison." Same nodes across all
   subgroups; Jaccard recomputed within each.

7. NO SELF-LOOPS
   A condition co-occurring with itself is meaningless. Course slides
   (Lecture 05, Topic 1) flag this as a design decision.
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact

# ─── Configuration ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

PATIENT_THRESHOLD = 50       # Minimum patients per condition node
JACCARD_THRESHOLD = 0.05     # Minimum Jaccard for an edge
FISHER_ALPHA = 0.01          # Significance level (before Bonferroni)
TEMPORAL_MIN_PATIENTS = 5    # Minimum patients per temporal transition

# ─── Step 1: Load extracted data ───────────────────────────────────────────
print("Step 1: Loading extracted data...")

cohort = pd.read_csv(os.path.join(OUT_DIR, "t2d_cohort_conditions.csv"))
demo = pd.read_csv(os.path.join(OUT_DIR, "t2d_cohort_demographics.csv"))
prevalence = pd.read_csv(os.path.join(OUT_DIR, "condition_prevalence.csv"))

cohort["condition_start_DATE"] = pd.to_datetime(cohort["condition_start_DATE"])

n_patients_total = cohort["person_id"].nunique()
print(f"  Loaded {len(cohort):,} records, {n_patients_total} patients, "
      f"{cohort['condition_concept_id'].nunique()} conditions")

# ─── Step 2: Filter to nodes >= PATIENT_THRESHOLD ──────────────────────────
print(f"\nStep 2: Filtering conditions to >= {PATIENT_THRESHOLD} patients...")

valid_conditions = prevalence[prevalence["patient_count"] >= PATIENT_THRESHOLD].copy()
valid_cond_ids = set(valid_conditions["condition_concept_id"])

nodes = valid_conditions[
    ["condition_concept_id", "condition_name", "patient_count", "record_count", "patient_pct"]
].copy()
nodes = nodes.sort_values("patient_count", ascending=False).reset_index(drop=True)
nodes.to_csv(os.path.join(OUT_DIR, "network_nodes.csv"), index=False)
print(f"  Nodes: {len(nodes)} conditions")
print(f"  Saved: data/network_nodes.csv")

cohort_filtered = cohort[cohort["condition_concept_id"].isin(valid_cond_ids)].copy()
print(f"  Records after node filter: {len(cohort_filtered):,}")

# ─── Step 3: Build patient-condition binary matrix ─────────────────────────
print("\nStep 3: Building patient-condition binary matrix...")

all_patients = sorted(cohort_filtered["person_id"].unique())
all_conditions = sorted(valid_cond_ids)
n_conditions = len(all_conditions)

pid_to_idx = {pid: i for i, pid in enumerate(all_patients)}
cid_to_idx = {cid: i for i, cid in enumerate(all_conditions)}
cid_to_name = dict(zip(nodes["condition_concept_id"], nodes["condition_name"]))

patient_cond = cohort_filtered[["person_id", "condition_concept_id"]].drop_duplicates()
rows = patient_cond["person_id"].map(pid_to_idx).values
cols = patient_cond["condition_concept_id"].map(cid_to_idx).values

binary_matrix = sparse.csr_matrix(
    (np.ones(len(rows), dtype=np.int32), (rows, cols)),
    shape=(len(all_patients), n_conditions),
)

print(f"  Matrix: {binary_matrix.shape[0]} patients x {binary_matrix.shape[1]} conditions")
print(f"  Non-zero entries: {binary_matrix.nnz:,}")

# ─── Step 4: Co-occurrence network (undirected, Jaccard-weighted) ──────────
print("\nStep 4: Computing co-occurrence counts and Jaccard similarity...")

cooccurrence = (binary_matrix.T @ binary_matrix).toarray()
condition_counts = np.diag(cooccurrence).copy()

n_pairs = n_conditions * (n_conditions - 1) // 2
bonferroni_alpha = FISHER_ALPHA / n_pairs
print(f"  Pairs to evaluate: {n_pairs:,}")
print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.2e}")

edges = []
n_tested = 0

for i in range(n_conditions):
    if (i + 1) % 100 == 0:
        print(f"    Processed {i + 1}/{n_conditions} conditions "
              f"({len(edges)} edges so far)...")

    count_a = int(condition_counts[i])

    for j in range(i + 1, n_conditions):
        count_b = int(condition_counts[j])
        count_both = int(cooccurrence[i, j])

        if count_both == 0:
            continue

        jaccard = count_both / (count_a + count_b - count_both)

        if jaccard < JACCARD_THRESHOLD:
            continue

        # Fisher's exact test (one-sided, testing positive association)
        a_only = count_a - count_both
        b_only = count_b - count_both
        neither = len(all_patients) - count_a - count_b + count_both

        _, p_value = fisher_exact(
            [[count_both, a_only], [b_only, neither]],
            alternative="greater",
        )

        n_tested += 1

        if p_value < bonferroni_alpha:
            edges.append({
                "condition_id_a": all_conditions[i],
                "condition_id_b": all_conditions[j],
                "condition_name_a": cid_to_name.get(all_conditions[i], ""),
                "condition_name_b": cid_to_name.get(all_conditions[j], ""),
                "patients_both": count_both,
                "patients_a": count_a,
                "patients_b": count_b,
                "jaccard": round(jaccard, 6),
                "p_value": p_value,
            })

edges_df = pd.DataFrame(edges)
if len(edges_df) > 0:
    edges_df = edges_df.sort_values("jaccard", ascending=False).reset_index(drop=True)

edges_df.to_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"), index=False)

connected_nodes = set()
if len(edges_df) > 0:
    connected_nodes = set(edges_df["condition_id_a"]) | set(edges_df["condition_id_b"])

print(f"\n  Pairs tested (Jaccard >= {JACCARD_THRESHOLD}): {n_tested:,}")
print(f"  Significant edges: {len(edges_df):,}")
print(f"  Connected nodes: {len(connected_nodes)}/{n_conditions}")
print(f"  Edge density: {len(edges_df) / n_pairs * 100:.2f}%")
print(f"  Saved: data/network_cooccurrence_edges.csv")

if len(edges_df) > 0:
    print(f"\n  Top 10 edges by Jaccard:")
    for _, row in edges_df.head(10).iterrows():
        print(f"    {row['jaccard']:.4f}  "
              f"{row['condition_name_a'][:35]:35s} <-> {row['condition_name_b'][:35]}")

# ─── Step 5: Temporal transition network (directed) ────────────────────────
print("\nStep 5: Building temporal transition network (first-occurrence edges)...")

first_occ = (
    cohort_filtered.sort_values("condition_start_DATE")
    .groupby(["person_id", "condition_concept_id"])["condition_start_DATE"]
    .first()
    .reset_index()
)
first_occ.columns = ["person_id", "condition_concept_id", "first_date"]
print(f"  First-occurrence records: {len(first_occ):,}")

transition_counts = {}
n_patients_done = 0

for pid, group in first_occ.groupby("person_id"):
    n_patients_done += 1
    if n_patients_done % 500 == 0:
        print(f"    Processed {n_patients_done}/{len(all_patients)} patients...")

    group_sorted = group.sort_values("first_date")
    cids = group_sorted["condition_concept_id"].values
    dates = group_sorted["first_date"].values

    n = len(cids)
    for k in range(n):
        for l in range(k + 1, n):
            if dates[k] < dates[l]:
                pair = (int(cids[k]), int(cids[l]))
            elif dates[l] < dates[k]:
                pair = (int(cids[l]), int(cids[k]))
            else:
                continue  # same date — no direction
            transition_counts[pair] = transition_counts.get(pair, 0) + 1

print(f"  Unique directed transitions: {len(transition_counts):,}")

temporal_edges = []
for (src, tgt), count in transition_counts.items():
    if count >= TEMPORAL_MIN_PATIENTS:
        temporal_edges.append({
            "source_condition_id": src,
            "target_condition_id": tgt,
            "source_name": cid_to_name.get(src, "Unknown"),
            "target_name": cid_to_name.get(tgt, "Unknown"),
            "patient_count": count,
        })

temporal_df = pd.DataFrame(temporal_edges)
if len(temporal_df) > 0:
    temporal_df = temporal_df.sort_values("patient_count", ascending=False).reset_index(drop=True)

temporal_df.to_csv(os.path.join(OUT_DIR, "network_temporal_edges.csv"), index=False)

print(f"  Transitions >= {TEMPORAL_MIN_PATIENTS} patients: {len(temporal_df):,}")
print(f"  Saved: data/network_temporal_edges.csv")

if len(temporal_df) > 0:
    print(f"\n  Top 10 temporal transitions:")
    for _, row in temporal_df.head(10).iterrows():
        print(f"    {row['patient_count']:>5} patients  "
              f"{row['source_name'][:30]:30s} -> {row['target_name'][:30]}")

# ─── Step 6: Demographic subgroup networks ─────────────────────────────────
print("\nStep 6: Building demographic subgroup co-occurrence networks...")


def build_subgroup_edges(subgroup_pids, subgroup_name):
    """Build Jaccard-weighted co-occurrence edges for a subgroup (same node set)."""
    sub_pc = patient_cond[patient_cond["person_id"].isin(subgroup_pids)]
    sub_patients = sorted(sub_pc["person_id"].unique())
    n_sub = len(sub_patients)

    if n_sub == 0:
        print(f"  {subgroup_name}: 0 patients — skipped")
        return pd.DataFrame()

    sub_pid_map = {pid: i for i, pid in enumerate(sub_patients)}
    sub_rows = sub_pc["person_id"].map(sub_pid_map).values
    sub_cols = sub_pc["condition_concept_id"].map(cid_to_idx).values

    sub_matrix = sparse.csr_matrix(
        (np.ones(len(sub_rows), dtype=np.int32), (sub_rows, sub_cols)),
        shape=(n_sub, n_conditions),
    )

    sub_cooc = (sub_matrix.T @ sub_matrix).toarray()
    sub_counts = np.diag(sub_cooc).copy()

    sub_edges = []
    for i in range(n_conditions):
        ca = int(sub_counts[i])
        if ca == 0:
            continue
        for j in range(i + 1, n_conditions):
            cb = int(sub_counts[j])
            if cb == 0:
                continue
            cab = int(sub_cooc[i, j])
            if cab == 0:
                continue

            jac = cab / (ca + cb - cab)
            if jac < JACCARD_THRESHOLD:
                continue

            sub_edges.append({
                "condition_id_a": all_conditions[i],
                "condition_id_b": all_conditions[j],
                "patients_both": cab,
                "patients_a": ca,
                "patients_b": cb,
                "jaccard": round(jac, 6),
            })

    sub_df = pd.DataFrame(sub_edges)
    if len(sub_df) > 0:
        sub_df = sub_df.sort_values("jaccard", ascending=False).reset_index(drop=True)

    fname = f"network_subgroup_edges_{subgroup_name}.csv"
    sub_df.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"  {subgroup_name}: {n_sub} patients, {len(sub_df):,} edges -> {fname}")
    return sub_df


# Gender subgroups
male_pids = set(demo[demo["gender"] == "Male"]["person_id"])
female_pids = set(demo[demo["gender"] == "Female"]["person_id"])
build_subgroup_edges(male_pids, "male")
build_subgroup_edges(female_pids, "female")

# Age subgroups
under60_pids = set(demo[demo["age_group"] == "Under 60"]["person_id"])
age60_74_pids = set(demo[demo["age_group"] == "60-74"]["person_id"])
age75plus_pids = set(demo[demo["age_group"] == "75+"]["person_id"])
build_subgroup_edges(under60_pids, "under60")
build_subgroup_edges(age60_74_pids, "age60_74")
build_subgroup_edges(age75plus_pids, "age75plus")

# ─── Step 7: Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)

summary_lines = [
    "=" * 70,
    "T2D COMORBIDITY NETWORK CONSTRUCTION SUMMARY",
    "=" * 70,
    "",
    "--- Parameters ---",
    f"Patient threshold: >= {PATIENT_THRESHOLD} patients per condition",
    f"Co-occurrence window: Same-patient lifetime",
    f"Edge weighting: Jaccard similarity",
    f"Edge threshold: Fisher's exact p < {FISHER_ALPHA} (Bonferroni) AND Jaccard >= {JACCARD_THRESHOLD}",
    f"Temporal minimum: >= {TEMPORAL_MIN_PATIENTS} patients per transition",
    "",
    "--- Network A: Co-occurrence (Undirected) ---",
    f"Total nodes (conditions >= {PATIENT_THRESHOLD} patients): {len(nodes)}",
    f"Connected nodes (>= 1 edge): {len(connected_nodes)}",
    f"Edges: {len(edges_df):,}",
    f"Edge density: {len(edges_df) / n_pairs * 100:.2f}%",
    "",
    "--- Network B: Temporal Transitions (Directed) ---",
    f"Nodes: {len(nodes)} (same set)",
    f"Directed edges (>= {TEMPORAL_MIN_PATIENTS} patients): {len(temporal_df):,}",
    "",
    "--- Demographic Subgroups (Co-occurrence, Jaccard >= {}) ---".format(JACCARD_THRESHOLD),
    f"Male:     {len(male_pids)} patients",
    f"Female:   {len(female_pids)} patients",
    f"Under 60: {len(under60_pids)} patients",
    f"60-74:    {len(age60_74_pids)} patients",
    f"75+:      {len(age75plus_pids)} patients",
    "",
    "--- Design Decisions ---",
    "1. Node threshold: >= 50 patients (Jensen et al., 2014)",
    "2. Co-occurrence: same-patient lifetime (Hidalgo et al., 2009)",
    "3. Weighting: Jaccard similarity (course Lecture 05, Topic 5)",
    "4. Significance: Fisher's exact + Bonferroni (Hidalgo et al., 2009)",
    "5. Temporal: first-occurrence directed edges (Jensen et al., 2014)",
    "6. Subgroups: same node set, recomputed weights (course Lecture 05, Topic 5)",
    "7. No self-loops",
    "",
    "=" * 70,
]

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUT_DIR, "network_summary.txt"), "w") as f:
    f.write(summary_text)

print(summary_text)
print("\nDone. All network files saved to data/ folder.")
