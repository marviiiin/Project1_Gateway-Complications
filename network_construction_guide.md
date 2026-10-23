# Network Construction Guide for T2D Comorbidity Project

Procedures derived from course lectures (Lecture 05: Project Topics, Lecture 07: ML with Graphs)
and supporting literature.

---

## Common Recipe (from Lecture 05)

Every worked example follows the same 5-step pipeline:

**Research Question -> Tables & Code Filter -> Nodes & Edges -> Project 1 Analysis -> Project 2 GNN Task**

Your project maps most directly to **Topic 5: Comorbidity Communities and Bridge Diagnoses**.

---

## Step 0: The Course's Graph Construction Table (Topic 5)

| Element | Definition |
|---------|-----------|
| **Node** | One condition concept, rolled up to a consistent level of the hierarchy |
| **Edge** | Undirected, between two conditions recorded for the same patient |
| **Weight** | Jaccard similarity or lift, so common conditions do not dominate by volume alone |
| **Graphs** | One overall graph, plus one per demographic subgroup for comparison |

---

## Step 1: Define Nodes

From Lecture 07: *"What are nodes? What are edges? The choice determines what you can study."*

**For your co-occurrence network:**
- Each node = one standard SNOMED condition concept (already extracted in Step 1 of cohort extraction)
- Apply a **minimum patient threshold** to filter out noise. Your prevalence table shows:
  - >= 50 patients -> 578 conditions
  - >= 100 patients -> 262 conditions
  - Pick a threshold and justify it (the slides say: "drop codes seen in only a handful of patients")

**Design decision from slides:** "Roll specific codes up to a clinically meaningful level and attach
readable names." You already kept standard SNOMED codes (not rolled up to vague ancestors like
"Clinical finding"), which is the right call. The slides explicitly warn: *"too high and everything
collapses"*.

---

## Step 2: Define Edges — Choose the Co-occurrence Window

This is the single most impactful design decision. The slides state explicitly:

> **"Same-visit versus same-patient co-occurrence -- this single choice reshapes the whole network."**

Three options (each produces a different network):

| Window | What it means | Effect on network |
|--------|--------------|-------------------|
| **Same visit** | Two conditions diagnosed on the same encounter | Sparse, captures acute co-diagnoses |
| **Same time window** (e.g., 30, 90, 180 days) | Conditions within N days of each other | Medium density, tunable |
| **Same patient lifetime** | Both conditions ever recorded for the same patient | Dense, captures chronic comorbidity |

**Recommendation from Topic 5:** Use **same-patient lifetime** for the co-occurrence network (your
research is about the chronic comorbidity landscape). For the temporal/directed network, use a
windowed or first-occurrence approach.

For your **temporal/directed network**, the slides from Topics 1, 2, and 4 show:
- **Edge**: Directed from an earlier event to a later one
- **Weight**: Inverse of the gap in days, so tightly clustered events connect strongly
- **Window**: You must choose (30, 90, 180 days) -- "justify your choice"
- Use **first occurrence** of each condition per patient to establish temporal ordering

---

## Step 3: Weight Edges — Normalize to Avoid False Hubs

The slides give an explicit warning:

> **"High-prevalence conditions connect to everything. Normalize, or they become false hubs."**

Three weighting options the course recommends:

1. **Jaccard similarity**: `J(A,B) = |patients with both| / |patients with A or B|`
   - Normalizes by prevalence of both conditions
   - Recommended in Topic 5 explicitly

2. **Lift (PMI)**: `Lift(A,B) = P(A,B) / (P(A) * P(B))`
   - "Pointwise mutual information, not raw counts -- rare codes otherwise vanish" (Topic 3)
   - Good for finding unexpected associations

3. **Inverse time gap** (for temporal network): Weight = 1 / (days between first occurrences)
   - From Topics 1, 2, 4: "tightly clustered events connect strongly"

**Literature backing**: Hidalgo et al. (2009) uses Relative Risk (RR) as edge weight. Jaccard and lift
are both standard alternatives cited in the OHDSI community.

---

## Step 4: Apply an Edge-Weight Threshold

From Topic 5:

> **"An edge-weight threshold, so the diagram stays readable."**

After computing weights, keep only edges above a meaningful threshold. This is essential for:
- Visualization (otherwise you get a hairball)
- Community detection quality (Louvain works better on sparser graphs)
- Statistical significance (only keep edges where co-occurrence is statistically above chance)

Methods:
- Keep top-K edges per node
- Keep edges where Jaccard or lift > some percentile cutoff
- Use a statistical test (e.g., chi-squared or Fisher's exact on the 2x2 contingency table per edge)

---

## Step 5: Handle Demographic Subgroups

From Topic 5:

> **"Whether subgroup graphs use the same node set, which is required for a fair comparison."**

When building networks by age group, gender, or race:
- **Keep the same node set** across all subgroups (use the union of all conditions meeting threshold in the overall network)
- Recompute edge weights within each subgroup
- This allows direct comparison of community structure across demographics

---

## Step 6: Node Features (for Project 2 / GNN)

Lecture 07 provides the node feature hierarchy:

1. **Node degree** -- simplest structural feature
2. **Node importance** -- betweenness centrality (number of shortest paths through a node), closeness centrality (avg shortest path length)
3. **Substructures / Graphlets** -- "A count vector of rooted subgraphs at a given node" (Graphlet Degree Vector / GDV)
   - Counts how many times each small subgraph pattern appears around a node
   - More expressive than degree alone

For your comorbidity network, node features would include:
- Degree, betweenness, closeness, eigenvector centrality
- Patient count and prevalence percentage
- Community membership (from Louvain)
- Participation coefficient (fraction of edges going to other communities -- identifies bridge diagnoses)

---

## Step 7: Watch-Outs from the Slides

| Warning | Source | What to do |
|---------|--------|-----------|
| "High-prevalence conditions connect to everything" | Topic 5 | Use Jaccard/lift, not raw co-occurrence counts |
| "Communities are not diagnoses -- describe what each cluster means before interpreting it" | Topic 5 | After Louvain, read the member conditions and name each community clinically |
| "Which level of concept_ancestor to roll up to -- too high and everything collapses" | Topic 1 | Keep standard SNOMED (you already do this) |
| "Whether to allow repeat diagnoses to create self-loops" | Topic 1 | For co-occurrence: no self-loops. For temporal: decide if a recurrence creates a loop |
| "A long path may just mean a long observation period. Normalize by follow-up time" | Topic 1 | If computing temporal distances, divide by patient follow-up length |
| "Leakage: an escalation event must never appear inside the graph you classify" | Topics 2, 4 | If predicting future complications, use only conditions before the prediction point |

---

## Your Two Networks

### Network A — Co-occurrence (Undirected, for Project 1 community analysis)

| Element | Your definition |
|---------|----------------|
| Node | Standard SNOMED condition with >= N patients (pick 50 or 100) |
| Edge | Two conditions co-occurring in the same patient (lifetime) |
| Weight | Jaccard similarity (or lift) |
| Threshold | Remove edges below a cutoff (justify by statistical test or percentile) |
| Subgraphs | One overall + one per demographic group (same node set) |

### Network B — Temporal Transitions (Directed, for cascade/trajectory analysis)

| Element | Your definition |
|---------|----------------|
| Node | Same condition set as Network A |
| Edge | Directed from condition A -> B if A's first occurrence precedes B's |
| Weight | Number of patients showing A->B transition, or inverse time gap |
| Threshold | Minimum patient count for a transition (e.g., >= 5 patients) |
| Subgraphs | One overall + one per demographic group |

---

## The Link Prediction Task (Project 2)

From Lecture 07: "Predict new/missing/unknown links based on existing links."

For your project, this translates to: **Given the current comorbidity network, predict which
condition pairs are likely to become connected** -- i.e., which new comorbidities will emerge.

Approach:
- Mask some edges from the co-occurrence network
- Train GCN, GraphSAGE, and GAT to predict them
- Evaluate with ROC-AUC
- Ablation: weighted vs. unweighted edges; condition-only vs. patient-condition bipartite graph (Topic 5)

---

## Key Literature References

- Hidalgo et al., "A Dynamic Network Approach for the Study of Human Phenotypes," PLoS Computational Biology, 2009 -- uses full diagnostic history, RR-weighted edges
- Jensen et al., "Temporal Disease Trajectories Condensed from Population-Wide Registry Data," Nature Communications, 2014 -- directed temporal transitions, first-occurrence ordering
- Hripcsak et al., "Observational Health Data Sciences and Informatics (OHDSI)," 2015 -- concept_ancestor expansion methodology
- OHDSI Book, Chapter 5: "Standardized Vocabularies" -- standard concepts only
- Wornow et al., "EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models," NeurIPS 2023 -- single-occurrence phenotyping
