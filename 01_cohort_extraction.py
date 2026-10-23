"""
01_cohort_extraction.py
========================
Purpose:  Extract the Type 2 Diabetes (T2D) cohort from the EHRShot dataset
          and prepare condition records for network construction.

Input:    Raw EHRShot CSV files from ../ehrshot_new_data/ehrshot/
Output:   data/t2d_cohort_conditions.csv   — all condition records for T2D patients
          data/t2d_cohort_demographics.csv  — demographics for T2D patients
          data/t2d_concept_ids.csv          — T2D descendant concept IDs used
          data/cohort_summary.txt           — summary statistics

Design Decisions & Justifications
----------------------------------

1. COHORT DEFINITION — Phenotype via concept_ancestor expansion
   We identify T2D patients using SNOMED concept_id 201826 ("Type 2 diabetes
   mellitus") and expand to ALL descendants via concept_ancestor. This follows
   the standard OHDSI cohort definition methodology recommended by:
     - Hripcsak et al., "Observational Health Data Sciences and Informatics
       (OHDSI): Opportunities for Observational Researchers," Studies in Health
       Technology and Informatics, 2015.
     - The OMOP CDM documentation (https://ohdsi.github.io/CommonDataModel/)
   Using concept_ancestor ensures we capture variant codings (e.g., "T2D without
   complication", "insulin-treated T2D", "poorly controlled T2D") that all map
   to the same clinical entity.

2. INCLUSION — At least ONE T2D diagnosis record
   A patient enters the cohort if they have >= 1 condition_occurrence record
   matching any T2D descendant concept. This is consistent with:
     - Wornow et al., "EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of
       Foundation Models," NeurIPS 2023 — uses single-occurrence phenotyping.
     - Singh et al., "Predicting Obstetric and Non-obstetric Diagnoses Co-
       occurrences During Pregnancy," BICOB 2026 — uses the same approach on
       EHRShot for pregnancy cohort definition.

3. ALL DIAGNOSES EXTRACTED — Not just T2D-related
   For each cohort patient, we extract ALL condition_occurrence records (not
   only diabetes-related ones). This is essential because:
     - Our research question concerns the COMORBIDITY landscape — how T2D
       interacts with cardiovascular, renal, neurological, and other conditions.
     - Hidalgo et al., "A Dynamic Network Approach for the Study of Human
       Phenotypes," PLoS Computational Biology, 2009 — uses full diagnostic
       history to build phenotypic disease networks.
     - Jensen et al., "Temporal Disease Trajectories Condensed from Population-
       Wide Registry Data," Nature Communications, 2014 — extracts all
       diagnoses for trajectory analysis.

4. STANDARD CONCEPTS ONLY
   We keep only records where condition_concept_id maps to a standard SNOMED
   concept (standard_concept = 'S', domain_id = 'Condition'). Non-standard
   source codes (ICD-9, ICD-10 billing codes) are already mapped to their
   standard equivalents in EHRShot's OMOP CDM. This avoids double-counting
   and follows OHDSI best practices:
     - OHDSI Book, Chapter 5: "Standardized Vocabularies"
     - Observational Health Data Sciences and Informatics, "The Book of OHDSI,"
       2019 (https://ohdsi.github.io/TheBookOfOhdsi/).

5. DEDUPLICATION
   We remove duplicate records where the same patient has the same
   condition_concept_id on the same date. This prevents inflating edge
   weights in the network due to administrative duplicates (e.g., multiple
   billing entries for the same diagnosis on the same encounter).
"""

import os
import sys
import pandas as pd
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ehrshot_new_data", "ehrshot")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

T2D_ANCESTOR_CONCEPT_ID = 201826  # SNOMED: "Type 2 diabetes mellitus"

# ─── Step 1: Load required tables ────────────────────────────────────────
print("Step 1: Loading EHRShot tables...")

conditions = pd.read_csv(
    os.path.join(DATA_DIR, "condition_occurrence.csv"),
    usecols=[
        "person_id",
        "condition_concept_id",
        "condition_start_DATE",
        "visit_occurrence_id",
    ],
)
print(f"  condition_occurrence: {len(conditions):,} records")

concepts = pd.read_csv(
    os.path.join(DATA_DIR, "concept.csv"),
    usecols=[
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "standard_concept",
    ],
)
print(f"  concept: {len(concepts):,} entries")

ancestors = pd.read_csv(
    os.path.join(DATA_DIR, "concept_ancestor.csv"),
    usecols=[
        "ancestor_concept_id",
        "descendant_concept_id",
        "min_levels_of_separation",
    ],
)
print(f"  concept_ancestor: {len(ancestors):,} entries")

person = pd.read_csv(
    os.path.join(DATA_DIR, "person.csv"),
    usecols=[
        "person_id",
        "gender_concept_id",
        "year_of_birth",
        "race_concept_id",
    ],
)
print(f"  person: {len(person):,} entries")

# ─── Step 2: Find all T2D descendant concept IDs ────────────────────────
print("\nStep 2: Expanding T2D concept via concept_ancestor...")

t2d_descendants = ancestors[
    ancestors["ancestor_concept_id"] == T2D_ANCESTOR_CONCEPT_ID
]["descendant_concept_id"].unique()

t2d_concept_info = concepts[concepts["concept_id"].isin(t2d_descendants)].copy()
t2d_concept_info = t2d_concept_info.sort_values("concept_id")

print(f"  T2D descendant concept IDs found: {len(t2d_descendants)}")
print("  Descendant concepts:")
for _, row in t2d_concept_info.iterrows():
    std = row.get("standard_concept", "")
    print(f"    {row['concept_id']:>10}  [{std}]  {row['concept_name']}")

t2d_concept_info.to_csv(os.path.join(OUT_DIR, "t2d_concept_ids.csv"), index=False)

# ─── Step 3: Identify T2D patients ──────────────────────────────────────
print("\nStep 3: Identifying T2D patient cohort...")

t2d_desc_set = set(t2d_descendants)
t2d_patient_ids = conditions[
    conditions["condition_concept_id"].isin(t2d_desc_set)
]["person_id"].unique()
t2d_patient_set = set(t2d_patient_ids)

total_patients = conditions["person_id"].nunique()
print(f"  T2D patients: {len(t2d_patient_ids)}")
print(f"  Total EHRShot patients (with conditions): {total_patients}")
print(f"  Cohort proportion: {len(t2d_patient_ids)/total_patients*100:.1f}%")

# ─── Step 4: Extract ALL condition records for cohort patients ──────────
print("\nStep 4: Extracting all condition records for T2D cohort...")

cohort = conditions[conditions["person_id"].isin(t2d_patient_set)].copy()
print(f"  Total records: {len(cohort):,}")

# ─── Step 5: Keep only standard condition concepts ──────────────────────
print("\nStep 5: Filtering to standard condition concepts only...")

std_condition_ids = set(
    concepts[
        (concepts["domain_id"] == "Condition") & (concepts["standard_concept"] == "S")
    ]["concept_id"]
)

before = len(cohort)
cohort = cohort[cohort["condition_concept_id"].isin(std_condition_ids)].copy()
print(f"  Records before: {before:,}")
print(f"  Records after (standard only): {len(cohort):,}")
print(f"  Dropped: {before - len(cohort):,} non-standard records")

# ─── Step 6: Deduplicate ────────────────────────────────────────────────
print("\nStep 6: Deduplicating (same patient + condition + date)...")

before = len(cohort)
cohort = cohort.drop_duplicates(
    subset=["person_id", "condition_concept_id", "condition_start_DATE"]
)
print(f"  Records before: {before:,}")
print(f"  Records after dedup: {len(cohort):,}")
print(f"  Duplicates removed: {before - len(cohort):,}")

# ─── Step 7: Add condition names ────────────────────────────────────────
print("\nStep 7: Adding condition names...")

concept_name_map = dict(zip(concepts["concept_id"], concepts["concept_name"]))
cohort["condition_name"] = cohort["condition_concept_id"].map(concept_name_map)

# ─── Step 8: Parse dates and sort ───────────────────────────────────────
print("Step 8: Parsing dates and sorting...")

cohort["condition_start_DATE"] = pd.to_datetime(
    cohort["condition_start_DATE"], errors="coerce"
)
cohort = cohort.dropna(subset=["condition_start_DATE"])
cohort = cohort.sort_values(["person_id", "condition_start_DATE"])

print(f"  Final cohort records: {len(cohort):,}")
print(f"  Unique conditions: {cohort['condition_concept_id'].nunique()}")
print(f"  Date range: {cohort['condition_start_DATE'].min()} to {cohort['condition_start_DATE'].max()}")

# ─── Step 9: Prepare demographics ──────────────────────────────────────
print("\nStep 9: Preparing demographics...")

demo = person[person["person_id"].isin(t2d_patient_set)].copy()
demo["age"] = 2026 - demo["year_of_birth"]
demo["age_group"] = pd.cut(
    demo["age"], bins=[0, 60, 75, 200], labels=["Under 60", "60-74", "75+"]
)

# Map gender codes to readable labels
gender_map = {8507: "Male", 8532: "Female"}
demo["gender"] = demo["gender_concept_id"].map(gender_map)

# Map race codes to readable labels (common OMOP race concept IDs)
race_map = {
    8515: "Asian",
    8516: "Black or African American",
    8527: "White",
    8557: "Native Hawaiian or Other Pacific Islander",
    8657: "American Indian or Alaska Native",
    0: "Unknown",
}
demo["race"] = demo["race_concept_id"].map(race_map).fillna("Other/Unknown")

print(f"  Patients: {len(demo)}")
print(f"\n  Age distribution:")
print(f"    Mean: {demo['age'].mean():.1f}, Median: {demo['age'].median():.0f}")
print(f"    Min: {demo['age'].min()}, Max: {demo['age'].max()}")
print(f"\n  Age groups:")
print(demo["age_group"].value_counts().sort_index().to_string(header=False))
print(f"\n  Gender:")
print(demo["gender"].value_counts().to_string(header=False))
print(f"\n  Race:")
print(demo["race"].value_counts().to_string(header=False))

# ─── Step 10: Save outputs ──────────────────────────────────────────────
print("\nStep 10: Saving outputs...")

cohort.to_csv(os.path.join(OUT_DIR, "t2d_cohort_conditions.csv"), index=False)
print(f"  Saved: data/t2d_cohort_conditions.csv ({len(cohort):,} rows)")

demo.to_csv(os.path.join(OUT_DIR, "t2d_cohort_demographics.csv"), index=False)
print(f"  Saved: data/t2d_cohort_demographics.csv ({len(demo)} rows)")

# ─── Step 11: Condition prevalence table ────────────────────────────────
print("\nStep 11: Building condition prevalence table...")

prevalence = (
    cohort.groupby("condition_concept_id")
    .agg(
        condition_name=("condition_name", "first"),
        patient_count=("person_id", "nunique"),
        record_count=("person_id", "count"),
    )
    .reset_index()
)
prevalence["patient_pct"] = (
    prevalence["patient_count"] / len(t2d_patient_ids) * 100
).round(2)
prevalence = prevalence.sort_values("patient_count", ascending=False)
prevalence.to_csv(os.path.join(OUT_DIR, "condition_prevalence.csv"), index=False)
print(f"  Saved: data/condition_prevalence.csv ({len(prevalence)} conditions)")

# Show thresholds
print("\n  Condition count at different patient thresholds:")
for thresh in [5, 10, 20, 30, 50, 75, 100]:
    n = (prevalence["patient_count"] >= thresh).sum()
    print(f"    >= {thresh:>3} patients: {n:>4} conditions")

# ─── Step 12: Summary report ────────────────────────────────────────────
summary_lines = [
    "=" * 70,
    "T2D COHORT EXTRACTION SUMMARY",
    "=" * 70,
    "",
    "Source: EHRShot dataset (Stanford STARR-OMOP)",
    f"T2D anchor concept: {T2D_ANCESTOR_CONCEPT_ID} (Type 2 diabetes mellitus)",
    f"T2D descendant concepts used: {len(t2d_descendants)}",
    "",
    "--- Cohort ---",
    f"T2D patients: {len(t2d_patient_ids)}",
    f"Total EHRShot patients: {total_patients}",
    f"Cohort proportion: {len(t2d_patient_ids)/total_patients*100:.1f}%",
    "",
    "--- Condition Records ---",
    f"Total records (after dedup, standard only): {len(cohort):,}",
    f"Unique condition concepts: {cohort['condition_concept_id'].nunique()}",
    f"Date range: {cohort['condition_start_DATE'].min()} to {cohort['condition_start_DATE'].max()}",
    "",
    "--- Demographics ---",
    f"Age: mean={demo['age'].mean():.1f}, median={demo['age'].median():.0f}, "
    f"min={demo['age'].min()}, max={demo['age'].max()}",
    f"Gender: Male={len(demo[demo['gender']=='Male'])}, Female={len(demo[demo['gender']=='Female'])}",
    "",
    "Age groups:",
    demo["age_group"].value_counts().sort_index().to_string(),
    "",
    "Race:",
    demo["race"].value_counts().to_string(),
    "",
    "--- Condition Prevalence Thresholds ---",
]
for thresh in [5, 10, 20, 30, 50, 75, 100]:
    n = (prevalence["patient_count"] >= thresh).sum()
    summary_lines.append(f"  >= {thresh:>3} patients: {n:>4} conditions")

summary_lines.extend([
    "",
    "--- Design Decisions ---",
    "1. Cohort: concept_ancestor expansion from 201826 (OHDSI standard)",
    "2. Inclusion: >= 1 T2D diagnosis record",
    "3. Scope: ALL diagnoses for cohort patients (full comorbidity landscape)",
    "4. Vocabulary: Standard SNOMED concepts only (standard_concept='S')",
    "5. Deduplication: same patient + condition + date",
    "",
    "=" * 70,
])

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUT_DIR, "cohort_summary.txt"), "w") as f:
    f.write(summary_text)
print(f"\n  Saved: data/cohort_summary.txt")

print("\n" + summary_text)
print("\nDone. All outputs saved to data/ folder.")
