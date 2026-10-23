"""
verify_extraction.py
====================
Verifies the output of 01_cohort_extraction.py against raw EHRShot tables.
Runs 20+ checks to ensure data integrity.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("VERIFICATION OF T2D COHORT EXTRACTION")
print("=" * 70)

# Load extracted data
cohort = pd.read_csv("data/t2d_cohort_conditions.csv")
demo = pd.read_csv("data/t2d_cohort_demographics.csv")
t2d_ids = pd.read_csv("data/t2d_concept_ids.csv")
prevalence = pd.read_csv("data/condition_prevalence.csv")

# Load raw source tables
DATA_DIR = "../ehrshot_new_data/ehrshot"
raw_cond = pd.read_csv(
    f"{DATA_DIR}/condition_occurrence.csv",
    usecols=["person_id", "condition_concept_id", "condition_start_DATE"],
)
raw_person = pd.read_csv(
    f"{DATA_DIR}/person.csv",
    usecols=["person_id", "gender_concept_id", "year_of_birth", "race_concept_id"],
)
ancestors = pd.read_csv(
    f"{DATA_DIR}/concept_ancestor.csv",
    usecols=["ancestor_concept_id", "descendant_concept_id"],
)
concepts = pd.read_csv(
    f"{DATA_DIR}/concept.csv",
    usecols=["concept_id", "concept_name", "domain_id", "standard_concept"],
)

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


# ─── CHECK 1: T2D Concept IDs ───────────────────────────────────────
print("\n--- CHECK 1: T2D Concept ID Expansion ---")

raw_t2d_desc = set(
    ancestors[ancestors["ancestor_concept_id"] == 201826]["descendant_concept_id"]
)
extracted_t2d = set(t2d_ids["concept_id"])

check(
    "T2D descendants match concept_ancestor",
    raw_t2d_desc == extracted_t2d,
    f"Raw: {len(raw_t2d_desc)}, Extracted: {len(extracted_t2d)}",
)

check("201826 is included in descendants (self-reference)", 201826 in extracted_t2d)

check(
    "All T2D descendants are standard concepts",
    all(t2d_ids["standard_concept"] == "S"),
)

# ─── CHECK 2: Patient Cohort ────────────────────────────────────────
print("\n--- CHECK 2: Patient Cohort ---")

raw_t2d_patients = set(
    raw_cond[raw_cond["condition_concept_id"].isin(raw_t2d_desc)]["person_id"].unique()
)
extracted_patients = set(cohort["person_id"].unique())
demo_patients = set(demo["person_id"].unique())

check(
    "Cohort patient count matches raw derivation",
    raw_t2d_patients == extracted_patients,
    f"Raw: {len(raw_t2d_patients)}, Extracted: {len(extracted_patients)}",
)

check(
    "Demographics covers all cohort patients",
    extracted_patients == demo_patients,
    f"Conditions: {len(extracted_patients)}, Demo: {len(demo_patients)}",
)

check("No non-T2D patients in cohort", extracted_patients.issubset(raw_t2d_patients))

# Verify every cohort patient has at least 1 T2D diagnosis
t2d_patients_in_cohort = set(
    cohort[cohort["condition_concept_id"].isin(raw_t2d_desc)]["person_id"].unique()
)
check(
    "Every cohort patient has at least 1 T2D diagnosis",
    t2d_patients_in_cohort == extracted_patients,
    f"With T2D dx: {len(t2d_patients_in_cohort)}, Total: {len(extracted_patients)}",
)

# ─── CHECK 3: Condition Records ─────────────────────────────────────
print("\n--- CHECK 3: Condition Records ---")

std_cond_ids = set(
    concepts[
        (concepts["domain_id"] == "Condition") & (concepts["standard_concept"] == "S")
    ]["concept_id"]
)
cohort_cond_ids = set(cohort["condition_concept_id"].unique())

check(
    "All conditions are standard SNOMED",
    cohort_cond_ids.issubset(std_cond_ids),
    f"Non-standard found: {len(cohort_cond_ids - std_cond_ids)}",
)

dup_check = cohort.duplicated(
    subset=["person_id", "condition_concept_id", "condition_start_DATE"]
)
check(
    "No duplicate (patient + condition + date) rows",
    dup_check.sum() == 0,
    f"Duplicates found: {dup_check.sum()}",
)

check(
    "All records belong to T2D cohort patients",
    set(cohort["person_id"]).issubset(raw_t2d_patients),
)

check(
    "No null dates in condition_start_DATE",
    cohort["condition_start_DATE"].isna().sum() == 0,
)

cohort["_date"] = pd.to_datetime(cohort["condition_start_DATE"], errors="coerce")
check("All dates are valid/parseable", cohort["_date"].isna().sum() == 0)

# ─── CHECK 4: Completeness ──────────────────────────────────────────
print("\n--- CHECK 4: Completeness ---")

raw_cohort_check = raw_cond[raw_cond["person_id"].isin(raw_t2d_patients)].copy()
raw_cohort_check = raw_cohort_check[
    raw_cohort_check["condition_concept_id"].isin(std_cond_ids)
]
raw_cohort_check = raw_cohort_check.drop_duplicates(
    subset=["person_id", "condition_concept_id", "condition_start_DATE"]
)
raw_cohort_check = raw_cohort_check.dropna(subset=["condition_start_DATE"])

check(
    "Record count matches independent raw derivation",
    len(cohort) == len(raw_cohort_check),
    f"Extracted: {len(cohort):,}, Independent: {len(raw_cohort_check):,}",
)

null_names = cohort["condition_name"].isna().sum()
check("All conditions have names", null_names == 0, f"Missing names: {null_names}")

# ─── CHECK 5: Demographics ──────────────────────────────────────────
print("\n--- CHECK 5: Demographics ---")

check("No null gender", demo["gender_concept_id"].isna().sum() == 0)

check("No null year_of_birth", demo["year_of_birth"].isna().sum() == 0)

check(
    "Age range is plausible (18-110)",
    int(demo["age"].min()) >= 18 and int(demo["age"].max()) <= 110,
    f"Min age: {demo['age'].min()}, Max age: {demo['age'].max()}",
)

check(
    "Gender values are valid (8507=M, 8532=F)",
    set(demo["gender_concept_id"].unique()).issubset({8507, 8532}),
    f"Found: {sorted(demo['gender_concept_id'].unique())}",
)

check("All patients assigned an age group", demo["age_group"].isna().sum() == 0)

male_n = len(demo[demo["gender"] == "Male"])
female_n = len(demo[demo["gender"] == "Female"])
check(
    "Gender counts sum to total",
    male_n + female_n == len(demo),
    f"Male: {male_n}, Female: {female_n}, Total: {len(demo)}",
)

# Cross-check demographics against raw person table
raw_demo = raw_person[raw_person["person_id"].isin(raw_t2d_patients)]
check(
    "Demographics match raw person table count",
    len(demo) == len(raw_demo),
    f"Extracted: {len(demo)}, Raw: {len(raw_demo)}",
)

# Spot-check a patient's year_of_birth
sample_pid = demo["person_id"].iloc[0]
raw_yob = int(raw_person[raw_person["person_id"] == sample_pid]["year_of_birth"].iloc[0])
ext_yob = int(demo[demo["person_id"] == sample_pid]["year_of_birth"].iloc[0])
check(
    f"Spot-check birth year for patient {sample_pid}",
    raw_yob == ext_yob,
    f"Raw: {raw_yob}, Extracted: {ext_yob}",
)

# ─── CHECK 6: Prevalence Table ──────────────────────────────────────
print("\n--- CHECK 6: Prevalence Table ---")

check(
    "Prevalence table has all unique conditions",
    len(prevalence) == cohort["condition_concept_id"].nunique(),
    f"Prevalence rows: {len(prevalence)}, Unique: {cohort['condition_concept_id'].nunique()}",
)

# Spot-check top condition
top = prevalence.iloc[0]
manual_top = cohort[
    cohort["condition_concept_id"] == top["condition_concept_id"]
]["person_id"].nunique()
check(
    f"Top condition count ({top['condition_name'][:40]})",
    int(top["patient_count"]) == manual_top,
    f"Table: {int(top['patient_count'])}, Manual: {manual_top}",
)

# Spot-check 5th condition
fifth = prevalence.iloc[4]
manual_5 = cohort[
    cohort["condition_concept_id"] == fifth["condition_concept_id"]
]["person_id"].nunique()
check(
    f"5th condition count ({fifth['condition_name'][:40]})",
    int(fifth["patient_count"]) == manual_5,
    f"Table: {int(fifth['patient_count'])}, Manual: {manual_5}",
)

# Spot-check a condition in the middle
mid = prevalence.iloc[len(prevalence) // 2]
manual_mid = cohort[
    cohort["condition_concept_id"] == mid["condition_concept_id"]
]["person_id"].nunique()
check(
    f"Mid-range condition count ({mid['condition_name'][:40]})",
    int(mid["patient_count"]) == manual_mid,
    f"Table: {int(mid['patient_count'])}, Manual: {manual_mid}",
)

# Verify patient_pct calculation
expected_pct = round(int(top["patient_count"]) / 1354 * 100, 2)
check(
    "patient_pct calculation is correct",
    abs(top["patient_pct"] - expected_pct) < 0.01,
    f"Table: {top['patient_pct']}, Expected: {expected_pct}",
)

# ─── CHECK 7: Data Integrity ────────────────────────────────────────
print("\n--- CHECK 7: Data Integrity ---")

# Check chronological sorting
cohort_sorted = cohort.copy()
cohort_sorted["_date"] = pd.to_datetime(cohort_sorted["condition_start_DATE"])
is_sorted = all(
    cohort_sorted.groupby("person_id")["_date"].apply(
        lambda x: x.is_monotonic_increasing
    )
)
check("Records are sorted by date within each patient", is_sorted)

# Check no future dates
max_date = pd.to_datetime(cohort["condition_start_DATE"]).max()
check(
    "No future dates (max date <= 2024)",
    max_date.year <= 2024,
    f"Max date: {max_date}",
)

# Check condition_concept_id has no nulls
check(
    "No null condition_concept_ids",
    cohort["condition_concept_id"].isna().sum() == 0,
)

# Check person_id has no nulls
check("No null person_ids", cohort["person_id"].isna().sum() == 0)

# ─── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED out of {passed + failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED - Data extraction is verified correct.")
else:
    print(f"WARNING: {failed} check(s) failed - review above.")
print("=" * 70)
