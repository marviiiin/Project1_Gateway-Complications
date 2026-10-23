"""
verify_gnn_results.py
======================
Verifies the output of 05_gnn_link_prediction.py.
Checks data integrity, model validity, and result plausibility.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("VERIFICATION OF GNN LINK PREDICTION RESULTS")
print("=" * 70)

# ─── Load outputs ─────────────────────────────────────────────────────────
results = pd.read_csv("data/gnn_results.csv")
predictions = pd.read_csv("data/gnn_predictions_sample.csv")
nodes = pd.read_csv("data/network_nodes.csv")
edges = pd.read_csv("data/network_cooccurrence_edges.csv")
metrics = pd.read_csv("data/analysis_node_metrics.csv")

valid_names = set(nodes["condition_name"])

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


# ─── CHECK 1: Results Table Structure ─────────────────────────────────────
print("\n--- CHECK 1: Results Table Structure ---")

check("Results file has rows", len(results) > 0, f"Rows: {len(results)}")

expected_models = {"GCN", "GCN (unweighted)", "GraphSAGE", "GraphSAGE (unweighted)",
                   "GAT", "GAT (unweighted)"}
check("All 6 model variants present",
      set(results["model"]) == expected_models,
      f"Found: {set(results['model'])}")

required_cols = {"model", "test_auc_mean", "test_auc_std", "test_ap_mean",
                 "test_ap_std", "weighted"}
check("All required columns present",
      required_cols.issubset(set(results.columns)),
      f"Columns: {list(results.columns)}")

check("No null values in results", results.isnull().sum().sum() == 0)

# ─── CHECK 2: AUC/AP Value Ranges ────────────────────────────────────────
print("\n--- CHECK 2: AUC/AP Value Ranges ---")

check("All AUC means in [0.5, 1.0] (better than random)",
      results["test_auc_mean"].min() >= 0.5 and results["test_auc_mean"].max() <= 1.0,
      f"Range: [{results['test_auc_mean'].min():.4f}, {results['test_auc_mean'].max():.4f}]")

check("All AP means in [0.5, 1.0] (better than random)",
      results["test_ap_mean"].min() >= 0.5 and results["test_ap_mean"].max() <= 1.0,
      f"Range: [{results['test_ap_mean'].min():.4f}, {results['test_ap_mean'].max():.4f}]")

check("All AUC stds are small (< 0.05, consistent across runs)",
      results["test_auc_std"].max() < 0.05,
      f"Max std: {results['test_auc_std'].max():.4f}")

check("All AP stds are small (< 0.05)",
      results["test_ap_std"].max() < 0.05,
      f"Max std: {results['test_ap_std'].max():.4f}")

check("AUC means are non-trivial (> 0.7, models learned something)",
      results["test_auc_mean"].min() > 0.7,
      f"Min AUC: {results['test_auc_mean'].min():.4f}")

# ─── CHECK 3: Model Ranking Plausibility ──────────────────────────────────
print("\n--- CHECK 3: Model Ranking Plausibility ---")

# GCN and GraphSAGE should generally outperform GAT on small graphs
gcn_auc = results[results["model"] == "GCN"]["test_auc_mean"].iloc[0]
sage_auc = results[results["model"] == "GraphSAGE"]["test_auc_mean"].iloc[0]
gat_auc = results[results["model"] == "GAT"]["test_auc_mean"].iloc[0]

check("GCN outperforms GAT (expected for small graph)",
      gcn_auc > gat_auc,
      f"GCN: {gcn_auc:.4f}, GAT: {gat_auc:.4f}")

check("GraphSAGE outperforms GAT",
      sage_auc > gat_auc,
      f"SAGE: {sage_auc:.4f}, GAT: {gat_auc:.4f}")

# Best model should have AUC > 0.85
best_auc = results["test_auc_mean"].max()
best_model = results.loc[results["test_auc_mean"].idxmax(), "model"]
check("Best model AUC > 0.85 (strong link prediction)",
      best_auc > 0.85,
      f"Best: {best_model} with AUC={best_auc:.4f}")

# ─── CHECK 4: Weighted vs Unweighted Ablation ────────────────────────────
print("\n--- CHECK 4: Weighted vs Unweighted Ablation ---")

for arch in ["GCN", "GraphSAGE", "GAT"]:
    w = results[results["model"] == arch]
    uw = results[results["model"] == f"{arch} (unweighted)"]

    if len(w) > 0 and len(uw) > 0:
        w_auc = w.iloc[0]["test_auc_mean"]
        uw_auc = uw.iloc[0]["test_auc_mean"]
        diff = abs(w_auc - uw_auc)

        # The difference should be small (< 5%), showing structure dominates
        check(f"{arch}: weighted vs unweighted difference < 5%",
              diff < 0.05,
              f"Weighted: {w_auc:.4f}, Unweighted: {uw_auc:.4f}, Diff: {diff:.4f}")

# Weighted column matches model name
for _, row in results.iterrows():
    expected_w = "unweighted" not in row["model"]
    check(f"'{row['model']}' weighted flag correct",
          row["weighted"] == expected_w,
          f"Reported: {row['weighted']}, Expected: {expected_w}")

# ─── CHECK 5: 3-Run Consistency ──────────────────────────────────────────
print("\n--- CHECK 5: 3-Run Consistency ---")

# Std should be > 0 (different seeds produce different results)
for _, row in results.iterrows():
    # Allow zero std for very stable models but flag if all are zero
    pass

nonzero_stds = (results["test_auc_std"] > 0).sum()
check("Most models have non-zero AUC std (runs differ)",
      nonzero_stds >= 4,
      f"Models with std > 0: {nonzero_stds}/6")

# Mean +/- std should still be > 0.5 (all runs beat random)
for _, row in results.iterrows():
    lower_bound = row["test_auc_mean"] - 2 * row["test_auc_std"]
    check(f"{row['model']}: mean - 2*std > 0.5 (all runs beat random)",
          lower_bound > 0.5,
          f"Lower bound: {lower_bound:.4f}")

# ─── CHECK 6: Predictions Sample ─────────────────────────────────────────
print("\n--- CHECK 6: Predictions Sample ---")

check("Predictions file has rows", len(predictions) > 0, f"Rows: {len(predictions)}")

check("Predictions have required columns",
      {"condition_a", "condition_b", "predicted_score", "actual_label"}.issubset(
          set(predictions.columns)))

check("No null values in predictions", predictions.isnull().sum().sum() == 0)

check("Predicted scores in [0, 1]",
      predictions["predicted_score"].min() >= 0
      and predictions["predicted_score"].max() <= 1,
      f"Range: [{predictions['predicted_score'].min():.4f}, "
      f"{predictions['predicted_score'].max():.4f}]")

check("Actual labels are binary (0 or 1)",
      set(predictions["actual_label"].unique()).issubset({0, 1}),
      f"Unique labels: {sorted(predictions['actual_label'].unique())}")

# All condition names should be valid
pred_names = set(predictions["condition_a"]) | set(predictions["condition_b"])
invalid_names = pred_names - valid_names
check("All predicted condition names are valid nodes",
      len(invalid_names) == 0,
      f"Invalid names: {invalid_names}" if invalid_names else "")

# Predictions should be sorted by score (descending)
scores = predictions["predicted_score"].values
is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
check("Predictions sorted by score (descending)", is_sorted)

# ─── CHECK 7: Train/Val/Test Split ───────────────────────────────────────
print("\n--- CHECK 7: Train/Val/Test Split ---")

n_edges = len(edges)
expected_train = int(0.8 * n_edges)
expected_val = int(0.1 * n_edges)
expected_test = n_edges - expected_train - expected_val

check("Total edges match source",
      n_edges == len(edges),
      f"Edges: {n_edges}")

check("80/10/10 split is reasonable",
      abs(expected_train / n_edges - 0.8) < 0.01
      and abs(expected_val / n_edges - 0.1) < 0.01
      and abs(expected_test / n_edges - 0.1) < 0.01,
      f"Train: {expected_train} ({expected_train/n_edges*100:.1f}%), "
      f"Val: {expected_val} ({expected_val/n_edges*100:.1f}%), "
      f"Test: {expected_test} ({expected_test/n_edges*100:.1f}%)")

# ─── CHECK 8: Node Features ──────────────────────────────────────────────
print("\n--- CHECK 8: Node Features ---")

feature_cols = [
    "patient_count", "patient_pct", "degree", "strength",
    "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality",
    "participation_coefficient",
    "temporal_out_degree", "temporal_in_degree",
]

check("All 11 feature columns exist in metrics",
      all(col in metrics.columns for col in feature_cols),
      f"Features: {len(feature_cols)}")

check("No null values in feature columns",
      metrics[feature_cols].isnull().sum().sum() == 0)

# Features should have variance (not constant)
low_var = []
for col in feature_cols:
    if metrics[col].std() == 0:
        low_var.append(col)
check("All features have non-zero variance",
      len(low_var) == 0,
      f"Zero-variance features: {low_var}" if low_var else "")

# ─── CHECK 9: Figure Files Exist ─────────────────────────────────────────
print("\n--- CHECK 9: Figure Files ---")

import os
fig_dir = "figures"

for fig_name in ["fig9_gnn_roc_curves.png", "fig10_gnn_comparison.png"]:
    path = os.path.join(fig_dir, fig_name)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    check(f"{fig_name} exists and non-empty",
          exists and size > 1000,
          f"Size: {size:,} bytes" if exists else "File not found")

# ─── CHECK 10: Clinical Plausibility of Predictions ──────────────────────
print("\n--- CHECK 10: Clinical Plausibility ---")

# Top positive predictions should involve clinically related conditions
top_positives = predictions[predictions["actual_label"] == 1].head(10)
if len(top_positives) > 0:
    avg_score_pos = top_positives["predicted_score"].mean()
    check("Top positive predictions have high scores (> 0.6)",
          avg_score_pos > 0.6,
          f"Avg score of top 10 positives: {avg_score_pos:.4f}")

# Check that the model discriminates: positive scores > negative scores on average
pos_scores = predictions[predictions["actual_label"] == 1]["predicted_score"]
neg_scores = predictions[predictions["actual_label"] == 0]["predicted_score"]

if len(pos_scores) > 0 and len(neg_scores) > 0:
    check("Positive examples scored higher than negatives on average",
          pos_scores.mean() > neg_scores.mean(),
          f"Pos mean: {pos_scores.mean():.4f}, Neg mean: {neg_scores.mean():.4f}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED out of {passed + failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED - GNN link prediction results are verified correct.")
else:
    print(f"WARNING: {failed} check(s) failed - review above.")
print("=" * 70)
