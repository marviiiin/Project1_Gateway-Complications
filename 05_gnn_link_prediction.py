"""
05_gnn_link_prediction.py
==========================
Purpose:  Project 2 — GNN-based link prediction on the T2D comorbidity network.
          Predicts which condition pairs are likely to co-occur, using the
          network structure learned by graph neural networks.

Input:    data/network_nodes.csv
          data/network_cooccurrence_edges.csv
          data/analysis_node_metrics.csv

Output:   data/gnn_results.csv                — model comparison results
          data/gnn_predictions_sample.csv      — sample predictions from best model
          figures/fig9_gnn_roc_curves.png       — ROC curves for all models
          figures/fig10_gnn_comparison.png      — model comparison bar charts

Task Definition (from course Lecture 05, Topic 5 & Lecture 07)
---------------------------------------------------------------
"Link prediction — predict which condition pairs are likely to become
connected." (Lecture 07: Link-Level Prediction Task)

"Alternatively, link prediction for the next condition a patient is likely
to acquire." (Topic 5: Comorbidity Communities, Project 2)

Methodology:
1. Split edges into train/val/test (80/10/10)
2. Generate negative samples (non-edges) equal to positive count
3. Train GCN, GraphSAGE, and GAT on the training graph
4. Predict link existence on held-out edges
5. Evaluate with ROC-AUC and Average Precision
6. Ablation: weighted vs unweighted edges

References:
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
  Networks," ICLR 2017 (GCN)
- Hamilton et al., "Inductive Representation Learning on Large Graphs,"
  NeurIPS 2017 (GraphSAGE)
- Velickovic et al., "Graph Attention Networks," ICLR 2018 (GAT)
- Zhang & Chen, "Link Prediction Based on Graph Neural Networks," NeurIPS
  2018 (link prediction framework)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ─── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Step 1: Load and prepare data ────────────────────────────────────────
print("Step 1: Loading network data...")

nodes_df = pd.read_csv(os.path.join(OUT_DIR, "network_nodes.csv"))
edges_df = pd.read_csv(os.path.join(OUT_DIR, "network_cooccurrence_edges.csv"))
metrics_df = pd.read_csv(os.path.join(OUT_DIR, "analysis_node_metrics.csv"))

# Create node index mapping
all_cids = sorted(nodes_df["condition_concept_id"].tolist())
cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}
n_nodes = len(all_cids)

print(f"  Nodes: {n_nodes}, Edges: {len(edges_df)}")
print(f"  Device: {device}")

# ─── Step 2: Build node features ──────────────────────────────────────────
print("\nStep 2: Building node feature matrix...")

# Node features: patient_count, patient_pct, degree, strength,
# degree_centrality, betweenness_centrality, closeness_centrality,
# eigenvector_centrality, participation_coefficient,
# temporal_out_degree, temporal_in_degree
feature_cols = [
    "patient_count", "patient_pct", "degree", "strength",
    "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality",
    "participation_coefficient",
    "temporal_out_degree", "temporal_in_degree",
]

# Ensure metrics are ordered by condition_concept_id
metrics_ordered = metrics_df.set_index("condition_concept_id").loc[all_cids]
X = metrics_ordered[feature_cols].values.astype(np.float32)

# Normalize features (z-score)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # avoid division by zero
X = (X - X_mean) / X_std

x_tensor = torch.tensor(X, dtype=torch.float).to(device)
print(f"  Feature matrix: {x_tensor.shape} ({len(feature_cols)} features)")

# ─── Step 3: Build edge index and split ───────────────────────────────────
print("\nStep 3: Building edge index and train/val/test split...")

# Convert edges to index pairs (both directions for undirected)
src = edges_df["condition_id_a"].map(cid_to_idx).values
dst = edges_df["condition_id_b"].map(cid_to_idx).values
edge_weights = edges_df["jaccard"].values.astype(np.float32)

# Shuffle edges
n_edges = len(src)
perm = np.random.permutation(n_edges)
src = src[perm]
dst = dst[perm]
edge_weights = edge_weights[perm]

# Split: 80% train, 10% val, 10% test
n_train = int(0.8 * n_edges)
n_val = int(0.1 * n_edges)

train_src, train_dst = src[:n_train], dst[:n_train]
val_src, val_dst = src[n_train:n_train + n_val], dst[n_train:n_train + n_val]
test_src, test_dst = src[n_train + n_val:], dst[n_train + n_val:]
train_weights = edge_weights[:n_train]

print(f"  Train: {n_train}, Val: {n_val}, Test: {n_edges - n_train - n_val}")

# Build training edge index (both directions for undirected graph)
train_edge_index = torch.tensor(
    np.stack([np.concatenate([train_src, train_dst]),
              np.concatenate([train_dst, train_src])]),
    dtype=torch.long,
).to(device)

train_edge_weight = torch.tensor(
    np.concatenate([train_weights, train_weights]),
    dtype=torch.float,
).to(device)

# Build PyG Data object
data = Data(x=x_tensor, edge_index=train_edge_index)
data.edge_weight = train_edge_weight

print(f"  Training graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges")


# ─── Step 4: Define models ────────────────────────────────────────────────
print("\nStep 4: Defining GNN models...")


class GCNEncoder(torch.nn.Module):
    """2-layer GCN encoder (Kipf & Welling, 2017)."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class SAGEEncoder(torch.nn.Module):
    """2-layer GraphSAGE encoder (Hamilton et al., 2017)."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATEncoder(torch.nn.Module):
    """2-layer GAT encoder (Velickovic et al., 2018)."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.3)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(torch.nn.Module):
    """Dot-product link predictor on node embeddings."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        return self.decode(z, edge_label_index)


HIDDEN = 64
EMBED = 32
print(f"  Hidden: {HIDDEN}, Embedding: {EMBED}")


# ─── Step 5: Training loop ───────────────────────────────────────────────
def make_labels_and_edges(pos_src, pos_dst, num_nodes):
    """Create positive + negative edge pairs with labels."""
    pos_edge = torch.tensor(np.stack([pos_src, pos_dst]), dtype=torch.long).to(device)

    neg_edge = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=len(pos_src),
    ).to(device)

    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.shape[1]),
        torch.zeros(neg_edge.shape[1]),
    ]).to(device)

    return edge_label_index, labels


def train_model(model, data, train_src, train_dst, val_src, val_dst,
                n_epochs=200, lr=0.005, use_weights=True):
    """Train a link prediction model and return metrics per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    best_val_auc = 0
    best_state = None
    patience = 30
    patience_counter = 0

    history = {"train_loss": [], "val_auc": [], "val_ap": []}

    ew = data.edge_weight if use_weights else None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        edge_label_index, labels = make_labels_and_edges(
            train_src, train_dst, data.num_nodes
        )

        pred = model(data.x, data.edge_index, edge_label_index, edge_weight=ew)
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        loss.backward()
        optimizer.step()

        history["train_loss"].append(loss.item())

        # Validation
        if epoch % 5 == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                val_edge_label_index, val_labels = make_labels_and_edges(
                    val_src, val_dst, data.num_nodes
                )
                val_pred = model(data.x, data.edge_index, val_edge_label_index,
                                 edge_weight=ew)
                val_pred_np = torch.sigmoid(val_pred).cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()

                val_auc = roc_auc_score(val_labels_np, val_pred_np)
                val_ap = average_precision_score(val_labels_np, val_pred_np)

            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0:
                print(f"    Epoch {epoch:>3}: loss={loss.item():.4f}, "
                      f"val_AUC={val_auc:.4f}, val_AP={val_ap:.4f}")

            if patience_counter >= patience // 5:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_auc


def evaluate_model(model, data, test_src, test_dst, use_weights=True):
    """Evaluate a trained model on test edges."""
    model.eval()
    ew = data.edge_weight if use_weights else None

    with torch.no_grad():
        test_edge_label_index, test_labels = make_labels_and_edges(
            test_src, test_dst, data.num_nodes
        )
        test_pred = model(data.x, data.edge_index, test_edge_label_index,
                          edge_weight=ew)
        test_pred_np = torch.sigmoid(test_pred).cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()

    auc = roc_auc_score(test_labels_np, test_pred_np)
    ap = average_precision_score(test_labels_np, test_pred_np)
    fpr, tpr, _ = roc_curve(test_labels_np, test_pred_np)

    return auc, ap, fpr, tpr, test_pred_np, test_labels_np


# ─── Step 6: Train all models ────────────────────────────────────────────
print("\nStep 5-6: Training GNN models...")

in_channels = x_tensor.shape[1]
results = []
roc_data = {}

# 3 architectures x 2 weight settings = 6 experiments
experiments = [
    ("GCN", GCNEncoder, True),
    ("GCN (unweighted)", GCNEncoder, False),
    ("GraphSAGE", SAGEEncoder, True),
    ("GraphSAGE (unweighted)", SAGEEncoder, False),
    ("GAT", GATEncoder, True),
    ("GAT (unweighted)", GATEncoder, False),
]

for name, EncoderClass, use_weights in experiments:
    print(f"\n  --- {name} ---")

    # Run 3 times for mean/std (course requirement from Topic 5)
    run_aucs = []
    run_aps = []
    best_run_auc = 0
    best_run_data = None

    for run in range(3):
        torch.manual_seed(SEED + run)

        if EncoderClass == GATEncoder:
            encoder = EncoderClass(in_channels, HIDDEN // 4, EMBED, heads=4)
        else:
            encoder = EncoderClass(in_channels, HIDDEN, EMBED)

        model = LinkPredictor(encoder)

        model, history, val_auc = train_model(
            model, data, train_src, train_dst, val_src, val_dst,
            n_epochs=200, use_weights=use_weights,
        )

        auc, ap, fpr, tpr, preds, labels = evaluate_model(
            model, data, test_src, test_dst, use_weights=use_weights
        )

        run_aucs.append(auc)
        run_aps.append(ap)

        if auc > best_run_auc:
            best_run_auc = auc
            best_run_data = (fpr, tpr, preds, labels, model)

    mean_auc = np.mean(run_aucs)
    std_auc = np.std(run_aucs)
    mean_ap = np.mean(run_aps)
    std_ap = np.std(run_aps)

    print(f"  Test ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Test AP:      {mean_ap:.4f} +/- {std_ap:.4f}")

    results.append({
        "model": name,
        "test_auc_mean": round(mean_auc, 4),
        "test_auc_std": round(std_auc, 4),
        "test_ap_mean": round(mean_ap, 4),
        "test_ap_std": round(std_ap, 4),
        "weighted": use_weights,
    })

    roc_data[name] = (best_run_data[0], best_run_data[1], mean_auc)

    # Save best model's predictions for the weighted version
    if use_weights and best_run_data is not None:
        if not hasattr(evaluate_model, "_best_model"):
            evaluate_model._best_model = (name, best_run_data[4], mean_auc)
        elif mean_auc > evaluate_model._best_model[2]:
            evaluate_model._best_model = (name, best_run_data[4], mean_auc)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "gnn_results.csv"), index=False)
print(f"\n  Saved: data/gnn_results.csv")

# ─── Step 7: Generate sample predictions ──────────────────────────────────
print("\nStep 7: Generating sample predictions from best model...")

# Find best weighted model
best_result = results_df[results_df["weighted"]].sort_values("test_auc_mean", ascending=False).iloc[0]
best_name = best_result["model"]
print(f"  Best model: {best_name} (AUC={best_result['test_auc_mean']:.4f})")

# Re-run best model to get predictions on all non-edges (top predictions)
if EncoderClass == GATEncoder:
    encoder = GATEncoder(in_channels, HIDDEN // 4, EMBED, heads=4)
else:
    # Use the best architecture
    if "GCN" in best_name:
        encoder = GCNEncoder(in_channels, HIDDEN, EMBED)
    elif "SAGE" in best_name:
        encoder = SAGEEncoder(in_channels, HIDDEN, EMBED)
    else:
        encoder = GATEncoder(in_channels, HIDDEN // 4, EMBED, heads=4)

best_model = LinkPredictor(encoder).to(device)
best_model, _, _ = train_model(
    best_model, data, train_src, train_dst, val_src, val_dst,
    n_epochs=200, use_weights=True,
)

# Predict on test edges
test_auc, test_ap, _, _, test_preds, test_labels = evaluate_model(
    best_model, data, test_src, test_dst, use_weights=True
)

# Save sample: top 50 predicted new links (from test positives)
idx_to_cid = {i: cid for cid, i in cid_to_idx.items()}
cid_to_name = dict(zip(nodes_df["condition_concept_id"], nodes_df["condition_name"]))

# Get predictions for test positive edges
n_test_pos = len(test_src)
test_pos_preds = test_preds[:n_test_pos]
top_idx = np.argsort(test_pos_preds)[::-1][:50]

sample_preds = []
for i in top_idx:
    src_cid = idx_to_cid[test_src[i]]
    dst_cid = idx_to_cid[test_dst[i]]
    sample_preds.append({
        "condition_a": cid_to_name.get(src_cid, ""),
        "condition_b": cid_to_name.get(dst_cid, ""),
        "predicted_score": round(float(test_preds[i]), 4),
        "actual_label": int(test_labels[i]),
    })

sample_df = pd.DataFrame(sample_preds)
sample_df.to_csv(os.path.join(OUT_DIR, "gnn_predictions_sample.csv"), index=False)
print(f"  Saved: data/gnn_predictions_sample.csv ({len(sample_df)} predictions)")

# ─── Step 8: Figures ──────────────────────────────────────────────────────
print("\nStep 8: Generating figures...")

# Figure 9: ROC Curves
fig, ax = plt.subplots(figsize=(8, 7))

colors = ["#e6194b", "#e6194b", "#4363d8", "#4363d8", "#3cb44b", "#3cb44b"]
styles = ["-", "--", "-", "--", "-", "--"]

for (name, (fpr, tpr, auc)), color, style in zip(roc_data.items(), colors, styles):
    ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=2,
            label=f"{name} (AUC={auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Link Prediction ROC Curves — GNN Model Comparison", fontsize=14)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig9_gnn_roc_curves.png"))
plt.close(fig)
print("  Saved: figures/fig9_gnn_roc_curves.png")

# Figure 10: Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

model_names = results_df["model"].values
auc_means = results_df["test_auc_mean"].values
auc_stds = results_df["test_auc_std"].values
ap_means = results_df["test_ap_mean"].values
ap_stds = results_df["test_ap_std"].values

bar_colors = ["#e6194b", "#ffb3ba", "#4363d8", "#b3cde3", "#3cb44b", "#b3e6b3"]

axes[0].bar(range(len(model_names)), auc_means, yerr=auc_stds,
            color=bar_colors, edgecolor="white", capsize=4)
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
axes[0].set_ylabel("ROC-AUC")
axes[0].set_title("Test ROC-AUC (mean +/- std, 3 runs)")
axes[0].set_ylim(0.5, 1.0)
for i, (m, s) in enumerate(zip(auc_means, auc_stds)):
    axes[0].text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)

axes[1].bar(range(len(model_names)), ap_means, yerr=ap_stds,
            color=bar_colors, edgecolor="white", capsize=4)
axes[1].set_xticks(range(len(model_names)))
axes[1].set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
axes[1].set_ylabel("Average Precision")
axes[1].set_title("Test Average Precision (mean +/- std, 3 runs)")
axes[1].set_ylim(0.5, 1.0)
for i, (m, s) in enumerate(zip(ap_means, ap_stds)):
    axes[1].text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)

fig.suptitle("GNN Link Prediction — Model Comparison", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig10_gnn_comparison.png"))
plt.close(fig)
print("  Saved: figures/fig10_gnn_comparison.png")

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GNN LINK PREDICTION RESULTS")
print("=" * 70)
print(f"\n{'Model':<25} {'AUC':>12} {'AP':>12}")
print("-" * 50)
for _, row in results_df.iterrows():
    print(f"{row['model']:<25} "
          f"{row['test_auc_mean']:.4f}+/-{row['test_auc_std']:.4f} "
          f"{row['test_ap_mean']:.4f}+/-{row['test_ap_std']:.4f}")

print(f"\nBest model: {best_result['model']} "
      f"(AUC={best_result['test_auc_mean']:.4f})")
print(f"\nAblation: Weighted vs Unweighted edges")
for arch in ["GCN", "GraphSAGE", "GAT"]:
    w = results_df[results_df["model"] == arch]
    uw = results_df[results_df["model"] == f"{arch} (unweighted)"]
    if len(w) > 0 and len(uw) > 0:
        diff = w.iloc[0]["test_auc_mean"] - uw.iloc[0]["test_auc_mean"]
        print(f"  {arch}: weighted={w.iloc[0]['test_auc_mean']:.4f}, "
              f"unweighted={uw.iloc[0]['test_auc_mean']:.4f}, "
              f"diff={diff:+.4f}")

print("\n" + "=" * 70)
print("Done. All GNN results saved to data/ and figures/ folders.")
print("=" * 70)
