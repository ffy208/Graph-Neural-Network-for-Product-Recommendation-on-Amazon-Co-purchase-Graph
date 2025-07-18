import os
import torch
from data_utils_inductive import build_inductive_split

# Paths to your preprocessed data (update these as needed)
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Group-Project"
EDGE_CSV_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "edges.csv")
FEATURE_NPY_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "features.npy")

print("🔍 Loading data and building inductive split...\n", flush=True)

# Call the data builder
train_g, features, train_edges, train_labels, val_g, val_edges, val_labels = build_inductive_split(
    edge_csv_path=EDGE_CSV_PATH,
    feature_npy_path=FEATURE_NPY_PATH,
    test_ratio=0.2,
    neg_ratio = 1
)

# === Simple sanity checks ===
print("✅ Train Graph:", flush=True)
print(f"  - Num nodes: {train_g.num_nodes()}", flush=True)
print(f"  - Num edges: {train_g.num_edges()}", flush=True)
print()

print("✅ Validation Graph:", flush=True)
print(f"  - Num nodes: {val_g.num_nodes()}", flush=True)
print(f"  - Num edges: {val_g.num_edges()}", flush=True)
print()

print("✅ Train Edges / Labels:", flush=True)
print(f"  - Positive + Negative edges: {len(train_edges)}", flush=True)
print(f"  - Labels (sample): {train_labels[:5]}", flush=True)
# print(f"  - Feature pairs shape: {train_feat_pairs.shape}", flush=True)
print()

print("✅ Validation Edges / Labels:", flush=True)
print(f"  - Positive + Negative edges: {len(val_edges)}", flush=True)
print(f"  - Labels (sample): {val_labels[:5]}", flush=True)
# print(f"  - Feature pairs shape: {val_feat_pairs.shape}", flush=True)
print()

print("🎉 Data utility pipeline works! You can now feed train_feat_pairs and train_labels into a model.", flush=True)
