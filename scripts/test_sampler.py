import os
import torch
from data_utils_inductive import build_inductive_split
from sampler import get_dataloaders, SimpleNeighborSampler, SimpleCollator, PairDataset
from torch.utils.data import DataLoader

# === Setup paths ===
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Group-Project"
EDGE_CSV_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "edges.csv")
FEATURE_NPY_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "features.npy")

# === Load graph and features ===
print("🔍 Loading data and building inductive split...\n")
train_g, features, train_edges, train_labels, val_g, val_edges, val_labels = build_inductive_split(
    edge_csv_path=EDGE_CSV_PATH,
    feature_npy_path=FEATURE_NPY_PATH,
    test_ratio=0.2
)

# === Build both train and val DataLoaders ===
train_loader, val_loader = get_dataloaders(
    train_edges=train_edges,
    train_labels=train_labels,
    val_edges=val_edges,
    val_labels=val_labels,
    features=features,
    train_graph=train_g,
    val_graph=val_g,
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# === Run through one batch of train_loader ===
print("\n🚀 Testing one batch from train_loader...\n")
for heads, tails, labels, blocks in train_loader:
    print(f"✅ Train heads: {heads.tolist()}")
    print(f"✅ Train tails: {tails.tolist()}")
    print(f"✅ Labels: {labels.tolist()}")
    print(f"✅ Train Blocks: {[block.num_nodes() for block in blocks]}")
    break

# === Run through one batch of val_loader ===
print("\n🧪 Testing one batch from val_loader...\n")
for heads, tails, labels, blocks in val_loader:
    print(f"✅ Val heads: {heads.tolist()}")
    print(f"✅ Val tails: {tails.tolist()}")
    print(f"✅ Labels: {labels.tolist()}")
    print(f"✅ Val Blocks: {[block.num_nodes() for block in blocks]}")
    break
