import torch
import os
from precompute_walks import precompute_neighbors, save_neighbors
from data_utils_inductive import build_inductive_split

# === Setup paths ===
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Group-Project"
EDGE_CSV_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "edges.csv")
FEATURE_NPY_PATH = os.path.join(GOOGLE_DRIVE_PATH, "data", "features.npy")
SAVE_DIR = os.path.join(GOOGLE_DRIVE_PATH, "data")
os.makedirs(SAVE_DIR, exist_ok=True)

print("🔍 Loading graph...\n")
train_g, _, _, _, val_g, _, _ = build_inductive_split(
    edge_csv_path=EDGE_CSV_PATH,
    feature_npy_path=FEATURE_NPY_PATH,
    test_ratio=0.2
)

# === Precompute for training graph ===
train_neighbors = precompute_neighbors(
    train_g,
    num_walks=15,        
    walk_length=5,       
    device='cuda'
)
save_neighbors(train_neighbors, os.path.join(SAVE_DIR, "train_neighbors_precomputed.npy"))

# === Precompute for validation graph ===
val_neighbors = precompute_neighbors(
    val_g,
    num_walks=15,         
    walk_length=5,     
    device='cuda'
)
save_neighbors(val_neighbors, os.path.join(SAVE_DIR, "val_neighbors_precomputed.npy"))
