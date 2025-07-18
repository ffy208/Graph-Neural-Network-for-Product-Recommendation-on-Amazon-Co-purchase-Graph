import os
os.environ["MPLBACKEND"] = "Agg"
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sampler import get_dataloaders
from pinsage_model import LinkPredictionModel
from data_utils_inductive import build_inductive_split
import dgl
import random
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import sys

# Redirect stdout to both console and log file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Setup logging
LOG_DIR = "/content/drive/MyDrive/Colab Notebooks/Group-Project/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "train_log.txt")
sys.stdout = Logger(log_file)

def plot_training_history(train_loss_list, train_auc_list, val_loss_list, val_auc_list, save_path):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_auc_list, label='Train AUC')
    plt.plot(epochs, val_auc_list, label='Val AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(train_loss_list, bins=20, alpha=0.6, label='Train Loss')
    plt.hist(val_loss_list, bins=20, alpha=0.6, label='Val Loss')
    plt.title('Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(train_auc_list, bins=20, alpha=0.6, label='Train AUC')
    plt.hist(val_auc_list, bins=20, alpha=0.6, label='Val AUC')
    plt.title('AUC Distribution')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_file = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_file)
    print(f"\U0001F4C8 Saved training history plot to: {plot_file}")

def train():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/content/drive/MyDrive/Colab Notebooks/Group-Project/data"
    GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Group-Project"
    MODEL_SAVE_PATH = os.path.join(GOOGLE_DRIVE_PATH, "model")
    PLOT_SAVE_PATH = os.path.join(GOOGLE_DRIVE_PATH, "plot")
    train_ratio = 0.8
    neg_ratio = 1

    print("\n🔍 Loading data and building inductive split...\n")
    train_g, features, train_edges, train_labels, val_g, val_edges, val_labels = build_inductive_split(
        edge_csv_path=f"{data_path}/edges.csv",
        feature_npy_path=f"{data_path}/features.npy",
        test_ratio=1 - train_ratio,
        neg_ratio=neg_ratio
    )

    print(f"✅ Graph: {train_g.num_nodes()} nodes, {train_g.num_edges()} edges")
    print(f"✅ Graph: {val_g.num_nodes()} nodes, {val_g.num_edges()} edges")
    print(f"✅ Train edges: {len(train_edges)}, Validation edges: {len(val_edges)}")

    param_grid = {
        "batch_size": [512], #[256, 512, 1024]
        "num_epochs": [35],
        "learning_rate": [1e-4], #[5e-4, 1e-3, 2e-3]
        "hidden_feats": [384], #[64, 128, 256]
        "out_feats": [128], #[64, 128]
        "num_layers": [2],
        "dropout": [0.3] #[0.05, 0.1, 0.2]
    }

    trials = 1
    best_auc = -1
    best_config = None
    best_state = None

    for trial in range(trials):
        print(f"\n🎲 Trial {trial + 1}/{trials}")
        batch_size = random.choice(param_grid["batch_size"])
        num_epochs = random.choice(param_grid["num_epochs"])
        learning_rate = random.choice(param_grid["learning_rate"])
        hidden_feats = random.choice(param_grid["hidden_feats"])
        out_feats = random.choice(param_grid["out_feats"])
        num_layers = random.choice(param_grid["num_layers"])
        dropout = random.choice(param_grid["dropout"])

        current_config = {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "hidden_feats": hidden_feats,
            "out_feats": out_feats,
            "num_layers": num_layers,
            "dropout": dropout,
            "train_ratio": train_ratio
        }

        print(f"🔧 Config: {current_config}")

        print("\n📦 Preparing dataloaders...\n")
        train_loader, val_loader = get_dataloaders(
            train_edges=train_edges,
            train_labels=train_labels,
            val_edges=val_edges,
            val_labels=val_labels,
            features=features,
            train_graph=train_g,
            val_graph=val_g,
            batch_size=batch_size,
            device=device
        )

        in_feats = features.shape[1]
        model = LinkPredictionModel(in_feats, hidden_feats, out_feats, num_layers, dropout).to(device)
        model.set_input_features(features)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=4e-5)

        train_loss_list, train_auc_list = [], []
        val_loss_list, val_auc_list = [], []

        best_trial_auc = -1
        best_trial_epoch = -1
        best_trial_state = None

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            all_preds = []
            all_labels = []
            start_time = time.time()

            for step, (heads, tails, labels, blocks) in enumerate(train_loader):
                labels = labels.to(dtype=torch.float32)
                optimizer.zero_grad()
                preds = model(blocks, heads, tails)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

            avg_loss = total_loss / len(train_loader.dataset)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            try:
                train_auc = roc_auc_score(all_labels, all_preds)
                train_ap = average_precision_score(all_labels, all_preds)
            except ValueError:
                train_auc = train_ap = float('nan')

            train_loss_list.append(avg_loss)
            train_auc_list.append(train_auc)

            model.eval()
            val_preds, val_true = [], []
            val_loss_total = 0
            with torch.no_grad():
                for heads, tails, labels, blocks in val_loader:
                    labels = labels.to(dtype=torch.float32)
                    preds = model(blocks, heads, tails)
                    val_loss_total += loss_fn(preds, labels).item() * len(labels)
                    val_preds.append(preds.cpu())
                    val_true.append(labels.cpu())

            val_preds = torch.cat(val_preds).numpy()
            val_true = torch.cat(val_true).numpy()
            val_loss = val_loss_total / len(val_loader.dataset)

            try:
                val_auc = roc_auc_score(val_true, val_preds)
                val_ap = average_precision_score(val_true, val_preds)
            except ValueError:
                val_auc = val_ap = float('nan')

            val_loss_list.append(val_loss)
            val_auc_list.append(val_auc)

            duration = time.time() - start_time
            gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

            print(f"✅ Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AUC: {train_auc:.4f} | AP: {train_ap:.4f} | Val AUC: {val_auc:.4f} | Time: {duration:.2f}s | GPU: {gpu_mb:.2f} MB")

            if val_auc > best_trial_auc:
                best_trial_auc = val_auc
                best_trial_epoch = epoch + 1
                best_trial_state = model.state_dict()

        print(f"🎯 Best epoch in trial {trial + 1}: Epoch {best_trial_epoch} with Val AUC: {best_trial_auc:.4f}")

        if best_trial_auc > best_auc:
            best_auc = best_trial_auc
            best_state = best_trial_state
            best_config = current_config

    print("\n🏆 Best Config Across All Trials:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"🥇 Best Validation AUC: {best_auc:.4f}")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_SAVE_PATH, 'best_pinsage_model.pth')
    torch.save(best_state, model_file)
    print(f"\n📮 Best model saved to: {model_file}")

    plot_training_history(train_loss_list, train_auc_list, val_loss_list, val_auc_list, PLOT_SAVE_PATH)

if __name__ == "__main__":
    train()