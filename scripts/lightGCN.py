import os
import time
import torch
import scipy.sparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import json


# —— DOK Patch ——
def _dok_update(self, other):
    for (i, j), v in other.items():
        self[i, j] = v


scipy.sparse.dok_matrix._update = _dok_update

# —— Logging & TQDM Force ——
import logging, sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="[%(levelname)s] %(message)s"
)
os.environ.setdefault("FORCE_TTY", "1")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer
from tqdm import tqdm, trange
import random


def log(message, log_file=None):
    """Log message to console and optionally to file"""
    print(message)
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")


def compute_roc_auc_ap_loss(model, test_data, device):
    """Compute ROC-AUC, Average Precision, and Loss scores following LightGCN paper design - OPTIMIZED VERSION"""
    model.eval()
    with torch.no_grad():
        # Get user and item embeddings
        user_embeddings = model.user_embedding.weight
        item_embeddings = model.item_embedding.weight

        # Get test interactions (positive samples only)
        test_interactions = test_data.dataset.inter_feat
        user_ids = test_interactions["user_id"].numpy()
        item_ids = test_interactions["item_id"].numpy()

        print(f"      validation positive samples: {len(user_ids)}")

        # OPTIMIZATION 1: Reduce computation by sampling only a subset
        if len(user_ids) > 10000:  # If too many samples, take a subset
            sample_indices = np.random.choice(len(user_ids), 10000, replace=False)
            user_ids = user_ids[sample_indices]
            item_ids = item_ids[sample_indices]
            print(f"      sampled positive samples: {len(user_ids)} (for speed)")

        # Get embeddings for positive samples
        user_emb = user_embeddings[user_ids].cpu().numpy()
        pos_item_emb = item_embeddings[item_ids].cpu().numpy()

        # Compute scores for positive samples
        pos_scores = np.sum(user_emb * pos_item_emb, axis=1)
        pos_probs = torch.sigmoid(torch.tensor(pos_scores)).numpy()

        # OPTIMIZATION 2: Use vectorized negative sampling
        num_items = item_embeddings.shape[0]
        num_pos = len(user_ids)

        # OPTIMIZATION 3: Reduce negative samples per user and use vectorized operations
        num_neg_per_user = 1  # equal to training stage
        total_neg_samples = num_pos * num_neg_per_user

        # Generate random negative items for all users at once
        neg_item_ids = np.random.randint(0, num_items, size=total_neg_samples)
        neg_item_emb = item_embeddings[neg_item_ids].cpu().numpy()

        # Reshape for vectorized computation
        neg_item_emb = neg_item_emb.reshape(num_pos, num_neg_per_user, -1)
        user_emb_expanded = user_emb[
            :, np.newaxis, :
        ]  # Shape: (num_pos, 1, embedding_dim)

        # Vectorized score computation
        neg_scores = np.sum(
            user_emb_expanded * neg_item_emb, axis=2
        )  # Shape: (num_pos, num_neg_per_user)
        neg_probs = torch.sigmoid(torch.tensor(neg_scores)).numpy()

        # Flatten negative probabilities
        neg_probs_flat = neg_probs.flatten()

        # Combine all predictions
        all_probs = np.concatenate([pos_probs, neg_probs_flat])
        all_labels = np.concatenate([np.ones(num_pos), np.zeros(len(neg_probs_flat))])

        # Compute metrics (AUC and AP are the main evaluation metrics for LightGCN)
        try:
            roc = roc_auc_score(all_labels, all_probs)
            print(f"      valid roc completed!")
            ap = average_precision_score(all_labels, all_probs)
            print(f"      valid ap completed!")
        except Exception as e:
            print(f"Error computing validation AUC: {e}")
            roc = float("nan")
            ap = float("nan")

        # Compute Loss using manual BCE calculation (for monitoring purposes)
        # Even though it's higher than train loss, it's useful for monitoring
        all_scores = np.concatenate([pos_scores, neg_scores.flatten()])
        all_probs = torch.sigmoid(torch.tensor(all_scores, dtype=torch.float))
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.float)
        loss = torch.nn.functional.binary_cross_entropy(all_probs, all_labels_tensor)

        print(f"      valid loss completed!")

        return roc, ap, loss.item()


def compute_topk_metrics(model, test_data, device, k=10):
    """Compute Hit@k and MRR@k metrics"""
    model.eval()
    with torch.no_grad():
        # Get user and item embeddings
        user_embeddings = model.user_embedding.weight
        item_embeddings = model.item_embedding.weight

        # Get test interactions
        test_interactions = test_data.dataset.inter_feat
        user_ids = test_interactions["user_id"].numpy()
        item_ids = test_interactions["item_id"].numpy()

        # Convert embeddings to CPU numpy
        z_cpu = user_embeddings.cpu().numpy()

        # Create positive item dictionary for each user
        pos_dict = {}
        for user_id, item_id in zip(user_ids, item_ids):
            if user_id not in pos_dict:
                pos_dict[user_id] = set()
            pos_dict[user_id].add(item_id)

        hit_list = []
        mrr_list = []

        for user_id, positives in pos_dict.items():
            if user_id >= len(z_cpu):
                continue

            # Compute scores for all items
            scores = (z_cpu[user_id] * z_cpu).sum(axis=1)

            # Get top-k items
            topk_idx = np.argsort(scores)[-k:][::-1]

            # Check if any positive items are in top-k
            hit = any(idx in positives for idx in topk_idx)
            hit_list.append(hit)

            # Compute MRR
            best_rank = float("inf")
            for pos_item in positives:
                if pos_item < len(scores):
                    rank = np.where(np.argsort(scores)[::-1] == pos_item)[0][0] + 1
                    best_rank = min(best_rank, rank)

            if best_rank != float("inf"):
                mrr_list.append(1.0 / best_rank)
            else:
                mrr_list.append(0.0)

        hit_k = np.mean(hit_list) if hit_list else 0.0
        mrr_k = np.mean(mrr_list) if mrr_list else 0.0

        return hit_k, mrr_k


def sample_params(param_space):
    params = {}
    for k, v in param_space.items():
        params[k] = random.choice(v)
    return params


# param_space = {
#     'learning_rate': [0.01, 0.005, 0.001],
#     'epochs': [30],
#     'train_batch_size': [2048, 4096],
#     'embedding_size': [64, 128],
#     'n_layers': [2, 3],
#     'loss_type': ['BCE'],
#     'reg_weight': [0, 1e-5, 1e-4, 1e-3],
# }

# test
param_space = {
    "learning_rate": [0.001],
    "epochs": [30],
    "train_batch_size": [1024],
    "embedding_size": [128],
    "n_layers": [2],
    "loss_type": ["BCE"],
    "reg_weight": [1e-3],
}


def main(config_dict, log_file):
    # 1. Configuration
    config = Config(model="LightGCN", config_dict=config_dict)

    # 2. Load data
    print("\U0001f4e5 Loading dataset...")
    log("Starting LightGCN training experiment", log_file)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Check data split information
    print(f"📊 Data split info:")
    print(f"   Train samples: {len(train_data.dataset.inter_feat)}")
    print(f"   Valid samples: {len(valid_data.dataset.inter_feat)}")
    print(f"   Test samples: {len(test_data.dataset.inter_feat)}")
    print(f"   Data split method: {config['data_split']}")

    # 3. Model initialization
    print("\U0001f527 Initializing model...")
    model = LightGCN(config, train_data.dataset).to(config["device"])

    # 4. Manual training-evaluation loop
    print("\U0001f680 Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_loss_list = []
    val_loss_list = []
    val_auc_list = []
    val_ap_list = []
    train_auc_list = []
    train_ap_list = []

    # Early stopping mechanism
    best_val_ap = 0
    patience = 3
    patience_counter = 0
    best_model_state = None

    for epoch in trange(1, config["epochs"] + 1, desc="Epochs", position=1):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config["device"])
        model.train()
        epoch_loss = 0.0
        steps = 0
        train_preds = []
        train_labels = []
        for batch_data in tqdm(
            train_data,
            desc=f"Epoch {epoch}/{config['epochs']}",
            leave=False,
            position=0,
        ):
            batch_data = batch_data.to(config["device"])
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                loss = model.calculate_loss(batch_data)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            steps += 1
            # Collect training set predictions and labels - manually handle positive and negative samples
            user_ids = batch_data.interaction["user_id"].cpu().numpy()
            item_ids = batch_data.interaction["item_id"].cpu().numpy()

            # Check if there is a label field, if not, manually handle positive and negative samples
            if "label" in batch_data.interaction:
                labels = batch_data.interaction["label"].cpu().numpy()
            else:
                # Check if there is a neg_item_id field (generated by negative sampling)
                if "neg_item_id" in batch_data.interaction:
                    # Step 1: Extract positive sample data
                    pos_user_ids = batch_data.interaction["user_id"].cpu().numpy()
                    pos_item_ids = batch_data.interaction["item_id"].cpu().numpy()
                    neg_item_ids = batch_data.interaction["neg_item_id"].cpu().numpy()

                    # Step 2: Merge positive and negative samples
                    all_user_ids = np.concatenate([pos_user_ids, pos_user_ids])
                    all_item_ids = np.concatenate([pos_item_ids, neg_item_ids])

                    # Step 3: Assign labels (1=positive sample, 0=negative sample)
                    pos_labels = np.ones_like(pos_user_ids)  # 正样本标签为1
                    neg_labels = np.zeros_like(pos_user_ids)  # 负样本标签为0
                    labels = np.concatenate([pos_labels, neg_labels])

                    # Step 4: Update variables used for calculation
                    user_ids = all_user_ids
                    item_ids = all_item_ids

                    if steps == 1:
                        print(
                            f"manual processing completed: positive samples {len(pos_user_ids)}, negative samples {len(neg_item_ids)}"
                        )
                        print(
                            f"label distribution: positive={np.sum(labels == 1)}, negative={np.sum(labels == 0)}"
                        )

                else:
                    # If there is no negative sampling, give a warning
                    print(
                        f"Warning: No 'label' or 'neg_item_id' field found in batch_data.interaction"
                    )
                    print(f"Available fields: {list(batch_data.interaction.keys())}")
                    # Temporarily use all 1s as labels, but this is not the correct approach
                    labels = np.ones_like(user_ids)
            user_embeddings = model.user_embedding.weight
            item_embeddings = model.item_embedding.weight
            user_emb = user_embeddings[user_ids].detach().cpu().numpy()
            item_emb = item_embeddings[item_ids].detach().cpu().numpy()
            scores = np.sum(user_emb * item_emb, axis=1)
            y_prob = torch.sigmoid(torch.tensor(scores)).numpy()
            train_preds.append(y_prob)
            train_labels.append(labels)
        avg_loss = epoch_loss / steps
        train_loss_list.append(avg_loss)
        print(f"✅ train loss calculated: {avg_loss:.4f}")

        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        try:
            train_auc = roc_auc_score(train_labels, train_preds)
            train_ap = average_precision_score(train_labels, train_preds)
            print(f"✅ train AUC and AP calculated: {train_auc:.4f}, {train_ap:.4f}")
        except Exception:
            train_auc = float("nan")
            train_ap = float("nan")
        train_auc_list.append(train_auc)
        train_ap_list.append(train_ap)

        # Validation
        print(f"\n🔍 evaluating valid set...")
        model.eval()
        with torch.no_grad():
            # Calculate custom ROC-AUC, AP and Loss (using the same batch of positive and negative samples)
            roc, ap, loss = compute_roc_auc_ap_loss(model, valid_data, config["device"])
            print(
                f"✅ valid set calculated: ROC-AUC={roc:.4f}, AP={ap:.4f}, Loss={loss:.4f}"
            )
        val_loss_list.append(loss)
        val_auc_list.append(roc)
        val_ap_list.append(ap)

        duration = time.time() - start_time
        gpu_mb = (
            torch.cuda.max_memory_allocated(config["device"]) / (1024**2)
            if torch.cuda.is_available()
            else 0
        )

        # Print and record logs for each epoch
        log_msg = (
            f"Epoch {epoch:>2}/{config['epochs']}  "
            f"Train Loss: {avg_loss:.4f}  "
            f"Train AUC: {train_auc:.4f}  AP: {train_ap:.4f}  "
            f"Val Loss: {loss:.4f}  "
            f"Val ROC-AUC: {roc:.4f}  Val AP: {ap:.4f}  "
            f"Time: {duration:.2f}s  GPU: {gpu_mb:.2f} MB"
        )
        print(log_msg)
        print("Batch label counts:", np.unique(labels, return_counts=True))
        log(log_msg, log_file)

        # Early stopping check
        if ap > best_val_ap:
            best_val_ap = ap
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"🎯 new best valid AP: {ap:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  valid AP not improved, patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                f"🛑 early stopping triggered! valid AP not improved for {patience} epochs"
            )
            # Restore best model
            model.load_state_dict(best_model_state)
            break

    # 5. Final results
    # Reuse the training and validation set results from the last epoch as final results
    final_train_auc = train_auc_list[-1] if train_auc_list else 0
    final_train_ap = train_ap_list[-1] if train_ap_list else 0
    final_train_loss = train_loss_list[-1] if train_loss_list else 0
    print(
        f"\U0001f4ca final train metrics - AUC: {final_train_auc:.4f} | AP: {final_train_ap:.4f} | loss: {final_train_loss:.4f}"
    )

    final_roc = val_auc_list[-1] if val_auc_list else 0
    final_ap = val_ap_list[-1] if val_ap_list else 0
    final_loss = val_loss_list[-1] if val_loss_list else 0
    print(
        f"\U0001f4ca final validation metrics - ROC-AUC: {final_roc:.4f} | AP: {final_ap:.4f} | loss: {final_loss:.4f}"
    )

    # 6. Plot training history
    plot_training_history(
        train_loss_list,
        train_auc_list,
        val_loss_list,
        val_auc_list,
        save_path="./training_history/",
    )

    # Calculate RecBole ranking metrics on validation set
    print("📊 calculating RecBole ranking metrics on validation set...")
    valid_result = Trainer.evaluate(
        Trainer(config, model), valid_data, load_best_model=False
    )
    recall10 = valid_result.get("Recall@10", 0)
    ndcg10 = valid_result.get("NDCG@10", 0)
    mrr = valid_result.get("MRR", 0)

    # Record final metrics
    log(
        f"== Final Train - AUC: {final_train_auc:.4f} | AP: {final_train_ap:.4f} | loss: {final_train_loss:.4f} ==",
        log_file,
    )
    log(
        f"== Final Validation - AUC: {final_roc:.4f} | AP: {final_ap:.4f} | loss: {final_loss:.4f} ==",
        log_file,
    )
    log(f"== RocBole Ranking metrics: {valid_result} ==", log_file)

    os.makedirs("./saved_models/", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save best model (based on AP)
    best_model_path = f"./saved_models/lightgcn_best_ap_{timestamp}.pth"
    torch.save(best_model_state, best_model_path)
    print(f"💾 Saved best model (AP: {best_val_ap:.4f}) to: {best_model_path}")

    # Save final model
    current_model_state = model.state_dict().copy()
    final_model_path = f"./saved_models/lightgcn_final_{timestamp}.pth"
    torch.save(current_model_state, final_model_path)
    print(f"💾 Saved final model to: {final_model_path}")

    # Save training configuration and results
    results = {
        "config": config_dict,
        "final_train_auc": final_train_auc,
        "final_train_ap": final_train_ap,
        "final_train_loss": final_train_loss,
        "final_val_auc": final_roc,
        "final_val_ap": final_ap,
        "final_val_loss": final_loss,
        "valid_result": valid_result,
        "best_val_ap": best_val_ap,
        "training_history": {
            "train_loss_list": train_loss_list,
            "train_auc_list": train_auc_list,
            "val_loss_list": val_loss_list,
            "val_auc_list": val_auc_list,
        },
    }

    results_path = f"./saved_models/lightgcn_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved training results to: {results_path}")

    return (
        final_train_auc,
        final_train_ap,
        final_train_loss,
        final_roc,
        final_ap,
        final_loss,
        valid_result,
    )


def random_search_main():
    log_file = "./lightgcn_random_search.log"
    trials = 10
    for trial in range(1, trials + 1):
        # Randomly sample parameters
        params = sample_params(param_space)
        # Build config_dict
        config_dict = {
            "model": "LightGCN",
            "dataset": "interactions",
            "data_path": "./dataset_recbole/",
            "data_split": "leave_one_out",  # Leave-one-out split for each user
            "epochs": params["epochs"],
            "learning_rate": params["learning_rate"],
            "train_batch_size": params["train_batch_size"],
            "embedding_size": params["embedding_size"],
            "n_layers": params["n_layers"],
            "use_gpu": True,
            "gpu_id": 0,
            "loss_type": params["loss_type"],
            # Negative sampling related configuration
            "train_negative_sampling": True,
            "neg_sampling": True,
            "neg_sample_num": 1,
            "neg_sample_strategy": "random",
            "neg_sample_distribution": "uniform",
            "reg_weight": params["reg_weight"],
            "metrics": ["Recall", "NDCG"],
            "topk": [10],
            "show_progress": True,
            "progress_bar": True,
            "save_log": True,
            "logging_level": "INFO",
        }
        log(f"\n===== Random Search Trial {trial} =====", log_file)
        log(f"Params: {config_dict}", log_file)
        train_auc, train_ap, train_loss, roc, ap, val_loss, valid_result = main(
            config_dict, log_file
        )
        log(
            f"time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Train AUC: {train_auc:.4f} | Train AP: {train_ap:.4f} | Train Loss: {train_loss:.4f} | Val AUC: {roc:.4f} | Val AP: {ap:.4f} | Val Loss: {val_loss:.4f} | Valid Result: {valid_result}",
            log_file,
        )


def plot_training_history(
    train_loss_list, train_auc_list, val_loss_list, val_auc_list, save_path
):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, val_loss_list, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_auc_list, label="Train AUC")
    plt.plot(epochs, val_auc_list, label="Val AUC")
    plt.title("Training and Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(train_loss_list, bins=20, alpha=0.6, label="Train Loss")
    plt.hist(val_loss_list, bins=20, alpha=0.6, label="Val Loss")
    plt.title("Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(train_auc_list, bins=20, alpha=0.6, label="Train AUC")
    plt.hist(val_auc_list, bins=20, alpha=0.6, label="Val AUC")
    plt.title("AUC Distribution")
    plt.xlabel("AUC")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_file = os.path.join(
        save_path, "training_history-" + time.strftime("%Y%m%d_%H%M%S") + ".png"
    )
    plt.savefig(plot_file)
    print(f"\U0001f4c8 Saved training history plot to: {plot_file}")


if __name__ == "__main__":
    random_search_main()
