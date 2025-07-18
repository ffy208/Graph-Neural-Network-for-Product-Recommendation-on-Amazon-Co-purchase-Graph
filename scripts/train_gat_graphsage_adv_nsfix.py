#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GAT Training Script
#  ----------------------------------------------------------------------
# Purpose: Train a Graph Attention Network (GAT) for link prediction on the
# Amazon products similarity graph while retaining the original GraphSage data
# preparation pipeline.
# 
# Key Fixes in this version:
# 1. Corrected negative sampling logic using proper mask-based filtering
# 2. Added robust checkpoint saving for network filesystems
# 3. Enhanced device compatibility for tensors and masks
# 4. Standalone implementation - no external dependencies on other training scripts
# 
# Workflow Overview
# 1. DataProcessor – loads `items_cleaned.csv`, standardises numerical features
#    and builds an undirected PyG `Data` graph via product "similar" relations.
# 2. GATModel – constructs a stack of `GATConv` layers; dimension checks ensure
#    the final output size equals `hidden_dim * heads` for multi-layer setups.
# 3. GATTrainer – handles full-graph training with corrected negative sampling, computes
#    AUC/AP metrics, logs history, manages early stopping and checkpointing.
# 4. CLI + main() – parses extensive hyper-parameters, supports inductive (node
#    split) vs transductive (edge split) evaluation, optional focal/mixed loss
#    functions and learning-rate schedulers, and finally saves plots & JSON
#    summaries.
# ----------------------------------------------------------------------

import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
import gc
from datetime import datetime
from collections import defaultdict
import random
import argparse
from typing import Optional

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.loader import LinkNeighborLoader

# Scikit-learn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ===== Added patch to ensure robust checkpoint saving =====
# On some network filesystems the default zip-based serialization can fail with
# "PytorchStreamWriter failed writing file data.pkl".  Falling back to the legacy
# pickle+tar format avoids this issue and keeps checkpoints small enough for our
# use-case.
_original_torch_save = torch.save  # type: ignore[attr-defined]

def _legacy_torch_save(obj, f, **kwargs):  # type: ignore[override]
    """Wrapper around torch.save forcing legacy (non-zip) serialization."""
    kwargs.setdefault("_use_new_zipfile_serialization", False)
    return _original_torch_save(obj, f, **kwargs)

# Monkey-patch globally so *any* torch.save call will use the safer behaviour
torch.save = _legacy_torch_save  # type: ignore[assignment]

# ---------------------------------------
# Custom Loss Functions
# ---------------------------------------

class FocalLoss(torch.nn.Module):
    """Binary Focal Loss implementation compatible with logits."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute the standard BCE with logits to obtain pt
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # p_t is prob corresponding to the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        return loss.mean()

# Mixed BCE + Focal loss
class MixedLoss(torch.nn.Module):
    """Blend BCEWithLogitsLoss and FocalLoss: total_loss = alpha*BCE + (1-alpha)*Focal"""

    def __init__(self, alpha: float = 0.5, focal_gamma: float = 1.5, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else torch.nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.bce(logits, targets) + (1 - self.alpha) * self.focal(logits, targets)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up logging
def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"gat_training_{timestamp}.log"
    
    # Redirect stdout and stderr to log file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
    
    return log_file

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class DataProcessor:
    """Data processor based on GraphSage.ipynb pipeline"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.asin2idx = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        
    def load_and_process(self):
        """Load and process data following GraphSage.ipynb pipeline"""
        log_message(f"Loading data from {self.csv_path}...")
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()
        log_message(f"Loaded {len(self.df)} items")
        
        # Create ASIN to index mapping
        self.asin2idx = {asin: i for i, asin in enumerate(self.df['ASIN'])}
        log_message(f"Created mapping for {len(self.asin2idx)} unique ASINs")
        
        # Process features
        self._process_features()
        
        # Build graph
        edge_index = self._build_graph()
        
        # Create node features
        x = self._create_node_features()
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Clear memory
        del self.df
        gc.collect()
        
        return data
    
    def _process_features(self):
        """Process features following GraphSage.ipynb approach"""
        log_message("Processing features...")
        
        # Define feature columns (same as GraphSage.ipynb)
        self.feature_cols = [
            'salesrank_log', 'reviews_total_log', 'reviews_downloaded_log',
            'reviews_avg_ratings', 'reviews_avg_votes', 'reviews_avg_helpful',
            'category_count'
        ]
        
        # Fill missing values
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0.0)
        
        # Log transform for skewed features
        for col in ['salesrank_log', 'reviews_total_log', 'reviews_downloaded_log']:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col].clip(lower=0))
        
        # Standardize features
        features = self.df[self.feature_cols].values
        features_scaled = self.scaler.fit_transform(features)
        self.df[self.feature_cols] = features_scaled
        
        log_message(f"Processed {len(self.feature_cols)} features")
    
    def _build_graph(self):
        """Build graph from similarity relationships"""
        log_message("Building graph from similarity relationships...")
        
        src, dst = [], []
        
        # Process similarities
        for idx, row in self.df.iterrows():
            sims = str(row['similar']).split(',') if pd.notna(row['similar']) else []
            for s in sims:
                s = s.strip()
                if s and s in self.asin2idx:
                    j = self.asin2idx[s]
                    if idx != j:  # Remove self-loops
                        src.append(idx)
                        dst.append(j)
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = to_undirected(edge_index)  # Make undirected
        
        log_message(f"Graph built: {len(self.asin2idx)} nodes, {edge_index.size(1)//2} edges")
        return edge_index
    
    def _create_node_features(self):
        """Create node feature matrix"""
        features = self.df[self.feature_cols].values
        return torch.tensor(features, dtype=torch.float)

class GATModel(torch.nn.Module):
    """GAT model based on GraphSage architecture but with GATConv layers"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, num_layers=2, dropout=0.2):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = torch.nn.ModuleList()
        
        # First layer: in_dim -> hidden_dim
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, 
                                 dropout=dropout, concat=True))
        
        # Middle layers: hidden_dim * heads -> hidden_dim
        for _ in range(num_layers - 2):
            in_channels = hidden_dim * heads
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads,
                                     dropout=dropout, concat=True))
        
        # Final layer: hidden_dim * heads -> out_dim
        if num_layers > 1:
            in_channels = hidden_dim * heads
            # Ensure out_dim matches the expected dimension
            expected_out_dim = hidden_dim * heads
            if out_dim != expected_out_dim:
                print(f"Warning: Model out_dim ({out_dim}) != expected ({expected_out_dim}), using {expected_out_dim}")
                out_dim = expected_out_dim
            
            self.convs.append(GATConv(in_channels, out_dim, heads=1,
                                     dropout=dropout, concat=False))
        else:
            # Single layer: in_dim -> out_dim
            # Ensure out_dim doesn't exceed hidden_dim for single layer
            if out_dim > hidden_dim:
                print(f"Warning: Single layer out_dim ({out_dim}) > hidden_dim ({hidden_dim}), using {hidden_dim}")
                out_dim = hidden_dim
            
            self.convs.append(GATConv(in_dim, out_dim, heads=1,
                                     dropout=dropout, concat=False))
    
    def forward(self, x, edge_index):
        """Forward pass through GAT layers"""
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.elu(x)
        
        return x

def dot_product_decode(z, edge_index):
    """Dot product decoder for link prediction"""
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

class GATTrainer:
    """GAT trainer with corrected negative-sampling logic and logging"""
    
    def __init__(self, model, device, lr: float = 0.01, weight_decay: float = 5e-4, *, criterion: Optional[torch.nn.Module] = None):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Use provided criterion or default to BCEWithLogitsLoss
        self.criterion = criterion if criterion is not None else torch.nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        
    def train_epoch(self, train_data, neg_ratio: float = 1.0):
        """Train for one epoch with corrected negative sampling.

        Negative edges are now restricted to nodes that *really* belong to the
        training split via `train_data.train_mask` instead of relying on a
        fragile index-based check.
        """
        self.model.train()

        # Generate *candidate* negative edges over the **whole** graph
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=int(train_data.edge_index.size(1) * neg_ratio),
        )

        # Keep only negatives whose both endpoints are in the training split
        train_mask = train_data.train_mask.to(neg_edge_index.device)
        valid_neg = train_mask[neg_edge_index[0]] & train_mask[neg_edge_index[1]]
        neg_edge_index = neg_edge_index[:, valid_neg]

        # Labels: 1 for positive, 0 for negative
        edge_label_index = torch.cat([train_data.edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(train_data.edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device),
            ],
            dim=0,
        )

        self.optimizer.zero_grad()
        z = self.model(train_data.x.to(self.device), train_data.edge_index.to(self.device))
        out = dot_product_decode(z, edge_label_index.to(self.device))
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, data, neg_ratio: float = 1.0, split_name: str = "val"):
        """Evaluate with split-consistent negative sampling."""
        self.model.eval()
        with torch.no_grad():
            # Determine node mask for the requested split; default to all nodes
            mask_attr = f"{split_name}_mask"
            if hasattr(data, mask_attr):
                split_mask = getattr(data, mask_attr)
                if split_mask is None or split_mask.sum() == 0:
                    split_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)
            else:
                split_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)

            # Generate candidate negatives over the full graph
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=int(data.edge_index.size(1) * neg_ratio),
            )
            # Keep only negatives fully inside the split
            split_mask = split_mask.to(neg_edge_index.device)
            valid_neg = split_mask[neg_edge_index[0]] & split_mask[neg_edge_index[1]]
            neg_edge_index = neg_edge_index[:, valid_neg]

            edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(data.edge_index.size(1), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device),
                ],
                dim=0,
            )

            z = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            out = dot_product_decode(z, edge_label_index.to(self.device))

            loss = self.criterion(out, edge_label)
            auc = roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())
            ap = average_precision_score(edge_label.cpu().numpy(), out.cpu().numpy())
            return loss.item(), auc, ap
    
    def plot_training_history(self, save_path):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC plot
        axes[0, 1].plot(self.train_aucs, label='Train AUC')
        axes[0, 1].plot(self.val_aucs, label='Val AUC')
        axes[0, 1].set_title('Training and Validation AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss distribution
        axes[1, 0].hist(self.train_losses, bins=20, alpha=0.7, label='Train Loss')
        axes[1, 0].hist(self.val_losses, bins=20, alpha=0.7, label='Val Loss')
        axes[1, 0].set_title('Loss Distribution')
        axes[1, 0].set_xlabel('Loss')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # AUC distribution
        axes[1, 1].hist(self.train_aucs, bins=20, alpha=0.7, label='Train AUC')
        axes[1, 1].hist(self.val_aucs, bins=20, alpha=0.7, label='Val AUC')
        axes[1, 1].set_title('AUC Distribution')
        axes[1, 1].set_xlabel('AUC')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_message(f"Training history plot saved to {save_path}")

def validate_args(args):
    """Validate and auto-fix model arguments"""
    print(f"Original arguments: hidden_dim={args.hidden_dim}, out_dim={args.out_dim}, heads={args.heads}, num_layers={args.num_layers}")
    
    if args.num_layers == 1:
        # Single layer: force heads=1 and ensure out_dim is reasonable
        if args.heads != 1:
            print(f"Warning: Single layer detected, auto-fixing heads from {args.heads} to 1")
            args.heads = 1
        
        # For single layer, ensure out_dim doesn't exceed input dimension
        if args.out_dim > args.hidden_dim:
            print(f"Warning: Single layer out_dim ({args.out_dim}) > hidden_dim ({args.hidden_dim}), auto-fixing out_dim to {args.hidden_dim}")
            args.out_dim = args.hidden_dim
    
    elif args.num_layers > 1:
        # Multi-layer: ensure dimensions are compatible
        expected_out_dim = args.hidden_dim * args.heads
        
        if args.out_dim != expected_out_dim:
            print(f"Warning: out_dim ({args.out_dim}) != hidden_dim*heads ({expected_out_dim})")
            print(f"Auto-fixing out_dim from {args.out_dim} to {expected_out_dim}")
            args.out_dim = expected_out_dim
    
    print(f"Fixed arguments: hidden_dim={args.hidden_dim}, out_dim={args.out_dim}, heads={args.heads}, num_layers={args.num_layers}")
    return args

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GAT Training - Standalone Fixed Version')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=16, help='Output dimension')
    parser.add_argument('--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Data parameters
    parser.add_argument('--neg_ratio', type=float, default=1.0, help='Negative sampling ratio (float)')
    parser.add_argument('--inductive', action='store_true', help='Use inductive split (node-based) instead of transductive')

    # Loss parameters
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Positive class weight for BCE/FocalLoss.')
    parser.add_argument('--use_focal', action='store_true', help='Use FocalLoss instead of BCEWithLogitsLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma value for FocalLoss')
    parser.add_argument('--mix_alpha', type=float, default=0.5, help='Alpha for MixedLoss (BCE + Focal)')

    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, choices=['none', 'multistep', 'cosine'], default='none', help='LR scheduler type')
    parser.add_argument('--milestones', type=str, default='80,120', help='Comma separated milestones for MultiStepLR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    
    args = parser.parse_args()
    return validate_args(args)

def main():
    """Main training function"""
    # Parse arguments
    args = get_args()
    
    # Parameters are already validated in get_args()
    
    # Setup
    set_seed(42)
    log_file = setup_logging()
    
    # Memory optimization for large graphs
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    log_message("Starting GAT training - Standalone Fixed Version")
    log_message(f"Log file: {log_file}")
    log_message(f"Arguments: {args}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    
    # Data processing
    data_processor = DataProcessor('data/items_cleaned.csv')
    data = data_processor.load_and_process()
    
    # Data split based on inductive flag
    if args.inductive:
        log_message("Performing inductive split (node-based split)...")
        
        num_nodes = data.num_nodes
        all_nodes = torch.arange(num_nodes)
        
        # Split nodes: 70% train, 15% val, 15% test
        num_train_nodes = int(0.7 * num_nodes)
        num_val_nodes = int(0.15 * num_nodes)
        
        # Randomly shuffle nodes
        node_indices = torch.randperm(num_nodes)
        train_nodes = all_nodes[node_indices[:num_train_nodes]]
        val_nodes = all_nodes[node_indices[num_train_nodes:num_train_nodes + num_val_nodes]]
        test_nodes = all_nodes[node_indices[num_train_nodes + num_val_nodes:]]
        
        log_message(f"Node split - Train: {len(train_nodes)}, Val: {len(val_nodes)}, Test: {len(test_nodes)}")
        
        # Create masks for each split
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True
        
        # Split edges based on node membership
        def split_edges_by_nodes(edge_index, train_nodes, val_nodes, test_nodes):
            """Split edges based on node membership (inductive split)"""
            train_edges = []
            val_edges = []
            test_edges = []
            
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                
                # Check if both nodes are in train set
                if train_mask[src] and train_mask[dst]:
                    train_edges.append(i)
                # Check if both nodes are in val set
                elif val_mask[src] and val_mask[dst]:
                    val_edges.append(i)
                # Check if both nodes are in test set
                elif test_mask[src] and test_mask[dst]:
                    test_edges.append(i)
                # Mixed edges (cross-split) are excluded for inductive learning
            
            return train_edges, val_edges, test_edges
        
        train_edge_indices, val_edge_indices, test_edge_indices = split_edges_by_nodes(
            data.edge_index, train_nodes, val_nodes, test_nodes
        )
        
        # Create split data
        train_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, train_edge_indices],
            num_nodes=data.num_nodes
        )
        
        val_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, val_edge_indices],
            num_nodes=data.num_nodes
        )
        
        test_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, test_edge_indices],
            num_nodes=data.num_nodes
        )
        
        log_message(f"Inductive edge split - Train: {len(train_edge_indices)} edges, "
                    f"Val: {len(val_edge_indices)} edges, "
                    f"Test: {len(test_edge_indices)} edges")
        
        # Store node masks for evaluation
        train_data.train_mask = train_mask
        train_data.val_mask = val_mask
        train_data.test_mask = test_mask
        
    else:
        log_message("Performing transductive split (edge-based split)...")
        
        # Traditional edge-based split
        transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                                  add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data)
        
        log_message(f"Transductive edge split - Train: {train_data.edge_index.size(1)} edges, "
                    f"Val: {val_data.edge_index.size(1)} edges, "
                    f"Test: {test_data.edge_index.size(1)} edges")
        
        # Create dummy masks for transductive learning (all nodes visible)
        num_nodes = data.num_nodes
        train_data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
        train_data.val_mask = torch.ones(num_nodes, dtype=torch.bool)
        train_data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Model setup
    in_dim = data.x.size(1)
    
    model = GATModel(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                    heads=args.heads, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    log_message(f"Model created - Input dim: {in_dim}, Hidden dim: {args.hidden_dim}, "
                f"Output dim: {args.out_dim}, Heads: {args.heads}, Layers: {args.num_layers}")
    
    # -----------------------------
    # Loss function setup
    # -----------------------------
    pos_weight_tensor = torch.tensor(args.pos_weight, device=device) if args.pos_weight != 1.0 else None
    
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight_tensor)
        log_message(f"Using FocalLoss with gamma={args.focal_gamma}, pos_weight={args.pos_weight}")
    elif args.mix_alpha is not None and 0.0 < args.mix_alpha < 1.0:
        criterion = MixedLoss(alpha=args.mix_alpha, focal_gamma=args.focal_gamma, pos_weight=pos_weight_tensor)
        log_message(f"Using MixedLoss with alpha={args.mix_alpha}, focal_gamma={args.focal_gamma}, pos_weight={args.pos_weight}")
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if pos_weight_tensor is not None else torch.nn.BCEWithLogitsLoss()
        log_message(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight}")

    # Training setup (optimizer in trainer)
    trainer = GATTrainer(model, device, lr=args.lr, weight_decay=args.weight_decay, criterion=criterion)

    # -----------------------------
    # Scheduler setup
    # -----------------------------
    scheduler = None
    if args.scheduler == 'multistep':
        milestones = [int(m.strip()) for m in args.milestones.split(',') if m.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer, milestones=milestones, gamma=args.lr_gamma)
        log_message(f"Using MultiStepLR with milestones={milestones}, gamma={args.lr_gamma}")
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        log_message(f"Using CosineAnnealingLR with T_max={args.epochs}, eta_min={args.lr * 0.01}")
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    
    log_message(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch(train_data, neg_ratio=args.neg_ratio)
        
        # Evaluate
        val_loss, val_auc, val_ap = trainer.evaluate(val_data, neg_ratio=args.neg_ratio, split_name="val")
        train_loss_eval, train_auc, train_ap = trainer.evaluate(train_data, neg_ratio=args.neg_ratio, split_name="train")
        
        # Store history
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.train_aucs.append(train_auc)
        trainer.val_aucs.append(val_auc)
        
        epoch_time = time.time() - start_time
        
        # Log progress
        log_message(f"Epoch {epoch+1:3d}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save best model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            model_suffix = f"h{args.hidden_dim}_hd{args.heads}_neg{args.neg_ratio}"
            best_model_path = model_dir / f"gat_best_{model_suffix}.pt"
            torch.save(model.state_dict(), best_model_path)
            log_message(f"New best model saved with Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_message(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Final evaluation
    log_message("Training completed. Evaluating on test set...")
    
    # Load best model with same suffix
    model_suffix = f"h{args.hidden_dim}_hd{args.heads}_neg{args.neg_ratio}"
    best_model_path = f"models/gat_best_{model_suffix}.pt"
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_auc, test_ap = trainer.evaluate(test_data, neg_ratio=args.neg_ratio, split_name="test")
    
    log_message(f"Final Test Results - Loss: {test_loss:.4f}, "
                f"AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
    
    # Plot training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/gat_training_history_{timestamp}.png"
    trainer.plot_training_history(plot_path)
    
    # Save training results
    results = {
        'best_val_auc': best_val_auc,
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'epochs_trained': len(trainer.train_losses),
        'model_config': {
            'in_dim': in_dim,
            'hidden_dim': args.hidden_dim,
            'out_dim': args.out_dim,
            'heads': args.heads,
            'num_layers': args.num_layers
        },
        'training_args': vars(args)
    }
    
    results_path = f"logs/gat_training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Training results saved to {results_path}")
    log_message("Training completed successfully!")

if __name__ == "__main__":
    main() 
