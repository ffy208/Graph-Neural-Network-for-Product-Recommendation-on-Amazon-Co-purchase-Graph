import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def move_to_device(tensor, device):
    """Helper function to move tensors to a specified device."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor


class PairDataset(Dataset):
    """
    PyTorch Dataset for edge pairs and labels.
    Each item is a tuple ((u, v), label), where (u, v) is an edge.
    """
    def __init__(self, edge_list, labels):
        self.edge_list = edge_list
        self.labels = labels

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, idx):
        return self.edge_list[idx], self.labels[idx]


class SimpleNeighborSampler:
    """
    Sampler for multi-layer neighbor sampling using uniform sampling.

    This sampler collects a neighborhood around the input seed nodes by
    performing fanout sampling for a fixed number of GNN layers (num_layers).

    Attributes:
        graph (DGLGraph): The training graph to sample from.
        fanout (int): Number of neighbors to sample per layer.
        num_layers (int): Number of layers (hops) for neighbor sampling.

    Methods:
        sample_blocks(seeds):
            For each layer, samples a set of neighbors from current seeds,
            and converts them to a message flow block.
    """
    def __init__(self, graph, fanout=10, num_layers=2, device='cpu'):
        self.graph = graph.to(device)
        self.fanout = fanout
        self.num_layers = num_layers
        self.device = device

    def sample_blocks(self, seeds):
        # print(f"\n[Sampler] Sampling blocks for seeds: {seeds.tolist()}")
        blocks = []
        seeds = seeds.to(self.device)

        # ⚠️ Check for out-of-bounds seed node IDs
        if seeds.max().item() >= self.graph.num_nodes():
            raise ValueError(
                f"[Sampler] Seed node ID {seeds.max().item()} exceeds graph node count {self.graph.num_nodes()}"
            )

        for layer in range(self.num_layers):
            # print(f"[Sampler] Layer {self.num_layers - layer - 1} sampling from seeds: {seeds.tolist()}")
            frontier = dgl.sampling.sample_neighbors(self.graph, seeds, self.fanout)
            # print(f"[Sampler] Frontier edges: {frontier.num_edges()}")
            block = dgl.to_block(frontier, seeds)
            # print(f"[Sampler] Block - src nodes: {block.num_src_nodes()}, dst nodes: {block.num_dst_nodes()}")
            block = block.to(self.device)
            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]
        return blocks


class SimpleCollator:
    """
    Collator to batch edge samples and attach neighbor blocks.

    This class creates training/validation batches by processing (u,v) edges
    and sampling neighbor blocks. It also attaches features to each block.

    Attributes:
        sampler (SimpleNeighborSampler): Neighbor sampler instance.
        features (Tensor): Full node feature tensor.

    Methods:
        collate(batch):
            Processes a batch of (u, v, label) tuples, samples neighbors,
            and attaches features to blocks.
    """
    def __init__(self, sampler, features):
        self.sampler = sampler
        self.features = features.to(sampler.device)

    def collate(self, batch):
        edges, labels = zip(*batch)
        heads = torch.tensor([u for u, v in edges], device=self.sampler.device)
        tails = torch.tensor([v for u, v in edges], device=self.sampler.device)
        seeds = torch.cat([heads, tails]).unique()

        # print(f"\n[Collator] Processing batch with {len(edges)} edges")
        # print(f"[Collator] Heads: {heads.tolist()} | Tails: {tails.tolist()}")
        # print(f"[Collator] Unique seeds for sampling: {seeds.tolist()}")

        # ⚠️ Check that seeds have valid feature indices
        if seeds.max().item() >= self.features.shape[0]:
            raise ValueError(
                f"[Collator] Max seed ID {seeds.max().item()} exceeds feature matrix size {self.features.shape[0]}"
            )

        blocks = self.sampler.sample_blocks(seeds)

        for i, block in enumerate(blocks):
            src_nids = block.srcdata[dgl.NID]
            dst_nids = block.dstdata[dgl.NID]
            if src_nids.max() >= self.features.shape[0] or dst_nids.max() >= self.features.shape[0]:
                raise ValueError(
                    f"[Error] Block contains node ID >= features.shape[0]: max src={src_nids.max().item()}, "
                    f"max dst={dst_nids.max().item()}, feature dim={self.features.shape[0]}"
                )
            # print(f"[Collator] Attaching features to block {i}...")
            block.srcdata['feat'] = self.features[src_nids]
            block.dstdata['feat'] = self.features[dst_nids]

        return heads, tails, torch.tensor(labels, device=self.sampler.device), blocks


def get_dataloaders(train_edges, train_labels, val_edges, val_labels, features, train_graph, val_graph, batch_size=1024, device='cpu'):
    """
    Constructs PyTorch DataLoaders for training and validation using neighbor sampling.

    Args:
        train_edges (List[Tuple[int, int]]): List of training edge tuples.
        train_labels (List[int]): Labels for training edges.
        val_edges (List[Tuple[int, int]]): List of validation edge tuples.
        val_labels (List[int]): Labels for validation edges.
        features (Tensor): Full feature matrix for nodes.
        train_graph (DGLGraph): Graph containing only training nodes/edges.
        val_graph (DGLGraph): Graph containing only validation nodes/edges.
        batch_size (int): Mini-batch size.
        device (str): Device to run sampling and tensor ops ("cpu" or "cuda").

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation.
    """
    train_dataset = PairDataset(train_edges, train_labels)
    val_dataset = PairDataset(val_edges, val_labels)

    train_sampler = SimpleNeighborSampler(train_graph, device=device)
    val_sampler = SimpleNeighborSampler(val_graph, device=device)

    train_collator = SimpleCollator(train_sampler, features)
    val_collator = SimpleCollator(val_sampler, features)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator.collate,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator.collate,
        drop_last=False,
    )

    return train_loader, val_loader

