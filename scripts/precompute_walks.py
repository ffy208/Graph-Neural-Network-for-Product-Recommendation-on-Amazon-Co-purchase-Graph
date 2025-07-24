import os
import torch
import dgl
import numpy as np
from tqdm import tqdm

def precompute_neighbors(graph, num_walks=2, walk_length=2, device='cpu'):
    """
    Precompute neighbors and associated sampling probabilities for each node
    using random walks.

    Stores full neighbor sets with sampling weights, to allow for subsampling
    during training based on importance.

    Returns:
        all_neighbors: dict[node_id] = {
            'neighbors': List[int],
            'probs': List[float]
        }
    """
    graph = graph.to(device)
    all_neighbors = {}

    print(f"🚶 Starting precomputation: {graph.num_nodes()} nodes")
    for node_id in tqdm(range(graph.num_nodes())):
        # Run multiple short random walks from this node
        start_nodes = torch.full((num_walks,), node_id, dtype=torch.long, device=device)
        walks, _ = dgl.sampling.random_walk(graph, start_nodes, length=walk_length)

        # Flatten all walk steps (except the root)
        walk_nodes = walks[:, 1:].reshape(-1)
        walk_nodes = walk_nodes[walk_nodes != -1]

        if len(walk_nodes) == 0:
            all_neighbors[node_id] = {'neighbors': [], 'probs': []}
            continue

        # Count frequency and convert to probabilities
        unique_nodes, counts = torch.unique(walk_nodes, return_counts=True)
        probs = counts.float() / counts.sum()

        all_neighbors[node_id] = {
            'neighbors': unique_nodes.cpu().tolist(),
            'probs': probs.cpu().tolist()
        }

    return all_neighbors


def save_neighbors(neighbors_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, neighbors_dict, allow_pickle=True)
    print(f"✅ Saved precomputed neighbors (with probs) to {save_path}")
