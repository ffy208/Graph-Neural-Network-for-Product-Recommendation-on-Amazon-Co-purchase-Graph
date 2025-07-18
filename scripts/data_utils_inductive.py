import os
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.model_selection import train_test_split

def load_features(feature_path):
    """
    Load node features from .npy file.

    Args:
        feature_path (str): Path to .npy file.

    Returns:
        Tensor: Node feature matrix.
    """
    # print(f"Loading features from: {feature_path}", flush=True)
    features = np.load(feature_path)
    # print(f"Feature shape: {features.shape}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(features, dtype=torch.float32).to(device)
    return features

def load_graph(edge_path, num_nodes):
    """
    Create graph from CSV edge list (only positive edges).

    Args:
        edge_path (str): Path to CSV file with 'source', 'target', 'label'.
        num_nodes (int): Total number of nodes.

    Returns:
        DGLGraph: Positive-edge graph.
    """
    # print(f"Loading edges from: {edge_path}", flush=True)
    df = pd.read_csv(edge_path)
    # print(f"Total edges read: {len(df)}", flush=True)
    pos_edges = df[df['label'] == 1][['source', 'target']].values
    # print(f"Positive edges count: {len(pos_edges)}", flush=True)
    src, dst = torch.tensor(pos_edges[:, 0]), torch.tensor(pos_edges[:, 1])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    # print(f"Graph loaded with {g.num_nodes()} nodes and {g.num_edges()} edges", flush=True)
    return g

def split_nodes(num_nodes, test_ratio=0.2, seed=42):
    """
    Randomly split nodes into train/test sets.

    Args:
        num_nodes (int): Total number of nodes.
        test_ratio (float): Fraction of nodes to reserve for test.
        seed (int): Random seed.

    Returns:
        train_nids (np.array), test_nids (np.array)
    """
    # print(f"Splitting {num_nodes} nodes into train/test with ratio {1 - test_ratio}/{test_ratio}", flush=True)
    np.random.seed(seed)
    all_nodes = np.arange(num_nodes)
    np.random.shuffle(all_nodes)
    split_idx = int(num_nodes * (1 - test_ratio))
    # print(f"Train nodes: {split_idx}, Test nodes: {num_nodes - split_idx}", flush=True)
    return all_nodes[:split_idx], all_nodes[split_idx:]

def extract_edges(g, node_set):
    """
    Extract edges where both ends are in node_set.

    Args:
        g (DGLGraph): Original graph.
        node_set (array): Set of node IDs to keep.

    Returns:
        List[Tuple[int, int]]: List of edges.
    """
    # print(f"Extracting edges within node set of size: {len(node_set)}", flush=True)
    src, dst = g.edges()
    mask = np.isin(src.numpy(), node_set) & np.isin(dst.numpy(), node_set)
    filtered_edges = list(zip(src[mask].tolist(), dst[mask].tolist()))
    # print(f"Extracted {len(filtered_edges)} edges", flush=True)
    return filtered_edges

def generate_negative_edges(pos_edges, valid_nodes, neg_ratio=1, exclude_set=None):
    """
    Generate negative edges where both nodes are from valid_nodes.

    Args:
        pos_edges (set): Set of true edges.
        valid_nodes (set): Nodes to sample from.
        neg_ratio (int): Number of negative edges per positive edge.
        exclude_set (set): Additional edges to exclude.

    Returns:
        List[Tuple[int, int]]: Negative edges.
    """
    num_samples = int(len(pos_edges) * neg_ratio)
    pos_edges = set(pos_edges)
    exclude_set = exclude_set or set()
    all_invalid = pos_edges.union(exclude_set)

    valid_nodes = np.array(list(valid_nodes))
    neg_edges = set()
    tries = 0
    max_tries = 20
    batch_size = max(2 * num_samples, 10000)  # Large batch to reduce iterations

    while len(neg_edges) < num_samples and tries < max_tries:
        us = np.random.choice(valid_nodes, size=batch_size)
        vs = np.random.choice(valid_nodes, size=batch_size)
        mask = us != vs
        pairs = set(zip(us[mask], vs[mask]))
        new_candidates = pairs - all_invalid - neg_edges

        for edge in new_candidates:
            neg_edges.add(edge)
            if len(neg_edges) >= num_samples:
                break

        tries += 1

    if len(neg_edges) < num_samples:
        print(f"⚠️ Warning: Only generated {len(neg_edges)} negative edges after {tries} tries", flush=True)
    else:
        print(f"Generated {len(neg_edges)} negative edges", flush=True)

    return list(neg_edges)



def build_inductive_split(edge_csv_path, feature_npy_path, test_ratio=0.2, neg_ratio = 1):
    """
    Build train graph + validation edges for strict inductive learning.

    Returns:
        train_graph (DGLGraph), features (Tensor),
        train_edges (list), train_labels (list), train_feat_pairs (Tensor),
        val_edges (list), val_labels (list), val_feat_pairs (Tensor)
    """
    # print("\n=== Building Inductive Split ===", flush=True)
    features = load_features(feature_npy_path)
    full_graph = load_graph(edge_csv_path, features.shape[0])

    train_nids, test_nids = split_nodes(features.shape[0], test_ratio=test_ratio)
    train_nid_set = set(train_nids)
    test_nid_set = set(test_nids)

    train_g = dgl.node_subgraph(full_graph, train_nids).to(features.device)
    # Create global-to-local node ID mapping after subgraph
    global_to_local = {nid.item(): i for i, nid in enumerate(train_nids)}
    train_g.ndata['feat'] = features[train_nids]
    # print(f"Train graph: {train_g.num_nodes()} nodes, {train_g.num_edges()} edges", flush=True)

    train_pos_edges = extract_edges(full_graph, train_nids)
    train_pos_labels = [1] * len(train_pos_edges)

    train_neg_edges = generate_negative_edges(
        pos_edges=set(train_pos_edges),
        neg_ratio = neg_ratio,
        valid_nodes=train_nid_set,
        exclude_set=set(train_pos_edges)
    )
    train_neg_labels = [0] * len(train_neg_edges)

    final_train_edges = train_pos_edges + train_neg_edges
    final_train_edges = [(global_to_local[u], global_to_local[v]) for u, v in final_train_edges]  
    final_train_labels = train_pos_labels + train_neg_labels

    # # Combine and filter final training edges to ensure all nodes exist in train_graph
    # final_train_edges_raw = train_pos_edges + train_neg_edges
    # final_train_labels_raw = train_pos_labels + train_neg_labels

    # train_node_set = set(train_nids.tolist())
    # final_train_edges = []
    # final_train_labels = []

    # for (u, v), label in zip(final_train_edges_raw, final_train_labels_raw):
    #     if u in train_node_set and v in train_node_set:
    #         final_train_edges.append((u, v))
    #         final_train_labels.append(label)

    # print(f"🧹 Filtered training edges: kept {len(final_train_edges)} / {len(final_train_edges_raw)}", flush=True)

    # train_feat_pairs = torch.stack([
    #     torch.cat([features[u], features[v]]) for u, v in final_train_edges
    # ])
    # print(f"Train feature pairs shape: {train_feat_pairs.shape}", flush=True)

    val_g = dgl.node_subgraph(full_graph, test_nids).to(features.device)
    # Create global-to-local node ID mapping after subgraph
    global_to_local_v = {nid.item(): i for i, nid in enumerate(test_nids)}
    val_g.ndata['feat'] = features[test_nids]
    test_pos_edges = extract_edges(full_graph, test_nids)
    test_pos_labels = [1] * len(test_pos_edges)

    test_neg_edges = generate_negative_edges(
        pos_edges=set(test_pos_edges),
        neg_ratio=neg_ratio,
        valid_nodes=test_nid_set,
        exclude_set=set(test_pos_edges)
    )
    test_neg_labels = [0] * len(test_neg_edges)

    val_edges = test_pos_edges + test_neg_edges
    val_edges = [(global_to_local_v[u], global_to_local_v[v]) for u, v in val_edges]
    val_labels = test_pos_labels + test_neg_labels

    # val_edges_raw = test_pos_edges + test_neg_edges
    # val_labels_raw = test_pos_labels + test_neg_labels

    # # Filter: keep only edges where both u and v are in train_nid_set (since we sample from train_graph)
    # val_edges = []
    # val_labels = []
    # for (u, v), label in zip(val_edges_raw, val_labels_raw):
    #     if u in test_nid_set and v in test_nid_set:
    #         val_edges.append((u, v))
    #         val_labels.append(label)

    # print(f"🧪 Filtered validation edges: kept {len(val_edges)} / {len(val_edges_raw)}", flush=True)

    # val_feat_pairs = torch.stack([
    #     torch.cat([features[u], features[v]]) for u, v in val_edges
    # ])
    # print(f"Validation feature pairs shape: {val_feat_pairs.shape}", flush=True)

    # print("=== Inductive Split Complete ===", flush=True)

    return train_g, features, final_train_edges, final_train_labels, val_g, val_edges, val_labels
