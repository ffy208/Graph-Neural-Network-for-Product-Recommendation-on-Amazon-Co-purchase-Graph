import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

def load_graph(path:str) -> Data:
    return torch.load(path)

def make_link_loaders(data, batch_size, num_neighbors, neg_ratio=1.0, nodes=None, shuffle=True):
    edge_label_index = data.edge_index
    return LinkNeighborLoader(
        data=data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label=torch.ones(edge_label_index.size(1)),
        neg_sampling_ratio=neg_ratio,
        input_nodes=nodes,
        shuffle=shuffle,
    )

def attach_features(data:Data, X):
    data.x = X
    return data
