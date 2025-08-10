import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList(
            [SAGEConv(in_dim if i==0 else hidden, hidden) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers)-1:
                x = self.act(x); x = self.dropout(x)
        return x

class DotPredictor(nn.Module):
    def forward(self, zi, zj):
        return (zi * zj).sum(dim=-1)  # logits

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, dropout=0.5):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden, num_layers, dropout)
        self.pred = DotPredictor()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.pred(z[edge_label_index[0]], z[edge_label_index[1]])
