import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


class DenseGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.2):
        super().__init__()
        self.fc_self = nn.Linear(in_feats, hidden_feats)
        self.fc_neigh = nn.Linear(in_feats, hidden_feats)
        self.out_fc = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, x):
        h_self = self.fc_self(x[:block.num_dst_nodes()])
        block.srcdata['h'] = x
        block.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'neigh'))
        h_neigh = block.dstdata['neigh']
        h_neigh = self.fc_neigh(h_neigh / block.in_degrees().clamp(min=1).unsqueeze(-1))
        h = torch.relu(h_self + h_neigh)
        h = self.dropout(h)
        return self.out_fc(h)


class PinSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_feats if i == 0 else out_feats
            self.layers.append(DenseGraphSAGE(in_dim, hidden_feats, out_feats, dropout))

    def forward(self, blocks):
        x = blocks[0].srcdata['feat']
        for i, layer in enumerate(self.layers):
            x = layer(blocks[i], x)
        return x


class DotProductPredictor(nn.Module):
    def forward(self, h_u, h_v):
        return torch.sum(h_u * h_v, dim=1)


class LinkPredictionModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = PinSAGEModel(in_feats, hidden_feats, out_feats, num_layers, dropout)
        self.decoder = DotProductPredictor()

    def set_input_features(self, features):
        self.input_features = features

    def forward(self, blocks, heads, tails):
        # Inject node features
        blocks[0].srcdata['feat'] = self.input_features[blocks[0].srcdata[dgl.NID]]
        h = self.encoder(blocks)

        # Convert global node IDs to local indices in output node list
        output_nids = blocks[-1].dstdata[dgl.NID]
        global_to_local = {nid.item(): i for i, nid in enumerate(output_nids)}
        heads_local = torch.tensor([global_to_local[h.item()] for h in heads.cpu()], device=heads.device)
        tails_local = torch.tensor([global_to_local[t.item()] for t in tails.cpu()], device=tails.device)

        h_u = h[heads_local]
        h_v = h[tails_local]
        return self.decoder(h_u, h_v)
