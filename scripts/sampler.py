import dgl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def move_to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor

class PairDataset(Dataset):
    def __init__(self, edge_list, labels):
        self.edge_list = edge_list
        self.labels = labels

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, idx):
        return self.edge_list[idx], self.labels[idx]

class PrecomputedSampler:
    def __init__(self, graph, precomputed_dict, num_neighbors=8, num_layers=3, device='cpu'):
        self.graph = graph.to(device)
        self.precomputed = precomputed_dict
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.device = device

    def sample_blocks(self, seeds):
        blocks = []
        seeds = seeds.to(self.device)

        for _ in range(self.num_layers):
            src_list = []
            dst_list = []

            for seed in seeds.tolist():
                neighbor_info = self.precomputed.get(seed, {'neighbors': [], 'probs': []})
                neighbors = neighbor_info['neighbors']
                probs = np.array(neighbor_info['probs'], dtype=np.float32)

                if len(neighbors) == 0 or len(probs) == 0:
                    continue

                # Normalize probs to ensure they sum to 1
                prob_sum = probs.sum()
                if prob_sum == 0 or np.isnan(prob_sum):
                    probs = np.ones(len(neighbors)) / len(neighbors)
                else:
                    probs = probs / prob_sum

                num_sample = min(self.num_neighbors, len(neighbors))
                sampled_indices = np.random.choice(len(neighbors), size=num_sample, replace=True, p=probs)
                sampled = torch.tensor([neighbors[i] for i in sampled_indices], device=self.device)

                src_list.append(sampled)
                dst_list.append(torch.full((len(sampled),), seed, dtype=torch.long, device=self.device))

            if len(src_list) == 0:
                continue

            src = torch.cat(src_list)
            dst = torch.cat(dst_list)

            frontier = dgl.graph((src, dst), num_nodes=self.graph.num_nodes()).to(self.device)
            block = dgl.to_block(frontier, seeds).to(self.device)

            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]

        return blocks


class NeighborCollator:
    def __init__(self, sampler, features):
        self.sampler = sampler
        self.features = features.to(sampler.device)

    def collate(self, batch):
        edges, labels = zip(*batch)
        heads = torch.tensor([u for u, v in edges], device=self.sampler.device)
        tails = torch.tensor([v for u, v in edges], device=self.sampler.device)
        seeds = torch.cat([heads, tails]).unique()

        blocks = self.sampler.sample_blocks(seeds)

        for block in blocks:
            src_nids = block.srcdata[dgl.NID]
            dst_nids = block.dstdata[dgl.NID]
            block.srcdata['feat'] = self.features[src_nids]
            block.dstdata['feat'] = self.features[dst_nids]

        return heads, tails, torch.tensor(labels, device=self.sampler.device), blocks

def get_dataloaders(train_edges, train_labels, val_edges, val_labels, features, train_graph, val_graph, train_precomp, val_precomp, batch_size=1024, num_neighbors=8, num_layers=3, device='cpu'):
    train_dataset = PairDataset(train_edges, train_labels)
    val_dataset = PairDataset(val_edges, val_labels)

    train_sampler = PrecomputedSampler(train_graph, train_precomp, num_neighbors=num_neighbors, num_layers=num_layers, device=device)
    val_sampler = PrecomputedSampler(val_graph, val_precomp, num_neighbors=num_neighbors, num_layers=num_layers, device=device)

    train_collator = NeighborCollator(train_sampler, features)
    val_collator = NeighborCollator(val_sampler, features)

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
