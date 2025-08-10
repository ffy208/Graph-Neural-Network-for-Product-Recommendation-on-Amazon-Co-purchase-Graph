from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # data
    processed_graph_path: str = "data/amazon_graph.pt"
    # training
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 1e-2
    weight_decay: float = 0.0
    num_neighbors: List[int] = field(default_factory=lambda: [10, 10])
    early_stop_patience: int = 20
    seed: int = 42
    # model
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    # io
    ckpt_dir: str = "results/graphsage_ckpts"
