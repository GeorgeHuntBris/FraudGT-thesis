"""
ETH dataset extended with soft auxiliary edge labels for phishing detection.

For each confirmed phishing node, incoming edges receive a soft label of
max(1 / in_degree, MIN_EDGE_LABEL). All other edges receive label 0.

This enables an auxiliary edge regression task to run alongside the main
node classification, encouraging the model to learn which transactions lead
to phishing accounts.
"""

import torch
from typing import Callable, Optional

from .eth_dataset import ETHDataset


class ETHAuxDataset(ETHDataset):
    """ETH dataset with soft auxiliary edge labels.

    Inherits all processing from ETHDataset (same raw files, same processed
    directory) and adds edge_soft_label in memory after loading.
    """

    MIN_EDGE_LABEL = 0.1

    def __init__(self, root: str, reverse_mp: bool = False,
                 add_ports: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, reverse_mp, add_ports, transform, pre_transform)
        self._add_soft_edge_labels()

    def _add_soft_edge_labels(self):
        for split in ['train', 'val', 'test']:
            data = self.data_dict[split]
            edge_index = data['node', 'to', 'node'].edge_index
            y = data['node'].y
            num_edges = edge_index.shape[1]
            num_nodes = data['node'].num_nodes

            dst_nodes = edge_index[1]

            # In-degree per node
            in_degree = torch.bincount(
                dst_nodes, minlength=num_nodes).float()

            # Soft label = max(1/in_degree, MIN_EDGE_LABEL) if dst is phishing
            dst_is_phishing = (y[dst_nodes] == 1)
            dst_in_degree = in_degree[dst_nodes].clamp(min=1)

            edge_soft_label = torch.where(
                dst_is_phishing,
                torch.clamp(1.0 / dst_in_degree,
                            min=self.MIN_EDGE_LABEL),
                torch.zeros(num_edges)
            )

            data['node', 'to', 'node'].edge_soft_label = edge_soft_label

    def __repr__(self) -> str:
        return 'ETHAuxDataset()'
