"""
ETH dataset extended with soft auxiliary edge labels for phishing detection.

For each edge originating FROM a confirmed phishing node, the edge receives a
soft label of max(1 / out_degree_src, MIN_EDGE_LABEL). All other edges get 0.

Weighting by source out-degree means a phishing node that sends to few addresses
(concentrated cashout) gets high-confidence labels per edge, while one that sends
to many places gets weaker labels. At inference, edge scores are aggregated per
SOURCE node so the signal directly boosts the fraud logit of the node whose
outgoing behaviour looks suspicious.
"""

import torch
from typing import Callable, Optional

from .eth_dataset import ETHDataset


class ETHAuxDataset(ETHDataset):
    """ETH dataset with soft auxiliary edge labels on outgoing phishing edges.

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

            src_nodes = edge_index[0]

            # Out-degree per node
            out_degree = torch.bincount(
                src_nodes, minlength=num_nodes).float()

            # Soft label = max(1/out_degree_src, MIN_EDGE_LABEL) if src is phishing
            src_is_phishing = (y[src_nodes] == 1)
            src_out_degree = out_degree[src_nodes].clamp(min=1)

            edge_soft_label = torch.where(
                src_is_phishing,
                torch.clamp(1.0 / src_out_degree, min=self.MIN_EDGE_LABEL),
                torch.zeros(num_edges)
            )

            data['node', 'to', 'node'].edge_soft_label = edge_soft_label

    def __repr__(self) -> str:
        return 'ETHAuxDataset()'
