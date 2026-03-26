"""
Node classification head with auxiliary edge scoring task.

The main task is node classification (same as hetero_node).
The auxiliary task supervises edge embeddings with soft phishing labels,
encouraging the transformer to learn which transactions lead to phishing nodes.

Auxiliary loss is stored as self.aux_loss after each forward pass so the
training loop can add it to the main classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_node_edge_aux')
class HeteroNodeEdgeAuxHead(nn.Module):
    """Node classification head with auxiliary soft edge classification."""

    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        # Main node classification head
        self.node_head = MLP(
            dim_in, dim_out,
            num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
            bias=True)

        # Auxiliary edge scoring head: edge_attr -> scalar suspicion score
        self.edge_aux_head = MLP(
            dim_in, 1,
            num_layers=2,
            bias=True)

        self.aux_lambda = cfg.gt.aux_lambda
        self.aux_loss = None

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        if isinstance(batch, HeteroData):
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return batch[task].x[:batch_size], \
                    batch[task].y[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return batch[task].x[batch[task][mask]], \
                    batch[task].y[batch[task][mask]]
        else:
            mask = f'{batch.split}_mask'
            return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        # --- Main node classification ---
        if isinstance(batch, HeteroData):
            x = batch[cfg.dataset.task_entity].x
            x = self.node_head(x)
            batch[cfg.dataset.task_entity].x = x
        else:
            batch.x = self.node_head(batch.x)

        pred, label = self._apply_index(batch)

        # --- Auxiliary edge task (training only) ---
        self.aux_loss = None
        if self.training:
            edge_type = ('node', 'to', 'node')
            if isinstance(batch, HeteroData) and \
                    hasattr(batch[edge_type], 'edge_soft_label') and \
                    hasattr(batch[edge_type], 'edge_attr'):

                edge_attr = batch[edge_type].edge_attr
                edge_soft_label = batch[edge_type].edge_soft_label.to(
                    edge_attr.device)

                edge_scores = self.edge_aux_head(edge_attr).squeeze(-1)
                aux_loss = F.binary_cross_entropy_with_logits(
                    edge_scores, edge_soft_label)
                self.aux_loss = self.aux_lambda * aux_loss

        return pred, label
