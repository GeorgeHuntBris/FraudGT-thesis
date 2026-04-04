"""
Node classification head with auxiliary edge scoring task.

Mirrors the logic of hetero_edge.py for building edge representations
(cat[src_emb, dst_emb, edge_attr]), but used as an auxiliary signal that
feeds back into node-level predictions rather than as a standalone edge task.

Training:
  - Node head trained with weighted cross-entropy against ground truth node labels.
  - Edge head: cat([src_emb, dst_emb, edge_attr]) -> scalar score.
    Soft BCE labels: max(1/out_degree_src, 0.1) for edges originating from
    confirmed phishing nodes, 0 otherwise.
  - Total loss = node_loss + aux_lambda * edge_loss.

Test/validation:
  - Edge head scores each edge using learned node embeddings + edge features.
  - Scores aggregated (max) per SOURCE node to produce a node-level suspicion
    signal from outgoing transactions.
  - Final prediction logit[:,1] += edge_combine_weight * aggregated_edge_score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_node_edge_aux')
class HeteroNodeEdgeAuxHead(nn.Module):
    """Node classification head with edge scoring combined at inference."""

    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        # Main node classification head (same as hetero_node.py)
        self.node_head = MLP(
            dim_in, dim_out,
            num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
            bias=True)

        # Edge scoring head: cat([src_emb, dst_emb, edge_attr]) -> scalar
        # Mirrors hetero_edge.py which uses dim_in * 3 for the same concatenation
        self.edge_aux_head = MLP(
            dim_in * 3, 1,
            num_layers=2,
            bias=True)

        self.aux_lambda = cfg.gt.aux_lambda
        self.edge_combine_weight = nn.Parameter(
            torch.tensor(cfg.gt.edge_combine_weight))
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
        edge_type = ('node', 'to', 'node')
        task = cfg.dataset.task_entity

        # --- Node head ---
        if isinstance(batch, HeteroData):
            node_logits = self.node_head(batch[task].x)
        else:
            node_logits = self.node_head(batch.x)

        # --- Edge head ---
        self.aux_loss = None
        if isinstance(batch, HeteroData) and \
                hasattr(batch[edge_type], 'edge_attr'):

            edge_attr = batch[edge_type].edge_attr
            edge_index = batch[edge_type].edge_index
            num_nodes = node_logits.shape[0]
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # Build edge representation: cat([src_emb, dst_emb, edge_attr])
            # Mirrors hetero_edge.py line 44-46
            node_emb = batch[task].x  # still holds embeddings (node_head didn't modify in place)
            edge_input = torch.cat([
                node_emb[src_nodes],
                node_emb[dst_nodes],
                edge_attr
            ], dim=-1)

            edge_scores = self.edge_aux_head(edge_input).squeeze(-1)  # (E,)

            # Training: soft BCE loss against pre-computed soft labels
            if self.training and hasattr(batch[edge_type], 'edge_soft_label'):
                edge_soft_label = batch[edge_type].edge_soft_label.to(
                    edge_attr.device)
                aux_loss = F.binary_cross_entropy_with_logits(
                    edge_scores, edge_soft_label)
                self.aux_loss = self.aux_lambda * aux_loss

            # Aggregate per SOURCE node: max outgoing suspicion score
            edge_scores_sigmoid = torch.sigmoid(edge_scores)
            node_edge_signal = scatter(
                edge_scores_sigmoid, src_nodes,
                dim=0, dim_size=num_nodes, reduce='max')  # (N,)

            # Add signal only to the phishing logit (col 1)
            # Adding to both cols equally (the old bug) cancels out in softmax
            node_logits = node_logits.clone()
            node_logits[:, 1] = node_logits[:, 1] + \
                self.edge_combine_weight * node_edge_signal

        # Write combined logits back for _apply_index
        if isinstance(batch, HeteroData):
            batch[task].x = node_logits
        else:
            batch.x = node_logits

        pred, label = self._apply_index(batch)
        return pred, label
