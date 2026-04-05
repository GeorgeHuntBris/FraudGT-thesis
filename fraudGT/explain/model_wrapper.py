import torch.nn as nn

"""
The explainer was written to work with models that take (x_dict, edge_index_dict,edge_attr=edge-attr_dict)
as separate arguments and retain a tensor

However, PNA takes a single HeteroData batch and returns (pred, label)

In: The wrapper receives the separate dicts the explainer sends and packs them into a HeteroData batch.
Out: Calls, model (PNA), gets (pred, label) in return and just returns the pred so it can be utilized by the explainer.
"""

class PNAExplainerWrapper(nn.Module):
    def __init__(self, model, batch):
        super().__init__()
        self.model = model
        self.batch = batch
        self.original_x = {nt: batch[nt].x.clone() for nt in batch.node_types if hasattr(batch[nt], 'x') and batch[nt].x is not None}

    def forward(self, x_dict, edge_index=None, **kwargs):
        for node_type, x in self.original_x.items():
            self.batch[node_type].x = x.clone()
        edge_attr_dict = kwargs.get('edge_attr', {})
        for edge_type, attr in edge_attr_dict.items():
            self.batch[edge_type].edge_attr = attr
        pred, label = self.model(self.batch)
        return pred
