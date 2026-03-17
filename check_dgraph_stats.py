import torch
from torch_geometric.datasets import DGraphFin

d = DGraphFin(root='data/DGraph/dgraph_pyg')[0]
n = d.num_nodes
print('Nodes:', n)
print('Edges:', d.edge_index.shape[1])
print('Train:', d.train_mask.sum().item())
print('Val:', d.val_mask.sum().item())
print('Test:', d.test_mask.sum().item())
illicit = (d.y == 1).sum().item()
licit = (d.y == 0).sum().item()
background = (d.y == 2).sum().item() + (d.y == 3).sum().item()
print('Illicit:', illicit, f'({100*illicit/n:.2f}%)')
print('Licit:', licit, f'({100*licit/n:.2f}%)')
print('Background:', background, f'({100*background/n:.2f}%)')
