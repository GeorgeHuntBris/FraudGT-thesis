"""
Compute DGraph node degree distribution statistics.
"""

import torch

data_dict = torch.load('data/DGraph/processed/data.pt', weights_only=False)
data = data_dict['test']
edge_index = data['node', 'to', 'node'].edge_index

degree = torch.bincount(edge_index[0])

print(f'Total nodes: {data["node"].num_nodes:,}')
print(f'Total edges: {edge_index.shape[1]:,}')
print(f'Max out-degree:  {degree.max().item():,}')
print(f'Min out-degree:  {degree.min().item()}')
print(f'Mean out-degree: {degree.float().mean().item():.2f}')
print(f'Median out-degree: {degree.float().median().item():.1f}')
print(f'\nDegree distribution:')
print(f'  Nodes with 0 out-edges:    {(degree==0).sum().item():,}')
print(f'  Nodes with 1 out-edge:     {(degree==1).sum().item():,}')
print(f'  Nodes with 2-5 out-edges:  {((degree>=2) & (degree<=5)).sum().item():,}')
print(f'  Nodes with 6-10 out-edges: {((degree>=6) & (degree<=10)).sum().item():,}')
print(f'  Nodes with >10 out-edges:  {(degree>10).sum().item():,}')
print(f'  Nodes with >100 out-edges: {(degree>100).sum().item():,}')
print(f'  Nodes with >1000 out-edges:{(degree>1000).sum().item():,}')
