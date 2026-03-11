"""
Compute average node degree for Elliptic, treating each timestep independently.
For each timestep, compute the avg degree of that snapshot, then average
across all 49 timesteps.
"""

import torch
import os

DATA_DIR = os.path.expanduser("~/FraudGT-thesis/data")
path = os.path.join(DATA_DIR, "Elliptic", "processed", "data.pt")

if not os.path.exists(path):
    print("Elliptic data not found")
    exit()

data_dict = torch.load(path, weights_only=False)

per_timestep = []

for split in ['train', 'val', 'test']:
    data = data_dict[split]
    x = data['node'].x                                   # [num_nodes, num_features]
    edge_index = data['node', 'to', 'node'].edge_index   # [2, num_edges]

    # x[:, 0] is the timestep each node (transaction) belongs to
    timesteps = x[:, 0]

    for t in timesteps.unique():
        # Boolean mask: which nodes belong to this timestep
        node_mask = (timesteps == t)

        # Node indices in this timestep
        node_ids = node_mask.nonzero(as_tuple=True)[0]

        num_nodes = node_ids.shape[0]

        # Keep only edges where the source node is in this timestep
        # (edges in Elliptic are within-timestep, so this captures all edges)
        src = edge_index[0]
        edge_mask = node_mask[src]
        num_edges = edge_mask.sum().item()

        avg_out = num_edges / num_nodes if num_nodes > 0 else 0.0

        per_timestep.append((int(t.item()), num_nodes, num_edges, avg_out))

# Sort by timestep for readability
per_timestep.sort(key=lambda x: x[0])

print(f"\n{'Timestep':<12} {'Nodes':<12} {'Edges':<12} {'Avg Out-Degree'}")
print("-" * 50)
for t, n, e, d in per_timestep:
    print(f"{t:<12} {n:<12,} {e:<12,} {d:.4f}")

avg_degree = sum(d for _, _, _, d in per_timestep) / len(per_timestep)
total_nodes = sum(n for _, n, _, _ in per_timestep)
total_edges = sum(e for _, _, e, _ in per_timestep)

print(f"\nTotal nodes across all timesteps : {total_nodes:,}")
print(f"Total edges across all timesteps : {total_edges:,}")
print(f"Mean avg out-degree across timesteps: {avg_degree:.4f}")
print(f"Mean avg total degree across timesteps: {avg_degree * 2:.4f}")
