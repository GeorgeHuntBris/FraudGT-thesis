"""
Compute average node degree for DGraph and ETH.
These datasets use a cumulative (growing) graph structure, so the test split
contains all nodes and edges seen up to the test period — giving the full graph.
"""

import torch
import os

DATA_DIR = os.path.expanduser("~/FraudGT-thesis/data")

datasets = [
    "ETH",
    "DGraph",
]

print(f"\n{'Dataset':<12} {'Nodes':<12} {'Edges':<12} {'Avg Out-Degree':<18} {'Avg Total Degree'}")
print("-" * 70)

for name in datasets:
    path = os.path.join(DATA_DIR, name, "processed", "data.pt")
    if not os.path.exists(path):
        print(f"{name:<12} data not found")
        continue

    # Load the test split — for cumulative datasets this contains all nodes/edges
    data_dict = torch.load(path, weights_only=False)
    data = data_dict['test']

    # edge_index has shape [2, num_edges] — row 0 is source, row 1 is destination
    edge_index = data['node', 'to', 'node'].edge_index

    # num_nodes is total nodes in the graph
    num_nodes = data['node'].num_nodes

    # num_edges is the number of directed edges
    num_edges = edge_index.shape[1]

    # Average out-degree = edges / nodes (each edge contributes 1 outgoing connection)
    avg_out = num_edges / num_nodes

    # Average total degree = 2 * edges / nodes (each edge contributes to in AND out degree)
    avg_total = 2 * num_edges / num_nodes

    print(f"{name:<12} {num_nodes:<12,} {num_edges:<12,} {avg_out:<18.4f} {avg_total:.4f}")
