import os.path as osp
import numpy as np
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split

from .temporal_dataset import TemporalDataset


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def to_adj_nodes_with_times(data):
    num_nodes = data['node'].num_nodes
    timestamps = data['node', 'to', 'node'].timestamps
    if timestamps is None:
        timestamps = torch.zeros(data['node', 'to', 'node'].edge_index.shape[1], 1)
    else:
        timestamps = timestamps.reshape(-1, 1)
    edges = torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = {i: [] for i in range(num_nodes)}
    adj_list_in = {i: [] for i in range(num_nodes)}
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u].append((v, t))
        adj_list_in[v].append((u, t))
    return adj_list_in, adj_list_out


def ports(edge_index, adj_list):
    ports_tensor = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1:
            continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:, [0]], return_index=True, axis=0)
        nbs_unique = a[np.sort(idx)][:, 0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u, v)] = i
    for i, e in enumerate(edge_index.T):
        ports_tensor[i] = ports_dict[tuple(e.numpy())]
    return ports_tensor


class BitcoinMDataset(TemporalDataset):
    """
    Bitcoin-M: Large-scale Bitcoin transaction graph for illicit activity detection.

    Nodes: ~2.5M transactions with 56 features
    Edges: ~19M directed transactions with 5 features
    Labels: -1 (unlabeled ~90%), 0 (licit), 1 (illicit)
    Task: Binary node classification on labeled nodes
    Split: Stratified random 50/25/25 on labeled nodes (seed=42, DIAM paper 2:1:1 ratio)

    Source: DIAM paper (arxiv 2309.02460). Download data.pt and place it at
    <root>/raw/data.pt before running.
    """

    def __init__(
        self,
        root: str,
        reverse_mp: bool = False,
        add_ports: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = 'BitcoinM'
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0], weights_only=False)

        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                if ('node', 'rev_to', 'node') in self.data_dict[split].edge_types:
                    del self.data_dict[split]['node', 'rev_to', 'node']

        if add_ports:
            self.ports_dict = torch.load(self.processed_paths[1], weights_only=False)
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.add_ports_func(
                    self.data_dict[split], self.ports_dict[split]
                )

    def add_ports_func(self, data, ports_data):
        in_ports, out_ports = ports_data
        if not self.reverse_mp:
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports, out_ports], dim=1
            )
        else:
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports], dim=1
            )
            data['node', 'rev_to', 'node'].edge_attr = torch.cat(
                [data['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1
            )
        return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'ports.pt']

    def download(self):
        raise FileNotFoundError(
            f"data.pt not found in {self.raw_dir}.\n"
            "Please download data.pt from the DIAM paper's repository\n"
            f"and place it at: {osp.join(self.raw_dir, 'data.pt')}"
        )

    def process(self):
        print("Loading Bitcoin-M dataset...")

        raw = torch.load(osp.join(self.raw_dir, 'data.pt'), weights_only=False)

        num_nodes = raw.X.shape[0]
        print(f"Number of nodes: {num_nodes:,}")
        print(f"Number of edges: {raw.edge_index.shape[1]:,}")
        print(f"Node feature dim: {raw.X.shape[1]}")
        print(f"Edge feature dim: {raw.edge_attr.shape[1]}")

        # Node features (56-dim): float64 -> float32 and z-normalise
        x = z_norm(raw.X.float())

        # Labels: -1=unlabeled, 0=licit, 1=illicit (already correct format)
        y = raw.y.long()

        # Edge features (5-dim): float64 -> float32, z-normalise
        # Column 4 is a Unix timestamp — use as port ordering key
        edge_attr = z_norm(raw.edge_attr.float())
        edge_index = raw.edge_index
        timestamps = raw.edge_attr[:, 4].float()

        # Stats
        labeled_mask = y >= 0
        n_labeled = labeled_mask.sum().item()
        n_illicit = (y == 1).sum().item()
        n_licit = (y == 0).sum().item()
        print(f"Labeled nodes: {n_labeled:,} ({100*n_labeled/num_nodes:.1f}%)")
        print(f"Illicit: {n_illicit:,} ({100*n_illicit/n_labeled:.2f}% of labeled)")
        print(f"Licit:   {n_licit:,} ({100*n_licit/n_labeled:.2f}% of labeled)")
        print(f"Licit:Illicit ratio: 1:{n_licit//n_illicit}")

        # Stratified 50/25/25 split on labeled nodes (seed=42)
        # Follows DIAM paper (arxiv 2309.02460): "ratio 2:1:1"
        labeled_inds = torch.where(labeled_mask)[0].numpy()
        labeled_y = y[torch.tensor(labeled_inds)].numpy()

        train_idx, temp_idx = train_test_split(
            labeled_inds, test_size=0.5, random_state=42, stratify=labeled_y
        )
        temp_y = y[torch.tensor(temp_idx)].numpy()
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=temp_y
        )

        train_mask = index_to_mask(torch.tensor(train_idx, dtype=torch.long), size=num_nodes)
        val_mask = index_to_mask(torch.tensor(val_idx, dtype=torch.long), size=num_nodes)
        test_mask = index_to_mask(torch.tensor(test_idx, dtype=torch.long), size=num_nodes)

        print(f"\nSplit statistics (labeled nodes only, 2:1:1 ratio):")
        print(f"Train: {train_mask.sum().item():,} nodes")
        print(f"Val:   {val_mask.sum().item():,} nodes")
        print(f"Test:  {test_mask.sum().item():,} nodes")

        # Compute ports once — all splits use full graph (no temporal ordering)
        print("\nComputing port numbers (this may take a few minutes for 19M edges)...")
        temp_data = HeteroData()
        temp_data['node'].num_nodes = num_nodes
        temp_data['node', 'to', 'node'].edge_index = edge_index
        temp_data['node', 'to', 'node'].timestamps = timestamps

        adj_list_in, adj_list_out = to_adj_nodes_with_times(temp_data)
        in_ports = ports(edge_index, adj_list_in)
        out_ports = ports(edge_index.flipud(), adj_list_out)
        print("Ports computed.")

        self.ports_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            split_mask_val = eval(f'{split}_mask')

            data = HeteroData()
            data['node'].x = x
            data['node'].y = y
            data['node'].num_nodes = num_nodes
            data['node'].train_mask = train_mask
            data['node'].val_mask = val_mask
            data['node'].test_mask = test_mask
            data['node'].split_mask = split_mask_val
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            # All splits use the full graph (transductive, no temporal leakage)
            data['node', 'to', 'node'].edge_index = edge_index
            data['node', 'to', 'node'].edge_attr = edge_attr
            data['node', 'to', 'node'].timestamps = timestamps

            data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = edge_attr

            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        print("Processing complete!")

    def __repr__(self) -> str:
        return f'BitcoinMDataset(name={self.name})'
