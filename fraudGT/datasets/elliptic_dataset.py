

import os
import os.path as osp
import pandas as pd
import numpy as np
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask

from .temporal_dataset import TemporalDataset


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def to_adj_nodes_with_times(data):
    """Build adjacency lists with timestamps for port computation."""
    num_nodes = data.num_nodes
    timestamps = (
        torch.zeros((data.edge_index.shape[1], 1))
        if data['node', 'to', 'node'].timestamps is None
        else data['node', 'to', 'node'].timestamps.reshape((-1, 1))
    )
    edges = torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out


def ports(edge_index, adj_list):
    """Compute port numberings for edges based on temporal ordering."""
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


class EllipticDataset(TemporalDataset):
    """
    Args:
        root: Root directory where the dataset should be saved
        name: Dataset name (default: 'elliptic')
        reverse_mp: Whether to include reverse edges for message passing
        add_ports: Whether to add port numbering features to edges
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply to data
    """

    # Train on time steps 1-34, validate on 35-42, test on 43-49
    TRAIN_TIME_STEPS = list(range(1, 35))   # 34 time steps
    VAL_TIME_STEPS = list(range(35, 43))    # 8 time steps
    TEST_TIME_STEPS = list(range(43, 50))   # 7 time steps

    def __init__(
        self,
        root: str,
        name: str = 'elliptic',
        reverse_mp: bool = False,
        add_ports: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.name = name
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
        """Add port numbering features to edge attributes."""
        in_ports, out_ports = ports_data

        if not self.reverse_mp:
            out_ports_list = [out_ports]
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports] + out_ports_list, dim=1
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
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'ports.pt']

    def download(self):
        for filename in self.raw_file_names:
            filepath = osp.join(self.raw_dir, filename)
            if not osp.exists(filepath):
                raise FileNotFoundError(
                    f"File '{filename}' not found in {self.raw_dir}. "
                    "Please download from Kaggle."
                )

    def process(self):
        """Process raw Elliptic data into PyG HeteroData format."""

        # Load raw data
        print("Loading Elliptic dataset...")

        # Features: txId, time_step, then 166 features
        df_features = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_features.csv'),
            header=None
        )

        # Edge list: txId1, txId2
        df_edges = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        )

        # Classes: txId, class (1=illicit, 2=licit, "unknown")
        df_classes = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_classes.csv')
        )

        # Create node ID mapping (original txId -> consecutive index)
        all_nodes = df_features[0].values
        node_id_map = {txid: idx for idx, txid in enumerate(all_nodes)}
        num_nodes = len(all_nodes)

        # Extract time steps and features
        time_steps = df_features[1].values  # Column 1 is time step
        node_features = df_features.iloc[:, 2:].values  # Columns 2-167 are features

        print(f"Number of nodes (transactions): {num_nodes}")
        print(f"Number of features per node: {node_features.shape[1]}")
        print(f"Time steps: {int(time_steps.min())} to {int(time_steps.max())}")

        # Process labels
        # Map: 1 -> 1 (illicit), 2 -> 0 (licit), "unknown" -> -1 (mask out)
        df_classes['mapped_class'] = df_classes['class'].apply(
            lambda x: 1 if x == '1' else (0 if x == '2' else -1)
        )

        # Create label array aligned with node indices
        labels = -1 * np.ones(num_nodes, dtype=np.int64)
        for _, row in df_classes.iterrows():
            txid = row['txId']
            if txid in node_id_map:
                labels[node_id_map[txid]] = row['mapped_class']

        labeled_mask = labels != -1
        n_labeled = labeled_mask.sum()
        n_illicit = (labels == 1).sum()
        n_licit = (labels == 0).sum()

        print(f"Labeled nodes: {n_labeled} / {num_nodes} ({100*n_labeled/num_nodes:.1f}%)")
        print(f"Illicit: {n_illicit} ({100*n_illicit/n_labeled:.2f}% of labeled)")
        print(f"Licit: {n_licit} ({100*n_licit/n_labeled:.2f}% of labeled)")

        # Process edges - map to consecutive node indices
        edge_src = df_edges['txId1'].map(node_id_map).values
        edge_dst = df_edges['txId2'].map(node_id_map).values

        # Remove edges with unmapped nodes (shouldn't happen but safety check)
        valid_edges = ~(np.isnan(edge_src) | np.isnan(edge_dst))
        edge_src = edge_src[valid_edges].astype(np.int64)
        edge_dst = edge_dst[valid_edges].astype(np.int64)

        print(f"Number of edges: {len(edge_src)}")

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
        timestamps = torch.tensor(time_steps, dtype=torch.float32)

        # Normalize features
        x = z_norm(x)

        # Create edge timestamps based on source node time step
        edge_timestamps = timestamps[edge_index[0]]

        # Create edge attributes (just timestamp for now, could add more)
        edge_attr = edge_timestamps.unsqueeze(1)

        # Split by time steps
        train_mask = torch.tensor(
            np.isin(time_steps, self.TRAIN_TIME_STEPS), dtype=torch.bool
        )
        val_mask = torch.tensor(
            np.isin(time_steps, self.VAL_TIME_STEPS), dtype=torch.bool
        )
        test_mask = torch.tensor(
            np.isin(time_steps, self.TEST_TIME_STEPS), dtype=torch.bool
        )

        # Also mask out unknown labels
        labeled_mask_tensor = torch.tensor(labeled_mask, dtype=torch.bool)
        train_mask = train_mask & labeled_mask_tensor
        val_mask = val_mask & labeled_mask_tensor
        test_mask = test_mask & labeled_mask_tensor

        print(f"\nSplit statistics (labeled nodes only):")
        print(f"Train: {train_mask.sum().item()} nodes (time steps 1-34)")
        print(f"Val: {val_mask.sum().item()} nodes (time steps 35-42)")
        print(f"Test: {test_mask.sum().item()} nodes (time steps 43-49)")

        # Get indices for each split
        train_inds = torch.where(train_mask)[0]
        val_inds = torch.where(val_mask)[0]
        test_inds = torch.where(test_mask)[0]

        # Cumulative node sets (for temporal consistency)
        node_train = train_inds
        node_val = torch.cat([train_inds, val_inds])
        node_test = torch.cat([train_inds, val_inds, test_inds])

        # Edge masks based on cumulative node sets
        e_train = torch.isin(edge_index[0], node_train) & torch.isin(edge_index[1], node_train)
        e_val = torch.isin(edge_index[0], node_val) & torch.isin(edge_index[1], node_val)
        e_test = torch.isin(edge_index[0], node_test) & torch.isin(edge_index[1], node_test)

        # Build data for each split
        self.ports_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')
            split_mask = eval(f'{split}_mask')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = edge_attr[e_mask]
            masked_timestamps = edge_timestamps[e_mask]

            data = HeteroData()
            data['node'].x = x
            data['node'].y = y
            data['node'].num_nodes = num_nodes

            # Store masks for this split (in node store)
            data['node'].train_mask = train_mask
            data['node'].val_mask = val_mask
            data['node'].test_mask = test_mask
            data['node'].split_mask = split_mask  # Current split's mask

            # Also store masks at top level for split_generator compatibility
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            # Edge data
            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            data['node', 'to', 'node'].timestamps = masked_timestamps

            # Reverse edges for bidirectional message passing
            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr

            # Compute ports
            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            out_ports = ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)

            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        # Save processed data
        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])

        print("\nProcessing complete!")

    def __repr__(self) -> str:
        return f'EllipticDataset(name={self.name})'
