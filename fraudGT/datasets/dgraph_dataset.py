import os
import os.path as osp
import shutil
import numpy as np
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DGraphFin
from torch_geometric.utils import index_to_mask

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


class DGraphDataset(TemporalDataset):
    """
    DGraph-Fin: Large-Scale Financial Graph Dataset for Fraud Detection.

    Nodes: ~3.7M users with 17 anonymised personal profile features
    Edges: ~4.3M directed relationships with 2 features (edge_type, edge_time)
    Task: Binary node classification (fraudster vs normal user)

    Download DGraphFin.zip from https://dgraph.xinye.com/dataset
    and place it at <root>/raw/DGraphFin.zip before running.
    """

    def __init__(
        self,
        root: str,
        reverse_mp: bool = False,
        add_ports: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = 'DGraph'
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
        return ['DGraphFin.zip']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'ports.pt']

    def download(self):
        raise FileNotFoundError(
            f"DGraphFin.zip not found in {self.raw_dir}.\n"
            "Please download DGraphFin.zip from https://dgraph.xinye.com/dataset\n"
            f"and place it at: {osp.join(self.raw_dir, 'DGraphFin.zip')}"
        )

    def process(self):
        print("Loading DGraph-Fin dataset via PyG...")

        # Use a separate subdirectory for PyG's DGraphFin to avoid
        # conflicts with our own processed files in self.processed_dir
        pyg_root = osp.join(self.root, 'dgraph_pyg')
        os.makedirs(osp.join(pyg_root, 'raw'), exist_ok=True)

        zip_src = osp.join(self.raw_dir, 'DGraphFin.zip')
        zip_dst = osp.join(pyg_root, 'raw', 'DGraphFin.zip')
        if not osp.exists(zip_dst):
            shutil.copy(zip_src, zip_dst)

        pyg_dataset = DGraphFin(root=pyg_root)
        raw = pyg_dataset[0]

        print(f"Number of nodes: {raw.num_nodes}")
        print(f"Number of edges: {raw.edge_index.shape[1]}")
        print(f"Node feature shape: {raw.x.shape}")

        # Node features (17-dim) normalised
        x = z_norm(raw.x.float())
        y = raw.y.squeeze().long()

        # Use provided train/val/test masks from DGraphFin
        train_mask = raw.train_mask
        val_mask = raw.val_mask
        test_mask = raw.test_mask

        # Background nodes (not in any split) have labels > 1 — mask them to -1
        # so the framework sees only binary labels (0=normal, 1=fraud)
        foreground_mask = train_mask | val_mask | test_mask
        y[~foreground_mask] = -1

        # Edge features: combine edge_type [E] and edge_time [E] into [E, 2]
        edge_type = raw.edge_type.float().unsqueeze(1)
        edge_time = raw.edge_time.float().unsqueeze(1)
        edge_attr = z_norm(torch.cat([edge_type, edge_time], dim=1))

        edge_index = raw.edge_index
        timestamps = raw.edge_time.float()

        print(f"Train nodes: {train_mask.sum().item()}")
        print(f"Val nodes: {val_mask.sum().item()}")
        print(f"Test nodes: {test_mask.sum().item()}")

        train_labeled = y[train_mask]
        illicit = (train_labeled == 1).sum().item()
        licit = (train_labeled == 0).sum().item()
        print(f"Train illicit: {illicit}, licit: {licit}, ratio=1:{licit // illicit}")

        # Cumulative node sets (same pattern as ETH/Elliptic)
        train_inds = torch.where(train_mask)[0]
        val_inds = torch.where(val_mask)[0]
        test_inds = torch.where(test_mask)[0]

        node_train = train_inds
        node_val = torch.cat([train_inds, val_inds])
        node_test = torch.cat([train_inds, val_inds, test_inds])

        e_train = torch.isin(edge_index[0], node_train) & torch.isin(edge_index[1], node_train)
        e_val = torch.isin(edge_index[0], node_val) & torch.isin(edge_index[1], node_val)
        e_test = torch.isin(edge_index[0], node_test) & torch.isin(edge_index[1], node_test)

        num_nodes = raw.num_nodes
        self.ports_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')
            split_mask_val = eval(f'{split}_mask')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = edge_attr[e_mask]
            masked_timestamps = timestamps[e_mask]

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

            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            data['node', 'to', 'node'].timestamps = masked_timestamps

            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr

            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            out_ports = ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)

            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        print("Processing complete!")

    def __repr__(self) -> str:
        return f'DGraphDataset(name={self.name})'
