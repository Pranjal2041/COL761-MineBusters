from torch_geometric.data import Data as TG_Data
import numpy as np
import pandas as pd
import torch


from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_sliding_window(arr: np.ndarray, window_size: int):
    n = arr.shape[0]
    indexer = np.arange(n - window_size + 1)[:, None] + np.arange(window_size)[None]
    return arr[indexer]


def load_test_dataset(
    X_file, adj_file, mu: torch.FloatTensor = None, sigma: torch.FloatTensor = None
):
    if X_file.endswith(".npz"):
        X = torch.from_numpy(np.load(X_file)["x"]).transpose(1, 2)
    else:
        X = torch.from_numpy(
            pd.read_csv(X_file, header=None).to_numpy().astype(np.float32)
        )
    if len(X.shape) != 3:
        X = X.unsqueeze(-1)
    adj = pd.read_csv(adj_file).to_numpy()[:, 1:]
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))
    data = TG_Data(X, edge_index, edge_attr.float())
    if mu is not None and sigma is not None:
        data.x = (data.x - mu.unsqueeze(1).unsqueeze(0)) / sigma.unsqueeze(1).unsqueeze(
            0
        )
        data.mu = mu
        data.sigma = sigma

    return data


def make_mu_sigma_tensors(data):
    num_nodes = data.x.shape[1]

    mu = torch.zeros((num_nodes,)).float()
    sigma = torch.ones((num_nodes,)).float()

    if hasattr(data, "mu"):
        mu = data.mu
    if hasattr(data, "sigma"):
        sigma = data.sigma
    return mu, sigma


def process_dataset(
    dataset_num=None,
    p=1,
    f=1,
    do_standardize=True,
    X_file=None,
    adj_file=None,
    splits_file=None,
) -> TG_Data:

    X_file = f"data/d{dataset_num}_X.csv" if X_file is None else X_file
    adj_file = f"data/d{dataset_num}_adj_mx.csv" if adj_file is None else adj_file
    splits_file = (
        f"data/d{dataset_num}_graph_splits.npz" if splits_file is None else splits_file
    )

    X = pd.read_csv(X_file)
    node_ids = X.columns[1:].astype(int)
    X = X.to_numpy()[:, 1:]
    if do_standardize:
        mu = X.mean(axis=0)
        sigma = X.mean(axis=0)
        sigma += 1e-16
        X = (X - mu) / (sigma)

    adj = pd.read_csv(adj_file).to_numpy()[:, 1:]
    splits = np.load(splits_file)
    train_node_ids = splits["train_node_ids"]
    val_node_ids = splits["val_node_ids"]
    test_node_ids = splits["test_node_ids"]

    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))
    if dataset_num == 2:
        edge_attr = 1 - edge_attr

    past_values = get_sliding_window(X, window_size=p)
    future_values = get_sliding_window(X, window_size=f)

    inputs = past_values[:-f].transpose(0, 2, 1)
    labels = future_values[p:].transpose(0, 2, 1)

    inputs = torch.from_numpy(inputs.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.float32))
    data = TG_Data(inputs, edge_index, edge_attr.float(), labels)
    data.train_mask = torch.tensor(
        [node_ids[i] in train_node_ids for i in range(data.x.shape[1])]
    )
    data.test_mask = torch.tensor(
        [node_ids[i] in test_node_ids for i in range(data.x.shape[1])]
    )
    data.val_mask = torch.tensor(
        [node_ids[i] in val_node_ids for i in range(data.x.shape[1])]
    )

    if do_standardize:
        data.mu = torch.from_numpy(mu.astype(np.float32))
        data.sigma = torch.from_numpy(sigma.astype(np.float32))

    return data


class GATDataset(Dataset):
    def __init__(self, data: TG_Data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.data.x[idx].to(device),
            "edge_index": self.data.edge_index.to(device),
            "edge_attr": self.data.edge_attr.to(device),
            "labels": self.data.y[idx].to(device) if self.data.y is not None else None,
            "mask": self.data.train_mask.to(device)
            if hasattr(self.data, "train_mask")
            else None,
        }


class STGMAN_Dataset(Dataset):
    def __init__(self, data: TG_Data, SE: torch.FloatTensor) -> None:
        super().__init__()
        self.data = data
        self.SE = SE

    def __len__(self):
        return self.data.x.shape[0]

    def __getitem__(self, idx):
        return {
            "SE": self.SE.to(device),
            "X": self.data.x.transpose(1, 2)[idx].unsqueeze(0).to(device),
            "labels": self.data.y.transpose(1, 2)[idx].unsqueeze(0).to(device)
            if self.data.y is not None
            else None,
            "mask": self.data.train_mask.to(device)
            if hasattr(self.data, "train_mask")
            else None,
        }
