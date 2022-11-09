from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# from torch_geometric.nn import GATv2Conv as GATConv
from tqdm import tqdm
import copy
import json
import os
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

DATASET = 1  # 1 or 2
NUM_EPOCHS = 15

NUM_HIDDEN_CHANNELS = 1
NUM_HEADS = 2
DROPOUT = 0.1
PATIENCE = 5


def process_dataset(dataset_num=1) -> Data:

    X_file = f"data/d{dataset_num}_X.csv"
    adj_file = f"data/d{dataset_num}_adj_mx.csv"
    splits_file = f"data/d{dataset_num}_graph_splits.npz"

    X = pd.read_csv(X_file)
    node_ids = X.columns[1:].astype(int)
    X = X.to_numpy()[:, 1:]
    adj = pd.read_csv(adj_file).to_numpy()[:, 1:]
    splits = np.load(splits_file)
    train_node_ids = splits["train_node_ids"]
    val_node_ids = splits["val_node_ids"]
    test_node_ids = splits["test_node_ids"]

    from torch_geometric.utils import dense_to_sparse

    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))
    if dataset_num == 2:
        edge_attr = 1 - edge_attr
    X = torch.from_numpy(X.astype(np.float32)).to(device)

    data = Data(
        X[:-1], edge_index.to(device), edge_attr.float().to(device), X[1:].to(device)
    )
    data.train_mask = torch.tensor(
        [node_ids[i] in train_node_ids for i in range(data.x.shape[1])]
    )
    data.test_mask = torch.tensor(
        [node_ids[i] in test_node_ids for i in range(data.x.shape[1])]
    )
    data.val_mask = torch.tensor(
        [node_ids[i] in val_node_ids for i in range(data.x.shape[1])]
    )

    return data


data = process_dataset(DATASET)


class CustomModel(torch.nn.Module):
    def __init__(
        self,
        num_features=1,
        hidden_channels=1,
        heads=1,
        num_classes=1,
        edge_dim=1,
        dropout=0.1,
    ):
        super().__init__()

        self.conv1 = GATConv(
            num_features,
            hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            fill_value=0,
        )
        self.act1 = torch.nn.ELU()
        self.conv2 = GATConv(
            hidden_channels,
            num_classes,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            fill_value=0,
        )
        self.act2 = torch.nn.ELU()

        self.loss_fct = torch.nn.MSELoss()

    def forward(self, x, edge_index, mask, edge_attr=None, labels=None):
        x = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act2(x)

        x = x[mask].squeeze()
        if labels is not None:
            labels = labels[mask]
            loss = self.loss_fct(x, labels)
            return loss, x
        return x


def train_epoch(model, data, epoch, optimizer):
    N = data.x.shape[0]
    model.train()
    batch_bar = tqdm(
        enumerate(np.random.choice(np.arange(N), N, replace=False)),
        total=N,
        desc=f"Epoch {epoch}",
    )
    avg_loss = 0
    for i, idx in batch_bar:
        optimizer.zero_grad()  # Clear gradients.
        loss, logits = model(
            x=data.x[idx].unsqueeze(1),
            mask=data.train_mask,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            labels=data.y[idx],
        )
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        avg_loss += loss.item()
        batch_bar.set_postfix(Loss=avg_loss / (i + 1))
        batch_bar.update()
    return loss


def test_split(model, data, split="test"):
    model.eval()
    N = data.x.shape[0]
    mae = 0
    node_mask = getattr(data, f"{split}_mask")
    with torch.no_grad():
        for idx in tqdm(np.arange(N), desc=f"Evaluating {split}"):

            logits = model(
                x=data.x[idx].unsqueeze(1),
                mask=node_mask,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
            )
            mae += torch.abs(
                logits - data.y[idx][node_mask]
            ).mean()  # Check against ground-truth labels.
    return (mae / N).item()


def test_all_splits(model, data):
    train_mae = test_split(model, data, split="train")
    val_mae = test_split(model, data, split="val")
    test_mae = test_split(model, data, split="test")

    print("\tTrain MAE:", train_mae)
    print("\tVal MAE:", val_mae)
    print("\tTest MAE:", test_mae)
    return train_mae, val_mae, test_mae


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    train_mae, val_mae, test_mae = test_all_splits(model, data)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump({"train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae}, f)
    print("Saved model to", path)


def train_model(model, num_epochs):
    best_model = None
    best_mae = float("inf")
    early_stopping_counter = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(num_epochs):
        train_epoch(model, data, epoch, optimizer)
        train_mae, val_mae, test_mae = test_all_splits(model, data)
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0
        elif early_stopping_counter < PATIENCE:
            early_stopping_counter += 1
        else:
            print("Ran out of patience! Early stopping")
            break

    return best_model


if __name__ == "__main__":
    model = train_model(
        model=CustomModel(
            num_features=1,
            hidden_channels=NUM_HIDDEN_CHANNELS,
            heads=NUM_HEADS,
            dropout=DROPOUT,
        ).to(device),
        num_epochs=NUM_EPOCHS,
    )
    save_model(
        model,
        f"models/d{DATASET}/gat",
    )
