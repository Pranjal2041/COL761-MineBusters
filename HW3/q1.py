import torch
from torch_geometric.nn import GATConv

import os

from data import GATDataset, process_dataset
from trainer import train_model, save_model

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

NUM_EPOCHS = 15

NUM_HIDDEN_CHANNELS = 2
NUM_HEADS = 2
DROPOUT = 0.1

DATASET = os.environ.get("DATASET", 1)  # 1 or 2
P = 1
F = 1


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
            fill_value=0,  # TODO need to experiment with this
            concat=False,
        )
        self.act1 = torch.nn.ELU()
        self.conv2 = GATConv(
            hidden_channels,
            num_classes,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            fill_value=0,
            concat=False,
        )
        self.act2 = torch.nn.ELU()  # TODO maybe change to ReLU or LeakyReLU

        self.loss_fct = torch.nn.MSELoss()  # TODO maybe change to L1Loss or HuberLoss

    def forward(self, x, edge_index, mask=None, edge_attr=None, labels=None):
        x = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act2(x)

        if labels is not None:

            loss = self.loss_fct(x[mask], labels[mask])
            return loss, x
        return x


if __name__ == "__main__":
    data = process_dataset(DATASET, p=P, f=F, do_standardize=False)
    dataset = GATDataset(data)
    model = CustomModel(
        num_features=data.x.shape[-1],
        hidden_channels=NUM_HIDDEN_CHANNELS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=F,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model = train_model(
        model=model,
        optimizer=optimizer,
        data_loader=dataset,
        num_epochs=NUM_EPOCHS,
    )
    save_model(
        model=model,
        data_loader=data,
        path=f"models/d{DATASET}_P{P}_F{F}/gat",
    )
