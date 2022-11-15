import json
import pickle
import sys
import torch
import pandas as pd
from torch_geometric.nn import GATConv

import os

from data import GATDataset, load_test_dataset, process_dataset
from trainer import train_model, save_model, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 15))

NUM_HIDDEN_CHANNELS = 2
NUM_HEADS = 2
DROPOUT = 0

DATASET = os.environ.get("DATASET", 1)  # 1 or 2
P = 1
F = 1

USERNAME = "cs1190431"
MODEL_SAVE_PATH = f"{USERNAME}_task1.model"
META_SAVE_PATH = f"{USERNAME}_meta_task1.pkl"


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
        self.config = {
            "num_features": num_features,
            "hidden_channels": hidden_channels,
            "heads": heads,
            "num_classes": num_classes,
            "edge_dim": edge_dim,
            "dropout": dropout,
        }

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

        # self.loss_fct = torch.nn.MSELoss()  # TODO maybe change to L1Loss or HuberLoss
        self.loss_fct = torch.nn.L1Loss()

    def forward(self, x, edge_index, mask=None, edge_attr=None, labels=None):
        x = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.act2(x)

        if labels is not None:

            loss = self.loss_fct(x[mask], labels[mask])
            return loss, x
        return x

    @staticmethod
    def load_pretrained(output_dir):
        config = json.load(open(os.path.join(output_dir, "config.json")))
        model = CustomModel(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, "model.pth")))
        return model

    def save_pretrained(self, output_dir):
        json.dump(self.config, open(os.path.join(output_dir, "config.json"), "w"))
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pth"))


def model_init():
    return CustomModel(
        num_features=P,
        hidden_channels=NUM_HIDDEN_CHANNELS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=F,
    ).to(device)


if __name__ == "__main__":
    action = sys.argv[1]

    if action == "train":
        X_file, adj_file, splits_file = (sys.argv[2], sys.argv[3], sys.argv[4])
        # data = process_dataset(DATASET, p=P, f=F, do_standardize=False)

        data = process_dataset(
            p=P,
            f=F,
            do_standardize=False,
            X_file=X_file,
            adj_file=adj_file,
            splits_file=splits_file,
        )

        pickle.dump(
            {
                "adj_file": adj_file,
                "splits_file": splits_file,
                "mu": getattr(data, "mu", None),
                "sigma": getattr(data, "sigma", None),
            },
            open(META_SAVE_PATH, "wb"),
        )

        dataset = GATDataset(data)
        model = model_init()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        model = train_model(
            model=model,
            optimizer=optimizer,
            data_loader=dataset,
            num_epochs=NUM_EPOCHS,
        )
        save_model(
            model=model,
            data_loader=dataset,
            path=f"models/d{DATASET}_P{P}_F{F}/gat",
        )
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    elif action == "test":
        X_file, output_file, model_path = (sys.argv[2], sys.argv[3], sys.argv[4])
        pkl_file = pickle.load(open(META_SAVE_PATH, "rb"))
        adj_file, splits_file, mu, sigma = (
            pkl_file["adj_file"],
            pkl_file["splits_file"],
            pkl_file["mu"],
            pkl_file["sigma"],
        )
        test_data = load_test_dataset(
            X_file=X_file, adj_file=adj_file, mu=mu, sigma=sigma
        )
        dataset = GATDataset(data=test_data)
        model = model_init()
        model.load_state_dict(torch.load(model_path))
        logits = predict(model=model, data_loader=dataset, verbose=True)

        logits = pd.DataFrame(logits.squeeze(-1))
        logits.to_csv(output_file, header=None, index=False)
