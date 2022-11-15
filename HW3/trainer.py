from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch

# from torch_geometric.nn import GATv2Conv as GATConv
from tqdm import tqdm
import copy
import json
import os

from data import make_mu_sigma_tensors

EVAL_RATIO = float(os.environ.get("EVAL_RATIO", 0.1))
PATIENCE = 5


def train_epoch(
    model,
    data_loader,
    epoch,
    optimizer,
):
    N = len(data_loader)
    eval_steps = int(EVAL_RATIO * N)
    best_model = copy.deepcopy(model)
    best_mae = float("inf")

    model.train()
    batch_bar = tqdm(
        enumerate(np.random.choice(np.arange(N), N, replace=False)),
        total=N,
        desc=f"Epoch {epoch}",
    )
    avg_loss = 0
    for i, idx in batch_bar:
        optimizer.zero_grad()  # Clear gradients.
        batch = data_loader[idx]
        loss, logits = model(**batch)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        avg_loss += loss.item()
        batch_bar.set_postfix(Loss=avg_loss / (i + 1))

        if (i + 1) % eval_steps == 0:
            _, val_mae, _ = compute_metrics(model, data_loader, verbose=True)
            if val_mae < best_mae:
                best_mae = val_mae
                best_model = copy.deepcopy(model)

    return best_model, best_mae


def predict(model, data_loader, verbose=True):
    model.eval()
    N = len(data_loader)
    mu, sigma = make_mu_sigma_tensors(data=data_loader.data)

    agg_logits = []

    with torch.no_grad():
        pbar = tqdm(np.arange(N), desc=f"Evaluating") if verbose else np.arange(N)
        for idx in pbar:

            batch = data_loader[idx]
            if "labels" in batch:
                del batch["labels"]
            logits = model(**batch)
            logits = logits.detach().cpu().squeeze(0)

            agg_logits.append((logits * sigma.unsqueeze(1) + mu.unsqueeze(1)).numpy())
    return np.array(agg_logits)


def compute_metrics(model, data_loader, verbose=True):
    masks = [
        data_loader.data.train_mask,
        data_loader.data.val_mask,
        data_loader.data.test_mask,
    ]

    # mae_s = np.array([0, 0, 0], dtype=np.float32)
    mae_s = [0.0, 0.0, 0.0]

    logits = predict(model=model, data_loader=data_loader, verbose=verbose)
    logits = torch.from_numpy(logits)
    # import pdb

    # pdb.set_trace()
    mu, sigma = make_mu_sigma_tensors(data=data_loader.data)
    labels = (data_loader.data.y * sigma.unsqueeze(1)) + mu.unsqueeze(1)

    if logits.shape[1] != masks[0].shape[0]:
        logits = logits.transpose(1, 2)

    for i in range(len(masks)):
        mae_s[i] += torch.abs(logits - labels)[
            :, masks[i]
        ].mean()  # Check against ground-truth labels.
    train_mae, val_mae, test_mae = mae_s
    if verbose:
        print("\tTrain MAE:", train_mae)
        print("\tVal MAE:", val_mae)
        print("\tTest MAE:", test_mae)
    return train_mae, val_mae, test_mae


def train_model(model, data_loader, optimizer, num_epochs):
    best_model = None
    best_mae = float("inf")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        best_ep_model, best_ep_mae = train_epoch(model, data_loader, epoch, optimizer)
        _, val_mae, _ = compute_metrics(model, data_loader)

        if best_ep_mae < val_mae:
            # NOTE should this be done?
            print(
                "Best epoch mae = ", best_ep_mae, " is better than val mae = ", val_mae
            )
            val_mae = best_ep_mae
            model = best_ep_model

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


def save_model(model, data_loader, path):
    os.makedirs(path, exist_ok=True)

    train_mae, val_mae, test_mae = compute_metrics(model, data_loader)
    metric_path = os.path.join(path, "metrics.json")

    # import pdb

    # pdb.set_trace()

    if os.path.exists(metric_path):
        d = json.load(open(metric_path, "r"))
        if d["test_mae"] < test_mae:
            print(f"Test MAE {test_mae} is worse than previous {d['test_mae']}")
            return

    with open(metric_path, "w") as f:
        json.dump(
            {
                "train_mae": train_mae.item(),
                "val_mae": val_mae.item(),
                "test_mae": test_mae.item(),
            },
            f,
        )
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    print("Saved model to", path)
