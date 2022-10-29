from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
# from torch_geometric.nn import GATv2Conv as GATConv
from tqdm import tqdm



DATASET = '2' # 1 or 2
NUM_EPOCHS = 5

NUM_HIDDEN_CHANNELS = 1
NUM_HEADS = 2
DROPOUT = 0.6



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_dataset() -> Data:

    X_file = f'data/d{DATASET}_X.csv'
    adj_file = f'data/d{DATASET}_adj_mx.csv'
    splits_file = f'data/d{DATASET}_graph_splits.npz'

    X = pd.read_csv(X_file).to_numpy()[:,1:]
    adj = pd.read_csv(adj_file).to_numpy()[:,1:]
    splits = np.load(splits_file)
    train_node_ids = splits["train_node_ids"]
    val_node_ids = splits["val_node_ids"]
    test_node_ids = splits["test_node_ids"]

    from torch_geometric.utils import dense_to_sparse
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))

    X = torch.from_numpy(X.astype(np.float32)).to(device)

    data = Data(X[:-1], edge_index.to(device), edge_attr.float().to(device), X[1:].to(device))
    data.train_mask = torch.tensor([i in train_node_ids for i in range(data.x.shape[1])])
    data.test_mask = torch.tensor([i in test_node_ids for i in range(data.x.shape[1])])
    data.val_mask = torch.tensor([i in val_node_ids for i in range(data.x.shape[1])])
    return data

class CustomModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, heads, num_classes = 1, edge_dim = 1):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels, edge_dim=edge_dim) 
        self.conv2 = GATConv(hidden_channels, num_classes, edge_dim=edge_dim)  

    def forward(self, x, edge_index, edge_attr = None):
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x



data = process_dataset()
model = CustomModel(num_features = 1, hidden_channels=NUM_HIDDEN_CHANNELS, heads=NUM_HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.MSELoss()


def train(data, epoch):
  N = data.x.shape[0]
  model.train()
  batch_bar = tqdm(enumerate(np.random.choice(np.arange(N), N, replace = False)), total = N, desc = f'Epoch {epoch}')
  avg_loss = 0
  for i, idx in batch_bar:
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x[idx].unsqueeze(1), data.edge_index, edge_attr = data.edge_attr).squeeze()
    loss = criterion(out[data.train_mask], data.y[idx][data.train_mask]) 
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    avg_loss += loss.item()
    batch_bar.set_postfix(
        Loss = avg_loss / (i + 1)
    )
    batch_bar.update()
  return loss
def test(data, mask):
  model.eval()
  N = data.x.shape[0]
  mae = 0
  with torch.no_grad():
    for idx in tqdm(np.arange(N)):
      out = model(data.x[idx].unsqueeze(1), data.edge_index, edge_attr = data.edge_attr).squeeze()
      mae += torch.abs(out - data.y[idx]).mean()  # Check against ground-truth labels.
  return (mae / N).item()

model.to(device)

for epoch in range(NUM_EPOCHS):
  train(data, epoch)
  print('Test MAE:',test(data, data.test_mask))