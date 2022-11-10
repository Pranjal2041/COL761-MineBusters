import os
from torch_geometric.nn import Node2Vec
import torch
from tqdm import tqdm
from torch_geometric_temporal.nn.attention.gman import GMAN, SpatioTemporalEmbedding
from data import STGMAN_Dataset, process_dataset
from trainer import save_model, train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

DATASET = os.environ.get("DATASET", 1)  # 1 or 2
NUM_EPOCHS = 15

P = 12
F = 12
K = 2
d = 2
L = 2
LR = 5e-4
BN_DECAY = 0.99

SE_SAVE_PATH = f"./data/d{DATASET}_SE_{K*d}.pth"


def node2vec(data):

    node_vectorizer = Node2Vec(
        data.edge_index,
        embedding_dim=K * d,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    loader = node_vectorizer.loader(batch_size=128, shuffle=True, num_workers=1)

    def vectorizer_train_epoch(model, loader):
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def vectorizer_train():
        pbar = tqdm(range(1, 101), desc="Training")
        for _ in pbar:
            loss = vectorizer_train_epoch(node_vectorizer, loader)
            pbar.set_description_str(f"Loss: {loss:.4f}")
        pbar.close()
        return node_vectorizer

    node_vectorizer = vectorizer_train()
    z = node_vectorizer()
    torch.save(z, SE_SAVE_PATH)
    return z


class ST_Embedding(SpatioTemporalEmbedding):
    def forward(
        self, SE: torch.FloatTensor, TE: torch.FloatTensor, T: int
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.

        Arg types:
                        * **SE** (PyTorch Float Tensor) - Spatial embedding, with shape (num_nodes, D).
                        * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, D)
                        * **T** (int) - Number of time steps in one day.

        Return types:
                        * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self._fully_connected_se(SE)
        TE = TE.unsqueeze(2)

        output = SE + TE
        return output


class ST_GMAN(torch.nn.Module):
    def __init__(
        self,
        L: int,
        K: int,
        d: int,
        num_his: int,
        num_pred: int,
        bn_decay: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        D = K * d
        self.num_his = num_his
        self.num_pred = num_pred
        self.temporal_embedding = torch.nn.Embedding(num_his + num_pred, D)
        self.gman = GMAN(
            L=L,
            K=K,
            d=d,
            num_his=num_his,
            bn_decay=bn_decay,
            use_bias=use_bias,
            mask=False,
            steps_per_day=1,
        )
        self.gman._st_embedding = ST_Embedding(
            D=D, bn_decay=bn_decay, steps_per_day=1, use_bias=use_bias
        )
        self.loss_fct = torch.nn.L1Loss()

    def forward(
        self,
        X: torch.FloatTensor,
        SE: torch.FloatTensor,
        mask: torch.BoolTensor,
        labels: torch.FloatTensor = None,
    ):
        """
        Making a forward pass of GMAN.

        Arg types:
                * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_hist, num of nodes).
                * **SE** (Pytorch Float Tensor) - Spatial embedding, with shape (number of nodes, D).

        Return types:
                * **X** (PyTorch Float Tensor) - Output sequence for prediction, with shape (batch_size, num_pred, num of nodes).
        """
        TE = self.temporal_embedding(
            torch.arange(self.num_his + self.num_pred)
            .unsqueeze(0)
            .repeat(X.shape[0], 1)
            .to(device)
        )
        X = self.gman(X, SE, TE)
        if labels is not None:
            loss = self.loss_fct(X[..., mask], labels[..., mask])
            return loss, X
        return X


if __name__ == "__main__":
    data = process_dataset(DATASET, p=P, f=F, do_standardize=True)
    if os.path.exists(SE_SAVE_PATH):
        z = torch.load(SE_SAVE_PATH)
    else:
        print("Running Node2Vec ....")
        z = node2vec(data)

    dataset = STGMAN_Dataset(data=data, SE=z)
    stgman = ST_GMAN(
        L=L, K=K, d=d, num_his=P, num_pred=F, bn_decay=BN_DECAY, use_bias=True
    ).to(device)

    optimizer = torch.optim.Adam(stgman.parameters(), lr=LR, weight_decay=5e-4)
    stgman = train_model(
        model=stgman, optimizer=optimizer, data_loader=dataset, num_epochs=NUM_EPOCHS
    )
    save_model(
        model=stgman,
        data_loader=data,
        path=f"models/d{DATASET}_P{P}_F{F}/stgman",
    )
