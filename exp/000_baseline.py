import os
import random
import sys
import uuid
from pathlib import Path
from typing import Optional, Union

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index
from tqdm.auto import tqdm


class LightGCN(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1.0 / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer("alpha", alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def k_fold(num_fold, all_df):
    skf = StratifiedKFold(num_fold, shuffle=True, random_state=12345)
    train_len = all_df["is_train"].sum()
    train_indices, val_indices, test_indices = [], [], []
    for _, idx in skf.split(torch.zeros(train_len), all_df.iloc[:train_len]["user_id"]):
        val_indices.append(torch.from_numpy(idx).to(torch.long))
        test_indices.append(torch.tensor(range(train_len, len(all_df))).to(torch.long))

    for i in range(num_fold):
        train_mask = torch.ones(train_len, dtype=torch.bool)
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, val_indices, test_indices


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{Path(sys.argv[0]).stem}_{str(uuid.uuid1())[:8]}"

    # data load
    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    sample_submission_df = pd.read_csv(Path(config.input_path) / "sample_submission.csv")
    anime_df = pd.read_csv(Path(config.input_path) / "anime.csv")

    # make Data
    all_df = pd.concat([train_df[["user_id", "anime_id"]], test_df[["user_id", "anime_id"]]]).reset_index(drop=True)
    all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
    all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])
    all_df["is_train"] = True
    all_df.loc[len(train_df) :, "is_train"] = False
    # userとanimeの番号が別になるようにずらす
    all_df["anime_label"] += len(user_idx)
    num_nodes = len(user_idx) + len(anime_idx)
    edges = all_df[["user_label", "anime_label"]].to_numpy()
    edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
    data = Data(num_nodes=num_nodes, edge_index=edge_index).to(device)
    data.edge_weight = torch.ones(len(all_df)).contiguous()

    # 学習
    wandb.init(
        project="atmacup-21",
        name=exp_name,  # {file}_{id}
        mode="online" if config.debug is False else "disabled",
        config=config.train,
    )

    oof_pred = np.zeros(len(train_df))
    test_preds = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*k_fold(config.train.num_folds, all_df))):
        model = LightGCN(
            num_nodes=data.num_nodes,
            embedding_dim=config.train.embedding_dim,
            num_layers=config.train.num_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch in tqdm(range(config.train.num_epochs if config.debug is False else 6), desc=f"Fold-{fold+1}"):
            # train
            model.train()
            optimizer.zero_grad()
            pred = model(data.edge_index[:, train_idx])
            target = torch.tensor(train_df.loc[train_idx.numpy(), "score"].to_numpy()).float().to(device)
            loss = F.mse_loss(pred, target).sqrt()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            with torch.no_grad():
                pred = model(data.edge_index[:, val_idx])
                target = torch.tensor(train_df.loc[val_idx.numpy(), "score"].to_numpy()).float().to(device)
                val_loss = F.mse_loss(pred, target).sqrt()
            wandb.log(
                {"epoch": epoch, f"loss/train/fold-{fold}": loss.item(), f"loss/valid/fold-{fold}": val_loss.item()}
            )
            if epoch % config.train.early_stopping == 0:
                tqdm.write(f"Epoch: {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"model_best_{fold}.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.train.early_stopping:
                    model.load_state_dict(torch.load(f"model_best_{fold}.pt"))
                    break

        # testing
        with torch.no_grad():
            oof_pred[val_idx.cpu().detach().numpy()] = model(data.edge_index[:, val_idx]).cpu().detach().numpy()
            test_pred = model(data.edge_index[:, test_idx]).cpu().detach().numpy()
            test_preds.append(test_pred)

    # calculate mean of predictions across all folds
    mean_test_preds = np.mean(test_preds, axis=0)
    # clip
    oof_pred = np.clip(oof_pred, 1.0, 10.0)
    mean_test_preds = np.clip(mean_test_preds, 1.0, 10.0)
    # calculate RMSE for oof predictions
    oof_rmse = mean_squared_error(train_df["score"], oof_pred, squared=False)
    wandb.log({"oof_rmse": oof_rmse})
    print({"oof_rmse": oof_rmse})
    wandb.finish()

    sample_submission_df["score"] = mean_test_preds
    sample_submission_df.to_csv(f"../output/sub_{exp_name}.csv", index=False)

    oof_df = pd.DataFrame({"score": oof_pred})
    oof_df.to_csv(f"../output/oof_{exp_name}.csv", index=False)


if __name__ == "__main__":
    main()
