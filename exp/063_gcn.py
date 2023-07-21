import os
import random
import sys
import uuid
from pathlib import Path
from typing import Optional, Union
import shutil

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
import torch.nn as nn
import torch_geometric
from cuml.preprocessing import StandardScaler
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

sys.path.append(os.pardir)
from utils import evaluate_score, load_datasets, load_sample_sub, load_target


# 同様のランダムシード設定関数
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


import torch
from torch_geometric.nn import SAGEConv, to_hetero, TransformerConv
from torch.nn import LayerNorm, Dropout


class GNNEncoder(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # 初めのレイヤー
        self.convs.append(TransformerConv((-1, -1), hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        # 中間のレイヤー
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv((-1, -1), hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        # 最後のレイヤー
        self.convs.append(TransformerConv((-1, -1), hidden_channels))
        self.norms.append(LayerNorm(out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index).relu()
            if i == 0:
                x_init = x
            else:
                x += x_init
            x = self.norms[i](x)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["user"][row], z_dict["anime"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, data_metadata, hidden_channels, num_layers=2):
        super().__init__()
        self.encoder = GNNEncoder(num_layers, hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data_metadata, aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    sample_submission_df = pd.read_csv(Path(config.input_path) / "sample_submission.csv")

    all_df = pd.concat([train_df[["user_id", "anime_id"]], test_df[["user_id", "anime_id"]]]).reset_index(drop=True)
    all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
    all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])
    all_df["is_train"] = True
    all_df.loc[len(train_df) :, "is_train"] = False

    edge_index = torch.stack(
        [torch.tensor(all_df["user_label"].values), torch.tensor(all_df["anime_label"].values)], dim=0
    )

    X_train_all, X_test = load_datasets(config.gcn.feats)

    # user feat
    features_df = pd.concat([X_train_all, X_test]).reset_index(drop=True)
    features_df["user_label"] = all_df["user_label"]
    user_features = features_df.groupby("user_label").mean().to_numpy()
    scaler = StandardScaler()
    scaled_user_features = np.nan_to_num(scaler.fit_transform(user_features), 0)  # nan をzero埋め

    # anime feat
    features_df = pd.concat([X_train_all, X_test]).reset_index(drop=True)
    features_df["anime_label"] = all_df["anime_label"]
    anime_features = features_df.groupby("anime_label").mean().to_numpy()
    scaler = StandardScaler()
    scaled_anime_features = np.nan_to_num(scaler.fit_transform(anime_features), 0)

    # data作成
    data = HeteroData()
    data["user"].x = torch.from_numpy(scaled_user_features).to(torch.float)
    data["anime"].x = torch.from_numpy(scaled_anime_features).to(torch.float)
    data["user", "score", "anime"].edge_index = edge_index
    all_df["score"] = np.nan  # テストの部分は np.nan　で埋めておく
    all_df.loc[: len(train_df), "score"] = train_df["score"]
    score = torch.from_numpy(all_df["score"].values).to(torch.float)
    data["user", "score", "anime"].edge_label = score
    data = T.ToUndirected()(data)
    del data["anime", "rev_score", "user"].edge_label

    wandb.init(
        project="atmacup-21",
        name=exp_name,  # {file}_{id}
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.gcn),
    )

    data = data.to(device)

    oof_pred = np.zeros(len(train_df))
    test_preds = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*k_fold(config.gcn.num_folds, all_df))):
        model = Model(
            data_metadata=data.metadata(), hidden_channels=config.gcn.hidden_channels, num_layers=config.gcn.num_layers
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.gcn.lr)

        best_val_loss = float("inf")
        early_stopping_counter = 0
        for epoch in tqdm(range(config.gcn.num_epochs if config.debug is False else 6), desc=f"Fold-{fold+1}"):
            # train
            model.train()
            optimizer.zero_grad()

            pred = model(
                data.x_dict,
                data.edge_index_dict,
                data["user", "anime"].edge_index[:, train_idx],
            )
            target = data["user", "anime"].edge_label[train_idx]
            loss = F.mse_loss(pred, target).sqrt()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            with torch.no_grad():
                pred = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data["user", "anime"].edge_index[:, val_idx],
                )
                target = data["user", "anime"].edge_label[val_idx]
                val_loss = F.mse_loss(pred, target).sqrt()

            wandb.log(
                {"epoch": epoch, f"loss/train/fold-{fold}": loss.item(), f"loss/valid/fold-{fold}": val_loss.item()}
            )
            if epoch % config.gcn.early_stopping == 0:
                tqdm.write(f"Epoch: {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_path / f"model_best_{fold}.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.gcn.early_stopping:
                    model.load_state_dict(torch.load(output_path / f"model_best_{fold}.pt"))
                    break

        # testing
        with torch.no_grad():
            oof_pred[val_idx.cpu().detach().numpy()] = (
                model(
                    data.x_dict,
                    data.edge_index_dict,
                    data["user", "anime"].edge_index[:, val_idx],
                )
                .cpu()
                .detach()
                .numpy()
            )
            test_pred = (
                model(
                    data.x_dict,
                    data.edge_index_dict,
                    data["user", "anime"].edge_index[:, test_idx],
                )
                .cpu()
                .detach()
                .numpy()
            )
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

    sample_submission_df.to_csv(output_path / "sub.csv", index=False)

    oof_df = pd.DataFrame({"score": oof_pred})
    oof_df.to_csv(output_path / "oof.csv", index=False)

    if config.debug:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    main()
