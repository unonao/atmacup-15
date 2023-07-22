import logging
import os
import pickle
import random
import sys
import time
import uuid
import warnings
from glob import glob
from pathlib import Path
from typing import Any, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import shutil

import implicit
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from timm.scheduler import CosineLRScheduler
from timm.utils import AverageMeter
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 最大表示列数の指定（ここでは50列を指定）
pd.set_option("display.max_columns", 50)

sys.path.append(os.pardir)

from utils import evaluate_score, load_datasets, load_sample_sub, load_target


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class UserDataset(Dataset):
    def __init__(self, merge_df: pd.DataFrame, max_padding: int = 531):
        """
        merge_df: すべてのデータを結合したもの。以下のカラムを持つ。
        - user_label: 0-indexed にした user_id
        - anime_label: 0-indexed にした anime_id
        - mode: その行について trainは1, validationは2, testは3 にしたもの
        - score: testに関しては適当な値(0)で良い
        """
        self.merge_df = merge_df
        self.max_padding = max_padding
        self.user2anime_dict = merge_df.groupby("user_label")["anime_label"].apply(list).to_dict()
        self.user2mode_dict = merge_df.groupby("user_label")["mode"].apply(list).to_dict()
        self.user2score = merge_df.groupby("user_label")["score"].apply(list).to_dict()

    def __len__(self):
        return self.merge_df["user_label"].nunique()

    def __getitem__(self, idx):
        """
        出力したいもの
        - input_tensor: user_id, anime_id 系列を結合したもの
        - mode_tensor: user_idか、train用の anime_id か、validation用のanime_idか、test用のanime_id かを判断するためのもの。
        損失計算の対象を決めるために設定する。{user_id: 0, train:1, valid:2, test:3}
        - attention_mask: 計算対象外のpaddingの位置をtransformerに教えるために必要
        - score_tensor: ラベルとなるスコア情報。ラベルが無いものは適当に0で埋めるが使わない
        """
        user_tensor = torch.Tensor([idx]).int()
        anime_tensor = torch.Tensor(self.user2anime_dict[idx]).int()
        mode_tensor = torch.Tensor(self.user2mode_dict[idx]).int()
        score_tensor = torch.Tensor(self.user2score[idx]).float()

        # ランダムに順序を変更する
        indices = torch.randperm(anime_tensor.size(0))
        anime_tensor = anime_tensor[indices]
        mode_tensor = mode_tensor[indices]
        score_tensor = score_tensor[indices]

        pad_length = self.max_padding - anime_tensor.size(0)

        """
        # unseen用 (user_tensorは入れない）
        attention_mask = torch.zeros([self.max_padding, self.max_padding], dtype=torch.bool)
        attention_mask[: anime_tensor.size(0), : anime_tensor.size(0)] = True
        input_tensor = torch.cat((anime_tensor, torch.zeros(pad_length, dtype=torch.int32)))
        mode_tensor = torch.cat(
            (
                mode_tensor,
                torch.zeros(pad_length, dtype=torch.int32),
            )
        )
        score_tensor = torch.cat(
            (
                score_tensor,
                torch.zeros(pad_length, dtype=torch.float),
            )
        )

        """
        # seen用
        attention_mask = torch.zeros([self.max_padding], dtype=torch.bool)
        attention_mask[: anime_tensor.size(0)] = True
        input_tensor = torch.cat((user_tensor, anime_tensor, torch.zeros(pad_length, dtype=torch.int32)))
        mode_tensor = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                mode_tensor,
                torch.zeros(pad_length, dtype=torch.int32),
            )
        )
        score_tensor = torch.cat(
            (
                torch.zeros(1, dtype=torch.float),
                score_tensor,
                torch.zeros(pad_length, dtype=torch.float),
            )
        )
        sample = {
            "user_ids": user_tensor,
            "input_tensor": input_tensor,
            "mode_tensor": mode_tensor,
            "attention_mask": attention_mask,
            "score_tensor": score_tensor,
        }
        return sample


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_layers=2,
        hidden_size: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 1024,
    ):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size

        # embedding
        self.user_embedding = nn.Embedding(2000, hidden_size)
        self.anime_embedding = nn.Embedding(2000, hidden_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x = self.anime_embedding(x[:, :])
        """
        # seen用
        user_x = self.user_embedding(x[:, 0:1])
        anime_x = self.anime_embedding(x[:, 1:])
        x = torch.cat([user_x, anime_x], dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        output = self.fc(x).squeeze(2)
        return output

    def get_losses(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mode_tensor: torch.Tensor,
        mode: int = 1,
    ) -> float:
        loss_fn = nn.MSELoss()
        loss = loss_fn(input[mode_tensor == mode], target[mode_tensor == mode])
        loss = torch.sqrt(loss)
        return loss

    # メインの学習ループ


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    anime = pd.read_csv(Path(config.input_path) / "anime.csv")
    train_user_ids = load_target("user_id")
    sub = load_sample_sub()

    if config.debug:
        n = 100
        sample_index = train_df.sample(n).index
        train_df = train_df.iloc[sample_index].reset_index(drop=True)
        test_df = test_df.head(n)
        train_user_ids = train_user_ids.iloc[sample_index].reset_index(drop=True)
        sub = sub.head(n)

    # Merge the train data with the anime meta data
    all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    # 0-indexedの連番にする
    all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
    all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])

    wandb.init(
        project="atmacup-21",
        name=exp_name,
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.tvtt),
    )

    def train_one_epoch(
        cfg,
        count_steps,
        current_epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        dataloader: torch.utils.data.DataLoader,
        n_iter: int,
        scaler: GradScaler,
    ):
        start_time = time.time()

        model.train()
        torch.set_grad_enabled(True)

        meters = {"loss_avg": AverageMeter()}
        for step, data in enumerate(dataloader):
            for k, v in data.items():
                data[k] = v.to(device)

            with autocast(enabled=cfg.use_amp):
                output = model(data["input_tensor"], data["attention_mask"])
                loss = model.get_losses(output, data["score_tensor"], data["mode_tensor"], 1)

            if cfg.accumulation_steps > 1:
                loss_bw = loss / cfg.accumulation_steps
                scaler.scale(loss_bw).backward()
                if cfg.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                if (step + 1) % cfg.accumulation_steps == 0 or step == n_iter:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step(count_steps)
                    optimizer.zero_grad()
                    count_steps += 1
            else:
                scaler.scale(loss).backward()
                if cfg.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(count_steps)
                optimizer.zero_grad()
                count_steps += 1

            meters["loss_avg"].update(loss.item(), dataloader.batch_size)

        return (meters["loss_avg"].avg, (time.time() - start_time) / 60)

    def val_one_epoch(
        cfg,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, float, float]:
        start_time = time.time()

        model.eval()
        torch.set_grad_enabled(False)

        meters = {"loss_avg": AverageMeter()}
        preds = []
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                for k, v in data.items():
                    data[k] = v.to(device)
                output = model(data["input_tensor"], data["attention_mask"])
                if (data["mode_tensor"] == 2).sum() > 0:
                    loss = model.get_losses(output, data["score_tensor"], data["mode_tensor"], 2)
                    preds.append(output.detach().cpu().numpy())
                    meters["loss_avg"].update(loss.item(), dataloader.batch_size)

        preds = np.concatenate(preds, axis=0)
        return (preds, meters["loss_avg"].avg, (time.time() - start_time) / 60)

    def infer(
        cfg,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        model.eval()
        torch.set_grad_enabled(False)

        pred_dict_val = {"user_label": [], "anime_label": [], "score": []}
        pred_dict = {"user_label": [], "anime_label": [], "score": []}
        with torch.no_grad():
            for step, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
                for k, v in data.items():
                    data[k] = v.to(device)
                output = model(data["input_tensor"], data["attention_mask"])

                data["user_ids"] = data["user_ids"].detach().cpu().numpy()
                data["input_tensor"] = data["input_tensor"].detach().cpu().numpy()
                data["score_tensor"] = data["score_tensor"].detach().cpu().numpy()
                data["mode_tensor"] = data["mode_tensor"].detach().cpu().numpy()
                output = output.detach().cpu().numpy()

                """
                user_tensor = np.copy(data["input_tensor"])
                user_tensor[:, :] = data["user_ids"][:, 0:1]  # user_id の値を突っ込む
                """
                user_tensor = np.copy(data["input_tensor"])
                user_tensor[:, 1:] = user_tensor[:, 0:1]

                pred_dict_val["user_label"].append(user_tensor[data["mode_tensor"] == 2])
                pred_dict_val["anime_label"].append(data["input_tensor"][data["mode_tensor"] == 2])
                pred_dict_val["score"].append(output[data["mode_tensor"] == 2])

                pred_dict["user_label"].append(user_tensor[data["mode_tensor"] == 3])
                pred_dict["anime_label"].append(data["input_tensor"][data["mode_tensor"] == 3])
                pred_dict["score"].append(output[data["mode_tensor"] == 3])

        pred_dict_val["user_label"] = np.concatenate(pred_dict_val["user_label"])
        pred_dict_val["anime_label"] = np.concatenate(pred_dict_val["anime_label"])
        pred_dict_val["score"] = np.concatenate(pred_dict_val["score"])
        pred_dict["user_label"] = np.concatenate(pred_dict["user_label"])
        pred_dict["anime_label"] = np.concatenate(pred_dict["anime_label"])
        pred_dict["score"] = np.concatenate(pred_dict["score"])
        return (pred_dict_val, pred_dict, (time.time() - start_time) / 60)

    all_df["mode"] = 0  # 初期化
    pred_dict_list = []
    pred_dict_val_list = []
    kf = StratifiedGroupKFold(n_splits=config.tvtt.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_df, train_df["score"], train_user_ids)):
        print(f"Fold {fold} start !")

        # ここで、trainとvalidに分ける。foldごとにデータセットを作らないとおかしくなるので注意
        # all_df[:len(train_df)] までのデータのmodeを決定
        all_df.loc[train_index, "mode"] = 1
        all_df.loc[valid_index, "mode"] = 2
        # all_df[len(train_df):] のデータのmode (test) を埋める
        all_df.loc[len(train_df) :, "mode"] = 3

        dataset = UserDataset(all_df)
        model = TransformerModel(
            hidden_size=config.tvtt.hidden_size, nhead=config.tvtt.nhead, dim_feedforward=config.tvtt.dim_feedforward
        )
        model.to(device)

        train_loader = DataLoader(
            dataset,
            batch_size=config.tvtt.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            dataset,
            batch_size=config.tvtt.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        # other
        best_loss = 1e9
        num_train_iter = len(train_loader)

        # optimizer & scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.tvtt.init_lr))

        num_train_optimization_steps = (
            int(len(train_loader) * config.tvtt.n_epochs // config.tvtt.accumulation_steps) + 1
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_train_optimization_steps,
            lr_min=float(config.tvtt.final_lr),
            warmup_t=int(num_train_optimization_steps * config.tvtt.num_warmup_steps_rate),
            warmup_lr_init=config.tvtt.warmup_lr_init,
            warmup_prefix=True,
        )
        scaler = GradScaler(enabled=config.tvtt.use_amp)

        # Training
        count_steps = 0
        early_stopping_counter = 0
        for epoch in tqdm(range(1, config.tvtt.n_epochs + 1 if config.debug is False else 2)):
            train_loss, train_time = train_one_epoch(
                config.tvtt,
                count_steps,
                current_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                n_iter=num_train_iter,
                scaler=scaler,
            )
            val_preds, val_loss, val_time = val_one_epoch(cfg=config.tvtt, model=model, dataloader=eval_loader)
            tqdm.write(f"Epoch {epoch} : Train loss {train_loss:.3f} Val loss: {val_loss:.3f}")
            wandb.log(
                {"epoch": epoch, f"nn/train_loss/fold-{fold}": train_loss, f"nn/valid_loss/fold-{fold}": val_loss}
            )

            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(
                    {"state_dict": model.state_dict()},
                    output_path / "best.pth",
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.tvtt.early_stopping:
                    break

        for _ in range(config.tvtt.num_tts):
            # Infernce
            model.load_state_dict(torch.load(output_path / "best.pth")["state_dict"])
            pred_dict_val, pred_dict, _ = infer(config.tvtt, model, eval_loader)

            pred_dict_val_list.append(pred_dict_val)
            pred_dict_list.append(pred_dict)

    result_dict = {"user_label": [], "anime_label": [], "score": []}

    for pred_dict_val in pred_dict_val_list:
        for key, val in pred_dict_val.items():
            result_dict[key] += list(val)

    for pred_dict in pred_dict_list:
        for key, val in pred_dict.items():
            result_dict[key] += list(val)

    result_df = pd.DataFrame(result_dict)
    result_df = result_df.groupby(["user_label", "anime_label"], as_index=False).mean()

    oof_df = all_df[["user_label", "anime_label"]].iloc[: len(train_df)].copy()
    test_preds_df = all_df[["user_label", "anime_label"]].iloc[len(train_df) :].copy()

    oof_df = oof_df.merge(result_df, on=["user_label", "anime_label"], how="left")
    print(oof_df["score"].isnull().sum())
    test_preds_df = test_preds_df.merge(result_df, on=["user_label", "anime_label"], how="left")
    print(test_preds_df["score"].isnull().sum())

    oof_pred = oof_df["score"].to_numpy()
    mean_y_preds = test_preds_df["score"].to_numpy()

    # 範囲内にする
    oof_pred = np.clip(oof_pred, 1.0, 10.0)
    mean_y_preds = np.clip(mean_y_preds, 1.0, 10.0)

    # CVスコア確認
    print("===CV scores===")
    rmse_all_valid = evaluate_score(train_df["score"], oof_pred, "rmse")
    print({"nn/rmse/all_val": rmse_all_valid})
    wandb.log({"nn/rmse/all_val": rmse_all_valid})

    # 保存
    oof_df = pd.DataFrame({"score": oof_pred})
    oof_df.to_csv(output_path / "oof.csv", index=False)

    sub[config.nn.target_name] = mean_y_preds
    sub.to_csv(output_path / "sub.csv", index=False)

    print(f"saved: {output_path}")
    if config.debug:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    main()
