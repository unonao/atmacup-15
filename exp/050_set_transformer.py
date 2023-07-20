import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
import uuid
from pathlib import Path
import random
import shutil
from tqdm.auto import tqdm
import random


sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub
from utils.model import SetTransformer


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AnimeDataset(Dataset):
    def __init__(self, num_animes, grouped_data, grouped_scores=None, num_samples=20):
        self.num_animes = num_animes
        self.data = list(grouped_data)
        self.scores = list(grouped_scores) if grouped_scores is not None else None
        self.num_samples = num_samples
        self.fill_score = 7.768770023680179  # 虚無データ用のスコア

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.num_samples == None:
            if self.scores is not None:
                return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(
                    self.scores[idx], dtype=torch.float32
                )
            else:
                return torch.tensor(self.data[idx], dtype=torch.long)

        # num_samples 件を超える場合、ランダムサンプリング
        # num_samples 件未満の場合、id: num_animes の虚無データ (スコアは平均値）で埋める
        data_len = len(self.data[idx])
        if data_len >= self.num_samples:
            indices = list(range(len(self.data[idx])))
            sampled_indices = random.sample(indices, self.num_samples)
            sampled_data = [self.data[idx][i] for i in sampled_indices]
            sampled_scores = [self.scores[idx][i] for i in sampled_indices]
        else:
            sampled_data = self.data[idx] + [self.num_animes for i in range(self.num_samples - data_len)]  # 虚無データ
            sampled_scores = self.scores[idx] + [self.fill_score for i in range(self.num_samples - data_len)]

        if self.scores is not None:
            return torch.tensor(sampled_data, dtype=torch.long), torch.tensor(sampled_scores, dtype=torch.float32)
        else:
            return torch.tensor(sampled_data, dtype=torch.long)


def make_chunk_index_list(n, k):
    """シャッフルして k 分割"""
    numbers = list(range(n))  # 0からN-1までのリストを作成
    random.shuffle(numbers)  # リストをシャッフル
    result = []
    for i in range(0, n, k):
        result.append(numbers[i : i + k])
    return result


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    wandb.init(
        project="atmacup-21-unseen",
        name=exp_name,
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.st),
    )

    # データをロード
    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    all_df = pd.concat([train_df, test_df])
    all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])  # 連番にする
    num_animes = len(anime_idx)  # ユニークなanime_idの数を取得
    train_df["anime_label"] = all_df[: len(train_df)]["anime_label"]
    test_df["anime_label"] = all_df[len(train_df) :]["anime_label"]
    train_user_ids = load_target("user_id")
    sub = load_sample_sub()

    # ハイパーパラメータ
    num_epochs = config.st.num_epochs
    num_samples = config.st.num_samples
    batch_size = config.st.batch_size
    embed_dim = config.st.embed_dim
    num_tts = config.st.num_tts
    dim_output = num_samples

    if config.debug:
        sample_index = train_df.sample(100).index
        train_df = train_df.iloc[sample_index].reset_index(drop=True)
        test_df = test_df.head(100)
        sub = sub.head(100)
        num_tts = 2

    oof_pred = np.zeros(train_df.shape[0])
    test_preds_all = []
    test_grouped_anime = test_df.groupby("user_id")["anime_label"].apply(list)

    kf = StratifiedGroupKFold(n_splits=config.st.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_df, train_df["score"], train_df["user_id"])):
        print(f"Fold {fold} start !")
        # user_idごとにanime_idのリストと平均スコアを取得
        train_grouped_anime = train_df.iloc[train_index].groupby("user_id")["anime_label"].apply(list)
        train_grouped_score = train_df.iloc[train_index].groupby("user_id")["score"].apply(list)
        valid_grouped_anime = train_df.iloc[valid_index].groupby("user_id")["anime_label"].apply(list)
        valid_grouped_score = train_df.iloc[valid_index].groupby("user_id")["score"].apply(list)
        test_grouped_anime = test_df.groupby("user_id")["anime_label"].apply(list)

        train_dataset = AnimeDataset(num_animes, train_grouped_anime, train_grouped_score, num_samples)
        valid_dataset = AnimeDataset(num_animes, valid_grouped_anime, valid_grouped_score, num_samples)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        model = SetTransformer(
            num_animes=num_animes,
            embed_dim=embed_dim,
            dim_output=dim_output,  # Since we want to predict scores, the dim_output is 1.
        ).to(device)

        criterion = nn.MSELoss()  # Using MSE as the loss for RMSE
        optimizer = torch.optim.Adam(model.parameters(), lr=config.st.lr, weight_decay=config.st.weight_decay)

        best_val_loss = float("inf")
        early_stopping_counter = 0

        print("Train")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            valid_loss = 0
            for _ in range(config.st.num_aug):
                for i, (data, scores) in enumerate(train_loader):
                    data, scores = data.to(device), scores.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = torch.sqrt(criterion(outputs, scores))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() ** 2

                # Validate the model
                model.eval()
                with torch.no_grad():
                    for i, (data, scores) in enumerate(valid_loader):
                        data, scores = data.to(device), scores.to(device)
                        outputs = model(data)
                        loss = torch.sqrt(criterion(outputs.squeeze(), scores))
                        valid_loss += loss.item() ** 2

            train_loss = np.sqrt(train_loss / len(train_loader) / config.st.num_aug)
            valid_loss = np.sqrt(valid_loss / len(valid_loader) / config.st.num_aug)

            print(f"Epoch{epoch}: Training Loss {train_loss},  Validation Loss {valid_loss}")
            wandb.log(
                {"epoch": epoch, f"nn/train_loss/fold-{fold}": train_loss, f"nn/valid_loss/fold-{fold}": valid_loss}
            )

            # Save the model if it has the best score so far
            if valid_loss < best_val_loss:
                torch.save(model.state_dict(), f"{output_path}/best_model.pt")
                best_val_loss = valid_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.st.early_stopping:
                    break

        model.load_state_dict(torch.load(f"{output_path}/best_model.pt"))
        model.eval()
        valid_dataset = AnimeDataset(num_animes, valid_grouped_anime, valid_grouped_score, None)
        test_dataset = AnimeDataset(num_animes, test_grouped_anime, None, None)
        valid_preds = []
        test_preds = []

        with torch.no_grad():
            print("Infer valid")
            for idx in tqdm(range(len(valid_dataset))):
                valid_pred = []
                for itts in range(num_tts):  # indexをシャッフルしてn個ずつに分割して推論を繰り返す
                    data, _ = valid_dataset[idx]
                    data_len = len(data)
                    one_time_pred = np.zeros(len(data))
                    data, scores = data.to(device), scores.to(device)

                    if data_len <= num_samples:  # １回でOK
                        one_time_pred = model(data.unsqueeze(dim=0)).cpu().numpy().flatten()[:data_len]
                        valid_pred.append(one_time_pred)
                        break
                    else:
                        chunk_index_list = make_chunk_index_list(len(data), num_samples)
                        for chunk_index in chunk_index_list:
                            one_time_pred[chunk_index] = (
                                model(data[chunk_index].unsqueeze(dim=0)).cpu().numpy().flatten()
                            )
                        valid_pred.append(one_time_pred)
                valid_preds.append(np.mean(valid_pred, axis=0))

            oof_pred[valid_index] = np.concatenate(valid_preds)  # group 化してあるものを 1d に戻す
            # この時点での valid_preds を利用したスコア算出
            rmse_valid = evaluate_score(train_df.iloc[valid_index]["score"], oof_pred[valid_index], "rmse")
            print(f"rmse_valid: {rmse_valid}")

            print("Infer test")
            for idx in tqdm(range(len(test_dataset))):
                test_pred = []
                for itts in range(num_tts):  # indexをシャッフルしてn個ずつに分割して推論を繰り返す
                    data = test_dataset[idx]
                    data_len = len(data)
                    one_time_pred = np.zeros(len(data))
                    data, scores = data.to(device), scores.to(device)

                    if data_len <= num_samples:  # １回でOK
                        one_time_pred = model(data.unsqueeze(dim=0)).cpu().numpy().flatten()[:data_len]
                        test_pred.append(one_time_pred)
                        break
                    else:
                        chunk_index_list = make_chunk_index_list(len(data), num_samples)
                        for chunk_index in chunk_index_list:
                            one_time_pred[chunk_index] = (
                                model(data[chunk_index].unsqueeze(dim=0)).cpu().numpy().flatten()
                            )
                        test_pred.append(one_time_pred)
                test_preds.append(np.mean(test_pred, axis=0))

        test_preds_all.append(np.concatenate(test_preds))

    mean_y_preds = np.mean(test_preds_all, axis=0)

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

    sub[config.st.target_name] = mean_y_preds
    sub.to_csv(output_path / "sub.csv", index=False)

    print(f"saved: {output_path}")
    if config.debug:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    main()
