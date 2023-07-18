import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
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

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


# 同様のランダムシード設定関数
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# データセットの定義
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ネットワークの定義
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, n_features, n_layers, dropout_rate=0.5):
        super(MyNet, self).__init__()

        hidden_dim = n_features // 2

        layers = []
        layers.append(nn.Linear(n_features, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout層を追加
        self.fc_layers = nn.Sequential(*layers)

        self.fc_final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.fc_layers(x)
        out = self.fc_final(h)
        return out


import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(X_train_all, X_test):
    # データ結合
    combined_df = pd.concat([X_train_all, X_test], axis=0)

    # 欠損値の平均値で埋める
    combined_df_filled = combined_df.fillna(combined_df.mean())

    # 正規化
    scaler = StandardScaler()
    combined_df_normalized = pd.DataFrame(scaler.fit_transform(combined_df_filled), columns=combined_df_filled.columns)

    # 結果のデータフレームを分割して返す
    X_train_processed = combined_df_normalized[: len(X_train_all)]
    X_test_processed = combined_df_normalized[len(X_train_all) :]

    return X_train_processed, X_test_processed


# メインの学習ループ
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    wandb.init(
        project="atmacup-21",
        name=exp_name,
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.nn),
    )

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(config.nn.feats)
    y_train_all = load_target(config.nn.target_name)
    train_user_ids = load_target("user_id")
    sub = load_sample_sub()

    if config.debug:
        sample_index = X_train_all.sample(100).index
        X_train_all = X_train_all.iloc[sample_index].reset_index(drop=True)
        y_train_all = y_train_all.iloc[sample_index].reset_index(drop=True)
        train_user_ids = train_user_ids.iloc[sample_index].reset_index(drop=True)
        X_test = X_test.head(100)
        sub = sub.head(100)
    X_train_all, X_test = prepare_data(X_train_all, X_test)

    oof_pred = np.zeros(X_train_all.shape[0])
    test_preds = []

    kf = StratifiedKFold(n_splits=config.nn.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_all, y_train_all, train_user_ids)):
        print(f"Fold {fold} start !")
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # Prepare the datasets
        train_dataset = MyDataset(X_train.values, y_train.values)
        valid_dataset = MyDataset(X_valid.values, y_valid.values)
        test_dataset = MyDataset(X_test.values, np.zeros(X_test.shape[0]))

        # tqdmでDataLoaderをラップします
        train_loader = DataLoader(train_dataset, batch_size=config.nn.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.nn.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.nn.batch_size, shuffle=False)

        # Define the model
        model = MyNet(
            n_features=X_train.shape[1], n_layers=config.nn.num_layers, dropout_rate=config.nn.dropout_rate
        ).to(device)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.nn.lr, weight_decay=config.nn.weight_decay)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=config.nn.lr * 0.1
        )

        best_val_loss = float("inf")
        early_stopping_counter = 0
        print("Train")
        # Train the model
        for epoch in range(config.nn.num_epochs):
            model.train()

            train_loss = 0
            for i, (inputs, targets) in enumerate(tqdm(train_loader)):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)

                # Forward pass
                outputs = model(inputs)
                loss = torch.sqrt(criterion(outputs.squeeze(), targets))

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            lr_sched.step()

            # Validate the model
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(valid_loader):
                    inputs = inputs.float().to(device)
                    targets = targets.float().to(device)
                    outputs = model(inputs)
                    loss = torch.sqrt(criterion(outputs.squeeze(), targets))
                    valid_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
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
                if early_stopping_counter >= config.nn.early_stopping:
                    break

        # best model での推論
        model.load_state_dict(torch.load(f"{output_path}/best_model.pt"))
        model.eval()
        valid_preds = []
        test_pred = []
        with torch.no_grad():
            for i, (inputs, _) in enumerate(valid_loader):
                inputs = inputs.float().to(device)
                outputs = model(inputs)
                valid_preds.extend(outputs.squeeze().tolist())
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.float().to(device)
                outputs = model(inputs)
                test_pred.extend(outputs.squeeze().tolist())

        oof_pred[valid_index] = valid_preds
        test_preds.append(test_pred)
        print("\n")
    mean_y_preds = np.mean(test_preds, axis=0)

    # 範囲内にする
    oof_pred = np.clip(oof_pred, 1.0, 10.0)
    mean_y_preds = np.clip(mean_y_preds, 1.0, 10.0)

    # CVスコア確認
    print("===CV scores===")
    rmse_all_valid = evaluate_score(y_train_all, oof_pred, "rmse")
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
