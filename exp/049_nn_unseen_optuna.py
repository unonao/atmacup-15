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

import torch
import torch.nn as nn

import pandas as pd
from sklearn.preprocessing import StandardScaler
import optuna

exp_name = f"{Path(sys.argv[0]).stem}"
output_path = Path(f"../output/{exp_name}")


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
class MyNet(nn.Module):
    def __init__(self, n_features, n_layers, hidden_dim, dropout_rate=0.5):
        super(MyNet, self).__init__()

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


def objective(trial, config: DictConfig, X_train_all, X_test, y_train_all, train_user_ids):
    seed_everything(config.seed)
    # ハイパーパラメータのサジェスチョン
    n_layers = trial.suggest_int("n_layers", 1, 6)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 1024, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)

    # 以下の部分はmain関数からのコードを取り入れ、必要な部分を変更しています。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kf = StratifiedGroupKFold(n_splits=config.nn.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_all, y_train_all, train_user_ids)):
        if fold > 0:
            break
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # Prepare the datasets
        train_dataset = MyDataset(X_train.values, y_train.values)
        valid_dataset = MyDataset(X_valid.values, y_valid.values)

        # tqdmでDataLoaderをラップします
        train_loader = DataLoader(train_dataset, batch_size=config.nn.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.nn.batch_size, shuffle=False)

        # Adjust the model with suggested hyperparameters
        model = MyNet(
            n_features=X_train.shape[1], n_layers=n_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.nn.lr, weight_decay=weight_decay)

        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=config.nn.lr * 0.1
        )
        best_val_loss = float("inf")
        early_stopping_counter = 0

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
                train_loss += loss.item() ** 2
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
                    valid_loss += loss.item() ** 2

            train_loss = np.sqrt(train_loss / len(train_loader))
            valid_loss = np.sqrt(valid_loss / len(valid_loader))
            # Save the model if it has the best score so far
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.nn.early_stopping:
                    break
        print(
            {
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay,
                "best_val_loss": best_val_loss,
            }
        )
        return best_val_loss  # You might want to return a validation metric instead.


# メインの学習ループ
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    os.makedirs(output_path, exist_ok=True)

    X_train_all, X_test = load_datasets(config.nn.feats)
    y_train_all = load_target(config.nn.target_name)
    train_user_ids = load_target("user_id")

    n_trials = 50
    if config.debug:
        sample_index = X_train_all.sample(100).index
        X_train_all = X_train_all.iloc[sample_index].reset_index(drop=True)
        y_train_all = y_train_all.iloc[sample_index].reset_index(drop=True)
        train_user_ids = train_user_ids.iloc[sample_index].reset_index(drop=True)
        X_test = X_test.head(100)
        n_trials = 2

    X_train_all, X_test = prepare_data(X_train_all, X_test)

    sys.stdout = open(output_path / "log.txt", "w")

    study = optuna.create_study(direction="minimize")  # minimize the objective in this case
    study.optimize(
        lambda trial: objective(trial, config, X_train_all, X_test, y_train_all, train_user_ids), n_trials=n_trials
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
