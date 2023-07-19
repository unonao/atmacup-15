import optuna
from optuna.integration import LightGBMPruningCallback

import os
import random
import sys
import uuid
from pathlib import Path
import shutil
import wandb
from wandb.lightgbm import wandb_callback, log_summary

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def objective(trial, X, y, config):
    seed_everything()
    # LightGBM parameters
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": trial.suggest_int("num_leaves", 32, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000),
        "max_depth": -1,
        "subsample_freq": 0,
        "bagging_seed": 0,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_bin": 200,
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "device": "gpu",  # Use GPU
    }

    # Cross-validation with the specified stratified K-fold
    kf = StratifiedKFold(n_splits=config.lgb.num_folds, shuffle=True, random_state=config.seed)

    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        if fold > 0:
            break

        X_train, X_valid = (X.iloc[train_index, :], X.iloc[valid_index, :])
        y_train, y_valid = (y.iloc[train_index], y.iloc[valid_index])

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_valid, y_valid)

        model_lgb = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=100), LightGBMPruningCallback(trial, "rmse")],
        )

        y_valid_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

        rmse_valid = evaluate_score(y_valid, y_valid_pred, "rmse")
        return rmse_valid


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything()
    exp_name = f"{Path(sys.argv[0]).stem}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(config.lgb.feats)
    y_train_all = load_target(config.lgb.target_name)
    sub = load_sample_sub()

    n_trials = 100
    if config.debug:
        sample_index = X_train_all.sample(100).index
        X_train_all = X_train_all.iloc[sample_index].reset_index(drop=True)
        y_train_all = y_train_all.iloc[sample_index].reset_index(drop=True)
        X_test = X_test.head(100)
        sub = sub.head(100)
        n_trials = 2

    sys.stdout = open(output_path / "log.txt", "w")

    # Using optuna for hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_all, y_train_all, config), n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
