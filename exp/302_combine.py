import os
import random
import sys
import uuid
from pathlib import Path
import shutil
import math
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from itertools import product
from sklearn.metrics import mean_squared_error

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


def cal_weighted_pred(dfs, weights):
    weighted_pred = np.zeros(len(dfs[0]))
    for i in range(len(weights)):
        weighted_pred += dfs[i].to_numpy().squeeze() * weights[i]
    return weighted_pred


def grid_search_weights(dfs, y_true):
    best_score = np.inf
    best_weights = None

    # 0.0から1.0まで0.1刻みで重みを試す
    for weights in product(np.arange(0.0, 1.1, 0.05), repeat=len(dfs)):
        if sum(weights) != 1.0:  # 重みの合計が1になる組み合わせだけ試す
            continue
        weighted_pred = cal_weighted_pred(dfs, weights)
        score = np.sqrt(mean_squared_error(y_true, weighted_pred))
        if score < best_score:
            best_score = score
            best_weights = weights
    return best_weights


def load_models(model_names, y_true):
    dfs_train = []
    dfs_test = []
    for model_name in model_names:
        df_train = pd.read_csv(f"../output/{model_name}/oof.csv")
        df_train.columns = [model_name]
        df_test = pd.read_csv(f"../output/{model_name}/sub.csv")
        df_test.columns = [model_name]
        dfs_train.append(df_train)
        dfs_test.append(df_test)

    def objective(weights):
        weighted_pred = cal_weighted_pred(dfs_train, weights)
        return np.sqrt(mean_squared_error(y_true, weighted_pred))

    constraints = {"type": "eq", "fun": lambda weights: 1 - sum(weights)}

    bounds = [(0, 1) for _ in range(len(dfs_train))]

    result = minimize(objective, [1.0 / len(dfs_train)] * len(dfs_train), bounds=bounds, constraints=constraints)
    weights = result.x

    print(f"Best weights: {weights}")

    train = pd.Series(cal_weighted_pred(dfs_train, weights))
    test = pd.Series(cal_weighted_pred(dfs_test, weights))
    return train, test


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    exp_name = f"{Path(sys.argv[0]).stem}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)
    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")

    seen_train_df, seen_test_df = load_models(config.combine.seen, train_df["score"])
    unseen_train_df, unseen_test_df = load_models(config.combine.unseen, train_df["score"])

    # testのうち unseenのindexを特定
    unseen_user_test_index = test_df[~test_df["user_id"].isin(train_df["user_id"])].index

    # seenとunseenの割合を計算
    total_len = len(test_df)
    unseen_len = len(unseen_user_test_index)
    seen_len = total_len - unseen_len
    rate_dict = {"seen_rate": seen_len / total_len, "unseen_rate": unseen_len / total_len}
    print(rate_dict)

    # oof を用いてスコアを算出する
    seen_score = evaluate_score(train_df["score"], seen_train_df, "rmse")
    unseen_score = evaluate_score(train_df["score"], unseen_train_df, "rmse")
    print(f"seen_score: {seen_score}")
    print(f"unseen_score: {unseen_score}")
    print(f"predicted score: {rate_dict['seen_rate']*seen_score + rate_dict['unseen_rate']*unseen_score}")
    print(
        f"predicted score2: {math.sqrt(rate_dict['seen_rate']*seen_score**2 + rate_dict['unseen_rate']*unseen_score**2)}"
    )

    # submissionを作る
    sub = load_sample_sub()
    sub["score"] = seen_test_df.to_numpy()  # まずはseenで埋める
    sub.loc[unseen_user_test_index, "score"] = unseen_test_df.loc[unseen_user_test_index].to_numpy()  # unseenで代入
    sub.to_csv(output_path / "sub.csv", index=False)


if __name__ == "__main__":
    main()
