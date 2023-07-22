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

from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from lightgbm import LGBMRegressor

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub

model_dict = {"linear": LinearRegression(), "elastic": ElasticNet(alpha=0.1, l1_ratio=0.1), "lgb": LGBMRegressor()}


def load_models(config, model_names, y_true):
    dfs_train = []
    dfs_test = []
    for model_name in model_names:
        df_train = pd.read_csv(f"../output/{model_name}/oof.csv")
        df_train.columns = [model_name]
        df_test = pd.read_csv(f"../output/{model_name}/sub.csv")
        df_test.columns = [model_name]
        dfs_train.append(df_train)
        dfs_test.append(df_test)

    train_x = pd.concat(dfs_train, axis=1)
    test_x = pd.concat(dfs_test, axis=1)

    preds = np.zeros(len(train_x))
    preds_test = np.zeros(len(test_x))
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=8)
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = y_true.iloc[tr_idx], y_true.iloc[va_idx]
        model = model_dict[config.model]
        model.fit(tr_x, tr_y)
        pred = model.predict(va_x)
        preds[va_idx] = pred
        pred_test = model.predict(test_x)
        preds_test += pred_test / n_splits

    preds = np.clip(preds, 1, 10)
    preds_test = np.clip(preds_test, 1, 10)

    train = pd.Series(preds)
    test = pd.Series(preds_test)
    return train, test


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    print()
    exp_name = f"{Path(sys.argv[0]).stem}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)
    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")

    seen_train_df, seen_test_df = load_models(config, config.combine.seen, train_df["score"])
    unseen_train_df, unseen_test_df = load_models(config, config.combine.unseen, train_df["score"])

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

    print()


if __name__ == "__main__":
    main()
