import os
import random
import sys
import uuid
from pathlib import Path
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import numpy as np
import pandas as pd


sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


def load_models(model_names):
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"../output/{model_name}/oof.csv")
        df.columns = [model_name]
        dfs.append(df)
    train = pd.concat(dfs, axis=1, sort=False).mean(axis=1)

    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"../output/{model_name}/sub.csv")
        df.columns = [model_name]
        dfs.append(df)
    test = pd.concat(dfs, axis=1, sort=False).mean(axis=1)
    return train, test


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    exp_name = f"{Path(sys.argv[0]).stem}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    seen_train_df, seen_test_df = load_models(config.combine.seen)
    unseen_train_df, unseen_test_df = load_models(config.combine.unseen)

    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")

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
    print(f"predicted score: {seen_score*rate_dict['seen_rate'] + unseen_score*rate_dict['unseen_rate']}")

    # submissionを作る
    sub = load_sample_sub()
    sub["score"] = seen_test_df.to_numpy()  # まずはseenで埋める
    sub.loc[unseen_user_test_index, "score"] = unseen_test_df.loc[unseen_user_test_index].to_numpy()  # unseenで代入
    sub.to_csv(output_path / "sub.csv", index=False)


if __name__ == "__main__":
    main()
