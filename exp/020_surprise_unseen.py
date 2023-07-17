import numpy as np
import pandas as pd
import wandb
from surprise import Dataset, Reader, SVD
from surprise import (
    NormalPredictor,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SVD,
    SVDpp,
    NMF,
    SlopeOne,
    CoClustering,
)
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Optional, Union
import shutil
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedGroupKFold

import hydra

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


algo_dict = {
    "BaselineOnly": BaselineOnly,
    "KNNWithMeans": KNNWithMeans,
    "KNNWithZScore": KNNWithZScore,
    "KNNBaseline": KNNBaseline,
    "SVD": SVD,
    "SVDpp": SVDpp,
    "SlopeOne": SlopeOne,
    "CoClustering": CoClustering,
    "KNNBasic": KNNBasic,
    "NMF": NMF,
    "NormalPredictor": NormalPredictor,
}


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    wandb.init(
        project="atmacup-21-unseen",
        name=exp_name,
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.surprise),
    )

    # Define a Reader object
    reader = Reader(rating_scale=(1, 10))

    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    sub = pd.read_csv(Path(config.input_path) / "sample_submission.csv")

    if config.debug:
        sample_index = train_df.sample(100).index
        train_df = train_df.iloc[sample_index].reset_index(drop=True)
        test_df = test_df.head(100)
        sub = sub.head(100)

    test_df["score"] = 0

    oof_pred = np.zeros(train_df.shape[0])
    y_preds = []

    kf = StratifiedGroupKFold(n_splits=config.lgb.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_df, train_df["score"], train_df["user_id"])):
        X_train, X_valid = (train_df.iloc[train_index, :], train_df.iloc[valid_index, :])
        reader = Reader(rating_scale=(1, 10))
        train_data = Dataset.load_from_df(X_train[["user_id", "anime_id", "score"]], reader).build_full_trainset()
        valid_set = (
            Dataset.load_from_df(X_valid[["user_id", "anime_id", "score"]], reader)
            .build_full_trainset()
            .build_testset()
        )
        test_set = (
            Dataset.load_from_df(test_df[["user_id", "anime_id", "score"]], reader)
            .build_full_trainset()
            .build_testset()
        )

        algo = algo_dict[config.surprise.model]()
        _ = algo.fit(train_data)

        y_valid_pred = algo.test(valid_set)
        oof_pred[valid_index] = [pred.est for pred in y_valid_pred]
        y_pred = algo.test(test_set)
        y_preds.append([pred.est for pred in y_pred])
    mean_y_preds = np.mean(y_preds, axis=0)

    # 範囲内にする
    oof_pred = np.clip(oof_pred, 1.0, 10.0)
    mean_y_preds = np.clip(mean_y_preds, 1.0, 10.0)

    # CVスコア確認
    print("===CV scores===")
    rmse_all_valid = evaluate_score(train_df["score"], oof_pred, "rmse")
    print({f"oof_rmse": rmse_all_valid})
    wandb.log({f"oof_rmse": rmse_all_valid})

    # 保存
    oof_df = pd.DataFrame({"score": oof_pred})
    oof_df.to_csv(output_path / "oof.csv", index=False)

    sub["score"] = mean_y_preds
    sub.to_csv(output_path / "sub.csv", index=False)

    print(f"saved: {output_path}")
    if config.debug:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    main()
