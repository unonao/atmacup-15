"""for seen"""
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
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.pardir)
from utils import load_datasets, load_target, evaluate_score, load_sample_sub


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything()
    exp_name = f"{Path(sys.argv[0]).stem}/{str(uuid.uuid1())[:8]}"
    output_path = Path(f"../output/{exp_name}")
    os.makedirs(output_path, exist_ok=True)

    wandb.init(
        project="atmacup-21",
        name=exp_name,
        mode="online" if config.debug is False else "disabled",
        config=OmegaConf.to_container(config.lgb),
    )

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(config.lgb.feats)
    y_train_all = load_target(config.lgb.target_name)
    sub = load_sample_sub()

    if config.debug:
        sample_index = X_train_all.sample(100).index
        X_train_all = X_train_all.iloc[sample_index].reset_index(drop=True)
        y_train_all = y_train_all.iloc[sample_index].reset_index(drop=True)
        X_test = X_test.head(100)
        sub = sub.head(100)

    train_df = pd.read_csv(Path(config.input_path) / "train.csv")
    test_df = pd.read_csv(Path(config.input_path) / "test.csv")
    agg_list = ["mean", "var", "min", "max"]
    cat_cols = ["user_id", "anime_id"]
    for c in cat_cols:
        for agg_name in agg_list:
            X_train_all[f"target_{c}_{agg_name}"] = np.nan

    oof_pred = np.zeros(X_train_all.shape[0])
    y_preds = []

    kf = StratifiedKFold(n_splits=config.lgb.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_all, y_train_all)):
        print(f"\nFold {fold}")
        X_train, X_valid = (
            X_train_all.iloc[train_index, :].copy().reset_index(drop=True),
            X_train_all.iloc[valid_index, :].copy().reset_index(drop=True),
        )
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # foldごとにtarget encoding
        for c in cat_cols:
            # test,validationはtrain全体を用いてエンコーディング
            data_tmp = pd.DataFrame({c: train_df.iloc[train_index][c], "target": y_train})
            for agg_name in agg_list:
                target_agg = data_tmp.groupby(c)["target"].agg(agg_name)
                X_valid[f"target_{c}_{agg_name}"] = train_df.iloc[valid_index][c].map(target_agg)
                X_test[f"target_{c}_{agg_name}"] = test_df[c].map(target_agg)
                X_train[f"target_{c}_{agg_name}"] = np.nan  # 初期化

            # 学習データはさらに fold を分けてエンコーディング
            kf_encoding = KFold(n_splits=8, shuffle=True, random_state=config.seed)
            for idx_1, idx_2 in kf_encoding.split(X_train):
                for agg_name in agg_list:
                    target_agg = data_tmp.iloc[idx_1].groupby(c)["target"].agg(agg_name)
                    X_train.loc[idx_2, f"target_{c}_{agg_name}"] = (
                        train_df.iloc[train_index][c].iloc[idx_2].map(target_agg)
                    )

        print(X_train.tail())
        print(X_valid.tail())
        print(X_test.tail())
        # Prepare the LightGBM datasets
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_valid, y_valid)

        # Train the model
        callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100), wandb_callback()]
        model_lgb = lgb.train(
            OmegaConf.to_container(config.lgb.params),
            lgb_train,
            num_boost_round=5000 if config.debug is False else 100,
            valid_sets=[lgb_train, lgb_val],
            callbacks=callbacks,
        )

        # Predict the validation data
        y_valid_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
        oof_pred[valid_index] = y_valid_pred
        y_pred = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
        y_preds.append(y_pred)

        # rmse_valid = evaluate_score(y_valid, y_valid_pred, "rmse")

        log_summary(model_lgb, save_model_checkpoint=False)

        # importance
        """
        """
        importance = pd.DataFrame(
            model_lgb.feature_importance(importance_type="gain"), index=X_train_all.columns, columns=["importance"]
        )
        importance = importance.sort_values("importance", ascending=False)
        print("head importance")
        print(importance.head())
        print("tail importance")
        print(importance.tail())
        # Set N
        N = 60
        # New: Plotting the top N feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x="importance", y=importance.index[:N], data=importance[:N])
        plt.title(f"Top {N} LightGBM Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path / f"top_feature_importance_{fold}.png")
        plt.close()

        # New: Plotting the bottom N feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x="importance", y=importance.index[-N:], data=importance[-N:])
        plt.title(f"Bottom {N} LightGBM Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path / f"bottom_feature_importance_{fold}.png")
        plt.close()
    mean_y_preds = np.mean(y_preds, axis=0)

    # 範囲内にする
    oof_pred = np.clip(oof_pred, 1.0, 10.0)
    mean_y_preds = np.clip(mean_y_preds, 1.0, 10.0)

    # CVスコア確認
    print("===CV scores===")
    rmse_all_valid = evaluate_score(y_train_all, oof_pred, "rmse")
    wandb.log({f"oof_rmse": rmse_all_valid})

    # 保存
    oof_df = pd.DataFrame({"score": oof_pred})
    oof_df.to_csv(output_path / "oof.csv", index=False)

    sub[config.lgb.target_name] = mean_y_preds
    sub.to_csv(output_path / "sub.csv", index=False)

    print(f"saved: {output_path}")
    if config.debug:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    main()
