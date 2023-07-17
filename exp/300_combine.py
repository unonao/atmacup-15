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


def load_models(model_names):
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"../output/{model_name}/oof.csv")
        df.columns = [model_name]
        dfs.append(df)
    X_train = pd.concat(dfs, axis=1, sort=False)

    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"../output/{model_name}/sub.csv")
        df.columns = [model_name]
        dfs.append(df)
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    seen_df = load_models(config.combine.seen)
    unseen_df = load_models(config.combine.unseen)


if __name__ == "__main__":
    main()
