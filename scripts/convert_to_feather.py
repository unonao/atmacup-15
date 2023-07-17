import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os

target = [
    "train",
    "test",
]

extension = "csv"


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    for t in target:
        df = pd.read_csv(Path(config.input_path) / f"{t}.{extension}", encoding="utf-8")
        df.to_feather(Path(config.input_path) / f"{t}.feather")


if __name__ == "__main__":
    main()
