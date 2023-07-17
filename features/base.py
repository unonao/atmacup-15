import argparse
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd
import contextlib
import time


@contextlib.contextmanager
def simple_timer(name):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def get_features(namespace):
    """
    名前空間内の Feature のサブクラスで抽象クラスでないものをgenerate
    """
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    """
    create.py内の特徴量を全て生成。overwrite で既にあるものも上書きする
    """
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, "was skipped")
        else:
            f.run().save()
            print(f"train:{f.train.shape}, Nan:{f.train.isnull().sum().sum()}")
            print(f"test:{f.test.shape}, Nan:{f.test.isnull().sum().sum()}")


# 特徴量の抽象基底クラス
class Feature(metaclass=ABCMeta):
    prefix = ""
    suffix = ""
    dir = "."

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip("_")

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f"{self.name}_train.feather"
        self.test_path = Path(self.dir) / f"{self.name}_test.feather"

    def run(self):
        with simple_timer(self.name):
            self.create_features()
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = "_" + self.suffix if self.suffix else ""
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))
