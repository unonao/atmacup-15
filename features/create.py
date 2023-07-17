import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder

Feature.dir = "features"


if __name__ == "__main__":
    args = get_arguments()
    train = pd.read_feather("./data/interim/train.feather")
    test = pd.read_feather("./data/interim/test.feather")
    features = pd.concat([train.drop(["Id", "SalePrice"], axis=1), test.drop("Id", axis=1)])

    features["MSSubClass"] = features["MSSubClass"].apply(str)

    temporal_features = [
        feature for feature in features.columns if "Yr" in feature or "Year" in feature or "Mo" in feature
    ]
    numeric_features = [
        feature for feature in features.columns if features[feature].dtype != "O" and feature not in temporal_features
    ]
    categorical_features = [
        feature for feature in features.columns if features[feature].dtype == "O" and feature not in temporal_features
    ]

    generate_features(globals(), args.force)
