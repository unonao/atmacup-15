import os
import re as re
from pathlib import Path

import numpy as np
import pandas as pd
from base import Feature, generate_features, get_arguments
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

Feature.dir = "."
label_feature = "score"
categorical_features = ["anime_id", "type", "source", "rating"]  # "user_id"
user_num_cols = ["members", "watching", "completed", "on_hold", "dropped", "plan_to_watch"]
numeric_features = ["episodes"]
other_featuers = ["genres", "japanese_name", "aired", "producers", "licensors", "studios", "duration"]

train = None
test = None
anime = None
features = None

agg_func_list = ["count", "sum", "mean", "var", "min", "max"]


def cal_user_grouped_stats(df) -> pd.DataFrame:
    """
    数値カラムからなるdfを受け取って、user_idでgroupした統計量を求め、元の特徴量との関係を作る
    - 元のカラム
    - userでgroupしたカラム
    - 上記2つのうち、meanを使って割合を作ったカラム
    """
    _df = df.copy()
    base_cols = df.columns
    _df["user_id"] = features["user_id"]

    # 集約
    user_stats = _df.groupby("user_id").agg(agg_func_list)
    user_stats.columns = ["_".join(col).strip() for col in user_stats.columns.values]
    user_stats.reset_index(inplace=True)
    _df = pd.merge(_df, user_stats, on="user_id", how="left").copy()

    # 集約結果との違いを生成
    for col in base_cols:
        _df[f"{col}_diff"] = _df[col] / _df[f"{col}_mean"]

    _df = _df.drop(["user_id"], axis=1)
    return _df


from rapidfuzz.distance import DamerauLevenshtein, Postfix, Prefix
from rapidfuzz.process import cdist
from rapidfuzz.fuzz import partial_ratio


def calculate_csim(df, aggway="sum", scorer="prefix"):
    """
    文字列の類似度を測る。transformで使うために用いる
    """
    queries = df
    if scorer == "partial":
        score = cdist(queries, queries, scorer=partial_ratio)
        result_df = pd.DataFrame(score, index=queries, columns=queries) / 100
        result = result_df.to_numpy()
        np.fill_diagonal(result, 0.0)  # 自身は0にする
    else:
        if scorer == "prefix":
            scorer = Prefix.normalized_distance
        elif scorer == "prefix":
            scorer = Postfix.normalized_distance
        else:
            scorer = DamerauLevenshtein.normalized_distance
        score = cdist(queries, queries, scorer=Prefix.normalized_distance)
        result_df = pd.DataFrame(score, index=queries, columns=queries)
        result_df = 1 - result_df  # distanceなので反転
        result = result_df.to_numpy()
        np.fill_diagonal(result, 0.0)  # 自身は0にする
    if aggway == "sum":
        result = np.sum(result, axis=1)
    elif aggway == "mean":
        result = np.mean(result, axis=1)
    elif aggway == "var":
        result = np.var(result, axis=1)
    elif aggway == "max":
        result = np.max(result, axis=1)
    return result


class JapaneseName(Feature):
    def create_features(self):
        """
        視聴した作品と類似タイトルがどれだけあるのかについて検索する
        """
        df = features[["user_id", "japanese_name"]].copy()

        use_cols = []
        for scorer in ["partial", "prefix", "postfix", "else"]:
            for aggway in ["sum", "mean", "var", "max"]:
                col = f"{scorer}_{aggway}"
                print(col)
                use_cols.append(col)
                df[col] = df.groupby("user_id")["japanese_name"].transform(calculate_csim, aggway, scorer)

        df = df[use_cols]
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class OtherString(Feature):
    def create_features(self):
        """
        視聴した作品と類似タイトルがどれだけあるのかについて検索する
        """
        cols = ["producers", "licensors", "studios"]
        df = features[["user_id"] + cols].copy()

        def sort_and_join(s):
            # split by comma, strip whitespace, sort, and rejoin
            return ", ".join(sorted(x.strip() for x in s.split(",")))

        for col in cols:
            df[col] = df[col].fillna("Undefined").apply(sort_and_join)

        use_cols = []
        for col in cols:
            for scorer in ["partial", "else"]:
                for aggway in ["sum", "mean", "var", "max"]:
                    new_col = f"{col}_{scorer}_{aggway}"
                    print(new_col)
                    use_cols.append(new_col)
                    df[new_col] = df.groupby("user_id")[col].transform(calculate_csim, aggway, scorer)

        df = df[use_cols]
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class CategoricalLabelEncoded(Feature):
    def create_features(self):
        """
        categorical_features をラベルエンコードした特徴量
        """
        df = features[categorical_features].copy()
        les = []
        for col in categorical_features:
            le = LabelEncoder()
            le.fit(df[col].fillna(""))
            df[col] = le.transform(df[col].fillna(""))
            les.append(le)
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class CategoricalLabelEncodedWithUser(Feature):
    def create_features(self):
        """
        categorical_features をラベルエンコードした特徴量。user_id付き
        """
        use_features = categorical_features + ["user_id"]
        df = features[use_features].copy()
        les = []
        for col in use_features:
            le = LabelEncoder()
            le.fit(df[col].fillna(""))
            df[col] = le.transform(df[col].fillna(""))
            les.append(le)
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class UserNum(Feature):
    def create_features(self):
        """
        作品に関係する人数の特徴量
        - {col}_norm (num): 全体の人数で考えたときの割合
        """
        df = features[user_num_cols].copy()

        sum_num = df[user_num_cols].sum(axis=1)

        for col in user_num_cols:
            df[f"{col}_norm"] = df[col] / sum_num

        df = cal_user_grouped_stats(df)

        # members_norm_max, members_norm_diff, members_norm_mean
        # 不要な特徴量を削除
        df = df.drop([f"members_norm_{agg_str}" for agg_str in ["max", "mean", "diff", "sum", "var", "min"]], axis=1)
        df = df.drop(["members_norm"], axis=1)
        # 元の特徴も一応残す
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class Aired(Feature):
    def create_features(self):
        """
        - start_year(num): 公開された年
        """
        df = features[["aired"]].copy()
        date_df = pd.DataFrame()
        date_df[["start_date", "end_date"]] = df["aired"].str.split(" to ", expand=True)

        # 年だけを取得するための関数定義
        def get_year(date_str):
            try:
                return pd.to_datetime(date_str).year
            except ValueError:
                return None

        df["start_year"] = date_df["start_date"].apply(get_year)
        df = df.drop(["aired"], axis=1)

        df = cal_user_grouped_stats(df)

        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class Genres(Feature):
    def create_features(self):
        """
        作られるカラム
        - "genres_num"(num): その作品のジャンル数
        - {genre} (num) : その作品が対応するジャンルをone-hotにエンコーディング
        """
        df = features[["genres"]].copy()

        # uniqueなジャンルを作る
        stacked_genres = anime["genres"].str.split(",").apply(pd.Series).stack().reset_index(drop=True)
        unique_genres = stacked_genres.unique()

        # そのジャンルの作品かどうかのフラグを立てる（multi-hot encoding)
        df["genres_num"] = 0
        for genre in unique_genres:
            df[genre] = 0
        for genre in unique_genres:
            df.loc[df["genres"].str.contains(genre), genre] = 1
            df["genres_num"] += df[genre]
        df = df.drop(["genres"], axis=1)

        df = cal_user_grouped_stats(df).copy()

        # 今まで見ているジャンルに一致する作品を見ているかのスコア
        df[[f"genres_{agg_str}_score" for agg_str in agg_func_list]] = 0.0
        for agg_str in agg_func_list:
            for genre in unique_genres:
                df[f"genres_{agg_str}_score"] += df[genre] * (1 / df["genres_num"]) * df[f"{genre}_{agg_str}"]
                # 不要な特徴量を削除
                df = df.drop([f"{genre}_{agg_str}"], axis=1)

        # 不要な特徴量を削除
        for genre in unique_genres:
            df = df.drop([genre, f"{genre}_diff"], axis=1)
        for agg_str in ["max", "min"]:
            df = df.drop([f"genres_{agg_str}_score"], axis=1)

        print(df.head())
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class Duration(Feature):
    def create_features(self):
        df = features[["duration"]].copy()

        def convert_to_minutes(s):
            if "hr. " in s and "min." in s:
                # 'hr.'と'min.'が含まれる場合、時間と分を分に変換
                hrs, mins = s.split(" hr. ")
                return int(hrs) * 60 + int(mins.split(" min.")[0])
            elif "hr." in s:
                # 'hr.'のみが含まれる場合、時間を分に変換
                return int(s.split(" hr.")[0]) * 60
            elif "min. per ep." in s:
                # 'min. per ep.'が含まれる場合、エピソードあたりの時間を取得
                return int(s.split(" min. per ep.")[0])
            elif "min." in s:
                # 'min.'が含まれる場合、そのまま分を取得
                return int(s.split(" min.")[0])
            else:
                # 上記のいずれにも該当しない場合、NaNにする
                return np.nan

        # 'duration'列に適用
        df["duration_in_minutes"] = df["duration"].apply(convert_to_minutes)
        df = df.drop(["duration"], axis=1)

        df = cal_user_grouped_stats(df)

        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


if __name__ == "__main__":
    args = get_arguments()

    with initialize(config_path="../yamls", version_base=None):
        config = compose(config_name="config.yaml")

    print("Load data")
    train = pd.read_feather(Path(config.input_path) / "train.feather")
    test = pd.read_feather(Path(config.input_path) / "test.feather")
    anime = pd.read_feather(Path(config.input_path) / "anime.feather")

    anime["genres"] = anime["genres"].str.replace(" ", "")

    features = pd.concat([train.drop([label_feature], axis=1), test])
    features = features.merge(anime, on="anime_id", how="left").reset_index(drop=True)
    print("Generate features")
    generate_features(globals(), args.force)
