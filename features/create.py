import os
import re as re
import sys
from pathlib import Path

import cuml
import implicit
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from base import Feature, generate_features, get_arguments
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from tqdm.auto import tqdm


sys.path.append(os.pardir)
from utils import load_datasets
from utils.embedding import TextEmbedder
from utils.gcn import LightGCN

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


def cal_user_grouped_stats(df, agg_list=agg_func_list) -> pd.DataFrame:
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
    user_stats = _df.groupby("user_id").agg(agg_list)
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


class TextEmbedding(Feature):
    def create_features(self):
        """
        言語モデルによる埋込情報と、同一user_id内でのコサイン類似度の統計量を算出する
        """
        # 文字列として扱う列を結合し、元の列を落とす
        concat_feature = ["japanese_name", "genres", "producers", "licensors", "studios", "rating"]
        anime_df = anime[concat_feature].copy()
        # スペース区切りで結合する
        anime_df[concat_feature] = anime_df[concat_feature].astype(str)
        anime_df["combined_features"] = anime_df[concat_feature].agg(" ".join, axis=1)
        embedder = TextEmbedder()
        anime_embeddings = embedder.get_embeddings(anime_df["combined_features"].values.tolist())

        df = features[["anime_id"]].copy()
        # anime の何行目にあるのかを求める
        df["row_number"] = df["anime_id"].map(anime[["anime_id"]].copy().reset_index().set_index("anime_id")["index"])
        embeddings = anime_embeddings[df["row_number"]]
        embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


import cupy as cp
from cuml.metrics import pairwise_distances


class TextCosineSim(Feature):
    def create_features(self):
        """
        言語モデルによる埋込情報と、同一user_id内でのコサイン類似度の統計量を算出する
        """
        # 文字列として扱う列を結合し、元の列を落とす
        concat_feature = ["japanese_name", "genres", "producers", "licensors", "studios", "rating"]
        anime_df = anime[concat_feature].copy()
        # スペース区切りで結合する
        anime_df[concat_feature] = anime_df[concat_feature].astype(str)
        anime_df["combined_features"] = anime_df[concat_feature].agg(" ".join, axis=1)
        embedder = TextEmbedder()
        anime_embeddings = embedder.get_embeddings(anime_df["combined_features"].values.tolist())
        df = features[["user_id", "anime_id"]].copy()
        df["row_number"] = df["anime_id"].map(anime[["anime_id"]].copy().reset_index().set_index("anime_id")["index"])
        embeddings_cp = cp.array(anime_embeddings)
        cosine_sim_matrix = pairwise_distances(embeddings_cp, embeddings_cp, metric="cosine")
        cosine_sim_matrix = cp.asnumpy(cosine_sim_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim_matrix)

        def calculate_cosine_sim(row_numbers, aggway="sum"):
            result = cosine_sim_df.iloc[row_numbers.to_numpy(), row_numbers.to_numpy()].to_numpy()
            np.fill_diagonal(result, 0.0)
            if aggway == "sum":
                result = np.sum(result, axis=1)
            elif aggway == "mean":
                result = np.mean(result, axis=1)
            elif aggway == "var":
                result = np.var(result, axis=1)
            elif aggway == "max":
                result = np.max(result, axis=1)
            return result

        use_cols = []
        for aggway in ["sum", "mean", "var", "max"]:
            col = f"cosine_sim_{aggway}"
            use_cols.append(col)
            print(col)
            df[col] = df.groupby("user_id")["row_number"].transform(calculate_cosine_sim, aggway)
        df = df[use_cols]
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


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


class StringLen(Feature):
    def create_features(self):
        """
        視聴した作品と類似タイトルがどれだけあるのかについて検索する
        """
        cols = ["producers", "licensors", "studios"]
        df = features[["user_id"] + cols].copy()

        def sort_and_join(s):
            # split by comma, strip whitespace, sort, and rejoin
            return ", ".join(sorted(x.strip() for x in s.split(",")))

        use_cols = []
        for col in cols:
            new_col = f"{col}_len"
            df[new_col] = df[col].fillna("Undefined").str.split(",").str.len()
            use_cols.append(new_col)

        df = df[use_cols]

        df = cal_user_grouped_stats(df)

        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class MultiHotFeatures(Feature):
    def create_features(self):
        df = features[["anime_id"]].copy()
        multilabel_cols = ["genres", "producers", "licensors", "studios"]
        multilabel_dfs = []

        all_cols = []
        for c in multilabel_cols:
            list_srs = anime[c].map(lambda x: x.split(",")).tolist()
            mlb = MultiLabelBinarizer()
            ohe_srs = mlb.fit_transform(list_srs)
            col_names = [f"ohe_{c}_{name}" for name in mlb.classes_]
            col_df = pd.DataFrame(ohe_srs, columns=col_names)
            all_cols += col_names
            multilabel_dfs.append(col_df)

        multilabel_df = pd.concat(multilabel_dfs, axis=1)

        # ユニーク数が多いので、SVDで次元圧縮する
        n_components = 30
        svd = cuml.TruncatedSVD(n_components=n_components)
        svd_df = svd.fit_transform(multilabel_df.astype(float))
        svd_df.columns = [f"svd_{ix}" for ix in range(n_components)]
        svd_df["anime_id"] = anime["anime_id"]
        df = df.merge(svd_df, on="anime_id", how="left")
        df = df.drop(["anime_id"], axis=1)

        df = cal_user_grouped_stats(df, ["sum", "mean", "var", "min", "max"]).copy()
        df = df.fillna(df.mean(axis=0))

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


from sklearn.preprocessing import OneHotEncoder


class CategoricalOneHotEncoded(Feature):
    def create_features(self):
        """
        categorical_features をOne-hotエンコードした特徴量
        """
        transformed_df = pd.DataFrame()

        for column in categorical_features:
            df = features[column].copy().to_frame()
            df = df.fillna("NA")  # Fill missing values
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(df)
            transformed_column = ohe.transform(df)
            transformed_column = pd.DataFrame(transformed_column, columns=ohe.get_feature_names_out(df.columns))

            # Merge the transformed column with the overall DataFrame
            transformed_df = pd.concat([transformed_df, transformed_column], axis=1)

        self.train = transformed_df[: train.shape[0]]
        self.test = transformed_df[train.shape[0] :]


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


class CategoricalOneHotEncodedWithUser(Feature):
    def create_features(self):
        """
        categorical_features をOne-hotエンコードした特徴量
        """
        transformed_df = pd.DataFrame()
        use_features = categorical_features + ["user_id"]
        for column in use_features:
            df = features[column].copy().to_frame()
            df = df.fillna("NA")  # Fill missing values
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(df)
            transformed_column = ohe.transform(df)
            transformed_column = pd.DataFrame(transformed_column, columns=ohe.get_feature_names_out(df.columns))

            # Merge the transformed column with the overall DataFrame
            transformed_df = pd.concat([transformed_df, transformed_column], axis=1)

        self.train = transformed_df[: train.shape[0]]
        self.test = transformed_df[train.shape[0] :]


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


class UserNumSecond(Feature):
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

        for i in range(len(user_num_cols)):
            for j in range(i + 1, len(user_num_cols)):
                col1 = user_num_cols[i]
                col2 = user_num_cols[j]
                if col1 == "members" or col2 == "members":
                    continue
                df[f"{col1}_div_{col2}"] = df[col1] / df[col2]
                df[f"{col1}_prod_{col2}"] = df[f"{col1}_norm"] * df[f"{col2}_norm"]

        # members_norm_max, members_norm_diff, members_norm_mean
        # 不要な特徴量を削除
        df = df.drop([f"members_norm_{agg_str}" for agg_str in ["max", "mean", "diff", "sum", "var", "min"]], axis=1)
        df = df.drop(["members_norm"], axis=1)
        df = df.drop(
            ["members", "watching_count", "dropped_count", "dropped_norm_count", "plan_to_watch_norm_count"], axis=1
        )
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


class AiredWithEnd(Feature):
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
            except:
                return None

        df["end_year"] = date_df["end_date"].apply(get_year)
        df["start_year"] = date_df["start_date"].apply(get_year)
        df["diff_year"] = df["start_year"] - df["end_year"]

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
        for agg_str in ["max", "min", "count", "sum"]:
            df = df.drop([f"genres_{agg_str}_score"], axis=1)

        print(df.head())
        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


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


class Duration(Feature):
    def create_features(self):
        df = features[["duration"]].copy()
        # 'duration'列に適用
        df["duration_in_minutes"] = df["duration"].apply(convert_to_minutes)

        df = df.drop(["duration"], axis=1)

        df = cal_user_grouped_stats(df)

        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class DurationEpisodes(Feature):
    def create_features(self):
        df = features[["duration", "episodes"]].copy()
        df.loc[df["episodes"] == "Unknown", "episodes"] = np.nan
        df["episodes"] = df["episodes"].astype(float)
        df["duration_in_minutes"] = df["duration"].apply(convert_to_minutes)
        df["total_len"] = df["duration_in_minutes"] * df["episodes"]

        df = df.drop(["duration"], axis=1)

        df = cal_user_grouped_stats(df)

        self.train = df[: train.shape[0]]
        self.test = df[train.shape[0] :]


class ImplicitFactors(Feature):
    def create_features(self):
        """ """
        df = features[["user_id", "anime_id"]].copy()
        # userとitemのIDをマッピング
        user_id_mapping = {id: i for i, id in enumerate(df["user_id"].unique())}
        anime_id_mapping = {id: i for i, id in enumerate(df["anime_id"].unique())}
        df["user_label"] = df["user_id"].map(user_id_mapping)
        df["anime_label"] = df["anime_id"].map(anime_id_mapping)

        item_user_data = csr_matrix((np.ones(len(df)), (df["user_label"], df["anime_label"])))

        model = implicit.gpu.als.AlternatingLeastSquares(factors=64)  # gpuを想定
        model.fit(item_user_data)

        user_factors = model.user_factors
        item_factors = model.item_factors
        embeddings = np.concatenate(
            (user_factors[df["user_label"]].to_numpy(), item_factors[df["anime_label"]].to_numpy()), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"item_factor_{j}" for j in range(item_factors.shape[1])
        ]
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class ImplicitFactorsLogisticMatrixFactor(Feature):
    def create_features(self):
        """ """
        df = features[["user_id", "anime_id"]].copy()
        # userとitemのIDをマッピング
        user_id_mapping = {id: i for i, id in enumerate(df["user_id"].unique())}
        anime_id_mapping = {id: i for i, id in enumerate(df["anime_id"].unique())}
        df["user_label"] = df["user_id"].map(user_id_mapping)
        df["anime_label"] = df["anime_id"].map(anime_id_mapping)

        item_user_data = csr_matrix((np.ones(len(df)), (df["user_label"], df["anime_label"])))

        model = implicit.cpu.lmf.LogisticMatrixFactorization(factors=30)
        model.fit(item_user_data)

        user_factors = model.user_factors
        item_factors = model.item_factors
        embeddings = np.concatenate((user_factors[df["user_label"]], item_factors[df["anime_label"]]), axis=1)
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"matrix_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"matrix_item_factor_{j}" for j in range(item_factors.shape[1])
        ]
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class ImplicitFactorsBpr(Feature):
    def create_features(self):
        """ """
        df = features[["user_id", "anime_id"]].copy()
        # userとitemのIDをマッピング
        user_id_mapping = {id: i for i, id in enumerate(df["user_id"].unique())}
        anime_id_mapping = {id: i for i, id in enumerate(df["anime_id"].unique())}
        df["user_label"] = df["user_id"].map(user_id_mapping)
        df["anime_label"] = df["anime_id"].map(anime_id_mapping)

        item_user_data = csr_matrix((np.ones(len(df)), (df["user_label"], df["anime_label"])))

        model = implicit.gpu.bpr.BayesianPersonalizedRanking(factors=64)  # gpuを想定
        model.fit(item_user_data)

        user_factors = model.user_factors
        item_factors = model.item_factors
        embeddings = np.concatenate(
            (user_factors[df["user_label"]].to_numpy(), item_factors[df["anime_label"]].to_numpy()), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"bpr_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"bpr_item_factor_{j}" for j in range(item_factors.shape[1])
        ]
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class ImplicitFactorsBprSvd(Feature):
    def create_features(self):
        """ """
        df = features[["user_id", "anime_id"]].copy()
        # userとitemのIDをマッピング
        user_id_mapping = {id: i for i, id in enumerate(df["user_id"].unique())}
        anime_id_mapping = {id: i for i, id in enumerate(df["anime_id"].unique())}
        df["user_label"] = df["user_id"].map(user_id_mapping)
        df["anime_label"] = df["anime_id"].map(anime_id_mapping)

        item_user_data = csr_matrix((np.ones(len(df)), (df["user_label"], df["anime_label"])))

        model = implicit.gpu.bpr.BayesianPersonalizedRanking(factors=64)  # gpuを想定
        model.fit(item_user_data)

        user_factors = model.user_factors
        item_factors = model.item_factors
        embeddings = np.concatenate(
            (user_factors[df["user_label"]].to_numpy(), item_factors[df["anime_label"]].to_numpy()), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"bpr_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"bpr_item_factor_{j}" for j in range(item_factors.shape[1])
        ]

        # ユニーク数が多いので、SVDで次元圧縮する
        n_components = 10
        svd = cuml.TruncatedSVD(n_components=n_components)
        svd.fit(item_factors.to_numpy())
        svd_arr = svd.transform(item_factors[df["anime_label"]].to_numpy())
        svd_df = pd.DataFrame(svd_arr, columns=[f"bpr_svd_{ix}" for ix in range(n_components)])

        svd_df = cal_user_grouped_stats(svd_df, ["sum", "mean", "var", "min", "max"]).copy()

        self.train = svd_df[: train.shape[0]]
        self.test = svd_df[train.shape[0] :]


class ImplicitFactorsUser(Feature):
    def create_features(self):
        """ """
        df = features[["user_id", "anime_id"]].copy()
        # userとitemのIDをマッピング
        user_id_mapping = {id: i for i, id in enumerate(df["user_id"].unique())}
        anime_id_mapping = {id: i for i, id in enumerate(df["anime_id"].unique())}
        df["user_label"] = df["user_id"].map(user_id_mapping)
        df["anime_label"] = df["anime_id"].map(anime_id_mapping)

        item_user_data = csr_matrix((np.ones(len(df)), (df["user_label"], df["anime_label"])))

        embeddings_df_list = []
        for key, model in {
            "mf": implicit.cpu.lmf.LogisticMatrixFactorization(factors=30),
            "bpr": implicit.cpu.bpr.BayesianPersonalizedRanking(factors=64),
            "als": implicit.cpu.als.AlternatingLeastSquares(factors=64),
        }.items():
            model.fit(item_user_data)
            factor = model.user_factors
            embeddings_df = pd.DataFrame(
                factor[df["user_label"]], columns=[f"{key}_user_{i}" for i in range(factor.shape[1])]
            )
            embeddings_df_list.append(embeddings_df)
        embeddings_df = pd.concat(embeddings_df_list, axis=1)
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class LightgcnEmbedding(Feature):
    def create_features(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_df = features[["user_id", "anime_id"]].copy()
        all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
        all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])
        all_df["is_train"] = True
        all_df.loc[len(train) :, "is_train"] = False
        # userとanimeの番号が別になるようにずらす
        all_df["anime_label"] += len(user_idx)
        num_nodes = len(user_idx) + len(anime_idx)
        edges = all_df[["user_label", "anime_label"]].to_numpy()
        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
        data = Data(num_nodes=num_nodes, edge_index=edge_index).to(device)
        data.edge_weight = torch.ones(len(all_df)).contiguous().to(device)

        # Negative Edge を追加
        transform = RandomLinkSplit(num_val=0, num_test=0, add_negative_train_samples=True, neg_sampling_ratio=1.0)
        train_data, _, _ = transform(data)

        model = LightGCN(
            num_nodes=data.num_nodes,
            embedding_dim=64,
            num_layers=3,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for epoch in tqdm(range(2001)):
            # train
            pred = model.predict_link(
                train_data.edge_index, train_data.edge_label_index, edge_weight=train_data.edge_weight, prob=True
            )
            loss = model.link_pred_loss(pred, train_data["edge_label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                tqdm.write(f"epoch {epoch} : loss {loss.item()}")

        # 埋め込み取得
        vectors = model.get_embedding(data.edge_index).detach().cpu().numpy()
        vectors /= np.linalg.norm(vectors)  # normalized
        user_factors = vectors[: len(user_idx)]
        item_factors = vectors[len(user_idx) :]
        embeddings = np.concatenate(
            (user_factors[all_df["user_label"]], item_factors[(all_df["anime_label"] - len(user_idx))]), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"lightgcn_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"lightgcn_item_factor_{j}" for j in range(item_factors.shape[1])
        ]
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class BprLightgcnEmbedding(Feature):
    def create_features(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_df = features[["user_id", "anime_id"]].copy()
        all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
        all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])
        all_df["is_train"] = True
        all_df.loc[len(train) :, "is_train"] = False
        # userとanimeの番号が別になるようにずらす
        all_df["anime_label"] += len(user_idx)
        num_nodes = len(user_idx) + len(anime_idx)
        edges = all_df[["user_label", "anime_label"]].to_numpy()
        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
        data = Data(num_nodes=num_nodes, edge_index=edge_index).to(device)
        data.edge_weight = torch.ones(len(all_df)).contiguous().to(device)

        # Negative Edge を追加
        transform = RandomLinkSplit(num_val=0, num_test=0, add_negative_train_samples=True, neg_sampling_ratio=1.0)
        train_data, _, _ = transform(data)

        model = LightGCN(
            num_nodes=data.num_nodes,
            embedding_dim=64,
            num_layers=3,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for epoch in tqdm(range(2001)):
            pred = model(train_data.edge_index, train_data.edge_label_index, train_data.edge_weight)
            loss = model.recommendation_loss(
                pos_edge_rank=pred[train_data.edge_label > 0],
                neg_edge_rank=pred[train_data.edge_label == 0],
                node_id=train_data.edge_index.unique(),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                tqdm.write(f"epoch {epoch} : loss {loss.item()}")

        # 埋め込み取得
        vectors = model.get_embedding(data.edge_index).detach().cpu().numpy()
        vectors /= np.linalg.norm(vectors)  # normalized
        user_factors = vectors[: len(user_idx)]
        item_factors = vectors[len(user_idx) :]
        embeddings = np.concatenate(
            (user_factors[all_df["user_label"]], item_factors[(all_df["anime_label"] - len(user_idx))]), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"lightgcn_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"lightgcn_item_factor_{j}" for j in range(item_factors.shape[1])
        ]
        self.train = embeddings_df[: train.shape[0]]
        self.test = embeddings_df[train.shape[0] :]


class SvdLightgcnEmbedding(Feature):
    def create_features(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_df = features[["user_id", "anime_id"]].copy()
        all_df["user_label"], user_idx = pd.factorize(all_df["user_id"])
        all_df["anime_label"], anime_idx = pd.factorize(all_df["anime_id"])
        all_df["is_train"] = True
        all_df.loc[len(train) :, "is_train"] = False
        # userとanimeの番号が別になるようにずらす
        all_df["anime_label"] += len(user_idx)
        num_nodes = len(user_idx) + len(anime_idx)
        edges = all_df[["user_label", "anime_label"]].to_numpy()
        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
        data = Data(num_nodes=num_nodes, edge_index=edge_index).to(device)
        data.edge_weight = torch.ones(len(all_df)).contiguous().to(device)

        # Negative Edge を追加
        transform = RandomLinkSplit(num_val=0, num_test=0, add_negative_train_samples=True, neg_sampling_ratio=1.0)
        train_data, _, _ = transform(data)

        model = LightGCN(
            num_nodes=data.num_nodes,
            embedding_dim=256,
            num_layers=4,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for epoch in tqdm(range(2001)):
            pred = model.predict_link(
                train_data.edge_index, train_data.edge_label_index, edge_weight=train_data.edge_weight, prob=True
            )
            loss = model.link_pred_loss(pred, train_data["edge_label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                tqdm.write(f"epoch {epoch} : loss {loss.item()}")

        # 埋め込み取得
        vectors = model.get_embedding(data.edge_index).detach().cpu().numpy()
        vectors /= np.linalg.norm(vectors)  # normalized
        user_factors = vectors[: len(user_idx)]
        item_factors = vectors[len(user_idx) :]
        embeddings = np.concatenate(
            (user_factors[all_df["user_label"]], item_factors[(all_df["anime_label"] - len(user_idx))]), axis=1
        )
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"lightgcn_user_factor_{i}" for i in range(user_factors.shape[1])] + [
            f"lightgcn_item_factor_{j}" for j in range(item_factors.shape[1])
        ]

        # ユニーク数が多いので、SVDで次元圧縮する
        n_components = 30
        svd = cuml.TruncatedSVD(n_components=n_components)
        svd.fit(embeddings)
        svd_arr = svd.transform(embeddings)
        svd_df = pd.DataFrame(svd_arr, columns=[f"bpr_lightgcn_{ix}" for ix in range(n_components)])

        self.train = svd_df[: train.shape[0]]
        self.test = svd_df[train.shape[0] :]


class PcaTwenty(Feature):
    def create_features(self):
        """ """
        conf = OmegaConf.create(
            {
                "num": ["duration_episodes", "genres", "user_num", "aired", "japanese_name", "other_string"],
                "cat": [],
                "models": [],
            }
        )
        n = 20

        X_train_all, X_test = load_datasets(conf)
        X = pd.concat([X_train_all, X_test])
        print(X.shape)
        X = X.astype(float)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = X.fillna(0)
        pca = cuml.PCA(n_components=n)
        Z_pca = pca.fit_transform(X)
        df = pd.DataFrame(Z_pca, columns=[f"pca{i}" for i in range(n)])
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
