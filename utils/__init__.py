import numpy as np
import pandas as pd
import time
import contextlib
from sklearn import metrics


@contextlib.contextmanager
def simple_timer(name="timer"):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def load_datasets(feats):
    dfs = [pd.read_feather(f"../features/{f}_train.feather") for f in feats.num + feats.cat]
    for model_name in feats.models:
        df = pd.read_csv(f"../output/{model_name}/oof.csv")
        df.columns = [model_name]
        dfs.append(df)
    X_train = pd.concat(dfs, axis=1, sort=False)

    dfs = [pd.read_feather(f"../features/{f}_test.feather") for f in feats.num + feats.cat]
    for model_name in feats.models:
        df = pd.read_csv(f"../output/{model_name}/sub.csv")
        df.columns = [model_name]
        dfs.append(df)
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


from sklearn.preprocessing import StandardScaler


def normalize_data(X_train_all, X_test):
    # データ結合
    combined_df = pd.concat([X_train_all, X_test], axis=0)

    # 欠損値の平均値で埋める
    combined_df_filled = combined_df.fillna(combined_df.mean())

    # 正規化
    scaler = StandardScaler()
    combined_df_normalized = pd.DataFrame(scaler.fit_transform(combined_df_filled), columns=combined_df_filled.columns)

    # 結果のデータフレームを分割して返す
    X_train_processed = combined_df_normalized[: len(X_train_all)]
    X_test_processed = combined_df_normalized[len(X_train_all) :]

    return X_train_processed, X_test_processed


def load_datasets_for_nn(feats):
    X_train, X_test = load_datasets(feats)  # 正規化込み
    X_train, X_test = normalize_data(X_train, X_test)

    dfs = [pd.read_feather(f"../features/{f}_train.feather") for f in feats.norm]
    for model_name in feats.models:
        df = pd.read_csv(f"../output/{model_name}/oof.csv")
        df.columns = [model_name]
        dfs.append(df)
    dfs.append(X_train)
    X_train = pd.concat(dfs, axis=1, sort=False)

    dfs = [pd.read_feather(f"../features/{f}_test.feather") for f in feats.norm]
    for model_name in feats.models:
        df = pd.read_csv(f"../output/{model_name}/sub.csv")
        df.columns = [model_name]
        dfs.append(df)
    dfs.append(X_test)
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_feather("../input/atmaCup15_dataset/train.feather")
    y_train = train[target_name]
    return y_train


def load_sample_sub():
    df = pd.read_feather("../input/atmaCup15_dataset/sample_submission.feather")
    return df


def evaluate_score(true, predicted, metric_name):
    if metric_name == "rmse":
        return np.sqrt(metrics.mean_squared_error(true, predicted))


def print_evaluate_regression(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Square", r2_square)
