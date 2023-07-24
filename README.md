# #15 atmacup 1st place solution

- 解法詳細：[1st place solution | #15 atmaCup](https://www.guruguru.science/competitions/21/discussions/eddaa54d-50d1-488d-b2d8-8f2c34e09851/)

## ディレクトリ構成

```.
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- compose.yaml
|-- exp : 実験用スクリプト
|-- features : 特徴量生成ファイル、生成した特徴用の保存先
|-- features_old : 旧版特徴量生成ファイル
|-- input : コンペデータの格納先
|-- notebook : ノートブック格納先。expに書く前の動作確認用がメイン
|-- output : 実行結果の格納先
|-- scripts : 実験用以外の雑多なスクリプト格納先
|-- utils
`-- yamls : hydra で使うための config ファイル群
```

### exp ディレクトリの実験用スクリプトの種類

- 00x: LightGCN 系 (seen 用)
- 01x: surprise によるモデル (seen 用)
- 02x: surprise によるモデル (unseen 用)
- 03x: NN (seen 用)
- 04x: NN (unseen 用)
- 05x: set transformer (unseen 用)
- 06x: Heterogeneous Graph Learning (seen 用)
- 07x: [TVTTFormer(Train-Valid-Test TransFoermer](https://www.guruguru.science/competitions/21/discussions/9864e927-0b29-4a2f-9df6-40e34fa0c89f/)
- 10x: LightGBM (seen 用)
- 20x: LightGBM (unseen 用)
- 30x: アンサンブル・stacking 用コード

## 実行方法

### 環境構築

kaggle docker をベースに

```sh
docker compose biuld
docker compose run kaggle bash # bash に入る
docker compose up # jupyter lab 起動
```

### 初期設定

input/atmaCup15_dataset にデータを配置後、feather 形式に変換

```sh
cd scripts/
python convert_to_feather.py
```

特徴量を作成

```sh
cd features
python create.py
```

### 実行方法

事前に特徴量を作成後、exp 内で実行する。

実行例

```sh
cd exp
python 000_baseline.py train.embedding_dim=6000 train.num_layers=5
python 000_baseline.py train=base005
python 001_lightgcn_agg.py debug=True train=train001
python 005_lightgcn_unseen.py debug=True train=000_lightgcn.yaml
python 010_surprise.py surprise.name=SVD debug=True
python 031_nn_seen_scheduler.py nn=nn031_006
python 033_nn.py nn=nn033_002
python 041_nn_seen_scheduler.py nn=nn040_007
python 060_gcn_seen.py gcn=gcn060_001
python 100_lgb.py lgb=lgb100_013
python 200_lgb.py lgb=lgb200
python 201_stratify.py lgb=lgb200_017
python 300_combine.py combine=combine306
python 301_combine.py combine=combine306
python 302_combine.py combine=combine310
python 041_nn_unseen_scheduler.py nn=nn040_007
```

## 最終提出

最終提出は以下のコードで生成しました。

```sh
cd exp
python 305_combine.py combine=combine319_v4
```

アンサンブルのためのモデルは実験ごとに一意の id を付与しており、wandb で管理していたので他の方はすぐに再現するのは難しいかもしれません。
