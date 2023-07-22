# atmacup-21

- 000(seen): LightGCN系
- 100(seen): lgb
- 200(unseen): lgb
- 300: stacking

"""sh
docker compose run kaggle bash
docker compose up
"""

## final submission

```sh
python 303_combine.py combine=combine318
python 304_combine.py combine=combine317 +model=linear
python 070_tvttformer_seen.py tvtt=base
python 080_tvttformer_unseen.py tvtt=base
```

どのようなアンサンブルを行うか

- seen
  [x] LightGCN (000_baseline.py) : `python 000_baseline.py  train=base005`
    [x] パラメータ調整
    [x] fold数の調整: 10
    [x] 学習率を下げる（もっと下げる。。。？）
  [x] GCN (062_gcn.py) : `python 064_gcn.py gcn=gcn060_005`
    [x] パラメータ調整 : `bash gcn.sh`
    [x] 特徴量の確定
    [x] fold数の調整: 10
    [x] 学習率を下げる
  [] LightGBM target無し (100_lgb.py) : `python 100_lgb.py lgb=lgb100_021`
    [x] パラメータ調整
    [x] 特徴量の確定
    [x] fold数の調整: 10
    [x] 学習率を下げる
  [] LightGBM targetあり (101_lgb_target.py): `python 101_lgb_target.py lgb=lgb100_021`
  [] NN  (034_nn.py):
    [] fold数の調整: 10
    [] 学習率を下げる
  [x] CF系  (010_surprise): `python 011_svdpp.py surprise=surprise011`
    [x] パラメータ調整
    [x] 特徴量の確定
    [x] fold数の調整: 10
- unseen
  - LightGBM target無し : (201_strutify_importance.py)
    [ ] パラメータ調整
    [x] 特徴量の確定
    [ ] fold数の調整: 10
    [ ] 学習率を下げる
  - LightGBM targetあり: (202_target.py)
  - NN : (042_nn.py)
  - CF系

## 初期設定

input/atmaCup15_dataset にデータを配置
"""sh
cd scripts/
python convert_to_feather.py
"""

feature create
"""sh
cd features
python create.py
"""

"""sh
cd exp
python 000_baseline.py  train.embedding_dim=6000 train.num_layers=5
python 000_baseline.py  train=base005

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

python 041_nn_unseen_scheduler.py  nn=nn040_007

"""
