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

どのようなアンサンブルを行うか

- seen
  - LightGCN: (000_baseline.py)
  - LightGBM(target無し): (100_lgb.py)
  - LightGBM(targetあり): (101_lgb_target.py)
  - NN : (034_nn.py)
  - CF系
- unseen
  - LightGBM(target無し): (201_strutify_importance.py)
  - LightGBM(targetあり): (101_lgb_target.py)
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
