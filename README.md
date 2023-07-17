# atmacup-21

- 000(seen): LightGCN系
- 100(seen): lgb
- 200(unseen): lgb
- 300: stacking

"""sh
docker compose run kaggle bash
docker compose up
"""

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
python 100_lgb.py lgb=lgb100
python 200_lgb.py lgb=lgb200
python 300_combine.py lgb=combine300
"""
