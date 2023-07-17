# atmacup-21

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

"""sh
cd exp
python 000_baseline.py  train.embedding_dim=6000 train.num_layers=5

python 001_lightgcn_agg.py debug=True train=train001
"""
