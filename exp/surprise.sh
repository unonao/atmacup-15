#!/bin/bash

models=(
  "BaselineOnly"
  "KNNBasic"
  "KNNWithMeans"
  "KNNWithZScore"
  "SVD"
  "SVDpp"
  "NMF"
  "SlopeOne"
  "NormalPredictor"
  "KNNBaseline"
  "CoClustering"
)

for model in "${models[@]}"; do
  python 020_surprise_unseen.py "surprise.model=${model}"
done
