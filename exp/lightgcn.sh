#!/bin/bash

aggrs=(
  "add"
  "sum"
  "mean"
  "min"
  "max"
  "mul"
)
embedding_dims=(
  "1024"
  "2048"
  "4096"
)
num_layerss=(
  "4"
  "5"
  "6"
  "7"
)

for aggr in "${aggrs[@]}"; do
for embedding_dim in "${embedding_dims[@]}"; do
for num_layers in "${num_layerss[@]}"; do
  python 002_gcn.py train=train002_001 "train.aggr=${aggr}" "train.embedding_dim=${embedding_dim}" "nn.num_layers=${num_layers}"
done
done
done
