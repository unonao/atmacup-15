#!/bin/bash

dropouts=(
  "0.5"
  "0.6"
)
weight_decays=(
  "0"
  "1e-6"
)
num_layerss=(
  "3"
  "5"
)

for dropout in "${dropouts[@]}"; do
for weight_decay in "${weight_decays[@]}"; do
for num_layers in "${num_layerss[@]}"; do
  python 041_nn_unseen_scheduler.py nn=nn040_002 "nn.dropout_rate=${dropout}" "nn.weight_decay=${weight_decay}" "nn.num_layers=${num_layers}"
done
done
done
