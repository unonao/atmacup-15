
hidden_channelss=(
  "32"
  "64"
  "256"
  "512"
  "1024"
)
num_layerss=(
  "2"
  "3"
  "5"
  "7"
  "9"
)

for hidden_channels in "${hidden_channelss[@]}"; do
for num_layers in "${num_layerss[@]}"; do
  python 063_gcn.py  gcn=gcn060_003 "gcn.hidden_channels=${hidden_channels}" "gcn.num_layers=${num_layers}"
done
done