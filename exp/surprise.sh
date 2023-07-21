
n_factorss=(
  "32"
  "64"
  "256"
  "512"
  "1024"
)
n_epochss=(
  "2"
  "3"
  "5"
  "7"
  "9"
)

for n_factors in "${n_factorss[@]}"; do
for n_epochs in "${n_epochss[@]}"; do
  python 011_svdpp.py surprise=surprise011 "surprise.n_factors=${n_factors}" "surprise.n_epochs=${n_epochs}"
done
done