
n_factorss=(
  "8"
  "16"
  "32"
)
n_epochss=(
  "3"
  "5"
  "8"
)

for n_factors in "${n_factorss[@]}"; do
for n_epochs in "${n_epochss[@]}"; do
  echo ${n_factors} ${n_epochs}
  python 011_svdpp.py surprise=surprise011 "surprise.n_factors=${n_factors}" "surprise.n_epochs=${n_epochs}"
  echo 
done
done
# 512 80