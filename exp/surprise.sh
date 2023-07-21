
n_factorss=(
  "16"
  "32"
  "64"
  "128"
  "256"
)
n_epochss=(
  "20"
  "40"
  "80"
  "160"
)

for n_factors in "${n_factorss[@]}"; do
for n_epochs in "${n_epochss[@]}"; do
  echo ${n_factors} ${n_epochs}
  python 011_svdpp.py surprise=surprise011 "surprise.n_factors=${n_factors}" "surprise.n_epochs=${n_epochs}"
  echo 
done
done
# 512 80