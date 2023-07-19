"""
手元の CV スコアと public LB のスコアを用いて、unseenの割合を求めるスクリプト
"""
from scipy.optimize import minimize
import numpy as np

# seen score,  unseen score,  public の順に格納
scores = [
    [1.143132400409859, 1.4139537431522253, 1.2218],
    [1.13855, 1.40413, 1.2124],
    [1.1302647706928934, 1.3961950862123547, 1.2095],
    [1.1293, 1.38101, 1.1998],
]
scores = np.array(scores)  # あなたのデータ


def calculate_a(x, y, z):
    if x != y:
        a = (z - y) / (x - y)
        return a
    else:
        return None


# scoresの値をループで回し、それぞれの組みに対するaの値を計算します
for score in scores:
    x, y, z = score
    a = calculate_a(x, y, z)
    print(f"{a} : For x={x}, y={y}, z={z}")

# 最小２乗誤差
# x, y, zをそれぞれ抽出
x = scores[:, 0]
y = scores[:, 1]
z = scores[:, 2]


# 最適化する関数を定義します。この関数は各トリプレットに対する二乗誤差の和を返します。
def func(a):
    return np.sum((np.array(z) - (np.array(x) * a[0] + np.array(y) * (1 - a[0]))) ** 2)


# 初期の推測値を設定します。ここでは0.5を使用します。
initial_guess = [0.5]

# 最適化の結果を見つけます。
result = minimize(func, initial_guess)

# 最適な a の値を表示します。
print(result.x)
