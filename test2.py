"""
手元の CV スコアと public LB のスコアを用いて、unseenの割合を求めるスクリプト
"""
from scipy.optimize import minimize
import numpy as np
import math

# seen score,  unseen score,  public の順に格納
scores = [
    [1.143132400409859, 1.4139537431522253, 1.2218],
    [1.13855, 1.40413, 1.2124],
    [1.1302647706928934, 1.3961950862123547, 1.2095],
    [1.1293, 1.38101, 1.1998],
    [1.1281078631771408, 1.368889612584224, 1.1964],
    [1.1237222255903876, 1.3669475462129905, 1.1880],
    [1.0998172745188464, 1.3615243180966956, 1.1793],
    [1.095543318751611, 1.3685943824106688, 1.1797],
    [1.0954229999551648, 1.3669106713764128, 1.1791],
]
scores = np.array(scores)

c_seen = 0.7726469288555016
c_unseen = 0.22735307114449846


def predict_seen_rate(x, y, z):
    if x != y:
        a = (z - y) / (x - y)
        return a
    else:
        return None


def predict_seen_rate2(x1, x2, y):
    # 方程式の両辺を二乗して c の値を計算する
    # これを行うために、式を変形して c の方程式を得る
    # c * x1**2 + x2**2 - c * x2**2 = y**2
    # c * (x1**2 - x2**2) = y**2 - x2**2
    # c = (y**2 - x2**2) / (x1**2 - x2**2)

    # 分母が0の場合は、解が存在しないためエラーを返す
    if x1**2 == x2**2:
        raise ValueError("x1^2 and x2^2 cannot be equal")

    c = (y**2 - x2**2) / (x1**2 - x2**2)

    return c


for score in scores:
    x, y, z = score
    score = c_seen * x + c_unseen * y
    score2 = math.sqrt(c_seen * x**2 + c_unseen * y**2)
    rate = predict_seen_rate(x, y, z)
    rate2 = predict_seen_rate2(x, y, z)
    print(
        f"seen: {x:0.4f}, unseen: {y:0.4f}, score: {score:0.4f}, score2: {score2:0.4f}, LB: {z:0.4f} (rate:{rate:0.4f}, rate2:{rate2:0.4f})"
    )


# 最小２乗誤差
# x, y, zをそれぞれ抽出
x = scores[:, 0]
y = scores[:, 1]
z = scores[:, 2]


def func(a):
    return np.sum((np.array(z) - (np.array(x) * a[0] + np.array(y) * (1 - a[0]))) ** 2)


def func2(a):
    return np.sum((np.array(z) - np.sqrt(np.power(x, 2) * a[0] + np.power(y, 2) * (1 - a[0]))) ** 2)


# 初期の推測値を設定します。ここでは0.5を使用します。
initial_guess = [0.5]

# 最適化の結果を見つけます。
result = minimize(func, initial_guess)
result2 = minimize(func2, initial_guess)
print(f"LB seen rate: rate1: {result.x[0]:0.4f}  rate2: {result2.x[0]:0.4f}")
