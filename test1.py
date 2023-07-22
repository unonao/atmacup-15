import math

c_seen = 0.7726469288555016
c_unseen = 0.22735307114449846


def print_score(seen, unseen):
    score = c_seen * seen + c_unseen * unseen
    score2 = math.sqrt(c_seen * seen**2 + c_unseen * unseen**2)
    print(f"score: {score}, score2: {score2}")


print_score(1.0862292649451508, 1.3519037725794139)  # 015 linear

# 未サブ
print_score(1.0838408226511915, 1.3519037725794139)  # 016 linear
print_score(1.0794302161264508, 1.35084852596117)  # 017 linear
