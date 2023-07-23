import math

c_seen = 0.7726469288555016
c_unseen = 0.22735307114449846


def print_score(seen, unseen):
    score = c_seen * seen + c_unseen * unseen
    score2 = math.sqrt(c_seen * seen**2 + c_unseen * unseen**2)
    print(f"seen:{seen:0.5f}, unseen: {unseen:0.5f},  score: {score:0.5f}, score2: {score2:0.5f}")


print_score(1.0838408226511915, 1.3519037725794139)  # 016 linear # 未サブ
print_score(1.0862292649451508, 1.3519037725794139)  # 015 linear 1.1721
print_score(1.0794302161264508, 1.35084852596117)  # 017 linear 1.1675
print_score(1.0788609685619033, 1.35084852596117)  # 018 # 未サブ
print_score(1.0765981445833601, 1.35084852596117)  # 019 # 未サブ 1.1683
print_score(1.0763797486419528, 1.3500891660509984)  # 020 # 未サブ
print_score(1.0763797486419528, 1.3485249836424367)  # 021 # 未サブ
print_score(1.0762322505007196, 1.3483221496198832)  # 022 # 未サブ
print_score(1.075646319584694, 1.3483221496198832)  # 023 # 未サブ
print_score(1.0728419192526708, 1.3483221496198832)  # 024 # 未サブ
