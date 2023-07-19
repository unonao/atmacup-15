import math

c_seen = 0.7726469288555016
c_unseen = 0.22735307114449846


def print_score(seen, unseen):
    score = c_seen * seen + c_unseen * unseen
    score2 = math.sqrt(c_seen * seen**2 + c_unseen * unseen**2)
    print(f"score: {score}, score2: {score2}")


print_score(1.143132400409859, 1.4139537431522253)
print_score(1.13855, 1.40413)
print_score(1.1302647706928934, 1.3961950862123547)
print_score(1.1293, 1.38101)
print_score(1.1281078631771408, 1.368889612584224)
print_score(1.1237222255903876, 1.3669475462129905)  # 005
print_score(1.0998172745188464, 1.3615243180966956)  # 006
print_score(1.095543318751611, 1.3685943824106688)  # 007
print_score(1.0954229999551648, 1.3606988560363749)  # 008
