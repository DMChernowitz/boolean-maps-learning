"""Does <L_l> converge as a function of x = l/2^n?  Test with an
exchangeable prior (mixture of iid coins, theta in {0.2, 0.7} with
weights 1/4, 3/4): G_k is closed-form, so we can push n up and compare
against the analytic limit
  L(x) = [g0 - (1-x) gbar] / (x gbar),
with g0 = h(mean theta) (the column entropy) and gbar = E h(theta)
(the conditional entropy for any t > 0)."""
import math
from math import comb

h = lambda x: 0 if x <= 0 or x >= 1 else -x*math.log2(x)-(1-x)*math.log2(1-x)
thetas = [(0.2, 0.25), (0.7, 0.75)]


def G(k):
    """mean block entropy of k answers under the mixture (log space)"""
    tot = 0.0
    lgc = lambda k, w: (math.lgamma(k+1) - math.lgamma(w+1)
                        - math.lgamma(k-w+1))
    for w in range(k + 1):
        logs = [math.log(wt) + w*math.log(th) + (k-w)*math.log(1-th)
                for th, wt in thetas]
        mx = max(logs)
        lpw = mx + math.log(sum(math.exp(z - mx) for z in logs))
        tot += math.exp(lgc(k, w) + lpw) * (-lpw / math.log(2))
    return tot


def L(n, l):
    Q = 2**n
    HM0 = Q * G(1)
    rec = G(l)
    rem = (Q - l) * (G(l + 1) - G(l))
    return (HM0 - rem) / rec


g0 = G(1)
gbar = sum(wt * h(th) for th, wt in thetas)
print(f"g0 = h(0.575) = {g0:.4f};  gbar = E h(theta) = {gbar:.4f}")
print("\n x   | n=4     n=6     n=8     n=10   | limit")
for x in (0.125, 0.25, 0.5, 0.75, 1.0):
    row = []
    for n in (4, 6, 8, 10):
        l = round(x * 2**n)
        row.append(L(n, l))
    lim = (g0 - (1 - x) * gbar) / (x * gbar)
    print(f" {x:.3f}| " + "  ".join(f"{v:.4f}" for v in row)
          + f" | {lim:.4f}")
