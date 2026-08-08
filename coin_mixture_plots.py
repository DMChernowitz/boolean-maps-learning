"""Leverage curves L(x) for coin-mixture priors with 1, 2, 3 and 10
coins: exact finite-n curves from the G_k laws against the limiting
hyperbola L(x) = 1 + c/x, c = (h(mean theta) - mean h(theta))/mean h.
Output: figures/coin_mixture_leverage.png"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = lambda x: 0 if x <= 0 or x >= 1 else -x*math.log2(x)-(1-x)*math.log2(1-x)


def G(k, coins):
    """mean block entropy of k answers (log-space, exact)"""
    lgc = lambda k, w: (math.lgamma(k+1) - math.lgamma(w+1)
                        - math.lgamma(k-w+1))
    tot = 0.0
    for w in range(k + 1):
        logs = [math.log(wt) + w*math.log(th) + (k-w)*math.log(1-th)
                for th, wt in coins]
        mx = max(logs)
        lpw = mx + math.log(sum(math.exp(z - mx) for z in logs))
        tot += math.exp(lgc(k, w) + lpw) * (-lpw / math.log(2))
    return tot


def L_curve(n, coins):
    Q = 2**n
    Gs = [G(k, coins) for k in range(Q + 2)]
    HM0 = Q * Gs[1]
    xs, Ls = [], []
    for l in range(1, Q + 1):
        rem = (Q - l) * (Gs[l + 1] - Gs[l]) if l < Q else 0.0
        xs.append(l / Q)
        Ls.append((HM0 - rem) / Gs[l])
    return xs, Ls


FAMILIES = [
    ("1 coin",  [(0.45, 1.0)]),
    ("2 coins", [(0.2, 0.25), (0.7, 0.75)]),
    ("3 coins", [(0.2, 0.2), (0.45, 0.5), (0.7, 0.3)]),
    ("10 coins", [(0.05 + 0.1*i, (i+1)/55) for i in range(10)]),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, (name, coins) in zip(axes, FAMILIES):
    tbar = sum(th*wt for th, wt in coins)
    gbar = sum(wt*h(th) for th, wt in coins)
    c = (h(tbar) - gbar) / gbar
    for n, col in ((4, "#9ec5f4"), (6, "#5598e7"), (8, "#2a78d6")):
        xs, Ls = L_curve(n, coins)
        ax.plot(xs, Ls, color=col, lw=1.4, label=f"$n={n}$")
    xs = [i/400 for i in range(8, 401)]
    ax.plot(xs, [1 + c/x for x in xs], "--", color="#eb6834",
            lw=1.6, label="$1 + c/x$")
    ax.set_title(f"{name}:  $c = {c:.3f}$")
    ax.set_xlabel("$x = \\ell/2^n$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.95, 3.4)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("$\\langle L \\rangle$")
axes[0].legend(fontsize=9)
fig.suptitle("Coin-mixture priors: exact finite-$n$ leverage against "
             "the limit hyperbola")
fig.tight_layout()
fig.savefig("figures/coin_mixture_leverage.png", dpi=130)
print("wrote figures/coin_mixture_leverage.png")
for name, coins in FAMILIES:
    tbar = sum(th*wt for th, wt in coins)
    gbar = sum(wt*h(th) for th, wt in coins)
    print(f"{name}: h(mean)={h(tbar):.4f} gbar={gbar:.4f} "
          f"c={(h(tbar)-gbar)/gbar:.4f}")
