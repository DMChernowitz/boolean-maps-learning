"""
Numerical verification and figures for the learning_agent.md section
"The expected trajectory: a discrete derivative of block entropies".

For arbitrary p, n, m, with the double average (question set uniform
over all l-subsets, truth psi ~ p, i.e. answers by their predictive
probabilities):

  received(l)  = E[cumulative surprisal]      = G_l
  remaining(l) = E[H(M) after l questions]    = (2^n - l)(G_{l+1} - G_l)
  deduced(l)   = H(M)_0 - remaining(l) - G_l  >= 0

with G_k the mean joint entropy of a random k-question block
(G_0 = 0, G_{2^n} = H(p)). The master law is verified against brute
force, and the conservation stack (remaining + received + deduced
= H(M)_0 at every l) is rendered for a (3,1) Gibbs prior with fields
gamma=1.2, lambda=0.25 and mu in {0, 0.25}
(figures/expected_trajectory_3to1.png).
"""

import csv
import math
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_INPUT = 3
NQ = 1 << N_INPUT
GAMMA, LAMBDA = 1.2, 0.25

h2 = lambda x: 0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)
H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)


def load31():
    C, F, W = [], [], []
    with open("output/table_3to1.csv") as fh:
        r = csv.reader(fh)
        hd = next(r)
        iC, iF, iW = (hd.index(k) for k in
                      ("circuit_complexity", "footprint", "weight_bias"))
        for row in r:
            C.append(int(row[iC])); F.append(int(row[iF])); W.append(int(row[iW]))
    return C, F, W


def gibbs(C, F, W, mu):
    E = [GAMMA * c + LAMBDA * f + mu * w for c, f, w in zip(C, F, W)]
    m = min(E)
    w = [math.exp(-(e - m)) for e in E]
    Z = sum(w)
    return [x / Z for x in w]


def block_entropy(p, S):
    d = {}
    for j, x in enumerate(p):
        if x <= 0:
            continue
        key = tuple((j >> q) & 1 for q in S)
        d[key] = d.get(key, 0) + x
    return H(list(d.values()))


def G(p, k):
    subs = list(combinations(range(NQ), k))
    return sum(block_entropy(p, S) for S in subs) / len(subs)


def brute_remaining(p, l):
    subs = list(combinations(range(NQ), l))
    tot = 0.0
    for S in subs:
        for qp in range(NQ):
            if qp in S:
                continue
            tot += block_entropy(p, tuple(sorted(S + (qp,)))) - block_entropy(p, S)
    return tot / len(subs)


def curves(p):
    Gs = [G(p, k) for k in range(NQ + 1)]
    HM0 = NQ * Gs[1] * 0 + sum(h2(sum(x for j, x in enumerate(p) if (j >> q) & 1))
                               for q in range(NQ))
    rem = [(NQ - l) * (Gs[l + 1] - Gs[l]) for l in range(NQ)] + [0.0]
    rec = Gs[:]
    ded = [HM0 - rem[l] - rec[l] for l in range(NQ + 1)]
    # verify master law and conservation
    for l in (0, 1, 3, 5):
        assert abs(rem[l] - brute_remaining(p, l)) < 1e-9, l
    assert abs(rem[0] - HM0) < 1e-9
    assert all(d > -1e-9 for d in ded), "deduced must be nonnegative"
    return HM0, rem, rec, ded, Gs


def main():
    C, F, W = load31()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, mu in zip(axes, (0.0, 0.25)):
        p = gibbs(C, F, W, mu)
        HM0, rem, rec, ded, Gs = curves(p)
        ls = list(range(NQ + 1))
        ax.bar(ls, rem, color="#2a78d6", label="remaining $H(M)$")
        ax.bar(ls, rec, bottom=rem, color="#eb6834",
               label="received $G_\\ell$ (cum. surprisal)")
        ax.bar(ls, ded, bottom=[a + b for a, b in zip(rem, rec)],
               color="#1baf7a", label="deduced (via correlations)")
        ax.axhline(HM0, color="#898781", lw=1)
        lev_end = HM0 / Gs[-1]
        ax.set_title(f"$\\mu={mu}$:  $H(M)_0={HM0:.2f}$,  $H(p)={Gs[-1]:.2f}$,"
                     f"  run leverage $={lev_end:.2f}$")
        ax.set_xlabel("unique questions answered $\\ell$")
        print(f"mu={mu}: H(M)0={HM0:.3f} H(p)={Gs[-1]:.3f} "
              f"run leverage={lev_end:.3f}")
        print("  l:        " + " ".join(f"{l:6d}" for l in ls))
        print("  remaining " + " ".join(f"{x:6.3f}" for x in rem))
        print("  received  " + " ".join(f"{x:6.3f}" for x in rec))
        print("  deduced   " + " ".join(f"{x:6.3f}" for x in ded))
    axes[0].set_ylabel("bits")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle(f"(3,1) Gibbs prior, $\\gamma={GAMMA}$, $\\lambda={LAMBDA}$: "
                 "conservation stack (bars sum to $H(M)_0$ at every $\\ell$)")
    fig.tight_layout()
    fig.savefig("figures/expected_trajectory_3to1.png", dpi=130)
    print("wrote figures/expected_trajectory_3to1.png")


if __name__ == "__main__":
    main()
