"""
Numerical verification and figure for the learning_agent.md section
"The continuum limit: recovering the classical theory".

Two archetype priors, both realized exactly on our finite hypothesis
spaces and compared against their known continuum laws:

A. Smooth / parametric: the exchangeable coin-mixture prior
   p_j = Integral_0^1 th^w (1-th)^(N-w) dth = 1 / ((N+1) C(N, w_j)),
   w_j = number of 1s in the truth table (m=1). The predictive is
   Laplace's rule of succession, gamma_l = (1/(l+1)) sum_k h((k+1)/(l+2)),
   independent of n, and the continuum tail is Clarke-Barron:
   gamma_l -> E[h(theta)] + 1/(2 ln2 l)  (d=1 parameter).
   We verify that the generic block-entropy machinery on the full
   2^16-map space reproduces the closed form exactly.

B. Condensed / discrete class: uniform on the gate-free tier
   (2 constants + 2n literals, m=1). gamma_l decays exponentially,
   ~ 2^-l per surviving confusable pair (annealed pair-collision rate),
   so in fraction-time t = l/2^n the remaining-entropy curve collapses
   to a step at t = 0+: the first-order/condensation archetype.

Output: figures/continuum_limit.png (three panels).
"""

import math
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = lambda x: 0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)
H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
EH = 1 / (2 * math.log(2))     # E_{theta~U[0,1]}[h(theta)] = 1/(2 ln 2)


# ---------- A: exchangeable coin-mixture on the (4,1) space ----------
def exchangeable_check():
    n = 4
    NQ, N = 1 << n, 1 << (1 << n)
    # p_j = 1/((NQ+1) * C(NQ, w_j))
    from math import comb
    pw = [1 / ((NQ + 1) * comb(NQ, w)) for w in range(NQ + 1)]
    # block entropy on any l-subset (exchangeable: all subsets equal);
    # verify with two different subsets via direct pattern sums
    def blockH(S):
        d = {}
        for j in range(N):
            w = bin(j).count("1")
            key = tuple((j >> q) & 1 for q in S)
            d[key] = d.get(key, 0) + pw[w]
        return H(list(d.values()))
    G = [0.0]
    for l in range(1, NQ + 1):
        S1 = tuple(range(l))
        G.append(blockH(S1))
    # spot check exchangeability with a scattered subset
    assert abs(blockH((1, 4, 9, 14)) - G[4]) < 1e-12
    gamma_machine = [G[l + 1] - G[l] for l in range(NQ)]
    # closed form: rule of succession
    gamma_closed = [sum(h((k + 1) / (l + 2)) for k in range(l + 1)) / (l + 1)
                    for l in range(NQ)]
    for l in range(NQ):
        assert abs(gamma_machine[l] - gamma_closed[l]) < 1e-10, l
    print("A: machinery == rule-of-succession closed form, all l (n=4)")
    # Clarke-Barron tail: (gamma_l - E[h]) * l -> 1/(2 ln 2), log-slowly
    for l in (8, 15, 200, 5000, 20000):
        g = (gamma_closed[l] if l < NQ else
             sum(h((k + 1) / (l + 2)) for k in range(l + 1)) / (l + 1))
        print(f"   l={l:6d}: (gamma_l - E[h]) * l = {(g - EH) * l:.4f}"
              f"   (CB predicts {EH:.4f})")
    return gamma_closed


# ---------- B: gate-free (freeze-out) class, n = 2, 3, 4 ----------
def gatefree_gamma(n):
    NQ = 1 << n
    tables = [0, (1 << NQ) - 1]
    for i in range(n):
        t = sum(1 << q for q in range(NQ) if (q >> i) & 1)
        tables.append(t)
        tables.append(t ^ ((1 << NQ) - 1))
    L = len(tables)
    def blockH(S):
        d = {}
        for t in tables:
            key = tuple((t >> q) & 1 for q in S)
            d[key] = d.get(key, 0) + 1 / L
        return H(list(d.values()))
    G = [0.0]
    for l in range(1, NQ + 1):
        subs = list(combinations(range(NQ), l))
        G.append(sum(blockH(S) for S in subs) / len(subs))
    return [G[l + 1] - G[l] for l in range(NQ)], G


def gatefree_gamma_mc(n, samples=4000, seed=0):
    """gamma_l for the gate-free class at larger n: subsets sampled,
    block entropies per sampled subset exact (Rao-Blackwellized)."""
    import random
    rng = random.Random(seed)
    NQ = 1 << n
    tables = [0, (1 << NQ) - 1]
    for i in range(n):
        t = sum(1 << q for q in range(NQ) if (q >> i) & 1)
        tables.append(t)
        tables.append(t ^ ((1 << NQ) - 1))
    L = len(tables)
    def blockH(S):
        d = {}
        for t in tables:
            key = tuple((t >> q) & 1 for q in S)
            d[key] = d.get(key, 0) + 1 / L
        return H(list(d.values()))
    gam = []
    for l in range(NQ):
        tot = 0.0
        for _ in range(samples):
            S = rng.sample(range(NQ), l)
            qp = rng.choice([q for q in range(NQ) if q not in S])
            tot += blockH(tuple(sorted(S + [qp]))) - blockH(tuple(sorted(S)))
        gam.append(tot / samples)
    return gam


def universal_annealed(x):
    """Annealed survivor curve: wrong survivors M ~ Poisson(2^-x); given
    M, the fresh column's predictive is (1+B)/(M+1), B ~ Binom(M, 1/2)."""
    from math import comb, exp
    mu = 2.0 ** (-x)
    tot = 0.0
    for M in range(0, 60):
        pM = exp(-mu) * mu ** M / math.factorial(M)
        if pM < 1e-14 and M > mu:
            break
        eb = sum(comb(M, b) * 0.5 ** M * h((1 + b) / (M + 1)) for b in range(M + 1))
        tot += pM * eb
    return tot


def main():
    gamA = exchangeable_check()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # panel 1: smooth archetype on LOG-LOG axes (the law is a tail law)
    ax = axes[0]
    lbig = [int(round(10 ** (i / 12))) for i in range(0, 55)]
    lbig = sorted(set(l for l in lbig if l >= 1))
    master = [sum(h((k + 1) / (l + 2)) for k in range(l + 1)) / (l + 1) - EH
              for l in lbig]
    ax.loglog(lbig, master, "-", color="#2a78d6",
              label="master curve $\\gamma_\\ell - E[h]$ (closed form)")
    ax.loglog(range(1, 16), [gamA[l] - EH for l in range(1, 16)], "o",
              color="#0d366b", ms=5,
              label="exact machinery, $(4,1)$ space")
    ax.loglog(lbig, [EH / l for l in lbig], "--", color="#eb6834",
              label="Clarke–Barron $\\frac{1}{2\\ell\\ln 2}$")
    ax.set_xlabel("questions answered $\\ell$ (log)")
    ax.set_ylabel("$\\gamma_\\ell - E[h(\\theta)]$ (log)")
    ax.set_title("A: coin-mixture — tail law on log–log")
    ax.legend(fontsize=8)

    # panel 2: condensation window, shifted axis x = l - log2(L-1)
    ax = axes[1]
    data = {}
    for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834")):
        gam, _ = gatefree_gamma(n)
        data[n] = gam
        L = 2 * n + 2
        xs = [l - math.log2(L - 1) for l in range(len(gam))]
        ax.plot(xs, gam, "o-", ms=4, color=col, label=f"$n={n}$ (exact)")
    for n, col in ((5, "#eda100"), (6, "#e87ba4")):
        gam = gatefree_gamma_mc(n)
        data[n] = gam
        L = 2 * n + 2
        xs = [l - math.log2(L - 1) for l in range(len(gam))]
        ax.plot(xs, gam, ".", ms=4, color=col, label=f"$n={n}$ (subset MC)")
    xg = [x / 10 for x in range(-40, 90)]
    ax.plot(xg, [universal_annealed(x) for x in xg], "--", color="#898781",
            label="universal annealed (Poisson survivors)")
    ax.set_xlim(-4, 9)
    ax.set_xlabel("shifted time  $x = \\ell - \\log_2(L-1)$")
    ax.set_ylabel("$\\gamma$ (bits)")
    ax.set_title("B: gate-free class — window collapse")
    ax.legend(fontsize=8)
    # collapse quality at fixed shifted x (linear interpolation in l)
    def interp(n, x):
        L = 2 * n + 2
        l = x + math.log2(L - 1)
        lo = int(math.floor(l))
        f = l - lo
        g = data[n]
        if lo < 0 or lo + 1 >= len(g):
            return None
        return (1 - f) * g[lo] + f * g[lo + 1]
    print("B collapse (values at fixed x, -> universal as n grows):")
    for x in (-1.0, 0.0, 1.0, 2.0):
        vals = "  ".join(f"n={n}:{interp(n, x):.3f}" for n in (2, 3, 4, 5, 6)
                         if interp(n, x) is not None)
        print(f"  x={x:4}: universal {universal_annealed(x):.3f}   {vals}")

    # panel 3: same data in fraction-time -> degenerate step (why the
    # t-axis is the wrong microscope for this archetype)
    ax = axes[2]
    for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834"),
                   (5, "#eda100"), (6, "#e87ba4")):
        gam = data[n]
        NQ = 1 << n
        rem = [(NQ - l) * gam[l] for l in range(NQ)] + [0.0]
        ts = [l / NQ for l in range(NQ + 1)]
        ax.plot(ts, [r / rem[0] for r in rem], "-", color=col, lw=1.5,
                label=f"$n={n}$")
    ax.plot([0, 0, 1], [1, 0, 0], "--", color="#898781",
            label="$n\\to\\infty$: step at $t=0^+$")
    ax.set_xlabel("asked fraction $t=\\ell/2^n$")
    ax.set_ylabel("remaining $H(M)_t / H(M)_0$")
    ax.set_title("B: fraction-time is degenerate (step at $0^+$)")
    ax.legend(fontsize=8)

    fig.suptitle("Continuum limits of the expected trajectory: correct scalings for the two archetypes")
    fig.tight_layout()
    fig.savefig("figures/continuum_limit.png", dpi=130)
    print("wrote figures/continuum_limit.png")


if __name__ == "__main__":
    main()
