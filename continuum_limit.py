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


def main():
    gamA = exchangeable_check()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # panel 1: smooth archetype
    ax = axes[0]
    ls = list(range(16))
    ax.plot(ls, gamA, "o", color="#2a78d6", label="discrete $\\gamma_\\ell$ (exact, $n{=}4$ space)")
    lfine = [x / 10 for x in range(5, 400)]
    ax.plot(lfine, [EH + 1 / (2 * math.log(2) * max(l, 1e-9)) for l in lfine],
            "-", color="#eb6834", label="Clarke–Barron  $E[h]+\\frac{1}{2\\ell\\ln 2}$")
    ax.axhline(EH, color="#898781", lw=1, label="$E_\\theta[h(\\theta)]=\\frac{1}{2\\ln 2}$")
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(0.7, 1.02)
    ax.set_xlabel("questions answered $\\ell$"); ax.set_ylabel("bits")
    ax.set_title("A: coin-mixture prior — smooth $1/\\ell$ tail")
    ax.legend(fontsize=8)

    # panel 2: condensation archetype, log scale
    ax = axes[1]
    guides = {}
    for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834")):
        gam, G = gatefree_gamma(n)
        guides[n] = (gam, G)
        ax.semilogy(range(len(gam)), [max(g, 1e-12) for g in gam], "o-",
                    color=col, ms=4, label=f"$n={n}$ (exact)")
    lref = list(range(1, 13))
    ax.semilogy(lref, [1.2 * 2 ** (-l) for l in lref], "--", color="#898781",
                label="$\\propto 2^{-\\ell}$ (pair-collision rate)")
    ax.set_xlabel("questions answered $\\ell$"); ax.set_ylabel("$\\gamma_\\ell$ (bits, log)")
    ax.set_title("B: gate-free class — exponential decay")
    ax.legend(fontsize=8)

    # panel 3: remaining entropy in fraction-time -> step at 0+
    ax = axes[2]
    for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834")):
        gam, G = guides[n]
        NQ = 1 << n
        rem = [(NQ - l) * gam[l] for l in range(NQ)] + [0.0]
        ts = [l / NQ for l in range(NQ + 1)]
        ax.plot(ts, [r / rem[0] for r in rem], "o-", color=col, ms=4,
                label=f"$n={n}$")
    ax.plot([0, 0, 1], [1, 0, 0], "--", color="#898781", label="$n\\to\\infty$: step at $t=0^+$")
    ax.set_xlabel("asked fraction $t=\\ell/2^n$")
    ax.set_ylabel("remaining $H(M)_t / H(M)_0$")
    ax.set_title("B: fraction-time collapse to condensation")
    ax.legend(fontsize=8)

    fig.suptitle("Continuum limits of the expected trajectory: the two classical archetypes")
    fig.tight_layout()
    fig.savefig("figures/continuum_limit.png", dpi=130)
    print("wrote figures/continuum_limit.png")


if __name__ == "__main__":
    main()
