"""Finite-n curves for the weight (sparsity) prior: iid Bernoulli
coefficients rho_S ~ Bern(theta), psi = rho^T Z^{tensor n}.

Exact per-subset block entropy via the Fourier identity
Pr(psi(S)=y) = 2^{-l} sum_c (-1)^{c.y} eps^{wt(G_S c)}, eps = 1-2theta,
computed with a fast Walsh-Hadamard transform.  n = 4 exact-MC.
Writes figures/weight_step.png and prints landmarks."""
import random
from itertools import combinations
from math import comb, log2

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(20260820)
CF = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100"]

n = 4
Q = 16
POP = np.array([bin(x).count("1") for x in range(1 << Q)], dtype=np.int8)
COLS = [sum(1 << msk for msk in range(Q) if (msk & q) == msk)
        for q in range(Q)]


def fwht(a):
    a = a.copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        a[:, 0, :] = x + a[:, 1, :]
        a[:, 1, :] = x - a[:, 1, :]
        a = a.reshape(-1)
        h *= 2
    return a


def block_entropy(subset, eps):
    combos = np.zeros(1, dtype=np.int64)
    for q in subset:
        combos = np.concatenate([combos, combos ^ COLS[q]])
    f = eps ** POP[combos].astype(float)
    P = fwht(f) / len(f)
    P = np.clip(P, 1e-300, None)
    return float(-(P * np.log2(P)).sum())


def G_sequence(theta, nsub=300):
    eps = 1 - 2 * theta
    G = [0.0]
    for l in range(1, Q + 1):
        if comb(Q, l) <= nsub:
            subs = list(combinations(range(Q), l))
        else:
            subs = [tuple(random.sample(range(Q), l))
                    for _ in range(nsub)]
        G.append(float(np.mean([block_entropy(s, eps) for s in subs])))
    return G


def leverage(G):
    rows = []
    for l in range(1, Q + 1):
        rem = (Q - l) * (G[l + 1] - G[l]) if l < Q else 0.0
        rows.append((l / Q, (Q * G[1] - rem) / G[l]))
    return rows


def h2(x):
    return -x * log2(x) - (1 - x) * log2(1 - x)


fig, (axg, axl) = plt.subplots(1, 2, figsize=(9.2, 3.5))
for i, theta in enumerate([0.11, 0.25]):
    R = h2(theta)
    G = G_sequence(theta)
    # landmarks: bijection makes the full-table entropy exact
    assert abs(G[Q] - Q * h2(theta)) < 0.02, (theta, G[Q], Q * h2(theta))
    eps = 1 - 2 * theta
    h0 = np.mean([comb(n, k) * h2((1 + eps ** (2 ** k)) / 2)
                  for k in range(n + 1)]) * (n + 1) / Q
    h0 = sum(comb(n, k) * h2((1 + eps ** (2 ** k)) / 2)
             for k in range(n + 1)) / Q
    print(f"theta={theta}: h2={R:.4f}  G1={G[1]:.4f}  "
          f"h0(formula)={h0:.4f}  G_Q={G[Q]:.3f} (exact {Q*R:.3f})")
    gam = np.diff(G)
    t = (np.arange(Q) + 0.5) / Q
    axg.plot(t, gam, color=CF[i], lw=1.4, marker="o", ms=2.6,
             label=rf"$\theta={theta}$, $n=4$")
    axg.plot([0, R, R, 1], [1, 1, 0, 0], color=CF[i], ls="--", lw=1.1)
    rows = leverage(G)
    axl.plot([r[0] for r in rows], [r[1] for r in rows],
             color=CF[i], lw=1.4, marker="o", ms=2.6)
    axl.plot([0, R, R, 1], [1, 1, 1 / R, 1 / R],
             color=CF[i], ls="--", lw=1.1)
axg.set_xlabel(r"$t$")
axg.set_ylabel(r"$\gamma_{n,\ell}$")
axg.set_title("profile: weight prior, conjectured steps dashed",
              fontsize=10)
axl.set_xlabel(r"$t$")
axl.set_ylabel(r"$\langle L_\ell\rangle$")
axl.set_title(r"leverage toward $1/h_2(\theta)$", fontsize=10)
axg.legend(fontsize=8, frameon=False)
for ax in (axg, axl):
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(r"figures/weight_step.png", dpi=160)
print("weight_step.png done")
