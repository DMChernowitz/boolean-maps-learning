"""Plots for thermodynamic_limit.md section 6: Reed-Muller step
profiles (exact n=3, Monte Carlo n=5,7) and Ising smoothness-prior
profiles from the cavity fixed point."""
import itertools
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CF = ["#1BAF7A", "#2A78D6", "#EB6834", "#EDA100", "#E87BA4"]
OUT = r"c:\Users\dc300\boolean_maps_learning\figures"

random.seed(20260814)


# ---------------- Reed-Muller: columns as ints over F2 ----------------
def rm_columns(n, d):
    monos = [s for r in range(d + 1)
             for s in itertools.combinations(range(n), r)]
    cols = []
    for q in range(2 ** n):
        bits = [(q >> i) & 1 for i in range(n)]
        c = 0
        for idx, s in enumerate(monos):
            val = 1
            for i in s:
                val &= bits[i]
            c |= val << idx
        cols.append(c)
    return cols, len(monos)


def rank_path(cols, order):
    """Rank after each prefix of columns in the given order."""
    basis = {}          # leading-bit -> vector
    path = [0]
    r = 0
    for j in order:
        v = cols[j]
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                r += 1
                break
        path.append(r)
    return path


def rm_exact(n, d):
    """Exact G_l by enumerating all subsets (n=3 only)."""
    cols, K = rm_columns(n, d)
    Q = 2 ** n
    tot = [0.0] * (Q + 1)
    cnt = [0] * (Q + 1)
    for mask in range(2 ** Q):
        idx = [j for j in range(Q) if (mask >> j) & 1]
        r = rank_path(cols, idx)[-1]
        tot[len(idx)] += r
        cnt[len(idx)] += 1
    return np.array([tot[l] / cnt[l] for l in range(Q + 1)]), K


def rm_mc(n, d, nperm):
    cols, K = rm_columns(n, d)
    Q = 2 ** n
    acc = np.zeros(Q + 1)
    order = list(range(Q))
    for _ in range(nperm):
        random.shuffle(order)
        acc += rank_path(cols, order)
    return acc / nperm, K


def curves(G, Q):
    """gamma_l (l=0..Q-1) and <L_l> (l=1..Q) from a G_l sequence."""
    gam = np.diff(G)
    ell = np.arange(1, Q + 1)
    lev = np.empty(Q)
    lev[:Q - 1] = (Q * G[1] - (Q - ell[:Q - 1]) * gam[1:]) / G[1:Q]
    lev[Q - 1] = Q * G[1] / G[Q]
    return gam, ell, lev


fig, (axg, axl) = plt.subplots(1, 2, figsize=(9.2, 3.5))
runs = [(3, 1, "exact", None), (5, 2, "mc", 6000), (7, 3, "mc", 2500)]
for i, (n, d, mode, nperm) in enumerate(runs):
    Q = 2 ** n
    G, K = rm_exact(n, d) if mode == "exact" else rm_mc(n, d, nperm)
    assert 2 * K == Q, (n, d, K)
    gam, ell, lev = curves(G, Q)
    tg = (np.arange(Q) + 0.5) / Q
    axg.plot(tg, gam, color=CF[i], lw=1.4,
             marker="o", ms=2.4, label=f"$n={n}$, $d={d}$")
    axl.plot(ell / Q, lev, color=CF[i], lw=1.4, marker="o", ms=2.4)

axg.plot([0, .5, .5, 1], [1, 1, 0, 0], "k--", lw=1.1,
         label=r"limit $\mathbf{1}[t<R]$")
axl.plot([0, .5, .5, 1], [1, 1, 2, 2], "k--", lw=1.1)
axg.set_xlabel(r"$t=\ell/Q$")
axg.set_ylabel(r"$\gamma_{n,\ell}$")
axg.set_title(r"profile: median-degree RM, $R=\frac{1}{2}$",
              fontsize=10)
axl.set_xlabel(r"$t=\ell/Q$")
axl.set_ylabel(r"$\langle L_\ell\rangle$")
axl.set_title(r"leverage: step $1\to 1/R$", fontsize=10)
axg.legend(fontsize=8, frameon=False)
for ax in (axg, axl):
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(OUT + r"\rm_step.png", dpi=160)
print("rm_step.png done")


# ---------------- Ising cavity fixed point ----------------
nodes, weights = np.polynomial.hermite.hermgauss(81)
z = nodes * np.sqrt(2.0)
w = weights / np.sqrt(np.pi)


def h2(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def gamma_ising(b, t):
    q = 0.0
    for _ in range(4000):
        v = b * b * (t + (1 - t) * q)
        qn = float(np.sum(w * np.tanh(np.sqrt(v) * z) ** 2))
        if abs(qn - q) < 1e-14:
            q = qn
            break
        q = qn
    v = b * b * (t + (1 - t) * q)
    return float(np.sum(w * h2((1 + np.tanh(np.sqrt(v) * z)) / 2)))


ts = np.unique(np.concatenate([np.linspace(0, 0.02, 101),
                               np.linspace(0.02, 1, 393)]))
fig, (axg, axl) = plt.subplots(1, 2, figsize=(9.2, 3.5))
for i, b in enumerate([0.35, 0.7, 0.95]):
    gam = np.array([gamma_ising(b, t) for t in ts])
    g = np.concatenate([[0], np.cumsum((gam[1:] + gam[:-1]) / 2
                                       * np.diff(ts))])
    L = np.where(g > 0, (1 - (1 - ts) * gam) / np.where(g > 0, g, 1), 1.0)
    L[0] = 1.0
    axg.plot(ts, gam, color=CF[i], lw=1.6, label=f"$b={b}$")
    axl.plot(ts, L, color=CF[i], lw=1.6)
axg.set_xlabel(r"$t$")
axg.set_ylabel(r"$\gamma(t)$")
axg.set_title(r"profile: smoothness prior, $\beta=b/\sqrt{n}$",
              fontsize=10)
axl.set_xlabel(r"$t$")
axl.set_ylabel(r"$L(t)$")
axl.set_title("leverage: finite start, smooth decline", fontsize=10)
axg.legend(fontsize=8, frameon=False)
for ax in (axg, axl):
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(OUT + r"\ising_smooth.png", dpi=160)
print("ising_smooth.png done")


# ------------- multicanonical beta bump: smooth interior peak -------------
ts = np.linspace(0, 1, 2001)
fig, (axg, axl) = plt.subplots(1, 2, figsize=(9.2, 3.5))
for i, wgt in enumerate([0.55, 0.8, 0.95]):
    gam = 1 - wgt * (3 * ts ** 2 - 2 * ts ** 3)
    L = ((1 + wgt * ts * (1 - ts) * (3 - 2 * ts))
         / (1 - wgt * ts ** 2 * (1 - ts / 2)))
    k = int(np.argmax(L))
    print(f"w={wgt}: peak L={L[k]:.4f} at t={ts[k]:.4f}, "
          f"L(1)={L[-1]:.4f}")
    axg.plot(ts, gam, color=CF[i], lw=1.6, label=f"$w={wgt}$")
    axl.plot(ts, L, color=CF[i], lw=1.6)
    axl.plot(ts[k], L[k], "o", color=CF[i], ms=4)
axg.set_xlabel(r"$t$")
axg.set_ylabel(r"$\gamma(t)$")
axg.set_title(r"profile: $\mu=w\,\mathrm{Beta}(2,2)+(1-w)\,\delta_1$",
              fontsize=10)
axl.set_xlabel(r"$t$")
axl.set_ylabel(r"$L(t)$")
axl.set_title("leverage: smooth interior maximum", fontsize=10)
axg.legend(fontsize=8, frameon=False)
for ax in (axg, axl):
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(OUT + r"\rm_peak.png", dpi=160)
print("rm_peak.png done")


# ------------- order-statistic priors: one and two uniform draws -------------
ts = np.linspace(0, 1, 1001)
fig, (axg, axl) = plt.subplots(1, 2, figsize=(9.2, 3.5))
cases = [(r"$R=U$ (one draw)", np.ones_like(ts) - ts,
          2 * np.ones_like(ts)),
         (r"$R=\min(U_1,U_2)$", (1 - ts) ** 2,
          3 * np.ones_like(ts)),
         (r"$R=\max(U_1,U_2)$", 1 - ts ** 2,
          3 * (1 + ts - ts ** 2) / (3 - ts ** 2))]
for i, (lab, gam, L) in enumerate(cases):
    axg.plot(ts, gam, color=CF[i], lw=1.6, label=lab)
    axl.plot(ts, L, color=CF[i], lw=1.6)
axg.set_xlabel(r"$t$")
axg.set_ylabel(r"$\gamma(t)$")
axg.set_title("profiles: order-statistic class priors", fontsize=10)
axl.set_xlabel(r"$t$")
axl.set_ylabel(r"$L(t)$")
axl.set_title(r"leverage: $2$, $3$, and the rising $\max$ curve",
              fontsize=10)
axl.set_ylim(0.9, 3.2)
axg.legend(fontsize=8, frameon=False)
for ax in (axg, axl):
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(OUT + r"\rm_orderstat.png", dpi=160)
print("rm_orderstat.png done")
