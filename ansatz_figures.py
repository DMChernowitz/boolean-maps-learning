"""Figures for the ansatz section of learning_agent.md: surprisal,
H(M) before, and H(M) after one question, in the correct / wrong /
expected cases, as functions of the spike weight p.

Twin ansatz: complementary twins (d = 2^n disagreement questions),
asking any q in D. H(M) after is 0 in every branch.

Spike ansatz: exact finite-N formulas. The confirming branch stays
spike+uniform on N' = N/2^m maps with p' = p/M*; the refuting branch
is uniform on N/2^m maps, so H(M) after = (2^n - 1) m.

Outputs figures/ansatz_{twin,spike}_{4to1,3to2}.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log2 = np.log2


def h(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 1)
    out[ok] = -x[ok] * log2(x[ok]) - (1 - x[ok]) * log2(1 - x[ok])
    return out


def spike_curves(n, m, p):
    NQ, K = 2 ** n, 2 ** m
    N = float(K) ** NQ
    beta = (1 - p) / (N - 1)
    Mstar = p + (N / K - 1) * beta
    Moth = (N / K) * beta
    Hcol = -Mstar * log2(Mstar) - (K - 1) * Moth * log2(Moth)
    Hbefore = NQ * Hcol
    # confirming branch: spike+uniform on N' = N/K, p' = p/M*
    Np = N / K
    pp = p / Mstar
    b2 = beta / Mstar
    Ms2 = pp + (Np / K - 1) * b2
    Mo2 = (Np / K) * b2
    Hcol2 = -Ms2 * log2(Ms2) - (K - 1) * Mo2 * log2(Mo2)
    H_ok = (NQ - 1) * Hcol2
    s_ok = -log2(Mstar)
    # refuting branch: uniform on N/K survivors
    H_bad = (NQ - 1) * m * np.ones_like(p)
    s_bad = -log2(Moth)
    # expectations over the answer
    P_ok = Mstar
    P_bad = (K - 1) * Moth
    s_exp = P_ok * s_ok + P_bad * s_bad          # = Hcol, the lemma
    H_exp = P_ok * H_ok + P_bad * H_bad
    return dict(Hbefore=Hbefore, s_ok=s_ok, s_bad=s_bad, s_exp=s_exp,
                H_ok=H_ok, H_bad=H_bad, H_exp=H_exp)


def twin_curves(n, m, p):
    NQ = 2 ** n
    d = NQ  # complementary twins
    Hbefore = d * h(p)
    zero = np.zeros_like(p)
    return dict(Hbefore=Hbefore,
                s_ok=-log2(p), s_bad=-log2(1 - p), s_exp=h(p),
                H_ok=zero, H_bad=zero, H_exp=zero)


CASES = [
    ("s_ok",  "surprisal, correct",     "#1baf7a", "-"),
    ("s_bad", "surprisal, wrong",       "#d62d2d", "-"),
    ("s_exp", "surprisal, expected",    "#555555", "-"),
    ("H_ok",  "$H(M)$ after, correct",  "#1baf7a", "--"),
    ("H_bad", "$H(M)$ after, wrong",    "#d62d2d", "--"),
    ("H_exp", "$H(M)$ after, expected", "#555555", "--"),
]


def make_fig(kind, n, m, fname):
    p = np.linspace(1e-4, 1 - 1e-4, 4000)
    c = spike_curves(n, m, p) if kind == "spike" else twin_curves(n, m, p)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(p, c["Hbefore"], color="#2a78d6", lw=2.2, ls=":",
            label="$H(M)$ before (all cases)")
    for key, lab, col, ls in CASES:
        ax.plot(p, c[key], color=col, ls=ls, lw=1.6, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, m * 2 ** n + 2)
    ax.set_xlabel("$p$")
    ax.set_ylabel("bits")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.set_title(f"{kind} ansatz, $(n,m) = ({n},{m})$"
                 + ("  (complementary twins, $d = 2^n$)"
                    if kind == "twin" else ""))
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("wrote", fname)


def make_leverage_fig(n, m, fname):
    p = np.linspace(1e-4, 1 - 1e-7, 6000)
    c = spike_curves(n, m, p)
    NQ, K = 2 ** n, 2 ** m
    dH_ok = c["Hbefore"] - c["H_ok"]
    dH_bad = c["Hbefore"] - c["H_bad"]
    L_ok = dH_ok / c["s_ok"]
    L_bad = dH_bad / c["s_bad"]
    # expected leverage = ratio of expected totals
    P_ok = 2.0 ** (-c["s_ok"])
    P_bad = 1 - P_ok
    L_exp = (P_ok * dH_ok + P_bad * dH_bad) / c["s_exp"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="#2a78d6", lw=1.0, ls=":",
               label="uniform-prior baseline $L = 1$")
    ax.plot(p, L_ok, color="#1baf7a", lw=1.8, label="leverage, correct")
    ax.plot(p, L_bad, color="#d62d2d", lw=1.8, label="leverage, wrong")
    ax.plot(p, L_exp, color="#555555", lw=1.8, label="leverage, expected")
    ax.set_xlim(0, 1)
    ax.set_ylim(-4, 12)
    ax.set_xlabel("$p$")
    ax.set_ylabel("$L_1 = \\Delta H(M) / s$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"spike ansatz leverage after one question, "
                 f"$(n,m) = ({n},{m})$")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("wrote", fname)


def make_culling_fig(fname):
    """<L_1>(p->1) * 2^-n in the n -> infinity limit: the culling
    ratio 1 - 2^-m, as a function of m."""
    ms = np.arange(1, 11)
    mc = np.linspace(1, 10, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(1, color="#555555", lw=1.0, ls="--",
               label="twin ceiling ($m \\to \\infty$)")
    ax.plot(mc, 1 - 2.0 ** (-mc), color="#2a78d6", lw=1.2, alpha=0.5)
    ax.plot(ms, 1 - 2.0 ** (-ms), "o", color="#2a78d6", ms=6,
            label="$1 - 2^{-m}$ (integer $m$)")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ms)
    ax.set_xlabel("$m$")
    ax.set_ylabel("$\\langle L_1 \\rangle \\, 2^{-n}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("spike ansatz: limiting leverage per question,\n"
                 "$p \\to 1$, $n \\to \\infty$")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("wrote", fname)


for n, m, tag in ((4, 1, "4to1"), (3, 2, "3to2")):
    make_fig("spike", n, m, f"figures/ansatz_spike_{tag}.png")
    make_fig("twin", n, m, f"figures/ansatz_twin_{tag}.png")
    make_leverage_fig(n, m, f"figures/ansatz_leverage_{tag}.png")
make_culling_fig("figures/ansatz_culling_ratio.png")
