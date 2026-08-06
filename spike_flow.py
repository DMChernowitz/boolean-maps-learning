"""Numerical verification for the learning_agent.md section on the
spike ansatz as a two-state machine (spike vs uniform, absorbing).

State after k questions: spike(p_k) on N_k = N 2^{-mk} maps with
probability q_k, uniform with probability 1 - q_k.  Recursions:

  q_{k+1} = q_k M*(p_k),      p_{k+1} = p_k / M*(p_k),

with M*(p) the spike-branch predictive probability of the special
answer.  Checks:

  1. the product q_k p_k is invariant (= p_0): the belief martingale;
  2. the recursion solution equals the direct Bayes answer
     q_k = P(special block) exactly, and q_k = p0 + (1-p0) 2^{-mk}
     in the large-N form;
  3. branch-weighted expected surprisal telescopes to the block
     entropy G_k, and branch-weighted <H(M_k)> equals the master law
     (2^n - k)(G_{k+1} - G_k)  [brute-forced at (2,2)];
  4. the delta-limit cumulative leverage
     <L_k> -> (1-2^-m)[2^n - (2^n-k) 2^{-mk}] / (1 - 2^{-mk})
     matches the exact machinery at p0 = 1 - 1e-9, and reproduces the
     k=1 and k=2^n limits already derived.
"""
import math

H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)


def exact_chain(n, m, p0, kmax):
    NQ, K = 2**n, 2**m
    N = float(K)**NQ
    beta0 = (1 - p0) / (N - 1)
    qs, ps = [1.0], [p0]
    q, p = 1.0, p0
    Nk = N
    for k in range(kmax):
        beta = (1 - p) / (Nk - 1)
        Mstar = p + (Nk / K - 1) * beta
        q, p, Nk = q * Mstar, p / Mstar, Nk / K
        qs.append(q)
        ps.append(p)
    return qs, ps


def G_block(n, m, p0, k):
    NQ, K = 2**n, 2**m
    N = float(K)**NQ
    beta = (1 - p0) / (N - 1)
    Pk = p0 + (N / K**k - 1) * beta
    uk = (N / K**k) * beta
    t = 0.0
    if 0 < Pk < 1:
        t -= Pk * math.log2(Pk)
    if uk > 0:
        t -= (K**k - 1) * uk * math.log2(uk)
    return t


def Hcol(m, p, Nk):
    K = 2**m
    beta = (1 - p) / (Nk - 1)
    Ms = p + (Nk / K - 1) * beta
    Mo = (Nk / K) * beta
    return H([Ms] + [Mo] * (K - 1))


# ---- checks 1 and 2 ----
for (n, m, p0) in ((4, 1, 0.35), (3, 2, 0.7), (4, 1, 0.995)):
    NQ, K = 2**n, 2**m
    N = float(K)**NQ
    beta0 = (1 - p0) / (N - 1)
    qs, ps = exact_chain(n, m, p0, NQ)
    for k in range(NQ + 1):
        assert abs(qs[k] * ps[k] - p0) < 1e-12          # martingale
        q_direct = p0 + (N / K**k - 1) * beta0           # Bayes block prob
        assert abs(qs[k] - q_direct) < 1e-12
        q_bigN = p0 + (1 - p0) * K**(-k)                 # large-N form
        # exact error is (1-p0)(1 - K^-k)/(N-1): uniformly O(1/N)
        assert abs(qs[k] - q_bigN) <= (1 - p0) / (N - 1) + 1e-12
print("checks 1-2: martingale q_k p_k = p0; recursion == Bayes block "
      "probability == large-N closed form")

# ---- check 3: chain vs G_k and master law, brute force at (2,2) ----
n, m, p0 = 2, 2, 0.55
NQ, K = 2**n, 2**m
N = K**NQ
beta = (1 - p0) / (N - 1)
pv = [beta] * N
pv[0] = p0                                  # special map j = 0
dig = lambda j, q: (j >> (m*q)) & (K-1)

def HMv(pvec, asked):
    tot = 0.0
    for q in range(NQ):
        if q in asked:
            continue
        col = [sum(x for j, x in enumerate(pvec) if dig(j, q) == a)
               for a in range(K)]
        tot += H(col)
    return tot

# enumerate all answer branches for k = 1, 2 questions (q = 0 then 1)
from itertools import product
for k in (1, 2):
    exp_HM, exp_s = 0.0, 0.0
    for ans in product(range(K), repeat=k):
        pr, s, pw = 1.0, 0.0, pv[:]
        ok = True
        for i, a in enumerate(ans):
            Z = sum(x for j, x in enumerate(pw) if dig(j, i) == a)
            if Z <= 0:
                ok = False
                break
            s += -math.log2(Z)
            pr *= Z
            pw = [x / Z if dig(j, i) == a else 0 for j, x in enumerate(pw)]
        if not ok:
            continue
        exp_HM += pr * HMv(pw, set(range(k)))
        exp_s += pr * s
    Gk = G_block(n, m, p0, k)
    Gk1 = G_block(n, m, p0, k + 1)
    assert abs(exp_s - Gk) < 1e-9
    assert abs(exp_HM - (NQ - k) * (Gk1 - Gk)) < 1e-9
    # chain prediction of <H(M_k)>
    qs, ps = exact_chain(n, m, p0, k)
    Nk = N / K**k
    chain_HM = (NQ - k) * (qs[k] * Hcol(m, ps[k], Nk) + (1 - qs[k]) * m)
    assert abs(chain_HM - exp_HM) < 1e-9
print("check 3: brute force == chain == G_k machinery at (2,2), k = 1, 2")

# ---- check 4: delta-limit leverage formula ----
def Lk_exact(n, m, p0, k):
    NQ = 2**n
    HM0 = NQ * G_block(n, m, p0, 1)
    rem = (NQ - k) * (G_block(n, m, p0, k+1) - G_block(n, m, p0, k))
    return (HM0 - rem) / G_block(n, m, p0, k)

def Lk_limit(n, m, k):
    NQ, u = 2**n, 2.0**(-m)
    return (1 - u) * (NQ - (NQ - k) * u**k) / (1 - u**k)

n, m = 4, 1
print(f"\n(4,1) cumulative leverage, delta limit vs exact at p0=1-1e-9:")
for k in (1, 2, 3, 4, 8, 16):
    print(f"  k={k:2d}: limit {Lk_limit(n, m, k):.4f}   "
          f"exact {Lk_exact(n, m, 1 - 1e-9, k):.4f}")
assert abs(Lk_limit(n, m, 1) - (2**n - (2**n - 1) * 2**(-m))) < 1e-12
assert abs(Lk_limit(n, m, 2**n) - 2**n * (1 - 2**(-m))
           * 1/(1 - 2**(-m*2**n))) < 1e-12
print("check 4: k=1 and k=2^n endpoints match the earlier limits")

# check 5: plateau + boundary-layer decomposition
#   L_k = 2^n (1-u) + (1-u) k u^k / (1 - u^k),   u = 2^-m
for n_, m_, k_ in ((4, 1, 1), (4, 1, 5), (6, 1, 3), (3, 2, 4)):
    u = 2.0**(-m_)
    dec = 2**n_ * (1 - u) + (1 - u) * k_ * u**k_ / (1 - u**k_)
    assert abs(Lk_limit(n_, m_, k_) - dec) < 1e-10
print("check 5: L_k = 2^n(1-u) + (1-u) k u^k/(1-u^k) decomposition")

# ---- n -> infinity at FINITE p: L(x) = 1 + [Hcol - (1-p)m]/(x(1-p)m) ----
def Hcol_bigN(m, p):
    K = 2**m
    Ps = p + (1 - p) / K
    up = (1 - p) / K
    return H([Ps] + [up] * (K - 1))


def Lx_finite_p(m, p, x):
    return 1 + (Hcol_bigN(m, p) - (1 - p) * m) / (x * (1 - p) * m)


# convergence check against the exact machinery (largest n before
# float overflow: m 2^n <= 1023)
for m_, n_ in ((1, 9), (2, 8)):
    NQ = 2**n_
    for p in (0.1, 0.5, 0.9):
        for x in (0.25, 0.5, 1.0):
            k = int(x * NQ)
            ex = Lk_exact(n_, m_, p, k)
            lim = Lx_finite_p(m_, p, x)
            assert abs(ex / lim - 1) < 0.1, (m_, p, x, ex, lim)
            # and the deviation shrinks with n
            ex_small = Lk_exact(n_ - 3, m_, p, int(x * 2**(n_-3)))
            assert abs(ex - lim) < abs(ex_small - lim) + 1e-9
print("check 6: n->inf finite-p hyperbola L(x) = 1 + c(p,m)/x, "
      "exact machinery converging onto it")

# ---- figure: leverage per question vs asked fraction x = k/2^n ----
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    u = 0.5   # m = 1 throughout

    ax = axes[0]
    for n_, col in ((3, "#1baf7a"), (4, "#2a78d6"), (5, "#eb6834"),
                    (6, "#eda100"), (8, "#e87ba4"), (10, "#777777")):
        ks = range(1, 2**n_ + 1)
        ax.plot([k / 2**n_ for k in ks],
                [Lk_limit(n_, 1, k) / 2**n_ for k in ks],
                ".-", ms=3, lw=1.1, color=col, label=f"$n={n_}$")
    ax.axhline(1 - u, color="black", lw=0.9, ls="--",
               label="culling ratio $1-2^{-m}$")
    ax.set_title("$p_0 \\to 1$ limit, varying $n$ ($m=1$)")
    ax.set_xlabel("$x = k/2^n$")
    ax.set_ylabel("$\\langle L_k \\rangle \\, 2^{-n}$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    n_ = 6
    for eps, col in ((1e-2, "#1baf7a"), (1e-4, "#2a78d6"),
                     (1e-8, "#eb6834"), (1e-12, "#eda100")):
        ks = range(1, 2**n_ + 1)
        ax.plot([k / 2**n_ for k in ks],
                [Lk_exact(n_, 1, 1 - eps, k) / 2**n_ for k in ks],
                ".-", ms=3, lw=1.1, color=col,
                label=f"$1 - p_0 = 10^{{{int(math.log10(eps))}}}$")
    ks = range(1, 2**n_ + 1)
    ax.plot([k / 2**n_ for k in ks],
            [Lk_limit(n_, 1, k) / 2**n_ for k in ks],
            "--", lw=1.4, color="black", label="$p_0 \\to 1$ limit")
    ax.set_title(f"approach at fixed $n = {n_}$, $m = 1$")
    ax.set_xlabel("$x = k/2^n$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Spike prior: cumulative leverage per question, "
                 "against the asked fraction")
    fig.tight_layout()
    fig.savefig("figures/spike_flow_leverage.png", dpi=130)
    print("wrote figures/spike_flow_leverage.png")

    # second figure: n -> infinity at finite p, L(x) hyperbolas
    import numpy as np
    xs = np.linspace(0.02, 1.0, 500)
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    PCOLS = ((0.001, "#1baf7a"), (0.1, "#2a78d6"), (0.5, "#eb6834"),
             (0.9, "#eda100"), (0.999, "#e87ba4"))
    for ax, m_ in zip(axes2, (1, 2)):
        for p, col in PCOLS:
            ax.plot(xs, [Lx_finite_p(m_, p, x) for x in xs],
                    color=col, lw=1.6, label=f"$p = {p}$")
        ax.axhline(1, color="black", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"$m = {m_}$")
        ax.set_xlabel("$x = k/2^n$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes2[0].set_ylabel("$\\lim_{n\\to\\infty} \\langle L_{x 2^n} \\rangle$")
    fig2.suptitle("Spike prior, $n \\to \\infty$ at finite $p$: "
                  "the leverage hyperbola $1 + c(p,m)/x$")
    fig2.tight_layout()
    fig2.savefig("figures/spike_flow_finite_p.png", dpi=130)
    print("wrote figures/spike_flow_finite_p.png")
