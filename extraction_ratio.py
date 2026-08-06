"""Figure for the extraction-ratio section of learning_agent.md.

Spike prior (weight p on one map, the rest uniform): closed forms via
exchangeability -- the block entropy G_k depends only on the block
size k:

  G_k = -P_k log2 P_k - ((2^m)^k - 1) u_k log2 u_k,
  P_k = p + (N/(2^m)^k - 1) beta,   u_k = (N/(2^m)^k) beta,
  beta = (1-p)/(N-1),

so received(1) = G_1, remaining(1) = (2^n - 1)(G_2 - G_1) by the
master trajectory law, and the deduced part of one question is
D_1 = H(M)_0 - remaining(1) - G_1.  The stored (convertible) knowledge
is C = H(M)_0 - H(p) = I - B.  The extraction ratio is X_1 = D_1 / C,
with the delta limit

  X_1 -> (2^n - 1)(1 - 2^-m)^2 / (2^n (1 - 2^-m) - 1)   as p -> 1.

Closed forms verified against brute force at (2,1) and (2,2).
Output: figures/extraction_ratio.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spike_X1(n, m, p):
    NQ, K = 2**n, 2**m
    N = float(K)**NQ
    beta = (1 - p) / (N - 1)

    def G(k):
        Pk = p + (N / K**k - 1) * beta
        uk = (N / K**k) * beta
        t = 0.0
        if 0 < Pk < 1:
            t -= Pk * math.log2(Pk)
        if uk > 0:
            t -= (K**k - 1) * uk * math.log2(uk)
        return t

    Hp = G(NQ)
    HM0 = NQ * G(1)
    C = HM0 - Hp
    rem1 = (NQ - 1) * (G(2) - G(1))
    D1 = HM0 - rem1 - G(1)
    return D1 / C if C > 0 else float("nan")


def delta_limit(n, m):
    NQ, c = 2**n, 1 - 2.0**(-m)
    return (NQ - 1) * c * c / (NQ * c - 1)


# ---- brute-force verification of the closed form ----
def brute_X1(n, m, p):
    NQ, K = 2**n, 2**m
    N = K**NQ
    beta = (1 - p) / (N - 1)
    pv = [beta] * N
    pv[0] = p
    H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
    dig = lambda j, q: (j >> (m*q)) & (K-1)

    def HM(pvec):
        return sum(H([sum(x for j, x in enumerate(pvec) if dig(j, q) == a)
                      for a in range(K)]) for q in range(NQ))

    HM0, Hp = HM(pv), H(pv)
    dHM1 = s1 = 0.0
    for a in range(K):
        Z = sum(x for j, x in enumerate(pv) if dig(j, 0) == a)
        post = [x / Z if dig(j, 0) == a else 0 for j, x in enumerate(pv)]
        dHM1 += Z * (HM0 - HM(post))
        s1 += Z * (-math.log2(Z))
    return (dHM1 - s1) / (HM0 - Hp)


for (n, m) in ((2, 1), (2, 2)):
    for p in (0.2, 0.7, 0.95):
        assert abs(spike_X1(n, m, p) - brute_X1(n, m, p)) < 1e-9
print("closed form == brute force at (2,1) and (2,2)")

# ---- figure: two panels, varying n and varying m ----
ps = np.linspace(1e-4, 1 - 1e-6, 1200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

ax = axes[0]
for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834"),
               (5, "#eda100"), (6, "#e87ba4")):
    ax.plot(ps, [spike_X1(n, 1, p) for p in ps], color=col, lw=1.6,
            label=f"$n={n}$")
    ax.axhline(delta_limit(n, 1), color=col, lw=0.8, ls="--", alpha=0.6)
ax.set_title("$m = 1$, varying $n$")
ax.set_xlabel("$p$")
ax.set_ylabel("$X_1 = D_1 / C$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper left")

ax = axes[1]
for m, col in ((1, "#1baf7a"), (2, "#2a78d6"), (3, "#eb6834"),
               (4, "#eda100")):
    ax.plot(ps, [spike_X1(3, m, p) for p in ps], color=col, lw=1.6,
            label=f"$m={m}$")
    ax.axhline(delta_limit(3, m), color=col, lw=0.8, ls="--", alpha=0.6)
ax.set_title("$n = 3$, varying $m$")
ax.set_xlabel("$p$")
ax.set_xlim(0, 1)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Spike prior: fraction of the stored knowledge $C$ freed "
             "by one question (dashes: $p \\to 1$ limits)")
fig.tight_layout()
fig.savefig("figures/extraction_ratio.png", dpi=130)
print("wrote figures/extraction_ratio.png")

for n, m in ((2, 1), (4, 1), (6, 1), (3, 2), (3, 4)):
    print(f"(n,m)=({n},{m}): delta limit {delta_limit(n, m):.4f}")
