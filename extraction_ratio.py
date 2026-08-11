"""Figure for the extraction-ratio section of learning_agent.md.

Spike prior (weight p on one map, the rest uniform): closed forms via
exchangeability -- the block entropy G_k depends only on the block
size k:

  G_k = -P_k log2 P_k - ((2^m)^k - 1) U_k log2 U_k,
  P_k = p + (N/(2^m)^k - 1) omega,   U_k = (N/(2^m)^k) omega,
  omega = (1-p)/(N-1),

so received(1) = G_1, remaining(1) = (2^n - 1)(G_2 - G_1) by the
master trajectory law.  The amount extracted from the dependency store
by one question is E[Delta H(M)] - G_1, where

  E[Delta H(M)] = H(M)_0 - remaining(1).

Writing C = H(M)_0 - H(P) for the dependency store, the extraction
ratio is

  R_1 = (E[Delta H(M)] - G_1) / C,

with the sharp-prior limit

  R_1 -> (|Q| - 1) N (1 - 2^-m)^2
         / (|Q| N (1 - 2^-m) - (N - 1))                  as p -> 1,

where |Q| = 2^n and N = (2^m)^(2^n).  The uniform prior occurs at
p = 1/N, not p = 0.  R_1 is undefined there, so the plotted curves
begin strictly on the spike side p > 1/N.

Closed forms verified against brute force at (2,1) and (2,2).
Output: figures/extraction_ratio.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spike_R1(n, m, p):
    num_questions, alphabet_size = 2**n, 2**m
    N = alphabet_size**num_questions
    omega = (1 - p) / (N - 1)

    def G(k):
        Pk = p + (N / alphabet_size**k - 1) * omega
        Uk = (N / alphabet_size**k) * omega
        t = 0.0
        if 0 < Pk < 1:
            t -= Pk * math.log2(Pk)
        if Uk > 0:
            t -= (alphabet_size**k - 1) * Uk * math.log2(Uk)
        return t

    Hp = G(num_questions)
    HM0 = num_questions * G(1)
    C = HM0 - Hp
    rem1 = (num_questions - 1) * (G(2) - G(1))
    extracted = HM0 - rem1 - G(1)
    return extracted / C if C > 0 else float("nan")


def sharp_limit(n, m):
    """Exact finite-N limit of R_1 as the special-map mass p -> 1."""
    num_questions, alphabet_size = 2**n, 2**m
    N = alphabet_size**num_questions
    culling_fraction = 1 - 1 / alphabet_size
    return ((num_questions - 1) * N * culling_fraction**2
            / (num_questions * N * culling_fraction - (N - 1)))


def uniform_point(n, m):
    """The value of p for which every one of the N maps has mass 1/N."""
    num_questions, alphabet_size = 2**n, 2**m
    return 1 / alphabet_size**num_questions


def spike_curve(n, m, samples=1200):
    """Sample the domain on which R_1 is defined: 1/N < p < 1."""
    p_uniform = uniform_point(n, m)
    ps = np.linspace(p_uniform, 1 - 1e-6, samples + 1)[1:]
    ratios = np.array([spike_R1(n, m, p) for p in ps])
    return ps, ratios


# ---- brute-force verification of the closed form ----
def brute_R1(n, m, p):
    num_questions, alphabet_size = 2**n, 2**m
    N = alphabet_size**num_questions
    omega = (1 - p) / (N - 1)
    pv = [omega] * N
    pv[0] = p
    H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
    dig = lambda j, q: (j >> (m*q)) & (alphabet_size - 1)

    def HM(pvec):
        return sum(H([sum(x for j, x in enumerate(pvec) if dig(j, q) == a)
                      for a in range(alphabet_size)])
                   for q in range(num_questions))

    HM0, Hp = HM(pv), H(pv)
    dHM1 = s1 = 0.0
    for a in range(alphabet_size):
        answer_prob = sum(x for j, x in enumerate(pv) if dig(j, 0) == a)
        post = [x / answer_prob if dig(j, 0) == a else 0
                for j, x in enumerate(pv)]
        dHM1 += answer_prob * (HM0 - HM(post))
        s1 += answer_prob * (-math.log2(answer_prob))
    return (dHM1 - s1) / (HM0 - Hp)


for (n, m) in ((2, 1), (2, 2)):
    for p in (0.2, 0.7, 0.95):
        assert abs(spike_R1(n, m, p) - brute_R1(n, m, p)) < 1e-9
print("closed form == brute force at (2,1) and (2,2)")

# The finite-N sharp guide is also the ratio of the leading entropy
# coefficients as p = 1 - epsilon approaches one.
for n, m in ((2, 1), (4, 1), (6, 1), (3, 2), (3, 4)):
    num_questions, alphabet_size = 2**n, 2**m
    N = alphabet_size**num_questions
    iota = lambda k: N / (N - 1) * (1 - alphabet_size**(-k))
    from_entropy_coefficients = (
        (num_questions - 1) * (2 * iota(1) - iota(2))
        / (num_questions * iota(1) - iota(num_questions))
    )
    assert math.isclose(sharp_limit(n, m), from_entropy_coefficients,
                        rel_tol=2e-15, abs_tol=2e-15)
print("exact finite-N p->1 guides == entropy-coefficient limits")

# ---- figure: two panels, varying n and varying m ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)

ax = axes[0]
for n, col in ((2, "#1baf7a"), (3, "#2a78d6"), (4, "#eb6834"),
               (5, "#eda100"), (6, "#e87ba4")):
    ps, ratios = spike_curve(n, 1)
    assert (np.all(np.isfinite(ratios))
            and np.all((0 <= ratios) & (ratios <= 1)))
    ax.plot(ps, ratios, color=col, lw=1.6, label=f"$n={n}$")
    ax.axhline(sharp_limit(n, 1), color=col, lw=0.8, ls="--", alpha=0.6)
ax.set_title("$m = 1$, varying $n$")
ax.set_xlabel("special-map mass $p$")
ax.set_ylabel("$R_1$ (fraction of $C$ extracted)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper left")

ax = axes[1]
for m, col in ((1, "#1baf7a"), (2, "#2a78d6"), (3, "#eb6834"),
               (4, "#eda100")):
    ps, ratios = spike_curve(3, m)
    assert (np.all(np.isfinite(ratios))
            and np.all((0 <= ratios) & (ratios <= 1)))
    ax.plot(ps, ratios, color=col, lw=1.6, label=f"$m={m}$")
    ax.axhline(sharp_limit(3, m), color=col, lw=0.8, ls="--", alpha=0.6)
ax.set_title("$n = 3$, varying $m$")
ax.set_xlabel("special-map mass $p$")
ax.set_xlim(0, 1)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Spike prior: one-question extraction ratio "
             "$R_1=[\\mathbb{E}\\,\\Delta H(M)-G_1]/C$\n"
             "solid: exact finite-$N$ curves; "
             "dashed: exact $p\\to1^-$ limits")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("figures/extraction_ratio.png", dpi=130)
print("wrote figures/extraction_ratio.png")

for n, m in ((2, 1), (4, 1), (6, 1), (3, 2), (3, 4)):
    print(f"(n,m)=({n},{m}): p_uniform={uniform_point(n, m):.6g}, "
          f"sharp-prior limit={sharp_limit(n, m):.6f}")
