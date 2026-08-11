"""Numerical verification of the spike prior as a two-state flow.

State after k questions: spike(p_k) on N_k = N u^k maps with
probability P_k, uniform with probability 1 - P_k.  Recursions:

  Pr(next confirmation | k confirmations) = P_{k+1}/P_k,
  p_{k+1} = p_k P_k/P_{k+1},

where u = |A|^-1. Checks:

  1. the product P_k p_k is invariant (= p): the belief martingale;
  2. the recursion solution equals the direct Bayes answer
     P_k = Pr(special block) exactly, and P_k = p + (1-p) u^k
     in the large-N form;
  3. branch-weighted expected surprisal telescopes to the block
     entropy G_k, and branch-weighted <H(M_k)> equals the master law
     (|Q| - k)(G_{k+1} - G_k)  [brute-forced at (2,2)];
  4. the sharp-prior cumulative-leverage limit
     <L_k> -> (1-u)[|Q| - (|Q|-k) u^k] / (1-u^k)
     is approached logarithmically slowly as p -> 1^- and reproduces
     the k=1 and k=|Q| limits already derived.
"""
import math

H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)


def exact_chain(n, m, p, kmax):
    num_questions, alphabet_size = 2**n, 2**m
    N = float(alphabet_size)**num_questions
    P_values, p_values = [1.0], [p]
    P_branch, p_cond = 1.0, p
    num_maps_remaining = N
    for k in range(kmax):
        omega_cond = (1 - p_cond) / (num_maps_remaining - 1)
        P_confirm = (p_cond
                     + (num_maps_remaining / alphabet_size - 1)
                     * omega_cond)
        P_branch, p_cond, num_maps_remaining = (
            P_branch * P_confirm,
            p_cond / P_confirm,
            num_maps_remaining / alphabet_size,
        )
        P_values.append(P_branch)
        p_values.append(p_cond)
    return P_values, p_values


def G_block(n, m, p, k):
    num_questions, alphabet_size = 2**n, 2**m
    N = float(alphabet_size)**num_questions
    omega = (1 - p) / (N - 1)
    Pk = p + (N / alphabet_size**k - 1) * omega
    Uk = (N / alphabet_size**k) * omega
    t = 0.0
    if 0 < Pk < 1:
        t -= Pk * math.log2(Pk)
    if Uk > 0:
        t -= (alphabet_size**k - 1) * Uk * math.log2(Uk)
    return t


def answer_entropy(m, p, num_maps_remaining):
    alphabet_size = 2**m
    omega = (1 - p) / (num_maps_remaining - 1)
    P = p + (num_maps_remaining / alphabet_size - 1) * omega
    U = (num_maps_remaining / alphabet_size) * omega
    return H([P] + [U] * (alphabet_size - 1))


# ---- checks 1 and 2 ----
for (n, m, p) in ((4, 1, 0.35), (3, 2, 0.7), (4, 1, 0.995)):
    num_questions, alphabet_size = 2**n, 2**m
    N = float(alphabet_size)**num_questions
    omega = (1 - p) / (N - 1)
    P_values, p_values = exact_chain(n, m, p, num_questions)
    for k in range(num_questions + 1):
        assert abs(P_values[k] * p_values[k] - p) < 1e-12  # martingale
        P_direct = (p + (N / alphabet_size**k - 1) * omega)
        assert abs(P_values[k] - P_direct) < 1e-12
        P_bigN = p + (1 - p) * alphabet_size**(-k)
        # Exact error: (1-p)(1-|A|^-k)/(N-1), uniformly O(1/N).
        assert abs(P_values[k] - P_bigN) <= (1 - p) / (N - 1) + 1e-12
print("checks 1-2: martingale P_k p_k = p; recursion == Bayes block "
      "probability == large-N closed form")

# ---- check 3: chain vs G_k and master law, brute force at (2,2) ----
n, m, p = 2, 2, 0.55
num_questions, alphabet_size = 2**n, 2**m
N = alphabet_size**num_questions
omega = (1 - p) / (N - 1)
pv = [omega] * N
pv[0] = p                                  # special map j = 0
dig = lambda j, q: (j >> (m*q)) & (alphabet_size - 1)

def HMv(pvec, asked):
    tot = 0.0
    for q in range(num_questions):
        if q in asked:
            continue
        col = [sum(x for j, x in enumerate(pvec) if dig(j, q) == a)
               for a in range(alphabet_size)]
        tot += H(col)
    return tot

# enumerate all answer branches for k = 1, 2 questions (q = 0 then 1)
from itertools import product
for k in (1, 2):
    exp_HM, exp_s = 0.0, 0.0
    for ans in product(range(alphabet_size), repeat=k):
        pr, s, pw = 1.0, 0.0, pv[:]
        ok = True
        for i, a in enumerate(ans):
            answer_prob = sum(x for j, x in enumerate(pw)
                              if dig(j, i) == a)
            if answer_prob <= 0:
                ok = False
                break
            s += -math.log2(answer_prob)
            pr *= answer_prob
            pw = [x / answer_prob if dig(j, i) == a else 0
                  for j, x in enumerate(pw)]
        if not ok:
            continue
        exp_HM += pr * HMv(pw, set(range(k)))
        exp_s += pr * s
    Gk = G_block(n, m, p, k)
    Gk1 = G_block(n, m, p, k + 1)
    assert abs(exp_s - Gk) < 1e-9
    assert abs(exp_HM - (num_questions - k) * (Gk1 - Gk)) < 1e-9
    # chain prediction of <H(M_k)>
    P_values, p_values = exact_chain(n, m, p, k)
    num_maps_remaining = N / alphabet_size**k
    chain_HM = (num_questions - k) * (
        P_values[k] * answer_entropy(m, p_values[k], num_maps_remaining)
        + (1 - P_values[k]) * m
    )
    assert abs(chain_HM - exp_HM) < 1e-9
print("check 3: brute force == chain == G_k machinery at (2,2), k = 1, 2")

# ---- check 4: sharp-prior leverage formula ----
def Lk_exact(n, m, p, k):
    num_questions = 2**n
    HM0 = num_questions * G_block(n, m, p, 1)
    rem = ((num_questions - k)
           * (G_block(n, m, p, k+1) - G_block(n, m, p, k)))
    return (HM0 - rem) / G_block(n, m, p, k)

def Lk_sharp_limit(n, m, k):
    num_questions, u = 2**n, 2.0**(-m)
    return ((1 - u)
            * (num_questions - (num_questions - k) * u**k)
            / (1 - u**k))

n, m = 4, 1
probe_ks = (1, 2, 3, 4, 8, 16)
previous_error = float("inf")
print("\ncheck 4: slow convergence toward the sharp-prior formula at (4,1)")
for epsilon in (1e-2, 1e-5, 1e-8):
    relative_errors = [
        abs(Lk_exact(n, m, 1 - epsilon, k)
            / Lk_sharp_limit(n, m, k) - 1)
        for k in probe_ks
    ]
    max_error = max(relative_errors)
    print(f"  1-p={epsilon:.0e}: maximum relative error "
          f"over k={probe_ks} is {max_error:.3f}")
    assert max_error < previous_error
    previous_error = max_error
assert abs(Lk_sharp_limit(n, m, 1)
           - (2**n - (2**n - 1) * 2**(-m))) < 1e-12
assert abs(Lk_sharp_limit(n, m, 2**n) - 2**n * (1 - 2**(-m))
           * 1/(1 - 2**(-m*2**n))) < 1e-12
print("check 4: errors decrease as p->1^-; endpoint formulas agree exactly")

# check 5: plateau + boundary-layer decomposition
#   L_k = |Q|(1-u) + (1-u) k u^k / (1-u^k),   u = |A|^-1
for n_, m_, k_ in ((4, 1, 1), (4, 1, 5), (6, 1, 3), (3, 2, 4)):
    u = 2.0**(-m_)
    dec = 2**n_ * (1 - u) + (1 - u) * k_ * u**k_ / (1 - u**k_)
    assert abs(Lk_sharp_limit(n_, m_, k_) - dec) < 1e-10
print("check 5: L_k = |Q|(1-u) + (1-u) k u^k/(1-u^k) decomposition")

# ---- n -> infinity at finite p: use h_spike(p,m) in the coefficient ----
def h_spike(p, m):
    """One-question answer entropy in the large-map-space limit."""
    alphabet_size = 2**m
    P = p + (1 - p) / alphabet_size
    U = (1 - p) / alphabet_size
    return H([P] + [U] * (alphabet_size - 1))


def Lx_finite_p(m, p, x):
    return 1 + (h_spike(p, m) - (1 - p) * m) / (x * (1 - p) * m)


# convergence check against the exact machinery (largest n before
# float overflow: m 2^n <= 1023)
for m_, n_ in ((1, 9), (2, 8)):
    num_questions = 2**n_
    for p in (0.1, 0.5, 0.9):
        for x in (0.25, 0.5, 1.0):
            k = int(x * num_questions)
            ex = Lk_exact(n_, m_, p, k)
            lim = Lx_finite_p(m_, p, x)
            assert abs(ex / lim - 1) < 0.1, (m_, p, x, ex, lim)
            # and the deviation shrinks with n
            ex_small = Lk_exact(n_ - 3, m_, p, int(x * 2**(n_-3)))
            assert abs(ex - lim) < abs(ex_small - lim) + 1e-9
print("check 6: n->inf finite-p hyperbola uses "
      "[h_spike(p,m)-(1-p)m]/[x(1-p)m], "
      "exact machinery converging onto it")

# ---- figure: leverage per question vs asked fraction x = ell/|Q| ----
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
                [Lk_sharp_limit(n_, 1, k) / 2**n_ for k in ks],
                ".-", ms=3, lw=1.1, color=col, label=f"$n={n_}$")
    ax.axhline(1 - u, color="black", lw=0.9, ls="--",
               label="culling ratio $1-2^{-m}$")
    ax.set_title("sharp-prior limit $p \\to 1^-$, varying $n$ ($m=1$)")
    ax.set_xlabel("$x = \\ell/|Q|$")
    ax.set_ylabel("$\\langle L_\\ell \\rangle/|Q|$")
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
                label=f"$1 - p = 10^{{{int(math.log10(eps))}}}$")
    ks = range(1, 2**n_ + 1)
    ax.plot([k / 2**n_ for k in ks],
            [Lk_sharp_limit(n_, 1, k) / 2**n_ for k in ks],
            "--", lw=1.4, color="black",
            label="sharp-prior limit $p \\to 1^-$")
    ax.set_title(f"approach at fixed $n = {n_}$, $m = 1$")
    ax.set_xlabel("$x = \\ell/|Q|$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Spike prior: normalized cumulative leverage, "
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
        ax.set_xlabel("$x = \\ell/|Q|$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes2[0].set_ylabel(
        "$\\lim_{n\\to\\infty} \\langle L_{x|Q|} \\rangle$")
    fig2.suptitle("Spike prior, $n \\to \\infty$ at finite $p$\n"
                  "$\\langle L\\rangle="
                  "1+[h_{\\mathrm{spike}}(p,m)-(1-p)m]/[x(1-p)m]$")
    fig2.tight_layout()
    fig2.savefig("figures/spike_flow_finite_p.png", dpi=130)
    print("wrote figures/spike_flow_finite_p.png")
