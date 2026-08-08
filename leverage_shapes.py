"""Leverage curves L(x) beyond the hyperbola: block-structured priors.

The limit is L(x) = [g0 - (1-x) gamma(x)] / int_0^x gamma, where
gamma(x) is the conditional-entropy rate of a fresh answer after a
fraction x of the questions.  Exchangeable priors force gamma to be a
step at zero (de Finetti), hence the 1 + c/x hyperbola.  Block priors
give tunable gamma:

  cliques (r equal answers):        gamma = (1-x)^{r-1}   -> L == r  (flat)
  parity groups (r answers sum 0):  gamma = 1 - x^{r-1}   -> L rises 1 -> r/(r-1)
  MDS code groups (any k of r
    determine the rest) + fresh:    gamma = w Fbar(x) + 1-w -> mid-run peak
  coins + code + fresh:             dive, dip, resurgent peak, decay

All finite-n curves are EXACT: for a product-of-groups prior,
G_k = sum_groups E_{j ~ Hypergeom(2^n, r, k)} H_group(j), with
H_group(j) closed form per block type.  Output:
figures/leverage_shapes.png"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2)
h = lambda x: 0 if x <= 0 or x >= 1 else -x*math.log2(x)-(1-x)*math.log2(1-x)


def lchoose(a, b):
    if b < 0 or b > a:
        return float("-inf")
    return math.lgamma(a+1) - math.lgamma(b+1) - math.lgamma(a-b+1)


# ---- coin-mixture block: G table for N iid-given-theta sites ----
COINS = [(0.2, 0.5), (0.7, 0.5)]
TBAR = sum(th*wt for th, wt in COINS)
GBAR = sum(wt*h(th) for th, wt in COINS)


def coin_G_table(Nmax):
    T = [0.0]*(Nmax+1)
    for N in range(1, Nmax+1):
        tot = 0.0
        for w in range(N+1):
            logs = [math.log(wt) + w*math.log(th) + (N-w)*math.log(1-th)
                    for th, wt in COINS]
            mx = max(logs)
            lpw = mx + math.log(sum(math.exp(z-mx) for z in logs))
            tot += math.exp(lchoose(N, w) + lpw) * (-lpw/LN2)
        T[N] = tot
    return T


# ---- exact finite-n machinery ----
def G_series(Q, blocks):
    """blocks: list of (count, group_size_r, Hfun(j)).  Exact G_k."""
    G = [0.0]*(Q+1)
    for k in range(Q+1):
        tot = 0.0
        for cnt, r, H in blocks:
            e = 0.0
            lo, hi = max(0, k-(Q-r)), min(r, k)
            for j in range(lo, hi+1):
                e += math.exp(lchoose(r, j) + lchoose(Q-r, k-j)
                              - lchoose(Q, k)) * H(j)
            tot += cnt*e
        G[k] = tot
    return G


def L_curve(Q, blocks):
    G = G_series(Q, blocks)
    HM0 = Q * G[1]
    xs, Ls = [], []
    for l in range(1, Q+1):
        rem = (Q-l)*(G[l+1]-G[l]) if l < Q else 0.0
        xs.append(l/Q)
        Ls.append((HM0-rem)/G[l])
    return xs, Ls


# ---- limit-curve helpers (per-site units) ----
def Fbar(x, r, k):
    """P(Bin(r-1, x) <= k-1): a fresh code answer is still uninferable."""
    if x <= 0:
        return 1.0
    if x >= 1:
        return 0.0
    return sum(math.exp(lchoose(r-1, i) + i*math.log(x)
                        + (r-1-i)*math.log(1-x)) for i in range(k))


def limit_from_gamma(gam0, gam, npts=2000):
    """L(x) from column entropy gam0 and rate gamma(x), by quadrature."""
    xs, Ls, acc = [], [], 0.0
    dx = 1.0/npts
    for i in range(1, npts+1):
        x = i*dx
        acc += gam(x - dx/2)*dx
        xs.append(x)
        Ls.append((gam0 - (1-x)*gam(x)) / acc)
    return xs, Ls


M = 4                      # answer bits per question (alphabet 16 for RS)
RCODE, KCODE = 16, 4       # MDS code: any 4 of 16 answers determine all


def blocks_clique(Q, r):
    return [(Q//r, r, lambda j: M*(j >= 1))]


def blocks_parity(Q, r):
    return [(Q//r, r, lambda j: M*min(j, r-1))]


def blocks_code_fresh(Q):   # 3/4 code, 1/4 fresh
    return [(3*Q//(4*RCODE), RCODE, lambda j: M*min(j, KCODE)),
            (Q//4, 1, lambda j: M*j)]


def blocks_cocktail(Q, GT):  # 1/4 coins, 1/2 code, 1/4 fresh
    return [(1, Q//4, lambda j: GT[M*j]),
            (Q//(2*RCODE), RCODE, lambda j: M*min(j, KCODE)),
            (Q//4, 1, lambda j: M*j)]


GT = coin_G_table(M*(2**10)//4)

WC = 0.75   # code weight in panel (c)
CO, CK, CI = 0.25, 0.5, 0.25   # cocktail weights

PANELS = [
    ("cliques of size $r$:  $L \\equiv r$",
     [(n, r, blocks_clique(2**n, r)) for n, r in ((8, 2), (8, 4))],
     [(lambda x: 2.0, "$r=2$"), (lambda x: 4.0, "$r=4$")],
     (0.9, 4.6)),
    ("parity groups ($r=4$):  rising $1 \\to 4/3$",
     [(n, None, blocks_parity(2**n, 4)) for n in (6, 8, 10)],
     [(lambda x: (1 - (1-x)*(1-x**3)) / (x - x**4/4), None)],
     (0.97, 1.4)),
    (f"MDS code ($\\rho=1/4$, $r={RCODE}$) + fresh:  peak",
     [(n, None, blocks_code_fresh(2**n)) for n in (6, 8, 10)],
     [(None, None)],
     (0.9, 3.2)),
    ("coins + code + fresh:  dive, dip, peak",
     [(n, None, blocks_cocktail(2**n, GT)) for n in (6, 8, 10)],
     [(None, None)],
     (0.9, 4.0)),
]


def limit_code_fresh(x):
    fb = Fbar(x, RCODE, KCODE)
    return WC*fb + (1-WC), 1.0


def limit_cocktail(x):
    fb = Fbar(x, RCODE, KCODE)
    gam = CO*GBAR + CK*fb + CI
    gam0 = CO*h(TBAR) + CK + CI
    return gam, gam0


fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
NCOL = {2: "#9ec5f4", 4: "#9ec5f4", 6: "#9ec5f4", 8: "#5598e7",
        10: "#2a78d6"}

for ax, (title, finites, limits, ylim) in zip(axes, PANELS):
    for n, tag, blocks in finites:
        xs, Ls = L_curve(2**n, blocks)
        lab = f"$n={n}$" + (f", $r={tag}$" if tag else "")
        ax.plot(xs, Ls, color=NCOL[tag or n], lw=1.4, label=lab)
    if title.startswith("cliques"):
        for gam, lab in limits:
            ax.axhline(gam(0), ls="--", color="#eb6834", lw=1.2)
    elif title.startswith("parity"):
        xs = [i/1000 for i in range(5, 1001)]
        ax.plot(xs, [limits[0][0](x) for x in xs], "--",
                color="#eb6834", lw=1.6, label="limit")
    else:
        gfun = limit_code_fresh if "MDS" in title else limit_cocktail
        gam0 = gfun(0.5)[1] if "MDS" in title else limit_cocktail(0)[1]

        def gam_only(x, f=gfun):
            return f(x)[0]
        xs, Ls = limit_from_gamma(gam0, gam_only)
        ax.plot(xs, Ls, "--", color="#eb6834", lw=1.6, label="limit")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$x = \\ell/2^n$")
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0].set_ylabel("$\\langle L \\rangle$")
fig.suptitle("Beyond the hyperbola: leverage shapes from block-structured "
             "priors (exact finite-$n$ vs limit)")
fig.tight_layout()
fig.savefig("figures/leverage_shapes.png", dpi=130)
print("wrote figures/leverage_shapes.png")

# ---- report the landmarks ----
for name, gfun in (("code+fresh", limit_code_fresh),
                   ("cocktail", limit_cocktail)):
    gam0 = gfun(0)[1] if name == "cocktail" else gfun(0.5)[1]
    xs, Ls = limit_from_gamma(gam0, lambda x: gfun(x)[0])
    body = [(x, L) for x, L in zip(xs, Ls) if x > 0.05]
    xp, Lp = max(body, key=lambda t: t[1])
    xd, Ld = min((t for t in body if t[0] < xp), key=lambda t: t[1],
                 default=(float("nan"), float("nan")))
    print(f"{name}: dip L={Ld:.3f} at x={xd:.3f}; "
          f"peak L={Lp:.3f} at x={xp:.3f}; L(1)={Ls[-1]:.3f}")

# flatness check for cliques
for r in (2, 4):
    xs, Ls = L_curve(2**8, blocks_clique(2**8, r))
    dev = max(abs(L-r) for L in Ls)
    print(f"clique r={r}: max |L - r| over all ell at n=8: {dev:.2e}")
