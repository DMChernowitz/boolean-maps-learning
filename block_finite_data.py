"""Exact finite-n leverage curves for three block priors, via the
hypergeometric increment formula and the finite leverage law.
Writes figures/data/blockfinite_*.dat for the appendix figure."""
from math import comb

OUT = r"figures/data"


def hyper(i, r, Q, ell):
    """P[i of the r-1 partners of a fresh question lie in a random
    ell-subset of the other Q-1 questions]."""
    if ell - i < 0 or ell - i > Q - r:
        return 0.0
    return comb(r - 1, i) * comb(Q - r, ell - i) / comb(Q - 1, ell)


def gamma_seq(Q, blocks):
    """blocks: list of (weight, r, eta list). Exact gamma_{n,l}."""
    out = []
    for ell in range(Q):
        g = 0.0
        for w, r, eta in blocks:
            g += w * sum(hyper(i, r, Q, ell) * eta[i] for i in range(r))
        out.append(g)
    return out


def leverage(Q, blocks):
    gam = gamma_seq(Q, blocks)
    G = [0.0]
    for g in gam:
        G.append(G[-1] + g)
    G1 = G[1]
    rows = []
    for ell in range(1, Q + 1):
        rem = (Q - ell) * gam[ell] if ell < Q else 0.0
        rows.append((ell / Q, (Q * G1 - rem) / G[ell]))
    return rows


def write(name, rows):
    with open(f"{OUT}/{name}.dat", "w") as fh:
        fh.write("t lev\n")
        for t, lev in rows:
            fh.write(f"{t:.6f} {lev:.6f}\n")


CLIQUE4 = [(1.0, 4, [1, 0, 0, 0])]
PARITY4 = [(1.0, 4, [1, 1, 1, 0])]
MDSFRESH = [(0.75, 4, [2, 2, 0, 0]), (0.25, 1, [2])]

for Q in (16, 64):
    write(f"blockfinite_clique4_q{Q}", leverage(Q, CLIQUE4))
    write(f"blockfinite_parity4_q{Q}", leverage(Q, PARITY4))
    write(f"blockfinite_mdsfresh_q{Q}", leverage(Q, MDSFRESH))

# landmarks: clique leverage is exactly r at every finite n
for Q in (16, 64):
    for t, lev in leverage(Q, CLIQUE4):
        assert abs(lev - 4) < 1e-9, (Q, t, lev)
# parity completes at r/(r-1)
assert abs(leverage(16, PARITY4)[-1][1] - 4 / 3) < 1e-9
# mds+fresh completes at 1/(1 - w/2) = 1.6
assert abs(leverage(64, MDSFRESH)[-1][1] - 1.6) < 1e-9

# peak of the mds+fresh limit curve
w = 0.75
best = (0, 0)
for i in range(1, 2000):
    t = i / 2000
    tau = 1 - 3 * t ** 2 + 2 * t ** 3
    L = (w * (1 - (1 - t) * tau) + (1 - w) * t) \
        / (w * (t - t ** 3 + t ** 4 / 2) + (1 - w) * t)
    if L > best[1]:
        best = (t, L)
print("mds+fresh limit peak: L=%.4f at t=%.3f; L(1)=%.4f"
      % (best[1], best[0], 1 / (1 - w / 2)))
print("all landmarks pass")
