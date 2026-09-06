"""Finite-n expected leverage of the spike prior, for figures/data.

All subsets of a given size share one block entropy, so the exact
finite law needs no simulation:

  P_k = p + (1-p) (A^(Q-k) - 1)/(A^Q - 1)        [eq:spikePk]
  G_k = h2(P_k) + (1-P_k) log2(A^k - 1)          [eq:spikeGsplit]
  <L_l> = (Q G_1 - (Q-l)(G_{l+1} - G_l)) / G_l   [eq:Lk-G]

against the thermodynamic curve

  L_p(t) = (h_spike(p,m) - (1-t)(1-p) m) / (t (1-p) m).

Writes figures/data/spikefin_m{m}p{100p}_n{n}.dat with columns t, lev.
"""
import math
import os
from fractions import Fraction

CASES = [(1, 0.3), (2, 0.8)]
ORDERS = (4, 5, 6, 8)
OUT = "figures/data"


def h2(x):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def block_entropies(n, m, p):
    """G_0..G_Q, exact in the big-integer counts"""
    Q, A = 2 ** n, 2 ** m
    denom = A ** Q - 1
    G = [0.0] * (Q + 1)
    for k in range(1, Q + 1):
        Pk = p + (1.0 - p) * float(Fraction(A ** (Q - k) - 1, denom))
        # log2(A^k - 1) rounds to k*m once the correction underflows
        lg = k * m if k * m > 60 else math.log2(A ** k - 1)
        G[k] = h2(Pk) + (1.0 - Pk) * lg
    return G


def leverage(n, m, p):
    Q = 2 ** n
    G = block_entropies(n, m, p)
    rows = []
    for l in range(1, Q + 1):
        rem = (Q - l) * (G[l + 1] - G[l]) if l < Q else 0.0
        rows.append((l / Q, (Q * G[1] - rem) / G[l]))
    return rows


def h_spike(m, p):
    A, u = 2 ** m, 2.0 ** -m
    v = h2(p + (1.0 - p) * u)
    if A > 2:
        v += (1.0 - p) * (1.0 - u) * math.log2(A - 1)
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    for m, p in CASES:
        hs = h_spike(m, p)
        c = (hs - (1.0 - p) * m) / ((1.0 - p) * m)
        print("m=%d p=%.2f: h_spike=%.5f, limit L(t) = 1 + %.5f/t, "
              "L(1)=%.4f" % (m, p, hs, c, 1.0 + c))
        for n in ORDERS:
            rows = leverage(n, m, p)
            f = "%s/spikefin_m%dp%d_n%d.dat" % (OUT, m, round(100 * p), n)
            with open(f, "w") as fh:
                fh.write("t lev\n")
                for t, L in rows:
                    fh.write("%.8f %.8f\n" % (t, L))
            print("   n=%2d  Q=%4d  L(1)=%.4f  ->  %s"
                  % (n, 2 ** n, rows[-1][1], f))


if __name__ == "__main__":
    main()
