"""Exact finite-n leverage for the polynomial-Occam prior of the
constructive-complexity section, for F(x)=x, F(x)=x^2 and
F(x)=1-(1-x)^2.

The mixture of nested uniform-code laws has exact subset entropy
H = -sum_e (2^{r_e}-2^{r_{e-1}}) P_e log2 P_e with
P_e = sum_{d>=e} w_d 2^{-r_d}, a function of the rank profile
(r_0,...,r_n) alone.  n=4: full subset enumeration (exact).
n=6,8: Monte Carlo over question orders, entropies exact per subset.
Usage: python rm_finite_data.py [n4|n6|n8|all]
Writes figures/data/rmfinite_{unif,max,min2}_n{4,6,8}.dat."""
import itertools
import random
import sys
from math import comb, log2

random.seed(20260819)
OUT = r"figures/data"
FS = [lambda x: x, lambda x: x * x, lambda x: 1 - (1 - x) ** 2]
TAGS = ["unif", "max", "min2"]
PERMS = {6: 8000, 8: 1200}


def columns(n):
    """Full-monomial column per question, monomials ordered by degree;
    class d uses the low K_d bits."""
    monos = [s for r in range(n + 1)
             for s in itertools.combinations(range(n), r)]
    K = [sum(comb(n, i) for i in range(d + 1)) for d in range(n + 1)]
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
    return cols, K


def weights(n, K, F):
    Q = 2 ** n
    R = [K[d] / Q for d in range(n + 1)]
    return [F(R[d]) - (F(R[d - 1]) if d else 0.0) for d in range(n + 1)]


def entropy(prof, w, n):
    """Exact mixture entropy from the rank profile prof[d]."""
    H = 0.0
    for e in range(n + 1):
        cnt = 2 ** prof[e] - (2 ** prof[e - 1] if e else 0)
        if cnt == 0:
            continue
        P = sum(w[d] * 2.0 ** (-prof[d]) for d in range(e, n + 1))
        H -= cnt * P * log2(P)
    return H


def rank_profile_path(cols, K, order):
    """Rank profiles after each prefix of the given question order.
    Skips classes whose basis is already full."""
    n = len(K) - 1
    bases = [{} for _ in range(n + 1)]
    prof = [0] * (n + 1)
    path = [tuple(prof)]
    masks = [(1 << K[d]) - 1 for d in range(n + 1)]
    for q in order:
        c = cols[q]
        for d in range(n + 1):
            if prof[d] == K[d]:
                continue
            v = c & masks[d]
            basis = bases[d]
            while v:
                h = v.bit_length() - 1
                if h in basis:
                    v ^= basis[h]
                else:
                    basis[h] = v
                    prof[d] += 1
                    break
        path.append(tuple(prof))
    return path


def curves_exact(n):
    cols, K = columns(n)
    Q = 2 ** n
    ws = [weights(n, K, F) for F in FS]
    tot = [[0.0] * (Q + 1) for _ in FS]
    cnt = [0] * (Q + 1)
    for mask in range(2 ** Q):
        idx = [q for q in range(Q) if (mask >> q) & 1]
        prof = rank_profile_path(cols, K, idx)[-1]
        for f, w in enumerate(ws):
            tot[f][len(idx)] += entropy(prof, w, n)
        cnt[len(idx)] += 1
    return [[tot[f][l] / cnt[l] for l in range(Q + 1)]
            for f in range(len(FS))]


def curves_mc(n, nperm):
    cols, K = columns(n)
    Q = 2 ** n
    ws = [weights(n, K, F) for F in FS]
    acc = [[0.0] * (Q + 1) for _ in FS]
    order = list(range(Q))
    for _ in range(nperm):
        random.shuffle(order)
        path = rank_profile_path(cols, K, order)
        for f, w in enumerate(ws):
            for l, prof in enumerate(path):
                acc[f][l] += entropy(prof, w, n)
    return [[a / nperm for a in acc[f]] for f in range(len(FS))]


def leverage(G, Q):
    rows = []
    for ell in range(1, Q + 1):
        rem = (Q - ell) * (G[ell + 1] - G[ell]) if ell < Q else 0.0
        rows.append((ell / Q, (Q * G[1] - rem) / G[ell]))
    return rows


def write(name, rows):
    with open(f"{OUT}/{name}.dat", "w") as fh:
        fh.write("t lev\n")
        for t, lev in rows:
            fh.write(f"{t:.6f} {lev:.6f}\n")


def run(n):
    Q = 2 ** n
    G = curves_exact(n) if n == 4 else curves_mc(n, PERMS[n])
    for f, tag in enumerate(TAGS):
        if n == 4:
            assert abs(G[f][1] - 1) < 1e-12   # fair columns, exact
        rows = leverage(G[f], Q)
        write(f"rmfinite_{tag}_n{n}", rows)
        print(f"n={n} F={tag:4s}: L(1/Q)={rows[0][1]:.4f} "
              f"L(1)={rows[-1][1]:.4f}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in ([4, 6, 8] if arg == "all" else [int(arg[1:])]):
        run(n)
