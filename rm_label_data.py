"""The label term: how much of each answer goes into identifying the
latent class rather than the map inside it.

Equation (89) splits the received entropy into a class average and what
the answers reveal about the label,

    H(psi(S)) = sum_d w_d H_d(psi(S)) + I(D; psi(S)),

and within a class H_d is the rank of the column submatrix.  Differencing
in the number of questions asked,

    gamma_{n,l} = sum_d w_d gamma^(d)_{n,l}  +  Delta_l I,

so the label is the gap between the true increment (not knowing D) and
the mixture of class increments (knowing D).

n = 4 is exact: every one of the 2^Q question sets is enumerated.
Larger n averages over random question orders, whose prefixes are
uniform subsets.

The mixture entropy is evaluated in log space.  Written directly it
needs 2^{r_e}, astronomically large, times P_e ~ 2^{-r_e}, which
underflows float64 once the ranks pass 1074 -- at n = 12 they reach
4096.  Factor P_e = 2^{-r_e} A_e with A_e of order one and both the
count and the logarithm come back into range.

Usage: python rm_label_data.py [n ...]     (default: 4 8 12)
Writes figures/data/rmlabel_n{n}.dat with columns t, mix, true.
"""
import itertools
import random
import sys
from math import comb, log2

random.seed(20260825)
OUT = r"figures/data"
F = lambda x: 1 - (1 - x) ** 2
ORDERS = {8: 400, 10: 400, 12: 300}
BINS = 200                      # target step count above the full-resolution size
FULL = 512                      # at or below this Q, one point per question


def columns(n):
    monos = [s for r in range(n + 1)
             for s in itertools.combinations(range(n), r)]
    K = [sum(comb(n, i) for i in range(d + 1)) for d in range(n + 1)]
    cols = []
    for q in range(2 ** n):
        bits = [(q >> i) & 1 for i in range(n)]
        c = 0
        for idx, s in enumerate(monos):
            v = 1
            for i in s:
                v &= bits[i]
            c |= v << idx
        cols.append(c)
    return cols, K


def weights(n, K):
    Q = 2 ** n
    R = [k / Q for k in K]
    w, prev = [], 0.0
    for d in range(n + 1):
        w.append(F(R[d]) - prev)
        prev = F(R[d])
    return w


def entropy(prof, w, n):
    """-sum_e (2^{r_e} - 2^{r_{e-1}}) P_e log2 P_e, in log space."""
    H = 0.0
    for e in range(n + 1):
        r = prof[e]
        A = sum(w[d] * 2.0 ** -(prof[d] - r) for d in range(e, n + 1))
        if A <= 0:
            continue
        share = 1.0 - (2.0 ** (prof[e - 1] - r) if e else 0.0)
        H += share * A * (r - log2(A))
    return H


def walk(cols, K, w, n, order):
    """(mixture, true) received entropy after each prefix of the order."""
    bases = [{} for _ in range(n + 1)]
    prof = [0] * (n + 1)
    masks = [(1 << K[d]) - 1 for d in range(n + 1)]
    mix = [0.0]
    tru = [entropy(prof, w, n)]
    for q in order:
        c = cols[q]
        for d in range(n + 1):
            if prof[d] == K[d]:
                continue
            v = c & masks[d]
            while v:
                h = v.bit_length() - 1
                if h in bases[d]:
                    v ^= bases[d][h]
                else:
                    bases[d][h] = v
                    prof[d] += 1
                    break
        mix.append(sum(w[d] * prof[d] for d in range(n + 1)))
        tru.append(entropy(prof, w, n))
    return mix, tru


def run(n):
    Q = 2 ** n
    cols, K = columns(n)
    w = weights(n, K)
    M = [0.0] * (Q + 1)
    H = [0.0] * (Q + 1)

    if n == 4:                                  # exact: every subset
        cnt = [0] * (Q + 1)
        for mask in range(2 ** Q):
            idx = [q for q in range(Q) if (mask >> q) & 1]
            mix, tru = walk(cols, K, w, n, idx)
            M[len(idx)] += mix[-1]
            H[len(idx)] += tru[-1]
            cnt[len(idx)] += 1
        M = [M[l] / cnt[l] for l in range(Q + 1)]
        H = [H[l] / cnt[l] for l in range(Q + 1)]
        tag = 'exact'
    else:
        nord = ORDERS[n]
        order = list(range(Q))
        for _ in range(nord):
            random.shuffle(order)
            mix, tru = walk(cols, K, w, n, order)
            for l in range(Q + 1):
                M[l] += mix[l]
                H[l] += tru[l]
        M = [x / nord for x in M]
        H = [x / nord for x in H]
        tag = '%d orders' % nord

    # Below the threshold every question is its own point: no binning at
    # all, so the staircase is the exact one.  Above it, bins of width h,
    # with one bin centred on each class rate: a class dies over a spread
    # of l about K_d, and a bin straddling K_d catches the whole burst of
    # label information instead of halving it across two.
    if Q <= FULL:
        edges = list(range(Q + 1))
    else:
        h = max(1, round(Q / BINS))
        edges = [0]

        def fill(upto):
            while edges[-1] + h <= upto:
                edges.append(edges[-1] + h)
            if upto > edges[-1]:
                edges.append(upto)

        for d in range(n):                      # interior rates only
            lo, hi = K[d] - h // 2, K[d] - h // 2 + h
            if lo <= edges[-1] or hi >= Q:
                continue
            fill(lo)
            edges.append(hi)
        fill(Q)
    with open(f"{OUT}/rmlabel_n{n}.dat", "w") as fh:
        fh.write("t mix true\n")
        for lo, hi in zip(edges, edges[1:]):
            fh.write("%.6f %.6f %.6f\n"
                     % (lo / Q, (M[hi] - M[lo]) / (hi - lo),
                        (H[hi] - H[lo]) / (hi - lo)))
    print("n=%2d (%s)  E[R_{n,D}]=%.5f  label area=%.5f  bound=%.5f"
          % (n, tag, M[Q] / Q, (H[Q] - M[Q]) / Q, log2(n + 1) / Q))


if __name__ == "__main__":
    for n in ([int(x) for x in sys.argv[1:]] or [4, 8, 12]):
        run(n)
