"""Mean rank increment of one Reed-Muller class along a random question
order: gamma^(d)_{n,l} = Pr[the (l+1)-th column is independent of the
first l].  This is the finite-n object that converges to the step
theta(R_d - t) of the thermodynamic limit.

For each n the class is the one whose rate K_d/Q sits closest to 1/3, so
that the three curves share a limiting rate and the sharpening is the
only thing the eye has to compare.

n=4: every question order is covered exactly by enumerating subsets is
unnecessary -- averaging the indicator over uniformly random orders is
already the definition, and at n=4 we take enough orders that the curve
is smooth to plotting accuracy.
Usage: python rm_gamma_data.py
Writes figures/data/rmgamma_n{4,6,8}.dat.
"""
import random
import sys
from math import comb

random.seed(20260824)
OUT = r"figures/data"
ORDERS = {6: 60000, 8: 12000, 10: 4000, 12: 1200}


def columns(n, d):
    """One integer per question: the allowed monomials of class d,
    evaluated there, packed as bits."""
    monos = [s for r in range(d + 1)
             for s in __import__('itertools').combinations(range(n), r)]
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
    return cols, len(monos)


TARGET = 0.6                    # the draw R; the class is whichever gap holds it
#  chosen so the selected rate closes on R from above as n grows,
#  rather than drifting away from it across the computable range


def pick_class(n):
    """The class the sampling rule selects for a draw at R = TARGET: the
    gap that contains it, so the rate is TARGET rounded up to the grid."""
    Q = 2 ** n
    K = [sum(comb(n, i) for i in range(d + 1)) for d in range(n + 1)]
    return min(d for d in range(n + 1) if K[d] / Q >= TARGET)


def gamma(n, nord):
    d = pick_class(n)
    cols, K = columns(n, d)
    Q = 2 ** n
    hits = [0] * Q                      # times column l+1 raised the rank
    order = list(range(Q))
    for _ in range(nord):
        random.shuffle(order)
        basis, rank = {}, 0
        for l, q in enumerate(order):
            if rank == K:               # class exhausted, nothing can help
                break
            v = cols[q]
            while v:
                h = v.bit_length() - 1
                if h in basis:
                    v ^= basis[h]
                else:
                    basis[h] = v
                    rank += 1
                    hits[l] += 1
                    break
    return d, K, [h / nord for h in hits]


def main():
    for n in (6, 8, 10, 12):
        Q = 2 ** n
        d, K, g = gamma(n, ORDERS[n])
        area = sum(g) / Q
        with open(f"{OUT}/rmgamma_n{n}.dat", "w") as fh:
            fh.write("t gamma\n")
            for l, val in enumerate(g):
                fh.write(f"{l / Q:.6f} {val:.6f}\n")
        print(f"n={n} d={d} K={K} rate={K/Q:.4f} "
              f"area={area:.4f} (exact {K/Q:.4f})")


if __name__ == "__main__":
    main()
