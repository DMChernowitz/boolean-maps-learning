"""Layered entropy books for the Gibbs ensembles, uniform question order.

At every l the three exact averages of eq:Lk-avg partition the starting
table entropy:

    H(M^(0))  =  <H(M^(l))>   +   G_l        +   (C_0 - <C_l>)
                  remaining        received       deduced

with H(M^(0)) = Q G_1, <H(M^(l))> = (Q-l)(G_{l+1} - G_l), and the
deduced layer the remainder.  G is the mean block entropy, computed by
the subset-lattice recursion of gibbs_curves.average_curve.

Emits the pgfplots coordinates for the nine stacked-bar panels.
"""
import numpy as np

from gibbs_curves import load_21, load_32, load_41, gibbs, ent, PARAMS

SYSTEMS = [("21", load_21, "(2,1)"),
           ("32", load_32, "(3,2)"),
           ("41", load_41, "(4,1)")]


def block_entropies(sys, p):
    """G_0 .. G_Q by the subset-lattice recursion"""
    Q, A = sys["Q"], sys["A"]
    arr = np.zeros((A,) * Q)
    np.add.at(arr, tuple(sys["D"][::-1]), p)
    G = np.zeros(Q + 1)
    cnt = np.zeros(Q + 1)

    def rec(a, k, bound):
        G[k] += ent(a.ravel())
        cnt[k] += 1
        for pos in range(min(k, bound)):
            rec(a.sum(axis=k - 1 - pos), k - 1, pos)

    rec(arr, Q, Q)
    return G / np.maximum(cnt, 1)


def layers(sys, p):
    Q = sys["Q"]
    G = block_entropies(sys, p)
    HM0 = Q * G[1]
    rows = []
    for l in range(Q + 1):
        rem = (Q - l) * (G[l + 1] - G[l]) if l < Q else 0.0
        rows.append((l, rem, G[l], HM0 - rem - G[l]))
    return HM0, rows


def main():
    for tag, load, name in SYSTEMS:
        sys = load()
        print("=" * 64)
        for label, b, a, mu in PARAMS:
            p = gibbs(sys, b, a, mu)
            HM0, rows = layers(sys, p)
            print("%% %s %s   H(M^(0)) = %.4f" % (name, label, HM0))
            for key, idx in (("rem", 1), ("rec", 2), ("ded", 3)):
                coords = " ".join("(%d,%.5f)" % (r[0], r[idx]) for r in rows)
                print("%%   %s: %s" % (key, coords))
            worst = max(abs(r[1] + r[2] + r[3] - HM0) for r in rows)
            assert worst < 1e-9, worst
            print("%%   conservation residual %.2e" % worst)


if __name__ == "__main__":
    main()
