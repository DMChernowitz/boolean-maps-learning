"""Gibbs complexity prior: expected leverage trajectories.

Prior p_j ~ exp(-(beta X_j + alpha F_j + mu B_j)) with X the AIG
complexity, F = n I + S the footprint, B the signed weight bias, read
from the classification tables (output/table_*.csv; the 2,1 values are
small enough to inline).

Strategies (Sec. 5.1 grid + the uniform average):
  average            uniform random order (the G_k laws)
  greedy fixed       grow the set by expected drop of H(M)
  greedy contingent  re-rank after every answer
  opt-l fixed        best set of size l* = 2^n/2, random order inside
  opt-l contingent   backward induction to horizon l*, rolled forward
For (4,1) the contingent optimum is skipped (3^16 posterior states).

All curves are <L_l> = (H(M)_0 - E[remaining_l]) / E[received_l],
ratios of expectations, in bits.  Prints pgfplots-ready coordinates
plus the class-count tables for the paper.
"""
import csv
import json
import math
from itertools import combinations

import numpy as np

LOG2 = math.log(2)


def ent(w):
    """entropy in bits of an unnormalized nonneg vector (sums to Z)"""
    Z = w.sum()
    if Z <= 0:
        return 0.0
    w = w[w > 1e-300]
    return float(math.log2(Z) - (w*np.log2(w)).sum()/Z)


# ---------------- systems ----------------
def load_21():
    X = [0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0]
    F = [2, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 5, 6, 6, 2]
    B = [2*bin(j).count("1") - 4 for j in range(16)]
    D = np.array([[(j >> q) & 1 for j in range(16)] for q in range(4)])
    return dict(Q=4, A=2, m=1, D=D, X=np.array(X), F=np.array(F),
                B=np.array(B), name="21")


def load_csv(path):
    X, F, B = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            X.append(int(row["circuit_complexity"]))
            F.append(int(row["footprint"]))
            B.append(int(row["weight_bias"]))
    return np.array(X), np.array(F), np.array(B)


def load_41():
    X, F, B = load_csv("output/table_4to1.csv")
    f = np.arange(65536)
    D = (f[None, :] >> np.arange(16)[:, None]) & 1
    return dict(Q=16, A=2, m=1, D=D, X=X, F=F, B=B, name="41")


def load_32():
    X, F, B = load_csv("output/table_3to2.csv")
    c = np.arange(65536)
    f0, f1 = c >> 8, c & 0xFF
    q = np.arange(8)[:, None]
    D = (((f0[None, :] >> q) & 1) << 1) | ((f1[None, :] >> q) & 1)
    return dict(Q=8, A=4, m=2, D=D, X=X, F=F, B=B, name="32")


# ---------------- machinery ----------------
def HM(D, w):
    """sum of column entropies of the (unnormalized) posterior w"""
    return sum(ent(np.bincount(D[q], weights=w, minlength=D.max()+1))
               for q in range(D.shape[0]))


def make_key(D, T, A):
    """pattern key over question set T (tuple), base-A packing"""
    key = np.zeros(D.shape[1], dtype=np.int64)
    for i, q in enumerate(T):
        key += D[q].astype(np.int64) * (A**i)
    return key


class Joint:
    """memoized joint entropies H(A_T) and remainders for set search"""

    def __init__(self, sys, p):
        self.D, self.A, self.Q, self.p = sys["D"], sys["A"], sys["Q"], p
        self.cache = {}

    def H(self, T):
        T = tuple(sorted(T))
        if T not in self.cache:
            if not T:
                self.cache[T] = 0.0
            else:
                key = make_key(self.D, T, self.A)
                w = np.bincount(key, weights=self.p,
                                minlength=self.A**len(T))
                self.cache[T] = ent(w)
        return self.cache[T]

    def remaining(self, T):
        """E over patterns of sum of unasked conditional col entropies"""
        T = tuple(sorted(T))
        return sum(self.H(T + (q,)) - self.H(T)
                   for q in range(self.Q) if q not in T)


def average_curve(sys, p):
    """uniform-order laws via the subset-lattice recursion"""
    Q, A = sys["Q"], sys["A"]
    arr = np.zeros((A,)*Q)
    idx = tuple(sys["D"][::-1])          # axis 0 <-> question Q-1
    np.add.at(arr, idx, p)
    G = np.zeros(Q + 1)
    cnt = np.zeros(Q + 1)

    def rec(a, k, bound):
        w = a.ravel()
        G[k] += ent(w)
        cnt[k] += 1
        for pos in range(min(k, bound)):
            rec(a.sum(axis=k - 1 - pos), k - 1, pos)

    # axis (k-1-pos) of a k-question array holds question index 'pos'
    rec(arr, Q, Q)
    G /= np.maximum(cnt, 1)
    HM0 = Q * G[1]
    out = []
    for l in range(1, Q + 1):
        rem = (Q - l)*(G[l+1] - G[l]) if l < Q else 0.0
        out.append((HM0 - rem)/G[l])
    return out, HM0


def greedy_fixed_curve(sys, p, HM0):
    J = Joint(sys, p)
    T = ()
    out = []
    for _ in range(sys["Q"]):
        cands = [q for q in range(sys["Q"]) if q not in T]
        q = min(cands, key=lambda q: J.remaining(tuple(T) + (q,)))
        T = tuple(sorted(T + (q,)))
        out.append((HM0 - J.remaining(T))/J.H(T))
    return out


def opt_fixed_curve(sys, p, HM0, lstar):
    J = Joint(sys, p)
    best = min(combinations(range(sys["Q"]), lstar), key=J.remaining)
    out = []
    for l in range(1, lstar + 1):
        subs = list(combinations(best, l))
        rec_ = sum(J.H(U) for U in subs)/len(subs)
        rem = sum(J.remaining(U) for U in subs)/len(subs)
        out.append((HM0 - rem)/rec_)
    return out, best


def greedy_contingent_curve(sys, p, HM0):
    D, Q, A = sys["D"], sys["Q"], sys["A"]
    nodes = [(np.arange(D.shape[1]), p.copy(), frozenset(), 0.0)]
    out = []
    for _ in range(Q):
        Erec = Erem = 0.0
        nxt = []
        for idx, w, asked, s in nodes:
            Z = w.sum()
            Dn = D[:, idx]
            best, bq = None, None
            for q in range(Q):
                if q in asked:
                    continue
                drop = 0.0
                for a in range(A):
                    wc = w[Dn[q] == a]
                    Za = wc.sum()
                    if Za > 0:
                        drop -= (Za/Z)*HM(Dn, w*(Dn[q] == a))
                # constant term HM(node) omitted: same for all q
                if best is None or drop > best + 1e-12:
                    best, bq = drop, q
            for a in range(A):
                sel = Dn[bq] == a
                Za = w[sel].sum()
                if Za <= 0:
                    continue
                child = (idx[sel], w[sel], asked | {bq},
                         s - math.log2(Za/Z))
                nxt.append(child)
                Erec += Za*child[3]
                Erem += Za*HM(D[:, child[0]], child[1])
        nodes = nxt
        out.append((HM0 - Erem)/Erec)
    return out


def opt_contingent_curve(sys, p, HM0, lstar):
    """backward induction to horizon l*, then the policy rolled
    forward; states are canonical sorted ((q, a), ...) tuples, with
    posterior supports passed down the recursion"""
    D, Q, A = sys["D"], sys["Q"], sys["A"]
    memo = {}

    def V(state, idx, w, t):
        if state in memo:
            return memo[state]
        if t == 0:
            memo[state] = (HM(D[:, idx], w), -1)
            return memo[state]
        askedqs = {q for q, _ in state}
        Z = w.sum()
        best, bq = None, None
        for q in range(Q):
            if q in askedqs:
                continue
            val = 0.0
            for a in range(A):
                sel = D[q, idx] == a
                Za = w[sel].sum()
                if Za > 0:
                    child = tuple(sorted(state + ((q, a),)))
                    val += (Za/Z)*V(child, idx[sel], w[sel], t - 1)[0]
            if best is None or val < best - 1e-12:
                best, bq = val, q
        memo[state] = (best, bq)
        return memo[state]

    root_idx = np.arange(D.shape[1])
    V((), root_idx, p.copy(), lstar)

    # roll the horizon-l* policy forward
    nodes = [((), root_idx, p.copy(), 0.0)]  # state, idx, w, surprisal
    out = []
    for _ in range(lstar):
        Erec = Erem = 0.0
        nxt = []
        for state, idx, w, s in nodes:
            q = memo[state][1]
            Z = w.sum()
            for a in range(A):
                sel = D[q, idx] == a
                Za = w[sel].sum()
                if Za <= 0:
                    continue
                child = (tuple(sorted(state + ((q, a),))), idx[sel],
                         w[sel], s - math.log2(Za/Z))
                nxt.append(child)
                Erec += Za*child[3]
                Erem += Za*HM(D[:, child[1]], child[2])
        nodes = nxt
        out.append((HM0 - Erem)/Erec)
    return out


# ---------------- run ----------------
PARAMS = [("Occam", 1.0, 0.0, 0.0),
          ("Occamer", 2.0, 0.0, 0.0),
          ("Pragmatist", 1.5, 1.0, -0.3)]


def gibbs(sys, beta, alpha, mu):
    E = beta*sys["X"] + alpha*sys["F"] + mu*sys["B"]
    w = np.exp(-(E - E.min()))
    return w/w.sum()


def main():
    results = {}
    for load in (load_21, load_32, load_41):
        sys = load()
        name = sys["name"]
        lstar = 2**{"21": 2, "32": 3, "41": 4}[name] // 2
        full = name != "41"
        for tag, beta, alpha, mu in PARAMS:
            p = gibbs(sys, beta, alpha, mu)
            avg, HM0 = average_curve(sys, p)
            r = {"HM0": HM0, "Hp": ent(p), "average": avg,
                 "greedy_fixed": greedy_fixed_curve(sys, p, HM0),
                 "greedy_contingent":
                     greedy_contingent_curve(sys, p, HM0)}
            of, bestset = opt_fixed_curve(sys, p, HM0, lstar)
            r["opt_fixed"] = of
            r["opt_fixed_set"] = list(bestset)
            if full:
                r["opt_contingent"] = opt_contingent_curve(
                    sys, p, HM0, lstar)
            results[f"{name}_{tag}"] = r
            print(f"== {name} params {tag} (beta={beta}, alpha={alpha},"
                  f" mu={mu}):  H(M)0={HM0:.4f}  H(p)={r['Hp']:.4f}")
            for k in ("average", "greedy_fixed", "greedy_contingent",
                      "opt_fixed", "opt_contingent"):
                if k in r:
                    print(f"  {k:18s}",
                          " ".join(f"{v:.4f}" for v in r[k]))
    with open("output/gibbs_curves.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote output/gibbs_curves.json")

    # class-count tables
    for load in (load_21, load_41, load_32):
        sys = load()
        vals, cnts = np.unique(sys["X"], return_counts=True)
        print(f"per-X counts {sys['name']}:",
              dict(zip(vals.tolist(), cnts.tolist())))
    s21 = load_21()
    tup = {}
    for j in range(16):
        t = (int(s21["X"][j]), int(s21["F"][j]), int(s21["B"][j]))
        tup[t] = tup.get(t, 0) + 1
    print("2,1 (X,F,B) breakdown:", sorted(tup.items()))


if __name__ == "__main__":
    main()
