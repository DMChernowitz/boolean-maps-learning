"""
Numerical verification and Monte Carlo for the learning_agent.md section
"Where the information flows: the answer, the columns, and the tower".

Exact identities (verified below on (2,1), any prior):
  1. expected surprisal = H_B(q_k) = the asked column's (deterministic)
     entropy drop;
  2. expected drop of the other columns = T = sum_{q'} I(A_q'; A_qk)
     = expected drop of the total correlation C  (the tower).

Round-resolved flows for a fixed question order (answers integrated
exactly -- no sampling needed):
  direct_k   = H(A_{q_k} | A_{S_{k-1}})
  transfer_k = sum_{q' not yet asked} [ H(A_q'|A_{S_{k-1}}) - H(A_q'|A_{S_k}) ]
with the invariants  sum_k direct_k = H(p),  sum_k transfer_k = C(p)
for EVERY order (verified per sampled order to machine precision).

Monte Carlo is only over what cannot be averaged analytically:
  - random policy: sample question orders uniformly;
  - greedy policy: sample the truth psi ~ p and play greedy (its choices
    depend on realized answers), recording realized ledger flows.
"""

import csv
import math

import numpy as np

REPO_TABLES = {
    ("4to1", 4, 1): "output/table_4to1.csv",
    ("3to1", 3, 1): "output/table_3to1.csv",
}


def load_table(path, column="circuit_complexity"):
    out = []
    with open(path) as fh:
        r = csv.reader(fh)
        idx = next(r).index(column)
        for row in r:
            out.append(int(row[idx]))
    return np.array(out, dtype=np.int64)


def entropy(w):
    w = w[w > 1e-300]
    return float(-(w * np.log2(w)).sum())


def make_digits(n, m):
    nq, N = 1 << n, (1 << m) ** (1 << n)
    j = np.arange(N, dtype=np.int64)
    return [( (j >> (m * q)) & ((1 << m) - 1) ).astype(np.int64)
            for q in range(nq)]


def block_entropy(p, dig, S, K):
    """H of the joint distribution of the answers in S (list of q)."""
    if not S:
        return 0.0
    key = np.zeros_like(dig[0])
    for i, q in enumerate(S):
        key = key * K + dig[q]
    return entropy(np.bincount(key, weights=p))


def flows_fixed_order(p, dig, order, K):
    """Exact expected (direct_k, transfer_k) per round for one order."""
    nq = len(dig)
    Hp = entropy(p)
    HS = 0.0                     # H(A_S) for the current prefix
    Wprev = {}
    for q in range(nq):          # W(q', empty) = H_B(q')
        Wprev[q] = block_entropy(p, dig, [q], K)
    Ctot = sum(Wprev.values()) - Hp
    S = []
    direct, transfer = [], []
    for qk in order:
        direct.append(Wprev[qk])
        S.append(qk)
        HSnew = block_entropy(p, dig, S, K)
        tr = 0.0
        Wnew = {}
        for q in Wprev:
            if q == qk:
                continue
            Wnew[q] = block_entropy(p, dig, S + [q], K) - HSnew
            tr += Wprev[q] - Wnew[q]
        transfer.append(tr)
        Wprev = Wnew
        HS = HSnew
    assert abs(sum(direct) - Hp) < 1e-8
    assert abs(sum(transfer) - Ctot) < 1e-8
    return np.array(direct), np.array(transfer)


def greedy_realized(p, dig, K, rng):
    """Sample psi ~ p, play greedy, record realized ledger flows."""
    nq = len(dig)
    psi = rng.choice(len(p), p=p)
    alive = np.ones(len(p), dtype=bool)
    w = p.copy()
    remaining = list(range(nq))
    direct, transfer = [], []
    HB = {q: entropy(np.bincount(dig[q], weights=w, minlength=K))
          for q in remaining}
    while remaining:
        mx = max(HB[q] for q in remaining)
        qk = rng.choice([q for q in remaining if HB[q] >= mx - 1e-12])
        a = dig[qk][psi]
        direct.append(HB[qk])          # asked-column drop (deterministic)
        keep = dig[qk] == a
        w = np.where(keep, w, 0.0)
        w /= w.sum()
        remaining.remove(qk)
        tr = 0.0
        for q in remaining:
            newH = entropy(np.bincount(dig[q], weights=w, minlength=K))
            tr += HB[q] - newH
            HB[q] = newH
        transfer.append(tr)
    return np.array(direct), np.array(transfer)


def run(name, n, m, beta, n_orders=40, n_greedy=120, seed=0):
    K, nq = 1 << m, 1 << n
    C = load_table(REPO_TABLES[(name, n, m)]) if (name, n, m) in REPO_TABLES \
        else None
    w = np.exp(-beta * C.astype(np.float64))
    p = w / w.sum()
    dig = make_digits(n, m)
    Hp = entropy(p)
    Ctot = sum(block_entropy(p, dig, [q], K) for q in range(nq)) - Hp
    print(f"\n=== {name}, beta={beta}: H(p)={Hp:.3f}, C(p)={Ctot:.3f}, "
          f"whole-run transfer share C/(H+C)={100*Ctot/(Hp+Ctot):.1f}% ===")

    rng = np.random.default_rng(seed)
    dsum = np.zeros(nq)
    tsum = np.zeros(nq)
    for _ in range(n_orders):
        order = rng.permutation(nq)
        d, t = flows_fixed_order(p, dig, list(order), K)
        dsum += d; tsum += t
    d, t = dsum / n_orders, tsum / n_orders
    print("random order (answers exact, MC over orders):")
    print("  k:        " + " ".join(f"{k+1:6d}" for k in range(nq)))
    print("  direct:   " + " ".join(f"{x:6.3f}" for x in d))
    print("  transfer: " + " ".join(f"{x:6.3f}" for x in t))
    print("  share%:   " + " ".join(f"{100*x/(x+y):6.1f}" if x+y > 1e-9
                                     else "     -" for x, y in zip(t, d)))

    dsum = np.zeros(nq); tsum = np.zeros(nq)
    for _ in range(n_greedy):
        dd, tt = greedy_realized(p, dig, K, rng)
        dsum += dd; tsum += tt
    d, t = dsum / n_greedy, tsum / n_greedy
    print(f"greedy (MC over psi ~ p, {n_greedy} runs, realized flows):")
    print("  direct:   " + " ".join(f"{x:6.3f}" for x in d))
    print("  transfer: " + " ".join(f"{x:6.3f}" for x in t))
    print("  share%:   " + " ".join(f"{100*x/(x+y):6.1f}" if x+y > 1e-9
                                     else "     -" for x, y in zip(t, d)))


if __name__ == "__main__":
    # exact identity check on (2,1), asked question q=2, beta=1.3
    C21 = np.array([0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0])
    w = np.exp(-1.3 * C21.astype(float)); p = w / w.sum()
    dig = make_digits(2, 1)
    d, t = flows_fixed_order(p, dig, [2, 0, 1, 3], 2)
    print(f"(2,1) fixed order [2,0,1,3]: direct_1={d[0]:.6f} (=H_B=1), "
          f"transfer_1={t[0]:.6f} (=0.124145 from the two-ledger check)")

    run("3to1", 3, 1, 2.0)
    run("4to1", 4, 1, 2.0, n_orders=30, n_greedy=80)
