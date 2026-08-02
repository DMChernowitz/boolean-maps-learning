"""
Numerical verification of the identities in learning_agent.md, section
"Correlated but still solvable: mixtures of product priors".

Setting: n=3, m=2 (8 questions, 4 possible answers, N = 4^8 = 65536
hypotheses), prior = mixture of L=3 noisy templates with symbol noise
eps. Every mixture-side prediction is checked against brute force over
the full hypothesis space.

Checks:
  1. Delta H(q) = H_B(q) for ANY prior (shown on the doc's n=2, m=1
     example prior AND on the mixture prior): the expected belief-entropy
     drop from asking q equals the current answer entropy of column q of M.
  2. M_{a,q} = sum_t w_t pi_t(a|q): columns of M are mixtures of the
     component columns.
  3. The update rule maps mixture to mixture: brute-force Bayes on all
     65536 hypotheses agrees with the L-dimensional weight update
     w_t <- w_t pi_t(a*|q_k) / M_{a*,q_k}, at every round of a full
     greedy trace.
  4. Pairwise joints P(A_q=a, A_q'=a') = sum_t w_t pi_t(a|q) pi_t(a'|q'),
     hence expected generalization
     E_a[H_B'(q')] = H_B(q') - I(A_q'; A_q) in elementary form.
  5. Belief entropy: H(p) = H(w) + sum_t w_t sum_q H(pi_t(.|q)) - H(T|A),
     with the identifiability leakage H(T|A) computed brute-force (tiny).
  6. Total correlation: C(p) = sum_q I(A_q;T) - I(A;T) > 0, extensive in
     the number of questions the templates disagree on.
"""

import math

log2 = math.log2


def H(vec):
    return -sum(x * log2(x) for x in vec if x > 0)


# ------------------------------------------------------------- check 1a
# Delta H(q) = H_B(q) holds for ANY prior: the doc's n=2, m=1 example.

FUNCS = {
 'FALSE': '0000', 'AND': '0001', 'a&~b': '0010', 'a(proj)': '0011',
 '~a&b': '0100', 'b(proj)': '0101', 'XOR': '0110', 'OR': '0111',
 'NOR': '1000', 'XNOR': '1001', '~b': '1010', 'b->a': '1011',
 '~a': '1100', 'a->b': '1101', 'NAND': '1110', 'TRUE': '1111',
}
PRIOR = {
 'FALSE': 0.150, 'AND': 0.149, 'a&~b': 0.094, 'a(proj)': 0.102,
 '~a&b': 0.060, 'b(proj)': 0.082, 'XOR': 0.012, 'OR': 0.014,
 'NOR': 0.141, 'XNOR': 0.114, '~b': 0.015, 'b->a': 0.013,
 '~a': 0.017, 'a->b': 0.016, 'NAND': 0.011, 'TRUE': 0.010,
}
LABELS = ['00', '01', '10', '11']


def check_deltaH_doc_prior():
    print("check 1a: Delta H(q) = H_B(q) on the doc's n=2,m=1 prior")
    Hp = H(PRIOR.values())
    for qi, q in enumerate(LABELS):
        col = [sum(w for f, w in PRIOR.items() if int(FUNCS[f][qi]) == a)
               for a in (0, 1)]
        expH = 0.0
        for a in (0, 1):
            raw = [w for f, w in PRIOR.items() if int(FUNCS[f][qi]) == a]
            Pa = sum(raw)
            expH += Pa * H([v / Pa for v in raw])
        dH = Hp - expH
        assert abs(dH - H(col)) < 1e-12
        print(f"  q={q}: Delta H = {dH:.3f} = H_B(q)")


# --------------------------------------------------------- mixture setup

n, m = 3, 2
NQ, K = 1 << n, 1 << m          # 8 questions, 4 answers
N = K ** NQ                     # 65536 hypotheses
EPS = 0.15
TEMPLATES = [
    [q & 3 for q in range(NQ)],               # copy the low two input bits
    [0] * NQ,                                 # constant 0
    [3 * ((q >> 2) & 1) for q in range(NQ)],  # both output bits = x2
]
L = len(TEMPLATES)
W0 = [0.5, 0.3, 0.2]
# true map: template 0 corrupted at q=6 (2 -> 1)
PSI = [0, 1, 2, 3, 0, 1, 1, 3]


def pi(t, a, q):
    return 1.0 - EPS if TEMPLATES[t][q] == a else EPS / (K - 1)


def digit(j, q):
    return (j >> (m * q)) & (K - 1)


def brute_p():
    p = []
    for j in range(N):
        p.append(sum(w * math.prod(pi(t, digit(j, q), q) for q in range(NQ))
                     for t, w in enumerate(W0)))
    return p


def brute_cols(p):
    cols = [[0.0] * K for _ in range(NQ)]
    for j, pj in enumerate(p):
        for q in range(NQ):
            cols[q][digit(j, q)] += pj
    return cols


def mix_col(w, q, asked):
    if q in asked:
        return [1.0 if a == asked[q] else 0.0 for a in range(K)]
    return [sum(w[t] * pi(t, a, q) for t in range(L)) for a in range(K)]


def brute_deltaH(p, q):
    """Expected belief-entropy drop from asking q, by brute force."""
    Hp = H(p)
    expH = 0.0
    for a in range(K):
        raw = [pj for j, pj in enumerate(p) if digit(j, q) == a]
        Pa = sum(raw)
        if Pa > 0:
            expH += Pa * H([v / Pa for v in raw])
    return Hp - expH


def main():
    check_deltaH_doc_prior()

    p = brute_p()
    assert abs(sum(p) - 1) < 1e-9

    # ------------------------------------------------------- check 2
    cols = brute_cols(p)
    for q in range(NQ):
        mc = mix_col(W0, q, {})
        assert max(abs(cols[q][a] - mc[a]) for a in range(K)) < 1e-12
    print("check 2: M columns = mixture of component columns "
          f"(all {NQ} columns, max err < 1e-12)")

    # ------------------------------------------------------- check 1b
    print("check 1b: Delta H(q) = H_B(q) on the mixture prior, round 1")
    for q in range(NQ):
        dH = brute_deltaH(p, q)
        assert abs(dH - H(cols[q])) < 1e-9
        print(f"  q={q} ({q:03b}): Delta H = {dH:.4f} = H_B(q)")

    # ------------------------------------------------------- check 4
    # pairwise joints and expected generalization, at round 1
    worst = 0.0
    for qk in range(NQ):
        for qp in range(NQ):
            if qp == qk:
                continue
            # brute pairwise joint
            joint_b = [[0.0] * K for _ in range(K)]
            for j, pj in enumerate(p):
                joint_b[digit(j, qk)][digit(j, qp)] += pj
            # mixture pairwise joint
            joint_m = [[sum(W0[t] * pi(t, a, qk) * pi(t, ap, qp)
                            for t in range(L))
                        for ap in range(K)] for a in range(K)]
            err = max(abs(joint_b[a][ap] - joint_m[a][ap])
                      for a in range(K) for ap in range(K))
            worst = max(worst, err)
            # expected generalization = mutual information
            Pa = [sum(joint_m[a]) for a in range(K)]
            expH = sum(Pa[a] * H([joint_m[a][ap] / Pa[a] for ap in range(K)])
                       for a in range(K) if Pa[a] > 0)
            I = H(cols[qp]) - expH
            # brute-force expectation of the post-update column entropy
            expH_b = 0.0
            for a in range(K):
                raw = {j: pj for j, pj in enumerate(p) if digit(j, qk) == a}
                Pa_b = sum(raw.values())
                col = [0.0] * K
                for j, pj in raw.items():
                    col[digit(j, qp)] += pj / Pa_b
                expH_b += Pa_b * H(col)
            assert abs(expH - expH_b) < 1e-9
            assert I >= -1e-12
    print(f"check 4: pairwise joints (all {NQ*(NQ-1)} ordered pairs, "
          f"max err {worst:.1e}); E[H_B'(q')] = H_B(q') - I(A_q';A_q)")

    # ------------------------------------------------------- checks 5, 6
    Hp = H(p)
    h_eps = H([1 - EPS] + [EPS / (K - 1)] * (K - 1))
    leak = 0.0     # H(T | A) = identifiability leakage
    for j, pj in enumerate(p):
        post = [W0[t] * math.prod(pi(t, digit(j, q), q) for q in range(NQ))
                / pj for t in range(L)]
        leak += pj * H(post)
    lhs = H(W0) + NQ * h_eps - leak
    assert abs(Hp - lhs) < 1e-9
    C = sum(H(cols[q]) for q in range(NQ)) - Hp
    C_pred = sum(H(cols[q]) - h_eps for q in range(NQ)) - (H(W0) - leak)
    assert abs(C - C_pred) < 1e-9
    print(f"check 5: H(p) = {Hp:.6f} = H(w) + 2^n h_eps - H(T|A) "
          f"= {H(W0):.4f} + {NQ * h_eps:.4f} - {leak:.2e}")
    print(f"check 6: C(p) = {C:.6f} = sum_q I(A_q;T) - I(A;T) > 0")

    # ------------------------------------------------------- check 3
    # full greedy trace: forecast from mixture, Bayes by brute force
    print("check 3: greedy trace, mixture weight-update vs brute-force Bayes")
    w = list(W0)
    asked = {}
    envelope = lambda k: m * (1 - k / NQ)
    for k in range(1, NQ + 1):
        cols = brute_cols(p)
        # forecast Delta H for every remaining question, mixture side
        remaining = [q for q in range(NQ) if q not in asked]
        forecasts = {}
        for q in remaining:
            mc = mix_col(w, q, asked)
            assert max(abs(cols[q][a] - mc[a]) for a in range(K)) < 1e-9
            dH_b = brute_deltaH(p, q)
            assert abs(dH_b - H(mc)) < 1e-9
            forecasts[q] = H(mc)
        qk = max(remaining, key=lambda q: forecasts[q])
        astar = PSI[qk]
        Hp_before = H(p)
        # brute Bayes update
        Pa = sum(pj for j, pj in enumerate(p) if digit(j, qk) == astar)
        p = [pj / Pa if digit(j, qk) == astar else 0.0
             for j, pj in enumerate(p)]
        # mixture weight update
        Z = sum(w[t] * pi(t, astar, qk) for t in range(L))
        w = [w[t] * pi(t, astar, qk) / Z for t in range(L)]
        asked[qk] = astar
        # verify the FULL updated M agrees
        cols_after = brute_cols(p)
        errM = max(abs(cols_after[q][a] - mix_col(w, q, asked)[a])
                   for q in range(NQ) for a in range(K))
        assert errM < 1e-9
        avgH = sum(H(c) for c in cols_after) / NQ
        print(f"  k={k}: asked q={qk} ({qk:03b}) -> a*={astar}, "
              f"forecast dH={forecasts[qk]:.4f}, realized "
              f"{Hp_before - H(p):.4f}, w=({', '.join(f'{x:.3f}' for x in w)}), "
              f"<H_B>={avgH:.4f} (envelope {envelope(k):.3f}), M err {errM:.0e}")

    print("all checks passed")


if __name__ == '__main__':
    main()
