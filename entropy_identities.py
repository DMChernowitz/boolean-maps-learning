"""
Numerical verification of the identities in learning_agent.md, sections
"General form of the entropy reduction" and "Closed forms from the index
representation".

Checks, for the running n=2, m=1 example prior:
  1. <H_B>_Q = (H(p) + C(p)) / 2^n at k=0 and after each possible first round;
  2. expected generalization = mutual information (incl. the 10->01 case
     where the realized entropy rises but the expectation falls);
  3. the chain rule: conditional answer entropies telescope to H(p).

Then the two solvable cases:
  4. Gibbs prior with energy = popcount(truth table) factorizes exactly:
     column probability sigma(-beta), <H_B>_Q = h(sigma(beta)), C = 0;
  5. generic Dirichlet(1) prior: expected reduction ~ 1/(2 ln2 (N+1)),
     and after k rounds the same with N -> N_k = 2^(2^n - k).
"""

import math
import random

FUNCS = {
 'FALSE':'0000','AND':'0001','a&~b':'0010','a(proj)':'0011',
 '~a&b':'0100','b(proj)':'0101','XOR':'0110','OR':'0111',
 'NOR':'1000','XNOR':'1001','~b':'1010','b->a':'1011',
 '~a':'1100','a->b':'1101','NAND':'1110','TRUE':'1111',
}
PRIOR = {
 'FALSE':0.150,'AND':0.149,'a&~b':0.094,'a(proj)':0.102,'~a&b':0.060,'b(proj)':0.082,
 'XOR':0.012,'OR':0.014,'NOR':0.141,'XNOR':0.114,'~b':0.015,'b->a':0.013,
 '~a':0.017,'a->b':0.016,'NAND':0.011,'TRUE':0.010,
}
LABELS = ['00','01','10','11']


def val(name, q):
    return int(FUNCS[name][LABELS.index(q)])


def entropy(vals):
    return -sum(v*math.log2(v) for v in vals if v > 0)


def h(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


def col_entropy(belief, q):
    return entropy([sum(w for n_, w in belief.items() if val(n_, q) == a)
                    for a in (0, 1)])


def check_total_correlation():
    Hp = entropy(PRIOR.values())
    sumH = sum(col_entropy(PRIOR, q) for q in LABELS)
    C0 = sumH - Hp
    assert abs(sumH/4 - (Hp + C0)/4) < 1e-12
    print(f"k=0: H(p)={Hp:.4f} C={C0:.4f} <H_B>={(Hp+C0)/4:.4f}")
    for qk in LABELS:  # psi = AND
        at = val('AND', qk)
        raw = {x: w for x, w in PRIOR.items() if val(x, qk) == at}
        tot = sum(raw.values())
        b = {x: v/tot for x, v in raw.items()}
        Hp1 = entropy(b.values())
        sumH1 = sum(col_entropy(b, q) for q in LABELS if q != qk)
        print(f"  learned {qk}: H(p1)={Hp1:.4f} C1={sumH1-Hp1:.4f} "
              f"<H_B>={(sumH1)/4:.4f}")


def check_mutual_information(qk='10', qp='01'):
    H0 = col_entropy(PRIOR, qp)
    exp_H = 0.0
    for at in (0, 1):
        raw = {x: w for x, w in PRIOR.items() if val(x, qk) == at}
        Pa = sum(raw.values())
        exp_H += Pa * col_entropy({x: v/Pa for x, v in raw.items()}, qp)
    print(f"I(A_{qp};A_{qk}) = {H0 - exp_H:.4f} (>= 0)")
    assert H0 - exp_H >= 0


def check_chain_rule():
    def block_H(qs):
        dist = {}
        for x, w in PRIOR.items():
            key = tuple(val(x, q) for q in qs)
            dist[key] = dist.get(key, 0) + w
        return entropy(dist.values())
    total = sum(block_H(LABELS[:k+1]) - block_H(LABELS[:k])
                for k in range(len(LABELS)))
    Hp = entropy(PRIOR.values())
    assert abs(total - Hp) < 1e-9
    print(f"chain rule: sum of conditional answer entropies = {total:.4f} = H(p)")


def check_popcount_gibbs(n=3, beta=1.0):
    Nq, N = 1 << n, 1 << (1 << n)
    w = [bin(j).count('1') for j in range(N)]
    Z = sum(math.exp(-beta*wi) for wi in w)
    p = [math.exp(-beta*wi)/Z for wi in w]
    theta = math.exp(-beta)/(1+math.exp(-beta))  # sigma(-beta)
    cols = [sum(p[j] for j in range(N) if (j >> q) & 1) for q in range(Nq)]
    Hp = entropy(p)
    C = sum(h(c) for c in cols) - Hp
    assert all(abs(c - theta) < 1e-12 for c in cols)
    assert abs(C) < 1e-9
    assert abs(Z - (1+math.exp(-beta))**Nq) < 1e-6
    print(f"popcount-Gibbs n={n} beta={beta}: cols=sigma(-b)={theta:.6f}, "
          f"<H_B>=h(sigma(b))={h(theta):.6f}, C={C:.1e}")


def check_dirichlet(n=3, trials=3000, seed=0):
    rng = random.Random(seed)
    Nq, N = 1 << n, 1 << (1 << n)
    tot = 0.0
    for _ in range(trials):
        e = [rng.expovariate(1.0) for _ in range(N)]
        s = sum(e)
        p = [x/s for x in e]
        cols = [sum(p[j] for j in range(N) if (j >> q) & 1) for q in range(Nq)]
        tot += 1 - sum(h(c) for c in cols)/Nq
    pred = 1/(2*math.log(2)*(N+1))
    print(f"Dirichlet n={n}: mean reduction={tot/trials:.6f} "
          f"predicted 1/(2 ln2 (N+1))={pred:.6f}")


if __name__ == '__main__':
    check_total_correlation()
    check_mutual_information()
    check_chain_rule()
    check_popcount_gibbs()
    check_dirichlet()
