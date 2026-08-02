"""
Numerical verification of the identities in learning_agent.md, section
"Computing the correlators for general m: a worked recipe at n=3, m=2".

Prior: a deliberately lumpy prior on the N = 4^8 = 65536 hypotheses
that singles out three maps (extra mass 0.40 / 0.25 / 0.15 on them)
and spreads the remaining 0.20 uniformly over everything.

Checks:
  1. master formula hat p(c) = sum_j p_j (-1)^popcount(j AND c) agrees
     with the closed form
       hat p(c) = sum_r v_r (-1)^popcount(j_r AND c)   (c != 0),
     the uniform background contributing nothing beyond shell 0;
  2. the worked examples A-D quoted in the text;
  3. a column of M reconstructed from its 3 on-site correlators;
  4. the 2^m-partner fiber update at m=2 (mask expands to 4 terms),
     against brute-force Bayes.
"""

import random

n, m = 3, 2
NQ, K = 1 << n, 1 << m
N = K ** NQ

# the three singled-out maps, by their digit tables (answers at q=0..7)
SPIKES = [
    ([q & 3 for q in range(NQ)],              0.40),  # low two input bits
    ([0] * NQ,                                0.25),  # constant 0
    ([3 * ((q >> 2) & 1) for q in range(NQ)], 0.15),  # both bits = x2
]
BG = 0.20
JR = [(sum(t[q] * K ** q for q in range(NQ)), v) for t, v in SPIKES]

p = [BG / N] * N
for j_r, v in JR:
    p[j_r] += v
assert abs(sum(p) - 1) < 1e-9


def digit(j, q):
    return (j >> (m * q)) & (K - 1)


def site(q, i):
    return 1 << (m * q + i)


def hat(belief, c):
    return sum(pj if bin(j & c).count('1') % 2 == 0 else -pj
               for j, pj in enumerate(belief))


def closed(c):
    if c == 0:
        return 1.0
    return sum(v * (-1) ** bin(j_r & c).count('1') for j_r, v in JR)


# ------------------------------------------------------------- check 1
rng = random.Random(1)
worst = max(abs(hat(p, c) - closed(c))
            for c in [rng.randrange(1, N) for _ in range(100)])
print(f"1. master formula = spike closed form (100 random c): "
      f"max err {worst:.1e}")

# ------------------------------------------------------------- check 2
cA = site(6, 1)
cB = site(6, 1) | site(6, 0)
cC = site(5, 1) | site(6, 1)
cD = site(4, 1) | site(5, 1) | site(6, 1)
vA, vB, vC, vD = (hat(p, c) for c in (cA, cB, cC, cD))
s51 = hat(p, site(5, 1))
connC = vC - s51 * vA
print(f"2. A: <s_6,1> = {vA:+.4f} (expect -0.3000)")
print(f"   B: <s_6,1 s_6,0> = {vB:+.4f} (expect 0)")
print(f"   C: <s_5,1 s_6,1> = {vC:+.4f} raw, connected "
      f"{connC:+.4f} (expect +0.1500)")
print(f"   D: <s_4,1 s_5,1 s_6,1> = {vD:+.4f} (expect -0.3000)")
assert abs(vA + 0.30) < 1e-12 and abs(vB) < 1e-12
assert abs(connC - 0.15) < 1e-12 and abs(vD + 0.30) < 1e-12

# ------------------------------------------------------------- check 3
q = 6
c1, c0, c10 = hat(p, site(q, 1)), hat(p, site(q, 0)), \
    hat(p, site(q, 1) | site(q, 0))
for a in range(K):
    a1, a0 = (a >> 1) & 1, a & 1
    pred = (1 + (-1) ** a1 * c1 + (-1) ** a0 * c0
            + (-1) ** (a1 + a0) * c10) / 4
    brute = sum(pj for j, pj in enumerate(p) if digit(j, q) == a)
    assert abs(pred - brute) < 1e-12
print(f"3. column q={q} of M from its on-site block "
      f"({c1:+.2f},{c0:+.2f},{c10:+.2f}): (0.30, 0.05, 0.45, 0.20)")

# ------------------------------------------------------------- check 4
qk, astar = 5, 1
block = [site(qk, 0), site(qk, 1)]
Ts = [0, block[0], block[1], block[0] | block[1]]
sig = {T: (-1) ** bin(T & sum(b for i, b in enumerate(block)
                              if (astar >> i) & 1)).count('1') for T in Ts}
post = [pj if digit(j, qk) == astar else 0.0 for j, pj in enumerate(p)]
Z = sum(post)
post = [x / Z for x in post]
denom = sum(sig[T] * hat(p, T) for T in Ts)
worst = max(abs(sum(sig[T] * hat(p, c ^ T) for T in Ts) / denom
                - hat(post, c))
            for c in [rng.randrange(N) for _ in range(100)])
print(f"4. 4-partner fiber update (learn q={qk} -> a*={astar}, "
      f"100 random c): max err {worst:.1e}")

# ------------------------------------------------------------- check 5
# the walkthrough quoted in the text: refresh column q'=6 of M from
# exactly 15 prior ledger entries (3 on-site q', 3 on-site q_k, 9 cross)
import math

D = denom                       # = 1 + 0.30 + 0.50 + 0 = 1.80 = 4 P(a*|q5)
assert abs(D - 1.80) < 1e-12
print(f"5. D = {D:.2f}, P(a*|q5) = {D/4:.4f}, "
      f"surprisal {-math.log2(D/4):.3f} bits")
qp = 6
patterns = [site(qp, 1), site(qp, 0), site(qp, 1) | site(qp, 0)]
primed = [sum(sig[T] * hat(p, c | T) for T in Ts) / D for c in patterns]
assert abs(primed[0] + 8 / 9) < 1e-12      # traced in the text: -0.889
H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
col_b, col_a = [], []
for a in range(K):
    a1, a0 = (a >> 1) & 1, a & 1
    signs = [(-1) ** a1, (-1) ** a0, (-1) ** (a1 + a0)]
    col_a.append((1 + sum(s * v for s, v in zip(signs, primed))) / 4)
    col_b.append((1 + sum(s * hat(p, c) for s, c in zip(signs, patterns)))
                 / 4)
    brute = sum(pj for j, pj in enumerate(post) if digit(j, qp) == a)
    assert abs(col_a[a] - brute) < 1e-12
print(f"   column q'=6: {[round(x,4) for x in col_b]} H={H(col_b):.3f} "
      f"-> {[round(x,4) for x in col_a]} H={H(col_a):.3f} bits")

print("all checks passed")
