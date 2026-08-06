"""Check the two 'most intelligent prior' ansatz cases at n=2, m=2
(N=256) against brute force: M before, M after one question in both
branches, surprisal, and entropy change of M."""
import math

n, m = 2, 2
NQ, K = 1 << n, 1 << m
N = K ** NQ
dig = lambda j, q: (j >> (m * q)) & (K - 1)
H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
h = lambda x: 0 if x <= 0 or x >= 1 else -x*math.log2(x)-(1-x)*math.log2(1-x)

def Mcols(p):
    return [[sum(p[j] for j in range(N) if dig(j, q) == a) for a in range(K)]
            for q in range(NQ)]

def HM(p):
    return sum(H(col) for col in Mcols(p))

def update(p, q, a):
    Z = sum(p[j] for j in range(N) if dig(j, q) == a)
    return [p[j] / Z if dig(j, q) == a else 0 for j in range(N)], Z

jstar = 27  # digits (0,1,2,3) at q=3,2,1,0 -> psi = NOT q map
pw = 0.3

# ---------- Case B: spike + uniform background ----------
beta = (1 - pw) / (N - 1)
p = [beta] * N
p[jstar] = pw

cols = Mcols(p)
Mstar_pred = pw + (N // K - 1) * beta
Moth_pred = (N // K) * beta
for q in range(NQ):
    a_star = dig(jstar, q)
    assert abs(cols[q][a_star] - Mstar_pred) < 1e-12
    for a in range(K):
        if a != a_star:
            assert abs(cols[q][a] - Moth_pred) < 1e-12
Hcol = H([Mstar_pred] + [Moth_pred] * (K - 1))
assert abs(HM(p) - NQ * Hcol) < 1e-12
print(f"B before: M* = {Mstar_pred:.6f}, M_oth = {Moth_pred:.6f}, "
      f"H_col = {Hcol:.6f}, H(M) = {HM(p):.6f}")

q0 = 1
# branch: fell on the special map
a_conf = dig(jstar, q0)
p_conf, Z = update(p, q0, a_conf)
s_conf = -math.log2(Z)
# predicted: still spike+uniform on N' = N/K maps, p' = pw/Z, beta' = beta/Z
pprime = pw / Z
assert abs(p_conf[jstar] - pprime) < 1e-12
bg = [p_conf[j] for j in range(N) if p_conf[j] > 0 and j != jstar]
assert all(abs(x - beta / Z) < 1e-15 for x in bg)
Nprime = N // K
Mstar2 = pprime + (Nprime // K - 1) * beta / Z
Moth2 = (Nprime // K) * beta / Z
Hcol2 = H([Mstar2] + [Moth2] * (K - 1))
assert abs(HM(p_conf) - (NQ - 1) * Hcol2) < 1e-12
print(f"B confirm: prob {Z:.4f}, surprisal {s_conf:.4f}, p' = {pprime:.4f}, "
      f"H(M') = {HM(p_conf):.6f} = 3 * {Hcol2:.6f}, "
      f"reduction = {HM(p) - HM(p_conf):.6f}")

# branch: fell elsewhere
a_dis = (a_conf + 1) % K
p_dis, Z2 = update(p, q0, a_dis)
s_dis = -math.log2(Z2)
assert abs(Z2 - Moth_pred) < 1e-12
live = [x for x in p_dis if x > 0]
assert len(live) == N // K and all(abs(x - 1/(N//K)) < 1e-15 for x in live)
assert abs(HM(p_dis) - (NQ - 1) * m) < 1e-12
print(f"B refute:  prob {Z2:.4f}, surprisal {s_dis:.4f}, "
      f"H(M') = {HM(p_dis):.6f} (= (2^n-1)m), "
      f"reduction = {HM(p) - HM(p_dis):.6f}  <-- negative!")

# expected reduction must be >= 0 (mutual information)
exp_red = Z * (HM(p) - HM(p_conf)) + (K - 1) * Z2 * (HM(p) - HM(p_dis))
print(f"B expected reduction = {exp_red:.6f} (must be >= 0)")

# ---------- Case A: two maps ----------
# rival differs from jstar at d = 2 questions (q = 0 and q = 2)
jriv = jstar ^ (1 << (m*0)) ^ (2 << (m*2))   # change digit0: 3->2, digit2: 1->3
d = sum(1 for q in range(NQ) if dig(jstar, q) != dig(jriv, q))
assert d == 2
pA = [0.0] * N
pA[jstar], pA[jriv] = pw, 1 - pw
assert abs(HM(pA) - d * h(pw)) < 1e-12
print(f"\nA before: d = {d}, H(M) = {HM(pA):.6f} = d*h(p) = {d*h(pw):.6f}")
# ask a disagreement question
qd = 0
pA_conf, Za = update(pA, qd, dig(jstar, qd))
pA_dis, Zb = update(pA, qd, dig(jriv, qd))
assert abs(Za - pw) < 1e-12 and abs(Zb - (1-pw)) < 1e-12
assert HM(pA_conf) < 1e-12 and HM(pA_dis) < 1e-12
print(f"A ask q in D: confirm prob {Za:.4f} surprisal {-math.log2(Za):.4f}, "
      f"refute prob {Zb:.4f} surprisal {-math.log2(Zb):.4f}; "
      f"both branches H(M') = 0, reduction = {d*h(pw):.4f}")
# ask an agreement question
qa = 1
pA_same, Zc = update(pA, qa, dig(jstar, qa))
assert abs(Zc - 1) < 1e-12 and abs(HM(pA_same) - HM(pA)) < 1e-12
print(f"A ask q not in D: prob 1, surprisal 0, M unchanged")
