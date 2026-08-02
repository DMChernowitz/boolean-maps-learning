"""
Numerical verification of the identities in learning_agent.md, section
"The update rule as a convolution: the correlator basis".

For the running n=2, m=1 example prior:
  1. the correlator ledger: all 16 Walsh coefficients
     hat p(S) = E[ prod_{q in S} (-1)^{A_q} ], printed shell by shell,
     with connected (covariance) values for shell 2;
  2. the convolution form of the update rule,
       hat p'(S) = [hat p(S) + sigma hat p(S xor {q_k})]
                   / [1 + sigma hat p({q_k})],   sigma = (-1)^a,
     checked against brute-force Bayes for every (q_k, a, S);
  3. the generalization formula for an unasked column,
       shift of <s_q'> = sigma * Cov(s_q', s_q_k) / (2 P(a|q_k)),
     reproducing the cross-table "anomaly" (learning q_k=10, a=0 raises
     H_B(01) from 0.764 to 0.795) from the sign of the connected
     correlator, and the other branch (a=1, H_B(01) -> 0.666).
"""

import math

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
LAB = ['00', '01', '10', '11']
NQ = 4


def corr(belief, S):
    """hat p(S) for S a bitmask over the four questions."""
    return sum(w * (-1) ** sum(int(FUNCS[f][q]) for q in range(NQ)
                               if (S >> q) & 1)
               for f, w in belief.items())


def posterior(belief, qk, a):
    raw = {f: w for f, w in belief.items() if int(FUNCS[f][qk]) == a}
    Z = sum(raw.values())
    return {f: w / Z for f, w in raw.items()}


def h(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def show_ledger():
    print("correlator ledger (shells 1 and 2; connected = covariance):")
    for q in range(NQ):
        print(f"  <s_{LAB[q]}> = {corr(PRIOR, 1 << q):+.4f}")
    for i in range(NQ):
        for j in range(i + 1, NQ):
            raw = corr(PRIOR, (1 << i) | (1 << j))
            conn = raw - corr(PRIOR, 1 << i) * corr(PRIOR, 1 << j)
            print(f"  <s_{LAB[i]} s_{LAB[j]}> = {raw:+.4f}   "
                  f"connected {conn:+.4f}")


def check_convolution():
    hp = {S: corr(PRIOR, S) for S in range(1 << NQ)}
    worst = 0.0
    for qk in range(NQ):
        for a in (0, 1):
            post = posterior(PRIOR, qk, a)
            sgn = (-1) ** a
            denom = 1 + sgn * hp[1 << qk]
            for S in range(1 << NQ):
                pred = (hp[S] + sgn * hp[S ^ (1 << qk)]) / denom
                worst = max(worst, abs(pred - corr(post, S)))
    assert worst < 1e-12
    print(f"convolution form: all (q_k, a, S), max err {worst:.1e}")


def check_anomaly(qk=2, qp=1):
    """Learning q_k=10; effect on unasked column q'=01, both branches."""
    m_k, m_p = corr(PRIOR, 1 << qk), corr(PRIOR, 1 << qp)
    pair = corr(PRIOR, (1 << qk) | (1 << qp))
    conn = pair - m_k * m_p
    print(f"pair ({LAB[qp]},{LAB[qk]}): raw {pair:+.4f}, "
          f"connected {conn:+.4f}")
    for a in (0, 1):
        sgn = (-1) ** a
        Pa = (1 + sgn * m_k) / 2
        new = (m_p + sgn * pair) / (1 + sgn * m_k)
        shift = sgn * conn / (2 * Pa)
        assert abs(new - corr(posterior(PRIOR, qk, a), 1 << qp)) < 1e-12
        assert abs(new - (m_p + shift)) < 1e-12
        print(f"  a={a}: <s_{LAB[qp]}> {m_p:+.4f} -> {new:+.4f} "
              f"(shift = sigma*conn/(2 P(a|q_k)) = {shift:+.4f}), "
              f"H_B({LAB[qp]}) {h((1 + m_p) / 2):.3f} -> "
              f"{h((1 + new) / 2):.3f}")


if __name__ == '__main__':
    show_ledger()
    check_convolution()
    check_anomaly()
    print("all checks passed")
