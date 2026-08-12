"""Generate pgfplots data tables for the thermodynamic-limit chapter.

Every formula here is implemented directly from thermodynamic_limit.md,
with that document's quoted landmark values used as assertions.  Output
is written to figures/data/*.dat as whitespace-separated tables with a
header row of column names, consumed by \\addplot table[x=..., y=...].

Analytic limit curves (hyperbolas, clique constants, the culling
coefficient) are plotted from closed forms inside the tex and are not
tabulated here.
"""
import math
import os

LOG2 = math.log(2.0)
OUT = os.path.join("figures", "data")


def h2(z):
    """Binary entropy in bits, with 0 log 0 = 0."""
    if z <= 0.0 or z >= 1.0:
        return 0.0
    return -z * math.log2(z) - (1.0 - z) * math.log2(1.0 - z)


def entropy(masses):
    """Shannon entropy in bits of an explicit list of probabilities."""
    return -sum(x * math.log2(x) for x in masses if x > 0.0)


def binom(a, b):
    if b < 0 or b > a:
        return 0.0
    return math.exp(math.lgamma(a + 1) - math.lgamma(b + 1)
                    - math.lgamma(a - b + 1))


def write_table(name, columns, rows):
    """Write one whitespace-separated table with a named header row."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".dat")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(" ".join(columns) + "\n")
        for row in rows:
            fh.write(" ".join(f"{value:.6g}" for value in row) + "\n")
    return path


def thin(values, cap):
    """Keep at most `cap` entries, endpoints included."""
    if len(values) <= cap:
        return values
    step = math.ceil(len(values) / cap)
    kept = values[::step]
    if kept[-1] != values[-1]:
        kept.append(values[-1])
    return kept


# ----------------------------------------------------------------------
# 1. The spike prior:  (1.2), (1.10), (1.17)-(1.25), (1.28)-(1.29),
#    (1.35), (1.42), (1.45)
# ----------------------------------------------------------------------
class Spike:
    """Spike-plus-uniform prior at (n, m) with special-map mass p."""

    def __init__(self, n, m, p):
        self.Q = 2 ** n
        self.A = 2 ** m
        self.m = m
        self.u = 1.0 / self.A
        self.N = float(self.A) ** self.Q
        self.p = p
        self.omega = (1.0 - p) / (self.N - 1.0)

    def P(self, k):
        """(1.2): probability of the all-zero answer block of length k."""
        return self.p + (self.N * self.u ** k - 1.0) * self.omega

    def U(self, k):
        """(1.8): the common probability of any other length-k block."""
        return self.N * self.u ** k * self.omega

    def G(self, k):
        """(1.10): entropy of the k-answer block distribution."""
        if k == 0:
            return 0.0
        return (-self.P(k) * math.log2(self.P(k))
                - (self.u ** -k - 1.0) * self.U(k) * math.log2(self.U(k)))

    def one_question(self):
        """(1.18)-(1.25): the branch anatomy of a single question."""
        P, U = self.P(1), self.U(1)
        P2, U2 = self.P(2), self.U(2)
        G1, G2 = self.G(1), self.G(2)
        h_c = (-(P2 / P) * math.log2(P2 / P)
               - (self.A - 1) * (U2 / P) * math.log2(U2 / P))
        h_before = self.Q * G1
        h_correct = (self.Q - 1) * h_c
        h_wrong = (self.Q - 1) * self.m
        return dict(
            h_before=h_before,
            s_correct=-math.log2(P), s_wrong=-math.log2(U), s_expected=G1,
            h_correct=h_correct, h_wrong=h_wrong,
            h_expected=(self.Q - 1) * (G2 - G1),
            l_correct=(h_before - h_correct) / (-math.log2(P)),
            l_wrong=(h_before - h_wrong) / (-math.log2(U)),
            l_expected=(h_before - (self.Q - 1) * (G2 - G1)) / G1,
        )

    def extraction_ratio(self):
        """(1.28): the share of the dependency store freed by one question."""
        G1, G2, GQ = self.G(1), self.G(2), self.G(self.Q)
        store = self.Q * G1 - GQ
        if store <= 0.0:
            return float("nan")
        return (self.Q - 1) * (2 * G1 - G2) / store

    def leverage(self, k):
        """(1.15)-(1.16): exact expected cumulative leverage."""
        if k == self.Q:
            return self.Q * self.G(1) / self.G(self.Q)
        return ((self.Q * self.G(1)
                 - (self.Q - k) * (self.G(k + 1) - self.G(k)))
                / self.G(k))


def extraction_sharp_limit(n, m):
    """(1.29): the exact finite-N limit of R_1 as p -> 1^-."""
    Q, A = 2 ** n, 2 ** m
    N = float(A) ** Q
    culled = 1.0 - 1.0 / A
    return ((Q - 1) * N * culled ** 2) / (Q * N * culled - (N - 1))


def leverage_sharp_limit(n, m, k):
    """(1.35): the p -> 1^- limit of the cumulative leverage."""
    Q, u = 2 ** n, 2.0 ** (-m)
    return (1.0 - u) * (Q - (Q - k) * u ** k) / (1.0 - u ** k)


def spike_column_entropy(p, m):
    """(1.42): the one-column entropy in the large-map-space limit."""
    A = 2 ** m
    return (h2(p + (1.0 - p) / A)
            + (1.0 - p) * (1.0 - 1.0 / A) * math.log2(A - 1)
            if A > 2 else h2(p + (1.0 - p) / A))


def spike_hyperbola(p, m, x):
    """(1.45): the fixed-p thermodynamic leverage curve."""
    return 1.0 + (spike_column_entropy(p, m) - (1.0 - p) * m) / (
        x * (1.0 - p) * m)


# ---- verification against the document's quoted values ----
assert math.isclose(extraction_sharp_limit(2, 1), 12.0 / 17.0, abs_tol=1e-12)
assert math.isclose(leverage_sharp_limit(4, 1, 1), 2 ** 4 - (2 ** 4 - 1) * 0.5,
                    abs_tol=1e-12)
for n_, m_, k_ in ((4, 1, 1), (4, 1, 5), (6, 1, 3), (3, 2, 4)):
    u_ = 2.0 ** (-m_)
    plateau = 2 ** n_ * (1 - u_) + (1 - u_) * k_ * u_ ** k_ / (1 - u_ ** k_)
    assert math.isclose(leverage_sharp_limit(n_, m_, k_), plateau,
                        abs_tol=1e-9)          # (1.36)
_probe = Spike(2, 1, 0.6).one_question()
assert math.isclose(
    _probe["h_expected"],
    _probe["s_correct"] and (
        2.0 ** -_probe["s_correct"] * _probe["h_correct"]
        + (1 - 2.0 ** -_probe["s_correct"]) * _probe["h_wrong"]),
    abs_tol=1e-12)                              # (1.23) branch average


def emit_spike_figures():
    grid = [1e-4 + (1 - 2e-4) * i / 399 for i in range(400)]
    for n, m, tag in ((4, 1, "4to1"), (3, 2, "3to2")):
        anatomy, leverage = [], []
        for p in grid:
            c = Spike(n, m, p).one_question()
            anatomy.append((p, c["h_before"], c["s_correct"], c["s_wrong"],
                            c["s_expected"], c["h_correct"], c["h_wrong"],
                            c["h_expected"]))
            leverage.append((p, c["l_correct"], c["l_wrong"], c["l_expected"]))
        write_table(f"spike_one_{tag}",
                    ["p", "hbefore", "scorrect", "swrong", "sexpected",
                     "hcorrect", "hwrong", "hexpected"], anatomy)
        write_table(f"spike_lev_{tag}",
                    ["p", "lcorrect", "lwrong", "lexpected"], leverage)

    # extraction ratio: defined on 1/N < p < 1
    for n in (2, 3, 4, 5, 6):
        start = 1.0 / 2.0 ** (2 ** n)
        rows = []
        for i in range(1, 401):
            p = start + (1 - 1e-6 - start) * i / 400
            rows.append((p, Spike(n, 1, p).extraction_ratio()))
        write_table(f"extraction_n{n}", ["p", "ratio"], rows)
    for m in (1, 2, 3, 4):
        start = 1.0 / 2.0 ** (m * 8)
        rows = []
        for i in range(1, 401):
            p = start + (1 - 1e-6 - start) * i / 400
            rows.append((p, Spike(3, m, p).extraction_ratio()))
        write_table(f"extraction_m{m}", ["p", "ratio"], rows)

    # sharp-prior plateau against the asked fraction
    for n in (3, 4, 5, 6, 8, 10):
        Q = 2 ** n
        rows = thin([(k / Q, leverage_sharp_limit(n, 1, k) / Q)
                     for k in range(1, Q + 1)], 160)
        write_table(f"spikeflow_sharp_n{n}", ["x", "lev"], rows)

    # approach at fixed n = 6, m = 1
    Q = 2 ** 6
    rows = []
    for k in range(1, Q + 1):
        row = [k / Q]
        for eps in (1e-2, 1e-4, 1e-8, 1e-12):
            row.append(Spike(6, 1, 1 - eps).leverage(k) / Q)
        row.append(leverage_sharp_limit(6, 1, k) / Q)
        rows.append(tuple(row))
    write_table("spikeflow_eps", ["x", "e2", "e4", "e8", "e12", "sharp"], rows)


# ----------------------------------------------------------------------
# 3. The coin mixture:  (3.10), (3.20), (3.21), (3.22)
# ----------------------------------------------------------------------
def block_bit_entropy(s, coins):
    """(3.20): the exact entropy F_s of s exchangeable answer bits."""
    total = 0.0
    for ones in range(s + 1):
        logs = [math.log(w) + ones * math.log(t) + (s - ones) * math.log1p(-t)
                for t, w in coins]
        peak = max(logs)
        log_mass = peak + math.log(sum(math.exp(v - peak) for v in logs))
        total -= math.exp(math.lgamma(s + 1) - math.lgamma(ones + 1)
                          - math.lgamma(s - ones + 1)
                          + log_mass) * log_mass / LOG2
    return total


def coin_parameters(m, coins):
    """(3.21)-(3.22): initial entropy, entropy density, and c_m."""
    initial = block_bit_entropy(m, coins)
    density = m * sum(w * h2(t) for t, w in coins)
    return initial, density, (initial - density) / density


def coin_leverage_curve(n, m, coins):
    """(3.21a)-(3.21b): exact finite-n cumulative leverage."""
    Q = 2 ** n
    blocks = [block_bit_entropy(m * ell, coins) for ell in range(Q + 2)]
    initial_matrix = Q * blocks[1]
    out = []
    for ell in range(1, Q + 1):
        remaining = ((Q - ell) * (blocks[ell + 1] - blocks[ell])
                     if ell < Q else 0.0)
        out.append((ell / Q, (initial_matrix - remaining) / blocks[ell]))
    return out


TWO_COIN = ((0.1, 0.5), (0.9, 0.5))                       # (3.15)


def matched_ten_coin():
    """Ten equal-weight biases matching F_1 and h_cond,1 of TWO_COIN."""
    fixed = (0.005, 0.04, 0.1, 0.2)
    required = 5.0 * h2(0.1) - sum(h2(t) for t in fixed)
    low, high = 0.2, 0.499999
    for _ in range(200):
        mid = 0.5 * (low + high)
        if h2(mid) < required:
            low = mid
        else:
            high = mid
    fitted = 0.5 * (low + high)
    biases = fixed + (fitted,)
    biases = biases + tuple(1.0 - t for t in reversed(biases))
    return tuple((t, 0.1) for t in biases), fitted


TEN_COIN, TEN_FITTED = matched_ten_coin()

# ---- verification against Section 3.3 and 3.4 ----
_G = [block_bit_entropy(s, TWO_COIN) for s in range(5)]
assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(
    _G, (0.0, 1.0, 1.680077, 2.269405, 2.797921)))          # (3.16)
assert all(math.isclose(a, b, abs_tol=5e-6) for a, b in zip(
    [row[1] for row in coin_leverage_curve(2, 1, TWO_COIN)],
    (1.95977, 1.67930, 1.52969, 1.42963)))
assert math.isclose(coin_parameters(1, TWO_COIN)[2], 1.132216, abs_tol=5e-7)
assert math.isclose(coin_parameters(2, TWO_COIN)[2], 0.791144, abs_tol=5e-7)
assert math.isclose(TEN_FITTED, 0.288171, abs_tol=5e-7)
assert math.isclose(coin_parameters(1, TEN_COIN)[2],
                    coin_parameters(1, TWO_COIN)[2], abs_tol=1e-12)


def emit_coin_figures():
    for label, coins, m in (("cointwo", TWO_COIN, 1),
                            ("cointen", TEN_COIN, 1),
                            ("coinm1", TWO_COIN, 1),
                            ("coinm2", TWO_COIN, 2)):
        for n in (4, 6, 8, 10):
            rows = [(x, lev) for x, lev in coin_leverage_curve(n, m, coins)
                    if x >= 0.15]
            write_table(f"{label}_n{n}", ["x", "lev"], thin(rows, 160))


# ----------------------------------------------------------------------
# 4-5. Block priors:  (4.2)-(4.4), (5.2)-(5.24), Appendices A-F
# ----------------------------------------------------------------------
def bernstein(r, i, x):
    """(4.3): the Bernstein basis polynomial beta_{r-1,i}."""
    return binom(r - 1, i) * x ** i * (1 - x) ** (r - 1 - i)


def bernstein_integral(r, i, x):
    """(A.9) in tail form: int_0^x beta_{r-1,i} = Pr[Bin(r,x) >= i+1]/r."""
    return sum(binom(r, j) * x ** j * (1 - x) ** (r - j)
               for j in range(i + 1, r + 1)) / r


def profile(coefficients, x):
    r = len(coefficients)
    return sum(bernstein(r, i, x) * c for i, c in enumerate(coefficients))


def profile_integral(coefficients, x):
    r = len(coefficients)
    return sum(bernstein_integral(r, i, x) * c
               for i, c in enumerate(coefficients))


def block_leverage(coefficients, initial, x):
    """(2.15) for one block type with initial marginal entropy `initial`."""
    return ((initial - (1 - x) * profile(coefficients, x))
            / profile_integral(coefficients, x))


def tilted_parity_coefficients(r, theta, sigma):
    """(C.15)-(C.17): the increments of a tilted parity block."""
    eps = 1 - 2 * theta
    out = []
    for i in range(r):
        v = r - i
        total = 0.0
        for z in (0, 1):
            weight = ((1 + (-1) ** z * eps ** i)
                      * (1 + (-1) ** ((sigma ^ z)) * eps ** v)
                      / (2 * (1 + (-1) ** sigma * eps ** r)))
            rem = sigma ^ z
            nxt = (theta * (1 - (-1) ** rem * eps ** (v - 1))
                   / (1 + (-1) ** rem * eps ** v))
            total += weight * h2(nxt)
        out.append(total)
    return out


def lapsed_mds_entropy(j, r, k, m, delta):
    """(D.18)-(D.20): subset entropy of a lapsed MDS block."""
    if j <= k:
        return j * m
    share = 2.0 ** (m * (k - j))
    kept = (1 - delta) + delta * share
    return (kept * (k * m - math.log2(kept))
            + delta * (1 - share) * (j * m - math.log2(delta)))


def mds_tail(r, k, x):
    """(5.9): tau_{r,k}(x) = Pr[Bin(r-1,x) <= k-1]."""
    return sum(bernstein(r, i, x) for i in range(k))


# ---- the cocktail of Section 5.6 / Appendix F, per output bit ----
COCKTAIL_M = 4
COCKTAIL_COINS = ((0.2, 0.25), (0.7, 0.75))
COCKTAIL_R, COCKTAIL_K, COCKTAIL_LAPSE = 16, 4, 0.1
FRESH_BIAS = 0.3

COIN_INITIAL = block_bit_entropy(COCKTAIL_M, COCKTAIL_COINS) / COCKTAIL_M
COIN_DENSITY = sum(w * h2(t) for t, w in COCKTAIL_COINS)
LAPSE_ETA = [(lapsed_mds_entropy(i + 1, COCKTAIL_R, COCKTAIL_K,
                                 COCKTAIL_M, COCKTAIL_LAPSE)
              - lapsed_mds_entropy(i, COCKTAIL_R, COCKTAIL_K,
                                   COCKTAIL_M, COCKTAIL_LAPSE)) / COCKTAIL_M
             for i in range(COCKTAIL_R)]
COCKTAIL_INITIAL = 0.25 * COIN_INITIAL + 0.5 + 0.25 * h2(FRESH_BIAS)


def cocktail_profile(x):
    """(F.9), per output bit."""
    return (0.25 * COIN_DENSITY + 0.5 * profile(LAPSE_ETA, x)
            + 0.25 * h2(FRESH_BIAS))


def cocktail_received(x):
    """(F.10), per output bit."""
    return (0.25 * COIN_DENSITY * x + 0.5 * profile_integral(LAPSE_ETA, x)
            + 0.25 * h2(FRESH_BIAS) * x)


def cocktail_leverage(x):
    """(F.11)."""
    return (COCKTAIL_INITIAL - (1 - x) * cocktail_profile(x)) \
        / cocktail_received(x)


# ---- verification against Sections 5.3, 5.6 and Appendix F ----
assert all(math.isclose(a, b, abs_tol=5e-6) for a, b in zip(
    tilted_parity_coefficients(4, 0.3, 1),
    (0.912441, 0.898876, 0.811317, 0.0)))                    # (5.8)
_tilt = tilted_parity_coefficients(4, 0.3, 1)
assert math.isclose(block_leverage(_tilt, _tilt[0], 1e-9), 1.0446, abs_tol=5e-4)
assert math.isclose(block_leverage(_tilt, _tilt[0], 1.0), 1.3915, abs_tol=5e-4)
assert math.isclose(COIN_INITIAL, 0.948287460, abs_tol=5e-9)     # (F.4)
assert math.isclose(COIN_DENSITY, 0.841450198, abs_tol=5e-9)     # (F.5)
assert math.isclose(profile_integral(LAPSE_ETA, 1.0), 0.332328056,
                    abs_tol=5e-9)                                # (F.7)
assert math.isclose(COCKTAIL_INITIAL, 0.957394590, abs_tol=5e-9)  # (F.8)
assert math.isclose(cocktail_profile(0.0), 0.930685274, abs_tol=5e-9)
assert math.isclose(cocktail_leverage(1.0), 1.60408, abs_tol=5e-5)  # (F.13)
assert math.isclose(
    (COCKTAIL_INITIAL - cocktail_profile(0.0)) / cocktail_profile(0.0),
    0.02870, abs_tol=5e-5)                                       # (F.12)
_fine = [i / 4000 for i in range(1, 4001)]
# the curve diverges as x -> 0, so the dip is the first interior minimum
# and the peak is the maximum of what follows it
_dip_x = min(_fine[:1200], key=cocktail_leverage)
_peak_x = max([x for x in _fine if x > _dip_x], key=cocktail_leverage)
assert math.isclose(_dip_x, 0.0836, abs_tol=2e-3)
assert math.isclose(cocktail_leverage(_dip_x), 1.499, abs_tol=2e-3)
assert math.isclose(_peak_x, 0.350, abs_tol=2e-3)
assert math.isclose(cocktail_leverage(_peak_x), 2.114, abs_tol=2e-3)


# ---- exact finite-n block machinery, (A.3) ----
def block_G(Q, blocks, ell):
    """(A.3): exact mean block entropy for a partition into blocks."""
    total = 0.0
    for count, size, subset_entropy in blocks:
        inner = 0.0
        for j in range(max(0, ell - (Q - size)), min(size, ell) + 1):
            inner += (binom(size, j) * binom(Q - size, ell - j)
                      / binom(Q, ell)) * subset_entropy(j)
        total += count * inner
    return total


def block_curve(Q, blocks):
    entropies = [block_G(Q, blocks, ell) for ell in range(Q + 2)]
    initial_matrix = Q * entropies[1]
    out = []
    for ell in range(1, Q + 1):
        remaining = ((Q - ell) * (entropies[ell + 1] - entropies[ell])
                     if ell < Q else 0.0)
        out.append((ell / Q, (initial_matrix - remaining) / entropies[ell]))
    return out


def emit_zoo_figures():
    m = COCKTAIL_M
    # cliques, exactly flat at every finite size
    for r in (2, 4):
        rows = block_curve(2 ** 8, [(2 ** 8 // r, r,
                                     lambda j, _m=m: _m * (j >= 1))])
        assert max(abs(lev - r) for _, lev in rows) < 1e-9      # (5.1)
        write_table(f"zoo_clique_r{r}", ["x", "lev"], thin(rows, 160))
    # parity blocks of size four
    for n in (6, 8, 10):
        rows = block_curve(2 ** n, [(2 ** n // 4, 4,
                                     lambda j, _m=m: _m * min(j, 3))])
        write_table(f"zoo_parity_n{n}", ["x", "lev"], thin(rows, 160))
    # MDS (16,4) blocks diluted with fresh questions, w = 3/4
    for n in (6, 8, 10):
        Q = 2 ** n
        rows = block_curve(Q, [
            (3 * Q // (4 * COCKTAIL_R), COCKTAIL_R,
             lambda j, _m=m: _m * min(j, COCKTAIL_K)),
            (Q // 4, 1, lambda j, _m=m: _m * j)])
        write_table(f"zoo_mds_n{n}", ["x", "lev"], thin(rows, 160))
    # the cocktail
    coin_table = [block_bit_entropy(s, COCKTAIL_COINS)
                  for s in range(m * 2 ** 10 // 4 + 1)]
    for n in (6, 8, 10):
        Q = 2 ** n
        rows = block_curve(Q, [
            (1, Q // 4, lambda j: coin_table[m * j]),
            (Q // (2 * COCKTAIL_R), COCKTAIL_R,
             lambda j: lapsed_mds_entropy(j, COCKTAIL_R, COCKTAIL_K, m,
                                          COCKTAIL_LAPSE)),
            (Q // 4, 1, lambda j, _m=m: _m * h2(FRESH_BIAS) * j)])
        write_table(f"zoo_cocktail_n{n}", ["x", "lev"], thin(rows, 160))

    # limiting curves that are not elementary enough to inline in the tex
    grid = [i / 400 for i in range(1, 401)]
    write_table("zoo_mds_limit", ["x", "lev"], [
        (x, (0.75 * (1 - (1 - x) * mds_tail(COCKTAIL_R, COCKTAIL_K, x))
             + 0.25 * x)
         / (0.75 * sum(bernstein_integral(COCKTAIL_R, i, x)
                       for i in range(COCKTAIL_K)) + 0.25 * x))
        for x in grid])
    write_table("zoo_cocktail_limit", ["x", "lev"],
                [(x, cocktail_leverage(x)) for x in grid])
    write_table("zoo_tilted_limit", ["x", "lev"],
                [(x, block_leverage(_tilt, _tilt[0], x)) for x in grid])
    # the Bernstein basis of degree three, for the profile-language figure
    write_table("bernstein3", ["x", "b0", "b1", "b2", "b3"],
                [(x, bernstein(4, 0, x), bernstein(4, 1, x),
                  bernstein(4, 2, x), bernstein(4, 3, x))
                 for x in [i / 200 for i in range(201)]])


def main():
    emit_spike_figures()
    emit_coin_figures()
    emit_zoo_figures()
    print(f"wrote {len(os.listdir(OUT))} tables to {OUT}")
    print(f"extraction sharp limits, m=1: "
          + ", ".join(f"n={n}: {extraction_sharp_limit(n, 1):.4f}"
                      for n in (2, 3, 4, 5, 6)))
    print(f"extraction sharp limits, n=3: "
          + ", ".join(f"m={m}: {extraction_sharp_limit(3, m):.4f}"
                      for m in (1, 2, 3, 4)))
    print(f"coin coefficients: c_1={coin_parameters(1, TWO_COIN)[2]:.6f}, "
          f"c_2={coin_parameters(2, TWO_COIN)[2]:.6f}, "
          f"ten-coin fitted bias={TEN_FITTED:.6f}")
    print(f"cocktail: dip {cocktail_leverage(_dip_x):.4f} at x={_dip_x:.4f}, "
          f"peak {cocktail_leverage(_peak_x):.4f} at x={_peak_x:.4f}, "
          f"L(1)={cocktail_leverage(1.0):.5f}")
    print("tilted parity eta at r=4, theta=0.3, sigma=1: "
          + ", ".join(f"{v:.6f}" for v in _tilt))


if __name__ == "__main__":
    main()
