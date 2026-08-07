"""Search for a new prior for the (2,1) worked example in
leveraged_learning.tex, by simulated annealing on the 16-simplex.

Truth stays AND (j=8).  Greedy question order = argmax expected drop
in H(M) at each step.  The new prior must reproduce the *conclusions*
of the current example while making them more pronounced:

  A. strict greedy best at every step (margin >= MARGIN bits),
  B. step-1 deduced NEGATIVE (the stack overshoots H(M)_0)  -- objective:
     make the overshoot s_1 - dHM_1 as large as possible,
  C. steps 2..4 deduced clearly positive (>= DED_FLOOR),
  D. cumulative leverage monotone increasing, final L_4 >= L_FLOOR,
  E. C_0 >= C_FLOOR (nonempty tower),
  F. p_j >= 0.01 for all j,
  G. windfall still dominates the deduced total: total windfall >= C_0
     (order-independent: H(p)_0 + log2 p_8 >= H(M)_0 - H(p)_0), so the
     "most of the realized leverage was luck" punchline survives.

Start point: the current example prior (Table 4 of the tex).
The search enforces MARGIN with a buffer so rounding to 3 decimals
keeps the report-level margin of 0.02; a final polish runs on the
0.001 lattice directly.
"""
import math
import random
import sys

NQ = 4
PSI = 8  # AND
PMIN = 0.01
MARGIN = 0.03      # strict-greedy margin in bits (search buffer)
MARGIN_CHECK = 0.02
DED_FLOOR = 0.02   # steps 2..4 deduced at least this
L_FLOOR = 1.20     # final cumulative leverage
C_FLOOR = 0.02     # initial total correlation
WF_ENFORCE = True  # require total windfall >= C_0 ("mostly luck" survives)

MAX_MARGIN = False  # objective = worst greedy margin, overshoot a floor
OS_FLOOR = 0.30     # required overshoot in max-margin mode

# overrides: --no-wf  --margin X  --lfloor X  --max-margin  --osfloor X
argv = sys.argv[1:]
if "--no-wf" in argv:
    WF_ENFORCE = False
if "--margin" in argv:
    MARGIN = float(argv[argv.index("--margin") + 1])
    MARGIN_CHECK = MARGIN - 0.01
if "--lfloor" in argv:
    L_FLOOR = float(argv[argv.index("--lfloor") + 1])
if "--max-margin" in argv:
    MAX_MARGIN = True
if "--osfloor" in argv:
    OS_FLOOR = float(argv[argv.index("--osfloor") + 1])

NAMES = ["FALSE", "NOR", "~b1&b0", "~b1", "b1&~b0", "~b0", "XOR", "NAND",
         "AND", "XNOR", "b0", "b1->b0", "b1", "b0->b1", "OR", "TRUE"]

P0 = [0.150, 0.141, 0.060, 0.017, 0.094, 0.015, 0.012, 0.011,
      0.149, 0.114, 0.082, 0.016, 0.102, 0.013, 0.014, 0.010]

bit = lambda j, q: (j >> q) & 1
H = lambda v: -sum(x * math.log2(x) for x in v if x > 0)
h = lambda x: 0.0 if x <= 0 or x >= 1 else -x*math.log2(x)-(1-x)*math.log2(1-x)


def Mcol1(p, q):
    return sum(x for j, x in enumerate(p) if bit(j, q) == 1)


def HM(p):
    return sum(h(Mcol1(p, q)) for q in range(NQ))


def update(p, q, a):
    Z = sum(x for j, x in enumerate(p) if bit(j, q) == a)
    return [x / Z if bit(j, q) == a else 0.0 for j, x in enumerate(p)], Z


def expected_dHM(p, q):
    before = HM(p)
    out = 0.0
    for a in (0, 1):
        Z = sum(x for j, x in enumerate(p) if bit(j, q) == a)
        if Z <= 0:
            continue
        post, _ = update(p, q, a)
        out += Z * (before - HM(post))
    return out


def simulate(p0):
    """Run the greedy game with truth=AND; return all diagnostics."""
    p = list(p0)
    asked = []
    HM0 = HM(p)
    Hp0 = H(p)
    out = {"HM0": HM0, "Hp0": Hp0, "C0": HM0 - Hp0,
           "order": [], "margins": [], "s": [], "dHM": [], "ded": [],
           "wind": [], "tower": [], "L": [], "forecast": []}
    cum_s = cum_d = 0.0
    for k in range(NQ):
        cand = [(expected_dHM(p, q), q) for q in range(NQ) if q not in asked]
        cand.sort(reverse=True)
        out["forecast"].append({q: e for e, q in cand})
        q = cand[0][1]
        out["margins"].append(cand[0][0] - cand[1][0] if len(cand) > 1
                              else float("inf"))
        a = bit(PSI, q)
        hm_b, hp_b = HM(p), H(p)
        p, Z = update(p, q, a)
        s = -math.log2(Z)
        dhm = hm_b - HM(p)
        dhp = hp_b - H(p)
        cum_s += s
        cum_d += dhm
        out["order"].append(q)
        out["s"].append(s)
        out["dHM"].append(dhm)
        out["ded"].append(dhm - s)
        out["wind"].append(dhp - s)
        out["tower"].append(dhm - dhp)
        out["L"].append(cum_d / cum_s)
        asked.append(q)
    return out


def score(p):
    """Objective: overshoot at step 1, minus penalties for broken checks."""
    r = simulate(p)
    overshoot = -r["ded"][0]
    pen = 0.0
    for m in r["margins"][:3]:
        pen += max(0.0, MARGIN - m)
    for d in r["ded"][1:]:
        pen += max(0.0, DED_FLOOR - d)
    pen += max(0.0, L_FLOOR - r["L"][-1])
    pen += max(0.0, 1.01 - r["L"][2])          # L3 above unity
    pen += max(0.0, C_FLOOR - r["C0"])
    if WF_ENFORCE:
        wf_tot = r["Hp0"] + math.log2(p[PSI])   # total windfall, order-free
        pen += max(0.0, r["C0"] - wf_tot)       # windfall dominance
    if MAX_MARGIN:
        pen += max(0.0, OS_FLOOR - overshoot)
        return (min(r["margins"][:3]) + 0.3 * min(overshoot, 0.5)
                - 50.0 * pen, r)
    bonus = 0.5 * min(min(r["margins"][:3]), 0.15)
    return overshoot + bonus - 50.0 * pen, r


def propose(p, rng, scale):
    q = list(p)
    i, j = rng.sample(range(16), 2)
    d = min(q[i] - PMIN, abs(rng.gauss(0, scale)))
    q[i] -= d
    q[j] += d
    return q


def anneal(seed, iters=60000):
    rng = random.Random(seed)
    p = list(P0)
    sc, _ = score(p)
    best_p, best_sc = list(p), sc
    for t in range(iters):
        T = 0.05 * (0.0005 / 0.05) ** (t / iters)
        scale = 0.05 * (0.005 / 0.05) ** (t / iters)
        q = propose(p, rng, scale)
        sq, _ = score(q)
        if sq > sc or rng.random() < math.exp((sq - sc) / T):
            p, sc = q, sq
            if sc > best_sc:
                best_p, best_sc = list(p), sc
    return best_p, best_sc


def polish(p, rounds=40):
    """Greedy coordinate transfers with shrinking step."""
    sc, _ = score(p)
    step = 0.01
    for _ in range(rounds):
        improved = False
        for i in range(16):
            for j in range(16):
                if i == j or p[i] - step < PMIN:
                    continue
                q = list(p)
                q[i] -= step
                q[j] += step
                sq, _ = score(q)
                if sq > sc:
                    p, sc = q, sq
                    improved = True
        if not improved:
            step /= 2
            if step < 1e-4:
                break
    return p, sc


def lattice_polish(p):
    """Hill-climb on the 0.001 grid (single-milli transfers)."""
    sc, _ = score(p)
    improved = True
    while improved:
        improved = False
        for i in range(16):
            if p[i] - 0.001 < PMIN - 1e-12:
                continue
            for j in range(16):
                if i == j:
                    continue
                q = list(p)
                q[i] = round(q[i] - 0.001, 3)
                q[j] = round(q[j] + 0.001, 3)
                sq, _ = score(q)
                if sq > sc + 1e-12:
                    p, sc = q, sq
                    improved = True
    return p, sc


def round3(p):
    """Round to 3 decimals keeping sum 1 and floors, largest-remainder."""
    scaled = [x * 1000 for x in p]
    fl = [max(10, int(x)) for x in scaled]
    rem = 1000 - sum(fl)
    order = sorted(range(16), key=lambda i: scaled[i] - fl[i], reverse=True)
    k = 0
    while rem != 0:
        i = order[k % 16]
        if rem > 0:
            fl[i] += 1
            rem -= 1
        elif fl[i] > 10:
            fl[i] -= 1
            rem += 1
        k += 1
    return [x / 1000 for x in fl]


def report(p, label):
    r = simulate(p)
    print(f"\n===== {label} =====")
    print("prior:")
    for j in range(16):
        print(f"  {j:2d} {NAMES[j]:8s} {p[j]:.3f}")
    print(f"H(M)0={r['HM0']:.3f}  H(p)0={r['Hp0']:.3f}  C0={r['C0']:.3f}")
    qlab = lambda q: format(q, "02b")
    cum_s = cum_d = 0.0
    for k in range(NQ):
        fc = r["forecast"][k]
        fstr = "  ".join(f"q={qlab(q)}:{e:.3f}" for q, e in
                         sorted(fc.items(), key=lambda kv: -kv[1]))
        print(f"\nstep {k+1}: forecast <dHM>  {fstr}   "
              f"margin={r['margins'][k]:.3f}" if k < 3 else
              f"\nstep {k+1}: (forced)")
        cum_s += r["s"][k]
        cum_d += r["dHM"][k]
        print(f"  ask q={qlab(r['order'][k])}  s={r['s'][k]:.3f}  "
              f"dHM={r['dHM'][k]:+.3f}  deduced={r['ded'][k]:+.3f} "
              f"(windfall {r['wind'][k]:+.3f} + tower {r['tower'][k]:+.3f})  "
              f"L{k+1}={r['L'][k]:.3f}")
    print(f"\nrun: received={cum_s:.3f}  destroyed={cum_d:.3f}  "
          f"L4={r['L'][-1]:.3f}")
    print(f"step-1 overshoot (stack above H(M)0): {-r['ded'][0]:.3f} bits")
    # figure-4 stack
    print("stack (l, remaining, received_cum, sum):")
    pp = list(p)
    cs = 0.0
    print(f"  l=0: {HM(pp):.3f} 0.000  sum={HM(pp):.3f}")
    for q in r["order"]:
        a = bit(PSI, q)
        pp, Z = update(pp, q, a)
        cs += -math.log2(Z)
        print(f"  l={r['order'].index(q)+1}: {HM(pp):.3f} {cs:.3f}  "
              f"sum={HM(pp)+cs:.3f}")
    wf_tot = r["Hp0"] + math.log2(p[PSI])
    print(f"total windfall={wf_tot:.3f}  tower(C0)={r['C0']:.3f}  "
          f"luck share of deduced={wf_tot/(wf_tot+r['C0']):.2f}")
    checks = {
        "strict greedy (margin>=0.02)":
            all(m >= MARGIN_CHECK - 1e-9 for m in r["margins"][:3]),
        "step-1 deduced negative": r["ded"][0] < 0,
        "steps 2-4 deduced positive":
            all(d >= DED_FLOOR - 1e-9 for d in r["ded"][1:]),
        f"final L >= {L_FLOOR}": r["L"][-1] >= L_FLOOR - 1e-9,
        "L3 > 1": r["L"][2] > 1.0,
        "C0 > 0": r["C0"] >= C_FLOOR - 1e-9,
        "windfall dominates" + ("" if WF_ENFORCE else " (not enforced)"):
            wf_tot >= r["C0"] - 1e-9 or not WF_ENFORCE,
        "all p_j >= 0.01": all(x >= PMIN - 1e-12 for x in p),
    }
    for name, val in checks.items():
        print(f"  [{'PASS' if val else 'FAIL'}] {name}")
    print(f"checklist: {'ALL PASS' if all(checks.values()) else 'FAIL'}")
    return r


if __name__ == "__main__":
    report(P0, "current example prior (baseline)")
    best_p, best_sc = None, -1e18
    for seed in range(6):
        p, sc = anneal(seed, iters=40000)
        p, sc = polish(p)
        print(f"seed {seed}: score {sc:.4f}  overshoot "
              f"{-simulate(p)['ded'][0]:.3f}", flush=True)
        if sc > best_sc:
            best_p, best_sc = p, sc
    pr, _ = lattice_polish(round3(best_p))
    report(best_p, "best found (raw)")
    report(pr, "best found (3-decimal lattice, polished)")
