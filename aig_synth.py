"""
Exact minimum-size AIG (And-Inverter Graph) synthesis via SAT.

Model: 2-input AND gates, unit cost. Any wire - either fanin of a gate, or
the final output - may be freely inverted (inversions are edge attributes,
not gates, and don't count toward size). This matches the standard AIG
circuit-complexity metric (e.g. used in ABC/mockturtle and the
krinkin/bounds reference data for the 222 four-variable NPN classes).

Known validation values: AND=1, OR=1, XOR=3, NAND=1 (all textbook AIG sizes).
"""

from pysat.formula import IDPool
from pysat.solvers import Cadical153


def _bit(x, row):
    return (x >> row) & 1


def try_size(n, targets, r, solver_name="cadical153"):
    num_rows = 1 << n
    base = 1 + n  # signal 0 = const0; 1..n = inputs; base.. = gates
    num_sig = base + r
    m = len(targets)

    vp = IDPool()

    def tvar(sig, row):
        return vp.id(("t", sig, row))

    def svar(g, a, b, ia, ib):
        return vp.id(("s", g, a, b, ia, ib))

    def ovar(o, sig, inv):
        return vp.id(("o", o, sig, inv))

    clauses = []

    for row in range(num_rows):
        clauses.append([-tvar(0, row)])  # const0 = 0
        for j in range(n):
            sig = 1 + j
            if _bit(row, j):
                clauses.append([tvar(sig, row)])
            else:
                clauses.append([-tvar(sig, row)])

    fanin_users = {sig: [] for sig in range(num_sig)}

    def candidates(g):
        upto = base + g
        cands = []
        for a in range(upto):
            for b in range(a, upto):
                if a == b:
                    invs = [(0, 0), (0, 1), (1, 1)]
                else:
                    invs = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for (ia, ib) in invs:
                    cands.append((a, b, ia, ib))
        return cands

    for g in range(r):
        gsig = base + g
        cands = candidates(g)
        svars = [svar(g, a, b, ia, ib) for (a, b, ia, ib) in cands]
        clauses.append(list(svars))
        for i in range(len(svars)):
            for j in range(i + 1, len(svars)):
                clauses.append([-svars[i], -svars[j]])
        for (a, b, ia, ib) in cands:
            s = svar(g, a, b, ia, ib)
            fanin_users[a].append(s)
            if b != a:
                fanin_users[b].append(s)
            for row in range(num_rows):
                y = tvar(gsig, row)
                A = tvar(a, row)
                B = tvar(b, row)
                p = A if ia == 0 else -A
                q = B if ib == 0 else -B
                clauses.append([-s, -y, p])
                clauses.append([-s, -y, q])
                clauses.append([-s, -p, -q, y])

    # Symmetry breaking: no duplicate gate truth tables (up to output polarity
    # would be too strong to forbid outright since a gate and its complement
    # can both be independently useful without extra cost via output
    # selection; so only forbid *exact* duplicates, matching-polarity).
    for g in range(r):
        gsig = base + g
        for other in range(base, gsig):
            diff_lits = []
            for row in range(num_rows):
                d = vp.id(("d", gsig, other, row))
                y1 = tvar(gsig, row)
                y2 = tvar(other, row)
                clauses.append([-d, y1, y2])
                clauses.append([-d, -y1, -y2])
                clauses.append([d, -y1, y2])
                clauses.append([d, y1, -y2])
                diff_lits.append(d)
            clauses.append(diff_lits)

    for g in range(r):
        gsig = base + g
        clause = list(fanin_users[gsig])
        for o in range(m):
            clause.append(ovar(o, gsig, 0))
            clause.append(ovar(o, gsig, 1))
        clauses.append(clause)

    for o in range(m):
        ovars = [ovar(o, sig, inv) for sig in range(num_sig) for inv in (0, 1)]
        clauses.append(list(ovars))
        for i in range(len(ovars)):
            for j in range(i + 1, len(ovars)):
                clauses.append([-ovars[i], -ovars[j]])
        for sig in range(num_sig):
            for inv in (0, 1):
                ov = ovar(o, sig, inv)
                for row in range(num_rows):
                    want = _bit(targets[o], row)
                    lit = tvar(sig, row) if inv == 0 else -tvar(sig, row)
                    if want:
                        clauses.append([-ov, lit])
                    else:
                        clauses.append([-ov, -lit])

    with Cadical153(bootstrap_with=clauses) as solver:
        if solver.solve():
            model = set(solver.get_model())

            def true_(v):
                return v in model

            gates = []
            for g in range(r):
                for (a, b, ia, ib) in candidates(g):
                    if true_(svar(g, a, b, ia, ib)):
                        gates.append((a, b, ia, ib))
                        break
            outs = []
            for o in range(m):
                for sig in range(num_sig):
                    for inv in (0, 1):
                        if true_(ovar(o, sig, inv)):
                            outs.append((sig, inv))
                            break
                    else:
                        continue
                    break
            return {"gates": gates, "outputs": outs, "base": base}
        return None


def synthesize(n, targets, max_gates=20):
    for r in range(0, max_gates + 1):
        sol = try_size(n, targets, r)
        if sol is not None:
            return r, sol
    raise RuntimeError("no solution found within max_gates")


if __name__ == "__main__":
    AND = 0b1000
    OR = 0b1110
    XOR = 0b0110
    NAND_ = 0b0111
    for name, f in [("AND", AND), ("OR", OR), ("XOR", XOR), ("NAND", NAND_)]:
        r, sol = synthesize(2, [f])
        print(name, "-> min AIG gates:", r, "gates:", sol["gates"], "outputs:", sol["outputs"])
