"""Build the self-contained interactive page (correlator_ledger.html)
from ledger_template.html by injecting the complexity data.

The template computes everything else client-side: Gibbs weights for the
three-field energy E = gamma*C + lambda*F + mu*W, the fast
Walsh-Hadamard transform of all posterior weights, columns of M from the
on-site correlators, the experiment (Bayes conditioning on answers), and
the learning-curve statistics. Only the circuit-complexity columns of
the classification tables are baked in (S, I, W, F are recomputed
in-page from the truth tables).

Run from anywhere: paths are resolved relative to this script.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def hexcol(path, nrows=65536):
    out = []
    with open(path) as fh:
        r = csv.reader(fh)
        idx = next(r).index("circuit_complexity")
        for row in r:
            v = int(row[idx])
            assert 0 <= v <= 15
            out.append(format(v, "x"))
    assert len(out) == nrows
    return "".join(out)


c41 = hexcol(os.path.join(REPO, "output", "table_4to1.csv"))
c32 = hexcol(os.path.join(REPO, "output", "table_3to2.csv"))
c31 = hexcol(os.path.join(REPO, "output", "table_3to1.csv"), nrows=256)
# spot checks against the synthesis validation values
assert c31[0x96] == "6" and c31[0xe8] == "4" and c31[0x88] == "1"

# (2,1): internal j is little-endian (bit q of j = answer at input q);
# names use the weight-indexed convention (b0 = rightmost bit of q).
NAMES = ["FALSE", "NOR", "¬b1∧b0", "¬b1", "b1∧¬b0",
         "¬b0", "XOR", "NAND", "AND", "XNOR", "b0", "b1→b0",
         "b1", "b0→b1", "OR", "TRUE"]
C21 = [0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0]
assert C21[6] == 3 and C21[9] == 3 and sum(1 for c in C21 if c == 0) == 6

with open(os.path.join(HERE, "ledger_template.html"), encoding="utf-8") as fh:
    html = fh.read()
html = html.replace("__C41__", c41).replace("__C32__", c32)
html = html.replace("__C31__", c31)
html = html.replace("__C21__", "[" + ",".join(map(str, C21)) + "]")
html = html.replace("__NAMES21__",
                    "[" + ",".join(f'"{n}"' for n in NAMES) + "]")
assert "__" + "C41" + "__" not in html

out = os.path.join(HERE, "correlator_ledger.html")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"wrote {out}: {len(html)/1024:.0f} KB")
