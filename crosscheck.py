import csv
import random

from groups import canonicalize_4to1
from aig_synth import synthesize

canon = canonicalize_4to1()

rows = []
with open("reference_npn4_opt_aig.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append((int(row["npn_rep_dec"]), int(row["opt_aig"]), row["status"]))

print(f"loaded {len(rows)} reference rows")

# sample a mix: smallest, largest, and some random ones for cross-check
random.seed(0)
sample = random.sample(rows, 10)

mismatches = 0
for dec, ref_val, status in sample:
    my_canon = int(canon[dec])
    r, sol = synthesize(4, [dec], max_gates=20)
    ok = "OK" if r == ref_val else "MISMATCH"
    if r != ref_val:
        mismatches += 1
    print(f"f={dec:5d} ref_opt_aig={ref_val:2d} ({status:5s}) mine={r:2d}  {ok}")

print(f"\n{mismatches} mismatches out of {len(sample)} checked")
