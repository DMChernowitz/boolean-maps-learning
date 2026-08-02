"""Build the 256-row table for 3-bit -> 1-bit maps.

Columns: input_000 .. input_111 (8 columns, each a 1-bit output string),
then circuit_complexity (minimum AIG size: 2-input AND gates, unit cost,
free wire inversion) — same metric and SAT synthesizer as the 3->2 and
4->1 tables. Only 14 NPN classes exist for n=3, so exact synthesis of the
class representatives is run inline rather than via a separate runner.

Validation values printed at the end: AND3 = 2, OR3 = 2, MAJ3 = 4,
MUX(x2 ? x1 : x0) = 3, and 2-input functions embed with their known
sizes (AND = 1, XOR = 3).
"""
import csv

import numpy as np

from groups import canonicalize_3to1
from aig_synth import synthesize


def main():
    canon = canonicalize_3to1()
    reps = sorted(int(x) for x in np.unique(canon))
    print(f"{len(reps)} representative classes to synthesize", flush=True)

    class_complexity = {}
    for rep in reps:
        r, _ = synthesize(3, [rep], max_gates=12)
        class_complexity[rep] = r
        print(f"  class rep 0x{rep:02x}: min AIG size {r}", flush=True)

    header = ([f"in_{format(k, '03b')}" for k in range(8)]
              + ["circuit_complexity", "support", "image_size",
                 "weight_bias", "footprint"])
    with open("output/table_3to1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for f in range(256):
            bits = [str((f >> k) & 1) for k in range(8)]
            support = sum(
                1 for i in range(3)
                if any((f >> q) & 1 != (f >> (q ^ (1 << i))) & 1
                       for q in range(8)))
            image = len({(f >> q) & 1 for q in range(8)})
            weight = 2 * bin(f).count("1") - 8
            w.writerow(bits + [class_complexity[int(canon[f])],
                               support, image, weight,
                               3 * image + support])

    comp = [class_complexity[int(canon[f])] for f in range(256)]
    print("rows written: 256")
    print("min complexity:", min(comp), "max complexity:", max(comp))
    print("counts per tier:", np.bincount(comp).tolist())

    AND3, OR3 = 0x80, 0xFE
    MAJ3 = sum(1 << q for q in range(8) if bin(q).count("1") >= 2)
    MUX = sum(1 << q for q in range(8)
              if ((q >> 1) & 1 if (q >> 2) & 1 else q & 1))
    AND2, XOR2 = 0x88, 0x66  # x0&x1, x0^x1 as 3-var tables
    PAR3 = sum(1 << q for q in range(8) if bin(q).count("1") & 1)
    for name, f, want in [("AND3", AND3, 2), ("OR3", OR3, 2),
                          ("MAJ3", MAJ3, 4), ("MUX", MUX, 3),
                          ("AND2", AND2, 1), ("XOR2", XOR2, 3)]:
        got = comp[f]
        flag = "OK" if got == want else "MISMATCH"
        print(f"  {name}: {got} (expected {want}) {flag}")
    print(f"  PARITY3: {comp[PAR3]}")


if __name__ == "__main__":
    main()
