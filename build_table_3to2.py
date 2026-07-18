"""Build the 65536-row table for 3-bit -> 2-bit maps.

Columns: input_000 .. input_111 (8 columns, each a 2-bit output string, e.g.
"01"), then circuit_complexity (joint/shared-gate circuit size for both
output bits together).
"""
import csv
import json

import numpy as np

from groups import canonicalize_3to2


def main():
    with open("class_complexity_3to2.json") as fh:
        class_complexity = {int(k): v for k, v in json.load(fh).items()}

    canon = canonicalize_3to2()

    header = [f"in_{format(k, '03b')}" for k in range(8)] + ["circuit_complexity"]

    with open("output/table_3to2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for combined_id in range(65536):
            f0 = combined_id >> 8  # output bit 1 (MSB of the pair)
            f1 = combined_id & 0xFF  # output bit 0 (LSB of the pair)
            cells = []
            for k in range(8):
                b1 = (f0 >> k) & 1
                b0 = (f1 >> k) & 1
                cells.append(f"{b1}{b0}")
            complexity = class_complexity[int(canon[combined_id])]
            w.writerow(cells + [complexity])

    comp_vals = [class_complexity[int(canon[c])] for c in range(65536)]
    print("min complexity:", min(comp_vals), "max complexity:", max(comp_vals))
    print("rows written: 65536")


if __name__ == "__main__":
    main()
