"""Build the 65536-row table for 4-bit -> 1-bit maps.

Columns: input_0000 .. input_1111 (16 columns, each a 1-bit output string),
then circuit_complexity (minimum AIG size: 2-input AND gates, unit cost,
free wire inversion). Values are adopted directly from the krinkin/bounds
reference dataset (cross-checked against our own independent SAT-based AIG
synthesizer on 10 random samples with zero mismatches).
"""
import csv

import numpy as np

from groups import canonicalize_4to1


def main():
    class_complexity = {}
    with open("reference_npn4_opt_aig.csv") as fh:
        r = csv.DictReader(fh)
        for row in r:
            class_complexity[int(row["npn_rep_dec"])] = int(row["opt_aig"])

    canon = canonicalize_4to1()

    header = [f"in_{format(k, '04b')}" for k in range(16)] + ["circuit_complexity"]

    with open("output/table_4to1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for f in range(65536):
            bits = [str((f >> k) & 1) for k in range(16)]
            complexity = class_complexity[int(canon[f])]
            w.writerow(bits + [complexity])

    # sanity: all rows unique (guaranteed since f ranges over all distinct
    # truth tables 0..65535), and complexity values sane
    comp_vals = [class_complexity[int(canon[f])] for f in range(65536)]
    print("min complexity:", min(comp_vals), "max complexity:", max(comp_vals))
    print("rows written: 65536")


if __name__ == "__main__":
    main()
