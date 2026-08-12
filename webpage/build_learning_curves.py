"""Build the interactive learning-curve page (learning_curves.html).

Bakes the circuit-complexity column of each classification table into
learning_curves_template.html as a hex string. Footprint and bias are
recomputed in-page from the truth tables, which this script verifies
against the tables first, so only one number per map has to be shipped.

Run from anywhere: paths are resolved relative to this script.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (2,1) complexities in the numeral convention of the paper's table:
# NOR at j = 1, XOR at 6, AND at 8, XNOR at 9.
C21 = [0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0]


def read_table(name):
    """Return the complexity, footprint and bias columns of a table."""
    complexity, footprint, bias = [], [], []
    with open(os.path.join(REPO, "output", name)) as fh:
        for row in csv.DictReader(fh):
            complexity.append(int(row["circuit_complexity"]))
            footprint.append(int(row["footprint"]))
            bias.append(int(row["weight_bias"]))
    return complexity, footprint, bias


def digits(j, n, m):
    """The Q answers of map j, in the layout the tables index by."""
    Q = 1 << n
    if m == 1:
        return [(j >> q) & 1 for q in range(Q)]
    if (n, m) == (3, 2):
        high, low = j >> 8, j & 0xFF
        return [(((high >> q) & 1) << 1) | ((low >> q) & 1)
                for q in range(Q)]
    raise ValueError(f"no digit layout for (n,m) = ({n},{m})")


def footprint_and_bias(j, n, m):
    """F = n I + S and B = 2 |table| - m Q, from the truth table alone."""
    Q = 1 << n
    d = digits(j, n, m)
    support = sum(1 for i in range(n)
                  if any(d[q] != d[q ^ (1 << i)] for q in range(Q)))
    return n * len(set(d)) + support, 2 * bin(j).count("1") - m * Q


def hex_column(values, ceiling=15):
    assert all(0 <= v <= ceiling for v in values), "complexity out of range"
    return "".join(format(v, "x") for v in values)


def main():
    baked = {}
    for key, name, n, m in (("41", "table_4to1.csv", 4, 1),
                            ("32", "table_3to2.csv", 3, 2)):
        complexity, footprint, bias = read_table(name)
        assert len(complexity) == 65536, f"{name}: unexpected row count"
        for j in (0, 1, 255, 4096, 30011, 65535):
            f, b = footprint_and_bias(j, n, m)
            assert (f, b) == (footprint[j], bias[j]), \
                f"{name}: in-page formula disagrees at j = {j}"
        # the page recomputes every row, so verify every row here
        for j in range(65536):
            assert footprint_and_bias(j, n, m) == (footprint[j], bias[j])
        baked[key] = hex_column(complexity)
        print(f"({n},{m}): complexity 0..{max(complexity)}, "
              f"footprint and bias verified on all 65536 maps")

    # (2,1) is small enough to carry inline, and its own tier structure
    # is checked against the paper: six maps free, eight at one gate,
    # XOR and XNOR at three.
    assert sum(1 for c in C21 if c == 0) == 6
    assert sum(1 for c in C21 if c == 1) == 8
    assert C21[6] == 3 and C21[9] == 3
    baked["21"] = hex_column(C21)

    path = os.path.join(HERE, "learning_curves_template.html")
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    for key, data in baked.items():
        token = f"__C{key}__"
        assert token in html, f"template is missing {token}"
        html = html.replace(token, data)
    assert "__C" not in html, "a placeholder survived substitution"

    out = os.path.join(HERE, "learning_curves.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out}: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
