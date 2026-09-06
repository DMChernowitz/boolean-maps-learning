# How the Gibbs circuit complexities were computed

The Gibbs prior of the paper weighs a Boolean map by three descriptors. Two of
them, the footprint and the bias, are read off a truth table in one pass. The
third, the circuit complexity `X`, is a *minimum over all circuits* and has to
be computed. This note records how.

Two spaces are tabulated, both of size `2^16 = 65,536`:

| space | questions | answers | maps | classes | `X` range |
|---|---|---|---|---|---|
| `(n,m) = (4,1)` | 16 | 2 | 65,536 | 222 | 0 .. 10 |
| `(n,m) = (3,2)` | 8 | 4 | 65,536 | 308 | 0 .. 9 |

## The cost model

`X_j` is the fewest two-input NAND gates in any feed-forward circuit computing
`phi_j`. The bookkeeping is done in the equivalent **and-inverter** form: AND
gates of unit cost, with inversion free on every wire. De Morgan's laws convert
between the two at equal gate count, so this is the standard AIG-size metric of
logic synthesis. Reference values: AND, OR, NAND, NOR cost 1; XOR costs 3;
constants, projections and negations cost 0.

For `m > 1` the circuit is **joint**: one circuit computes all output bits with
gates shared, so `X` is at most, and generally below, the sum of the per-output
costs.

## Step 1 — symmetry reduction (`groups.py`)

Relabelling wires is free in an and-inverter circuit. Permuting inputs is
rewiring; negating an input or an output is an edge attribute; for `m = 2`,
swapping the two outputs renames which wire is which. So `X` is constant on the
orbits of

- `(4,1)`: the standard NPN group, order `4! * 2^4 * 2 = 768`;
- `(3,2)`: inputs permuted and negated, each output negated, outputs swapped,
  order `3! * 2^3 * 2^2 * 2 = 384`.

`groups.py` applies every group element to every truth table and keeps the
numerically smallest image as the class representative. This collapses 65,536
maps to **222** classes at `(4,1)` and **308** at `(3,2)`, so exact synthesis
runs a few hundred times rather than 65,536 times.

## Step 2 — exact synthesis as SAT (`aig_synth.py`)

For a fixed gate budget `r`, `try_size(n, targets, r)` builds a CNF that is
satisfiable exactly when an `r`-gate and-inverter circuit computes the targets.

Variables:

- `t[s][row]` — the bit carried by signal `s` on truth-table row `row`. Signal
  `0` is the constant 0 and signals `1..n` are the inputs; all are pinned by
  unit clauses. Signals `n+1 ..` are the gates.
- `s[g][a,b,ia,ib]` — a one-hot selector: gate `g` takes its fanins from
  earlier signals `a <= b`, inverted according to `ia`, `ib`. Restricting to
  *earlier* signals makes the circuit acyclic by construction, so no ordering
  constraints are needed.
- `o[k][sig,inv]` — a one-hot selector naming which signal, at which polarity,
  carries output `k`.

Clauses:

- AND semantics, three per (gate, candidate, row): `y -> p`, `y -> q`,
  `p & q -> y`, with `p`, `q` the possibly-inverted fanin literals.
- Output correctness: the selected signal must match the target column on every
  row.
- At-most-one and at-least-one over each selector family.
- **No dangling gates**: every gate must feed a later gate or an output.
- **No duplicate gates**: no two gates may carry identical truth tables. (Only
  exact duplicates are forbidden — a gate and its complement can both be useful
  for free via the inversion attributes.)

`synthesize()` then tries `r = 0, 1, 2, ...` upward and returns the first
satisfiable size. Because every smaller budget was *refuted* by the solver, the
answer is a proven minimum, not a best effort. Back end is CaDiCaL via PySAT.

Self-test: running `python aig_synth.py` synthesizes the two-input gates and
prints AND = 1, OR = 1, XOR = 3, NAND = 1.

## Step 3 — the tables

- `(4,1)`: values are adopted from the published `github.com/krinkin/bounds`
  reproducibility dataset (`reference_npn4_opt_aig.csv`, 222 NPN classes).
  `build_table_4to1.py` maps each of the 65,536 truth tables to its class
  representative and writes `output/table_4to1.csv`.
- `(3,2)`: no reference table exists, so `run_synth_3to2.py` synthesizes all
  308 classes from scratch over a process pool, caching the result in
  `class_complexity_3to2.json`. `build_table_3to2.py` expands that to
  `output/table_3to2.csv`.
- `(3,1)`: `build_table_3to1.py`, 256 maps, small enough to be direct.

Each CSV carries the truth-table columns plus `circuit_complexity`, `support`,
`image_size`, `footprint` and `weight_bias` — everything `gibbs_curves.py`
needs to build the ensemble.

## Cross-check and the one caveat

`crosscheck.py` re-derives ten randomly chosen `(4,1)` classes (fixed seed) with
our own synthesizer and compares against the reference. Zero mismatches.

The reference dataset is honest about its own status column: **220 of the 222
classes are marked proven optimal, and two are not.** The classes of `0x1669`
and `0x166b` carry a best-known upper bound of 10 rather than a proof of
optimality. Their orbits contain 32 and 64 maps, so at most 96 of the 65,536
entries — 0.15% — could in principle be one gate too high. Both sit at the top
of the range, where the Occam weight `e^{-X}` is smallest, so nothing in the
paper's figures turns on them.

Our own synthesizer reproduces the difficulty rather than resolving it. For
`0x1669` it refutes `r = 5` in under a second and `r = 7` in 27 seconds, and
does not settle `r = 9` within 15 minutes. The instance size grows steeply
with the budget -- the selector family for gate `g` has `O((n+g)^2)` candidates
and the at-most-one encoding is quadratic in that -- which is why these two
classes are where the published proofs stop.

## Files

| file | what it does |
|---|---|
| `aig_synth.py` | SAT encoding and the `try_size` / `synthesize` loop |
| `groups.py` | NPN-style canonicalization for both spaces |
| `run_synth_3to2.py` | parallel driver, all 308 `(3,2)` classes |
| `crosscheck.py` | independent re-derivation of sampled `(4,1)` classes |
| `build_table_4to1.py`, `build_table_3to2.py`, `build_table_3to1.py` | expand class values to full truth-table CSVs |
| `gibbs_curves.py` | consumes the CSVs, builds the ensemble and the trajectories |
