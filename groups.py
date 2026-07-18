"""
NPN-style equivalence classes for NAND circuit complexity.

n=4, m=1 case: standard NPN group (permute inputs, negate inputs, negate output).
Group order = 4! * 2^4 * 2 = 768.

n=3, m=2 case: permute inputs, negate inputs, negate each output independently,
swap the two outputs. Group order = 3! * 2^3 * 2 * 2 * 2 = 384.

Both act on the space of truth tables by relabeling wires only, which NAND
circuits realize for free (permutation = rewiring, negation = free NOT gate,
output negation = free NOT gate, output swap = relabeling which output is
which). So minimum NAND circuit size is a class invariant, and we only need
to run exact synthesis once per class.
"""

import itertools
import numpy as np


def _kmaps(n):
    """For each (permutation, negation-mask) pair, build the bit-gather map
    kmap such that new_bits[k] = old_bits[kmap[k]]."""
    kmaps = []
    for pi in itertools.permutations(range(n)):
        for nmask in range(1 << n):
            kmap = [0] * (1 << n)
            for k in range(1 << n):
                y = 0
                for j in range(n):
                    xj = (k >> j) & 1
                    yj = xj ^ ((nmask >> pi[j]) & 1)
                    y |= yj << pi[j]
                kmap[k] = y
            kmaps.append(kmap)
    return kmaps


def _bitgather(arr, kmap, nbits):
    """arr: numpy array of ints each < 2**nbits. Returns array where bit k of
    result = bit kmap[k] of arr, vectorized."""
    result = np.zeros_like(arr)
    for k in range(nbits):
        src = kmap[k]
        bit = (arr >> src) & 1
        result |= (bit << k)
    return result


def canonicalize_4to1():
    """Returns numpy array `canon` of length 65536 where canon[f] is the
    minimum truth-table value in f's NPN orbit."""
    n = 4
    size = 1 << (1 << n)  # 65536
    f_arr = np.arange(size, dtype=np.int64)
    best = f_arr.copy()
    kmaps = _kmaps(n)
    for kmap in kmaps:
        transformed = _bitgather(f_arr, kmap, 1 << n)
        for out_neg in (0, 1):
            cand = transformed ^ (0xFFFF if out_neg else 0)
            np.minimum(best, cand, out=best)
    return best


def canonicalize_3to2():
    """Returns numpy array `canon` of length 65536 (indexed by combined id
    f0*256+f1) where canon[id] is the minimum combined id in the orbit."""
    n = 3
    rows = 1 << n  # 8
    size = 1 << (2 * rows)  # 65536
    ids = np.arange(size, dtype=np.int64)
    f0_arr = ids >> rows
    f1_arr = ids & 0xFF
    best = ids.copy()
    kmaps = _kmaps(n)
    for kmap in kmaps:
        g0 = _bitgather(f0_arr, kmap, rows)
        g1 = _bitgather(f1_arr, kmap, rows)
        for na in (0, 1):
            g0n = g0 ^ (0xFF if na else 0)
            for nb in (0, 1):
                g1n = g1 ^ (0xFF if nb else 0)
                for swap in (0, 1):
                    if swap:
                        h0, h1 = g1n, g0n
                    else:
                        h0, h1 = g0n, g1n
                    cand = (h0 << rows) | h1
                    np.minimum(best, cand, out=best)
    return best


if __name__ == "__main__":
    c4 = canonicalize_4to1()
    classes4 = np.unique(c4)
    print("n=4->1: total functions", len(c4), "classes", len(classes4))

    c32 = canonicalize_3to2()
    classes32 = np.unique(c32)
    print("n=3->2: total functions", len(c32), "classes", len(classes32))
