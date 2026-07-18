import json
import time
from multiprocessing import Pool

import numpy as np

from groups import canonicalize_3to2
from aig_synth import synthesize


def solve_one(combined_id):
    f0 = combined_id >> 8
    f1 = combined_id & 0xFF
    r, sol = synthesize(3, [f0, f1], max_gates=20)
    return combined_id, r


def main():
    canon = canonicalize_3to2()
    reps = sorted(int(x) for x in np.unique(canon))
    print(f"{len(reps)} representative classes to synthesize", flush=True)

    results = {}
    t0 = time.time()
    with Pool() as pool:
        for i, (cid, r) in enumerate(pool.imap_unordered(solve_one, reps)):
            results[cid] = r
            if (i + 1) % 10 == 0 or (i + 1) == len(reps):
                print(f"  {i+1}/{len(reps)} done, elapsed {time.time()-t0:.1f}s", flush=True)

    with open("class_complexity_3to2.json", "w") as fh:
        json.dump(results, fh)
    print("done, total time", time.time() - t0)


if __name__ == "__main__":
    main()
