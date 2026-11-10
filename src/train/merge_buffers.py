"""Merge consecutive replay buffers of a chain of runs into one file, without duplicates.

Each run of the chain loaded the previous run's buffer (ring of --cap rows) and then overwrote it from that buffer's
write position `i` onwards. So run k's own transitions are the rows (i_{k-1} + j) mod cap for j < (i_k - i_{k-1}) mod cap
(when it wrote fewer than cap rows; checked). The merged file keeps all of the first buffer, then appends each later
run's own rows in the order they were written, and is saved in the Replay format (n = i = total rows), so a run with a
larger --buffer resumes appending after them.

Usage: .venv/bin/python src/train/merge_buffers.py runs/sac_merged_buffer.npz runs/sac/buffer.npz runs/sac2/buffer.npz
"""

import os
import sys

import numpy as np

KEYS = ("s", "s2", "a", "r", "g")


def main():
    out, first, *rest = sys.argv[1:]
    prev = np.load(first)
    cap = len(prev["r"])
    if int(prev["n"]) != cap:
        raise SystemExit(f"{first} is not full ({prev['n']} of {cap} rows): the ring offsets below would not hold")
    parts = {k: [prev[k]] for k in KEYS}
    print(f"{first}: {cap} rows", flush=True)
    for path in rest:
        cur = np.load(path)
        written = (int(cur["i"]) - int(prev["i"])) % cap
        idx = (int(prev["i"]) + np.arange(written)) % cap
        keep = np.ones(cap, bool)
        keep[idx] = False
        if not np.array_equal(prev["s"][keep], cur["s"][keep]):      # rows it did not write must be the previous run's
            raise SystemExit(f"{path} does not continue {first}: untouched rows differ")
        for k in KEYS:
            parts[k].append(cur[k][idx])
        print(f"{path}: {written} new rows", flush=True)
        prev = cur
    merged = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(merged["r"])
    tmp = f"{out}.tmp.npz"
    np.savez(tmp, **merged, i=n, n=n)
    os.replace(tmp, out)
    print(f"saved {out}: {n} rows", flush=True)


if __name__ == "__main__":
    main()
