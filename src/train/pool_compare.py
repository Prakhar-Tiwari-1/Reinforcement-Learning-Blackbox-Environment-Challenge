"""Pool the results of contest_compare.sh logs: per checkpoint, runs, mean reward/1000, r=+5 and died per 1000.
Usage: .venv/bin/python src/train/pool_compare.py results/contest_compare_*.log"""
import re
import statistics as st
import sys
from collections import defaultdict
P = re.compile(r"\| (\S+) \| final: (\d+) steps .*reward/1000 ([+-][\d.]+) \(\+:(\d+) -:(\d+)\)")
runs = defaultdict(list)
for f in sys.argv[1:]:
    for line in open(f):
        m = P.search(line)
        if m:
            c, n, r, p, d = m.groups()
            runs[c].append((int(n), float(r), int(p), int(d)))
print(f"{'checkpoint':32} {'runs':>4} {'steps':>7} {'reward/1000':>11} {'sd(run)':>7} {'r=+5/1000':>9} {'died/1000':>9}")
for c, v in sorted(runs.items(), key=lambda kv: -sum(x[1] for x in kv[1]) / len(kv[1])):
    n = sum(x[0] for x in v)
    rw = [x[1] for x in v]
    print(f"{c:32} {len(v):4d} {n:7d} {st.mean(rw):+11.1f} {st.stdev(rw) if len(rw) > 1 else 0:7.1f} "
          f"{1000 * sum(x[2] for x in v) / n:9.1f} {1000 * sum(x[3] for x in v) / n:9.2f}")
