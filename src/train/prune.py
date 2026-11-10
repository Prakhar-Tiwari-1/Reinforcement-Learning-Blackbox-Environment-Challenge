"""Free disk space. Dry run by default: shows what would be deleted; add --yes to delete.

Always kept: latest.pt, best_*.pt, config.json, log.jsonl, selection.json of every run, the core datasets
(random, sticky, const, long_random .npz), all results/ text files, all docs.

Deleted:
  - runs/*/step_*.pt except the newest --keep-last of each run (and all of them for finished runs)
  - runs/*/latest_eval.pt (temporary copies made by choose_checkpoint.py)
  - runs/*/stdout.log larger than --max-log-mb: truncated to its last 2,000 lines
  - with --data-days N: data/eval_*.npz older than N days (evaluation recordings; their statistics are
    already saved in results/eval_summary.jsonl)

Usage: uv run python src/train/prune.py            # dry run
       uv run python src/train/prune.py --yes --data-days 7
"""

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true")
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--max-log-mb", type=float, default=20)
    p.add_argument("--data-days", type=float, default=None)
    args = p.parse_args()
    victims, freed = [], 0
    for run in sorted((ROOT / "runs").glob("*")):
        if not run.is_dir():
            continue
        steps = sorted(run.glob("step_*.pt"))
        keep = 0 if (run / "DONE").exists() else args.keep_last
        victims += steps[:-keep] if keep else steps
        victims += list(run.glob("latest_eval.pt"))
        log = run / "stdout.log"
        if log.exists() and log.stat().st_size > args.max_log_mb * 2**20:
            print(f"{'truncate' if args.yes else 'would truncate'} {log} ({log.stat().st_size / 2**20:.0f} MB)")
            if args.yes:
                lines = log.read_text(errors="replace").splitlines()[-2000:]
                log.write_text("\n".join(lines) + "\n")
    if args.data_days is not None:
        cutoff = time.time() - args.data_days * 86400
        victims += [f for f in (ROOT / "data").glob("eval_*.npz") if f.stat().st_mtime < cutoff]
    for v in victims:
        freed += v.stat().st_size
        print(f"{'delete' if args.yes else 'would delete'} {v.relative_to(ROOT)}")
        if args.yes:
            v.unlink()
    print(f"{'freed' if args.yes else 'would free'} {freed / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
