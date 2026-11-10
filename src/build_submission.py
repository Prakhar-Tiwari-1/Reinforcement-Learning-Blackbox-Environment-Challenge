"""Embed a trained checkpoint into prakhar.tiwari.py (the single-file submission).

Usage: uv run python src/build_submission.py data/ppo_cont.pt
"""

import base64
import io
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "prakhar.tiwari.py"


def main(ckpt):
    ck = torch.load(ROOT / ckpt, weights_only=False, map_location="cpu")
    if ck.get("algo") == "ddqn":  # DDQN policy: Q-network + its fixed normalisation
        keep = {"algo": "ddqn", "net": {k: v.detach().cpu().float() for k, v in ck["net"].items()}, "norm_mean": ck["norm_mean"],
                "norm_std": ck["norm_std"], "hist": ck.get("hist", 4), "source": str(ckpt)}
        return write(keep, ckpt)
    keep = {"net": {k: v.detach().cpu().float() for k, v in ck["net"].items()},
            "norm_mean": np.asarray(ck["norm_mean"], np.float64), "norm_var": np.asarray(ck["norm_var"], np.float64),
            "reward_scale": ck.get("reward_scale") or 0.2, "source": str(ckpt)}
    if "arch" in ck:  # v2 checkpoints describe their architecture; v1 (legacy) ones do not
        keep["arch"] = ck["arch"]
    # (the optimizer state and training counters are left out: not needed to play)
    return write(keep, ckpt)


def write(keep, ckpt):
    buf = io.BytesIO()
    torch.save(keep, buf)
    b64 = base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()
    lines = "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))
    src = SUB.read_text()
    new = re.sub(r'PRETRAINED = """\n.*?"""', lambda m: f'PRETRAINED = """\n{lines}\n"""', src, flags=re.S)
    assert new != src or lines in src
    SUB.write_text(new)
    print(f"embedded {ckpt} into {SUB.name}: {len(b64)} base64 chars, file now {len(new) / 1024:.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1])
