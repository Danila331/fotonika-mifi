"""Generate direct-GS teacher phases for an existing frozen target dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from doe.baselines import gs_batch
from doe.distill import validate_teacher_arrays


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    targets = np.load(args.targets)
    args.out.mkdir(parents=True, exist_ok=True)
    phases = []
    started = time.perf_counter()
    for first in range(0, len(targets), args.batch):
        current = targets[first:first + args.batch]
        phase = gs_batch(current, iters=args.iters, device=args.device).astype(np.float32)
        phases.append(phase)
        print(f"{min(first + args.batch, len(targets))}/{len(targets)}", flush=True)
    phases = np.concatenate(phases).astype(np.float32)
    validate_teacher_arrays(targets, phases)
    np.save(args.out / "phases.npy", phases)
    np.save(args.out / "targets.npy", targets)
    manifest = {"status": "complete", "method": f"plain_gs_{args.iters}", "iters": args.iters,
                "size": len(targets), "device": args.device, "source_targets": str(args.targets),
                "targets_sha256": sha256(args.out / "targets.npy"), "phases_sha256": sha256(args.out / "phases.npy"),
                "elapsed_seconds": time.perf_counter() - started}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("GS_TEACHER_READY", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
