"""Build a resumable teacher dataset from new training quadruples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from doe.data import QuadGenerator
from doe.distill import validate_teacher_arrays
from doe.infer import NeuralDOE
from doe.io import phase_to_u8
from doe.metrics import evaluate
from doe.quality import exported_metrics
from scripts.perceptual_pilot import perceptual_refine


def atomic_save_npy(path, array):
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def atomic_save_checkpoint(path, targets, phases):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, targets=targets, phases=phases)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200, help="perceptual steps after the base 100")
    parser.add_argument("--seed", type=int, default=88401)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 1 or args.batch < 1 or args.steps < 0:
        raise ValueError("size/batch must be positive and steps nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = NeuralDOE.load(device=device)
    generator = QuadGenerator("train", seed=args.seed)
    targets_parts, phases_parts = [], []
    checkpoint_path = args.out / "checkpoint.npz"
    start = 0
    if checkpoint_path.exists():
        saved = np.load(checkpoint_path)
        targets_parts.append(saved["targets"])
        phases_parts.append(saved["phases"])
        start = len(saved["targets"])
        if start > args.size:
            raise ValueError("checkpoint contains more samples than requested")
        for _ in range(0, start, args.batch):
            generator.sample(args.batch)
        print(f"Resuming at {start}/{args.size}", flush=True)
    started = time.perf_counter()
    for first in range(start, args.size, args.batch):
        count = min(args.batch, args.size - first)
        targets = generator.sample(count)
        baseline = net.predict_refined_batch(targets, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
        phases, _ = perceptual_refine(net, targets, baseline, args.steps, ms_weight=0.3, tv_weight=0.1)
        targets_parts.append(targets.astype(np.uint8, copy=False))
        phases_parts.append(phases.astype(np.float32))
        current_targets = np.concatenate(targets_parts)
        current_phases = np.concatenate(phases_parts).astype(np.float32)
        validate_teacher_arrays(current_targets, current_phases)
        atomic_save_checkpoint(checkpoint_path, current_targets, current_phases)
        print(json.dumps({"completed": first + count, "total": args.size,
                          "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    targets = np.concatenate(targets_parts)
    phases = np.concatenate(phases_parts).astype(np.float32)
    validate_teacher_arrays(targets, phases)
    # Deterministic target regeneration and independent saved-codec metric check.
    regenerated = QuadGenerator("train", seed=args.seed).sample(min(args.batch, args.size))
    assert np.array_equal(regenerated, targets[:len(regenerated)])
    reference = evaluate(phase_to_u8(phases[0]), targets[0])
    verified = exported_metrics(phases[0], targets[0])
    assert verified["pcc"] == reference["summary"]["pcc"]["mean"]
    atomic_save_npy(args.out / "targets.npy", targets)
    atomic_save_npy(args.out / "phases.npy", phases)
    checkpoint = Path(net.meta["path"])
    manifest = {
        "status": "complete", "seed": args.seed, "split": "train", "size": args.size,
        "batch": args.batch, "base_steps": 100, "perceptual_steps": args.steps,
        "ms_l1_weight": 0.3, "excess_tv_weight": 0.1, "device": device,
        "teacher_checkpoint": str(checkpoint), "teacher_sha256": sha256(checkpoint),
        "targets_sha256": sha256(args.out / "targets.npy"), "phases_sha256": sha256(args.out / "phases.npy"),
        "elapsed_seconds": time.perf_counter() - started,
        "verification": {"deterministic_targets": True, "independent_metric_match": True},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("TEACHER_READY", json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
