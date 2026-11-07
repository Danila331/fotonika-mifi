"""Unsupervised physics-in-the-loop training (ML agent).

net(targets) -> phase -> Fresnel intensity -> jury-style loss vs targets. No precomputed holograms.
Validation goes through the official path: doe.metrics.evaluate(phase_to_u8(phase), targets) with its
defaults (amplitude |K|, both images max-normalised: expert rules E1/E2).

Usage:
  python -m doe.train --run NAME [--init CKPT | --resume] [--steps 30000 --batch 16 --qat-frac 0.1 ...]
  python -m doe.train --run NAME --bench 20          # speed probe: median ms/step and ETA, then exit
Device: cuda > mps > cpu. Writes <results-dir>/NAME/{log.txt, history.json, curves.png} and
<ckpt-dir>/NAME/{best.pt (best val --best-metric), last.pt (full state every --save-every steps, atomic)}.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .io import phase_to_u8
from .losses import LossConfig, jury_loss
from .metrics import crosstalk, evaluate
from .model import build_model, count_params

ROOT = Path(__file__).resolve().parent.parent


class QuadStream(torch.utils.data.IterableDataset):
    """Endless stream of uint8 (B,4,N,N) batches; each DataLoader worker gets its own seed."""

    def __init__(self, batch: int, seed: int):
        self.batch, self.seed = batch, seed

    def __iter__(self):
        from .data import QuadGenerator
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        gen = QuadGenerator("train", self.seed * 1000 + wid)
        while True:
            yield torch.from_numpy(np.ascontiguousarray(gen.sample(self.batch)))


def load_val(n: int, split: str = "val") -> np.ndarray:
    p = ROOT / "data" / f"{split}.npy"
    if p.exists():
        v = np.load(p)
        return v[:n] if n > 0 else v
    from .data import QuadGenerator
    return QuadGenerator(split, 0).sample(n if n > 0 else 64)


@torch.no_grad()
def predict_batch(model, targets_u8: np.ndarray, device, bs: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(targets_u8), bs):
        x = torch.tensor(targets_u8[i:i + bs], dtype=torch.float32, device=device) / 255.0
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def official_metrics(phases: np.ndarray, targets_u8: np.ndarray) -> dict:
    """Mean over samples of evaluate() summaries, through the 8-bit phase."""
    keys = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")
    acc = {f"{k}_{s}": [] for k in keys for s in ("mean", "worst")}
    acc["selectivity"] = []
    acc["own_best"] = []
    for ph, t in zip(phases, targets_u8):
        rep = evaluate(phase_to_u8(ph), t)
        summ = rep["summary"]
        for k in keys:
            for s in ("mean", "worst"):
                v = summ[k][s]
                acc[f"{k}_{s}"].append(np.nan if v is None else v)
        xt = crosstalk(t, rep["recon"])
        acc["selectivity"].append(xt["selectivity"])
        acc["own_best"].append(xt["own_best"])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def stage_weights(n: int) -> list:
    """Deep supervision for unrolled models: final stage weight 1, earlier stages 0.3."""
    return [0.3] * (n - 1) + [1.0]


def plot_curves(hist: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tr = hist["train"]
    if tr:
        s = [r["step"] for r in tr]
        ax[0].plot(s, [r["loss"] for r in tr], lw=0.8, label="train loss")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    va = hist["val"]
    if va:
        s = [r["step"] for r in va]
        for k in ("pcc_mean", "pcc_worst", "ssim_global_mean", "ssim_skimage_mean", "ssim_gauss_mean", "selectivity"):
            ax[1].plot(s, [r.get(k, float("nan")) for r in va], marker=".", label=k)
        ax[1].set_xlabel("step"); ax[1].set_title("official val (8-bit phase, peak)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pick_device(name=None) -> torch.device:
    """cuda > mps > cpu (or an explicit --device)."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def default_workers() -> int:
    return max(0, min(8, (os.cpu_count() or 2) - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    # model (ignored when --init / --resume supplies a checkpoint)
    ap.add_argument("--kind", default="doe")
    ap.add_argument("--inputs", default="physics")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--mults", default="1,2,4,8")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--unroll-width", type=int, default=24)
    ap.add_argument("--unroll-mults", default="1,2,4")
    ap.add_argument("--bp-phase", default="random", help="random | zero | quad<a> (back-projection phase)")
    ap.add_argument("--quantity", default="amplitude", choices=["amplitude", "intensity"])
    # optimisation
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the CNNs only (physics stays fp32)")
    # loss
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--norm", default="soft", choices=["soft", "peak"])
    ap.add_argument("--soft-beta", type=float, default=50.0, help="sharpness of the soft max-normalisation")
    ap.add_argument("--w-pcc", type=float, default=1.0, help="weight of (1 - PCC) per plane")
    ap.add_argument("--xtalk", type=float, default=0.0, help="crosstalk penalty weight (depth separation)")
    ap.add_argument("--add-stages", type=int, default=0,
                    help="with --init: append N new (zero-initialised) refinement stages to the checkpoint's model")
    ap.add_argument("--quant", action="store_true", help="straight-through 8-bit phase quantisation from step 1")
    ap.add_argument("--qat-frac", type=float, default=0.0,
                    help="enable 8-bit QAT for the last fraction of the steps (e.g. 0.1)")
    # checkpoints / IO
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from (model cfg taken from it)")
    ap.add_argument("--resume", action="store_true", help="continue from <ckpt-dir>/<run>/last.pt if it exists")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "ml"))
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--best-metric", default="pcc_mean")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true", help="sanity: train on one fixed batch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bench", type=int, default=0, help="time N steps after warm-up, print ms/step and exit")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.results_dir) / args.run
    ck_dir = Path(args.ckpt_dir) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / ("bench.txt" if args.bench else "log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    resume_ck = None
    if args.resume and (ck_dir / "last.pt").exists():
        resume_ck = torch.load(ck_dir / "last.pt", map_location="cpu", weights_only=False)
        mcfg = resume_ck["model_cfg"]
        model = build_model(mcfg)
        model.load_state_dict(resume_ck["state_dict"])
        log(f"resume from {ck_dir / 'last.pt'} at step {resume_ck['step']}")
    elif args.init and args.add_stages:
        # deeper model: the checkpoint's architecture + `add_stages` new refinement stages. Old weights load
        # unchanged; new stages are zero-initialised, i.e. start as plain damped GS steps after the old model.
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        mcfg = dict(ck["model_cfg"])
        old_n = mcfg.get("unroll", 0)
        mcfg["unroll"] = old_n + args.add_stages
        model = build_model(mcfg)
        sd = dict(ck["state_dict"])
        mix = model.stage_mix.detach().clone()
        if "stage_mix" in sd:
            mix[:old_n] = sd.pop("stage_mix")
        sd["stage_mix"] = mix
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(tuple(f"stages.{i}." for i in range(old_n, mcfg["unroll"])))
                                      for k in missing), (missing, unexpected)
        log(f"init from {args.init} (its step {ck.get('step')}): {old_n} -> {mcfg['unroll']} refinement stages, "
            f"{len(missing)} new tensors")
    elif args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        mcfg = ck["model_cfg"]
        model = build_model(mcfg)
        model.load_state_dict(ck["state_dict"])
        log(f"init from {args.init} (its step {ck.get('step')})")
    else:
        mcfg = dict(kind=args.kind, inputs=args.inputs, residual=args.residual, width=args.width,
                    mults=tuple(int(m) for m in args.mults.split(",")), blocks=args.blocks,
                    attn=not args.no_attn, unroll=args.unroll, unroll_width=args.unroll_width,
                    unroll_mults=tuple(int(m) for m in args.unroll_mults.split(",")),
                    quantity=args.quantity, bp_phase=args.bp_phase)
        model = build_model(mcfg)
    model.to(dev)
    model.cnn_autocast = bool(args.amp)
    lcfg = LossConfig(alpha=args.alpha, beta=args.beta, gamma=args.gamma, quant=args.quant, norm=args.norm,
                      soft_beta=args.soft_beta, w_pcc=args.w_pcc, xtalk=args.xtalk,
                      quantity=model.cfg.get("quantity", args.quantity))
    qat_start = args.steps + 1 if args.qat_frac <= 0 else int(round(args.steps * (1 - args.qat_frac))) + 1
    log(f"args {vars(args)}")
    log(f"model {model.cfg} params {count_params(model) / 1e6:.2f}M device {dev}"
        + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    hist = {"train": [], "val": []}
    best = -1e9
    start = 1
    if resume_ck is not None:
        opt.load_state_dict(resume_ck["optimizer"])
        hist = resume_ck.get("hist", hist)
        best = resume_ck.get("best", best)
        start = resume_ck["step"] + 1
        try:
            torch.set_rng_state(resume_ck["rng_torch"])
            np.random.set_state(resume_ck["rng_numpy"])
        except Exception as e:  # RNG restore is best-effort
            log(f"rng restore skipped: {e}")

    if not args.bench:
        val_u8 = load_val(args.val_n)
        log(f"val set {val_u8.shape} ({'data/val.npy' if (ROOT / 'data' / 'val.npy').exists() else 'generator'})")

    if args.overfit:
        fixed = torch.from_numpy(load_val(args.batch))
        batches = iter(lambda: fixed, None)
    else:
        # new seed stream after a resume, so a resumed run does not replay the same batches
        dl = torch.utils.data.DataLoader(QuadStream(args.batch, args.seed * 7919 + start), batch_size=None,
                                         num_workers=args.workers,
                                         prefetch_factor=4 if args.workers else None,
                                         persistent_workers=bool(args.workers),
                                         pin_memory=dev.type == "cuda")
        batches = iter(dl)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    def full_state(step):
        return dict(model_cfg=model.cfg, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    loss_cfg=vars(lcfg), step=step, args=vars(args), optimizer=opt.state_dict(), hist=hist,
                    best=best, rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state())

    nst = 1 + model.cfg.get("unroll", 0)
    sw = stage_weights(nst)
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    bench_times = []
    end = args.steps if not args.bench else start + 5 + args.bench - 1
    for step in range(start, end + 1):
        if args.bench:
            sync()
            tb = time.perf_counter()
        model.train()
        lcfg.quant = args.quant or step >= qat_start
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        xb = next(batches).to(dev, non_blocking=True).float() / 255.0
        phases = model(xb, return_all=True)
        loss = 0.0
        for w, ph in zip(sw, phases):
            l, m = jury_loss(ph, xb, model.fresnel, lcfg)
            loss = loss + w * l
        loss = loss / sum(sw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if args.bench:
            sync()
            if step >= start + 5:  # first 5 steps are warm-up (cudnn autotune, MPS graph build)
                bench_times.append(time.perf_counter() - tb)
            continue
        run_loss += float(loss.detach())
        run_n += 1
        if step % 50 == 0:
            rec = dict(step=step, loss=run_loss / run_n, pcc=float(m["pcc"].mean()), lr=lr_at(step),
                       gnorm=float(gn), t=time.time() - t0, quant=lcfg.quant)
            hist["train"].append(rec)
            if step % 200 == 0:
                log(f"step {step} loss {rec['loss']:.4f} train_pcc {rec['pcc']:.4f} lr {rec['lr']:.2e} "
                    f"gn {rec['gnorm']:.2f} {rec['t'] / (step - start + 1) * 1000:.0f}ms/step"
                    + (" QAT" if lcfg.quant else ""))
            run_loss, run_n = 0.0, 0
        if step % args.val_every == 0 or step == args.steps:
            ph = predict_batch(model, val_u8, dev)
            vm = official_metrics(ph, val_u8)
            vm["J"] = 0.5 * vm["pcc_mean"] + 0.5 * (vm["ssim_global_mean"] + vm["ssim_skimage_mean"]
                                                    + vm["ssim_gauss_mean"]) / 3
            vm["Jsel"] = vm["J"] + vm["selectivity"]  # quality + depth separation, for --best-metric Jsel
            vm["step"] = step
            vm["quant"] = bool(lcfg.quant)
            hist["val"].append(vm)
            tag = ""
            if vm[args.best_metric] > best:
                best = vm[args.best_metric]
                st = full_state(step)
                st["val"] = vm
                for k in ("optimizer", "hist", "rng_torch", "rng_numpy"):
                    st.pop(k)
                atomic_save(st, ck_dir / "best.pt")
                tag = " *best*"
            log("VAL step {step} pcc {pcc_mean:.4f}/{pcc_worst:.4f} ssim_g {ssim_global_mean:.4f} "
                "ssim_sk {ssim_skimage_mean:.4f} ssim_gs {ssim_gauss_mean:.4f} J {J:.4f} "
                "sel {selectivity:.4f} own {own_best:.2f}/4".format(**vm) + tag)
            with open(out_dir / "history.json", "w") as f:
                json.dump(hist, f)
            try:
                plot_curves(hist, out_dir / "curves.png")
            except Exception as e:  # plotting must never kill a run
                log(f"plot failed: {e}")
        if step % args.save_every == 0 or step == args.steps:
            atomic_save(full_state(step), ck_dir / "last.pt")
    if args.bench:
        ms = float(np.median(bench_times) * 1000)
        log(f"BENCH device {dev} batch {args.batch} amp {args.amp}: median {ms:.1f} ms/step over "
            f"{len(bench_times)} steps -> {args.steps} steps ~ {ms * args.steps / 3.6e6:.2f} h (+validation)")
    else:
        log(f"done. best val {args.best_metric} {best:.4f}, {time.time() - t0:.0f}s")
    logf.close()
    os._exit(0)  # do not hang on persistent DataLoader workers


if __name__ == "__main__":
    main()
