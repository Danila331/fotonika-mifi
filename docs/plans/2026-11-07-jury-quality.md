# Jury quality experiment implementation plan

**Goal:** Improve saved 8-bit DOE reconstructions, comparing the original checkpoint and default 100-step refinement with experimental refinement and a fine-tuned checkpoint.

**Architecture:** Keep the original checkpoint and default behavior. Add an optional quality refinement path and a reproducible experiment runner. Select candidates on validation only, freeze the configuration, then evaluate once on held-out examples. Report individual metrics rather than claiming a known jury score.

**Tech Stack:** Python, PyTorch (Metal where supported), NumPy, Pillow, existing independent Fresnel evaluator, pytest.

## 1. Establish environment and data
- Create isolated worktree and Python environment; verify MPS forward/backward numerics and runtime.
- Download official source data needed for a representative pilot (Caltech-101 and BSDS500 if available), preserve source/class splits via existing preprocessing.
- Generate fixed training, validation (32 quads), and held-out test (64 quads) with distinct seeds and source splits. Save dataset hashes and disclose missing sources. Existing demo quads are qualitative examples only.
- Record baseline checkpoint hash/config and baseline pure-network and default-refinement scores.

## 2. Compare refinement options
- Implement `doe/quality.py`: configurable phase-only refinement using existing differentiable physics and losses; preserve baseline candidate and choose saved quantized candidates with a documented proxy score.
- Try a small predeclared validation grid: longer refinement, global-SSIM emphasis, and a transition toward exact peak normalization. Include local SSIM and crosstalk in reporting and guardrails.
- Meaningful tests in `tests/test_quality.py`: actual saved candidate score agrees with independent evaluator, candidate selection retains baseline when alternatives are worse, bounded valid phase and constant-target handling.

## 3. Pilot fine-tuning
- Fine-tune from the original weights with jury-aligned loss and quantization, using only training examples. Benchmark first and size a feasible run on this computer.
- Compare checkpoint candidates on validation, including against unchanged initial weights. Retain only if validation improves; do not force adoption of worse weights.
- Distillation is a subsequent option if the pilot and available runtime justify it; record what actually ran.

## 4. Freeze, evaluate, and deliver
- Freeze the winning validation configuration before touching held-out evaluation.
- Evaluate baseline and selected candidate on the same 64 held-out quads, saved PNG format, independent evaluator, paired bootstrap intervals, per-family results and runtime.
- Generate static comparison contact sheets and a local HTML gallery with targets, baseline, candidate, metrics, and all examples including regressions. Render unretouched reconstructions at consistent scale.
- Expose optional quality inference through the CLI only when supported by measured results. Document unchanged original mode and costs of quality mode.
- Run project tests, obtain code review, address findings, verify Docker smoke test and commit reproducible changes. Deliver metrics, limitations, checkpoint location and gallery.

## Supplied documents and frozen experiment
The supplied AI фотоника.pdf and Хакатон_задача_q2.docx specify a neural method, single-FFT Fresnel propagation, 532 nm, 10 µm pixels, four depths 24.1/25.3/26.5/27.7 mm, 128x128 grayscale 8-bit phase PNG and amplitude reconstructions. They name Pearson and SSIM (global formula illustrated), but give no scalar ranking formula, numerical SSIM constants, runtime limit or explicit approval of additional iterative optimization. Report pure-network and hybrid results separately; retain existing expert clarification on peak normalization.

Validation chose long_balanced: 100 original steps plus 300 extra, alpha1 beta0.5 xtalk0.25, cosine lr0.03 to0.003, check every50 steps. Mean PCC and all SSIM cannot regress; worst PCC may drop by0.01, selectivity by0.02 per quad. Proxy0.4PCC+0.3globalSSIM+0.3GaussianSSIM is not an official score.

Pilot: original checkpoint, 400 steps, batch2, lr1e-5, warmup20, quantization from start, alpha1 beta0.5 gamma0.3 xtalk0.25 soft_beta20. Compare original weights explicitly; no automatic promotion. Real sources Caltech101/BSDS500 plus procedural generator. Other source families fall back to available/synthetic data. New validation/test seeds use existing source split rules, so this is not a pristine external-source benchmark.
