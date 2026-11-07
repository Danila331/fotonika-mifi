"""End-to-end tests of the delivery CLI: synth.py -> files in the case format -> evaluate.py agrees exactly."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N = 128


def _method():
    """gs once the baselines agent has delivered doe/baselines.py, the trivial backprop phase before that."""
    return ("gs", "20") if importlib.util.find_spec("doe.baselines") else ("backprop", "0")


def _run(*args, check=True):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{args} failed:\n{r.stdout}\n{r.stderr}")
    return r


def _targets(d: Path):
    yy, xx = np.mgrid[:N, :N]
    shapes = [((xx - 40) ** 2 + (yy - 50) ** 2 < 400),
              (abs(xx - 80) < 20) & (abs(yy - 70) < 30),
              (abs(xx - yy) < 8) & (xx > 20) & (xx < 110),
              ((xx - 64) ** 2 + (yy - 64) ** 2 < 900) & ((xx - 64) ** 2 + (yy - 64) ** 2 > 300)]
    paths = []
    for i, s in enumerate(shapes):
        p = d / f"t{i + 1}.png"
        Image.fromarray((s * (150 + 30 * i)).astype(np.uint8), mode="L").save(p)
        paths.append(p)
    return paths


def _is_case_png(path):
    with Image.open(path) as img:
        a = np.asarray(img)
        return img.format == "PNG" and img.mode == "L" and img.size == (N, N) and a.dtype == np.uint8


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    targets = _targets(d)
    method, steps = _method()
    out = d / "out"
    _run("synth.py", *targets, "--out", out, "--method", method, "--steps", steps, "--policy", "peak")
    return targets, out, method


def test_outputs_exist_in_case_format(synth_run):
    _, out, method = synth_run
    for name in ["doe.png"] + [f"recon_{i}.png" for i in range(1, 5)]:
        assert _is_case_png(out / name), name
    for name in ("metrics.json", "metrics.csv", "comparison.png"):
        assert (out / name).stat().st_size > 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["method"] == method and m["policy"] == "peak" and m["quantity"] == "amplitude"
    assert len(m["planes"]) == 4
    assert [p["depth_mm"] for p in m["planes"]] == [24.1, 25.3, 26.5, 27.7]
    for k in ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss"):
        assert set(m["summary"][k]) >= {"mean", "worst"}
    assert set(m["summary_variants"]) == {f"{q}/{p}" for q in ("amplitude", "intensity")
                                          for p in ("peak", "minmax", "fit")}
    assert m["timings_ms"]["synth_ms"] > 0 and m["timings_ms"]["total_ms"] >= m["timings_ms"]["synth_ms"]
    assert len((out / "metrics.csv").read_text().strip().splitlines()) == 1 + 4 + 2


def test_evaluate_reproduces_metrics_exactly(synth_run, tmp_path):
    targets, out, _ = synth_run
    js = tmp_path / "ev.json"
    r = _run("evaluate.py", out / "doe.png", *targets, "--json", js, "--check-recon", out)
    assert "бит-в-бит" in r.stdout
    ev = json.loads(js.read_text())["evaluations"][0]
    m = json.loads((out / "metrics.json").read_text())
    assert ev["quantity"] == m["quantity"] == "amplitude"
    assert ev["planes"] == m["planes"]          # exact float equality, not approx
    assert ev["summary"] == m["summary"]
    for k in ("selectivity", "own_best", "diag", "off"):
        assert ev["crosstalk"][k] == m["crosstalk"][k]
    # the record-only intensity variant agrees exactly as well
    js2 = tmp_path / "ev_int.json"
    _run("evaluate.py", out / "doe.png", *targets, "--json", js2, "--quantity", "intensity", "--policy", "minmax")
    assert json.loads(js2.read_text())["evaluations"][0]["summary"] == m["summary_variants"]["intensity/minmax"]


def test_evaluate_all_policies_match_core(tmp_path):
    sys.path.insert(0, str(ROOT))
    import evaluate as ev
    from doe.metrics import evaluate as core

    rng = np.random.default_rng(1)
    phase_u8 = rng.integers(0, 256, (N, N), dtype=np.uint8)
    targets = rng.integers(0, 256, (4, N, N), dtype=np.uint8)
    targets[2] = 0                                  # constant target: PCC undefined on both sides
    targets[3] = (targets[3] // 2)                  # max != 255: exercises the max = 1 normalisation
    for q in ("amplitude", "intensity"):
        for pol in ("peak", "minmax", "fit"):
            rep, recon = ev.evaluate(phase_u8, targets, pol, q)
            ref = core(phase_u8, targets, policy=pol, quantity=q)
            assert np.array_equal(recon, ref["recon"])
            assert rep["quantity"] == ref["quantity"] and rep["policy"] == ref["policy"]
            assert rep["planes"] == ref["planes"] and rep["summary"] == ref["summary"]
    assert core(phase_u8, targets)["quantity"] == "amplitude"     # official default
    from doe.metrics import crosstalk as core_xt
    rep, recon = ev.evaluate(phase_u8, targets)
    a, b = ev.crosstalk(targets, recon), core_xt(targets, recon)
    assert np.array_equal(np.array(a["matrix"]), np.array(b["matrix"]), equal_nan=True)
    assert {k: a[k] for k in ("diag", "off", "selectivity", "own_best")} == \
           {k: b[k] for k in ("diag", "off", "selectivity", "own_best")}


def test_evaluate_does_not_import_torch(synth_run, tmp_path):
    targets, out, _ = synth_run
    code = ("import sys; sys.path.insert(0, '.'); import evaluate; "
            f"evaluate.main({[str(out / 'doe.png'), *map(str, targets), '--json', str(tmp_path / 'e.json')]!r}); "
            "bad = [m for m in sys.modules if m == 'torch' or m.startswith(('torch.', 'doe'))]; "
            "print('IMPORTED', bad)")
    r = _run("-c", code)
    assert "IMPORTED []" in r.stdout, r.stdout


def test_strict_rejects_bad_input(tmp_path):
    targets = _targets(tmp_path)
    Image.open(targets[0]).convert("RGB").save(targets[0])
    r = _run("synth.py", *targets, "--out", tmp_path / "o", "--method", "backprop", "--strict", "--no-figure",
             check=False)
    assert r.returncode != 0 and "mode RGB" in r.stderr
    r = _run("synth.py", *targets, "--out", tmp_path / "o", "--method", "backprop", "--no-figure")
    assert "converted" in r.stderr and _is_case_png(tmp_path / "o" / "doe.png")


def test_default_cli_uses_single_network_pass(tmp_path):
    targets = _targets(tmp_path)
    out = tmp_path / "default"
    _run("synth.py", *targets, "--out", out, "--no-figure")
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["method"] == "nn" and metrics["steps"] == 0
