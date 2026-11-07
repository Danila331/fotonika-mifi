import numpy as np
import pytest

from doe.quality import QualityConfig, acceptable, exported_metrics, proxy_score
from doe.io import phase_to_u8
import evaluate as independent


def test_exported_score_matches_independent_saved_codec():
    rng = np.random.default_rng(902)
    ph = rng.uniform(-np.pi, np.pi, (128, 128))
    t = rng.integers(0, 256, (4, 128, 128), dtype=np.uint8)
    m = exported_metrics(ph, t)
    r, _ = independent.evaluate(phase_to_u8(ph), t)
    for key in ('pcc', 'ssim_global', 'ssim_skimage', 'ssim_gauss'):
        assert m[key] == r['summary'][key]['mean']
    assert proxy_score(m) == pytest.approx(.4*m['pcc']+.3*m['ssim_global']+.3*m['ssim_gauss'])


def test_guardrails_reject_better_proxy_with_worse_single_metric():
    baseline = dict(pcc=.6, ssim_global=.5, ssim_skimage=.3, ssim_gauss=.3, pcc_worst=.4, selectivity=.2)
    candidate = {**baseline, 'ssim_global': .7, 'pcc': .59}
    assert proxy_score(candidate) > proxy_score(baseline)
    assert not acceptable(candidate, baseline)
    assert acceptable(baseline, baseline)
    assert not acceptable({**baseline, 'selectivity': .17}, baseline)


def test_constant_targets_have_defined_proxy():
    m = exported_metrics(np.zeros((128, 128)), np.zeros((4, 128, 128), dtype=np.uint8))
    assert m['pcc'] is None and np.isfinite(proxy_score(m))
    assert acceptable(m, m)


def test_config_validation():
    with pytest.raises(ValueError):
        QualityConfig(steps=-1)
    with pytest.raises(ValueError):
        QualityConfig(check_every=0)


def test_refinement_retains_baseline_and_returns_valid_quantized_phase():
    import torch
    from types import SimpleNamespace
    from doe.physics import Fresnel
    from doe.losses import LossConfig
    from doe.quality import refine_quality
    from doe.io import u8_to_phase
    torch.set_num_threads(4)
    rng = np.random.default_rng(99)
    t = rng.integers(0, 256, (1, 4, 128, 128), dtype=np.uint8)
    baseline = rng.uniform(-np.pi, np.pi, (1, 128, 128))
    net = SimpleNamespace(device=torch.device('cpu'), loss_cfg=LossConfig(),
                          model=SimpleNamespace(fresnel=Fresnel()),
                          _to_tensor=lambda a: torch.tensor(a / 255., dtype=torch.float32))
    p, info = refine_quality(net, t, QualityConfig(steps=0), baseline)
    assert np.array_equal(p, u8_to_phase(phase_to_u8(baseline)))
    q, choices = refine_quality(net, t, QualityConfig(steps=2, check_every=1), baseline)
    assert q.shape == baseline.shape and np.isfinite(q).all() and np.abs(q).max() <= np.pi
    assert acceptable(exported_metrics(q[0], t[0]), exported_metrics(p[0], t[0]))
    assert choices[0]['score'] >= info[0]['score']


def test_supplied_quality_engine_rejects_incompatible_export_policy(tmp_path):
    from types import SimpleNamespace
    import synth
    with pytest.raises(ValueError, match='amplitude with peak'):
        synth.run([], tmp_path, policy='minmax', synthesiser=SimpleNamespace(method='nn_quality'))
