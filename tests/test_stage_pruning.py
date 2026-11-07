import pytest

from doe.distill import build_student, truncate_stages


def test_truncate_stages_preserves_requested_prefix():
    model = build_student()
    truncate_stages(model, 1)
    assert len(model.stages) == 1
    assert model.stage_mix.shape == (1,)


def test_truncated_model_state_matches_its_config():
    from doe.model import build_model

    model = build_student()
    truncate_stages(model, 1)
    restored = build_model(model.cfg)
    restored.load_state_dict(model.state_dict())
    assert len(restored.stages) == 1
    assert restored.stage_mix.shape == (1,)


def test_truncate_stages_rejects_invalid_count():
    model = build_student()
    with pytest.raises(ValueError):
        truncate_stages(model, 3)
    with pytest.raises(ValueError):
        truncate_stages(model, -1)
