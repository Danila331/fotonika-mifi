import numpy as np
import pytest

from doe.distill import validate_teacher_arrays


def test_validate_teacher_arrays_accepts_quantized_phase_and_uint8_targets():
    targets = np.zeros((2, 4, 128, 128), dtype=np.uint8)
    phases = np.zeros((2, 128, 128), dtype=np.float32)
    validate_teacher_arrays(targets, phases)


def test_validate_teacher_arrays_rejects_shape_dtype_and_nonfinite_errors():
    with pytest.raises(ValueError, match="shape"):
        validate_teacher_arrays(np.zeros((1, 4, 128, 128), dtype=np.uint8),
                                np.zeros((2, 128, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        validate_teacher_arrays(np.zeros((1, 4, 128, 128), dtype=np.float32),
                                np.zeros((1, 128, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        validate_teacher_arrays(np.zeros((1, 4, 128, 128), dtype=np.uint8),
                                np.full((1, 128, 128), np.nan, dtype=np.float32))
