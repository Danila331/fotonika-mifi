from doe.distill import build_student
from doe.model import count_params


def test_student_is_compact_and_has_two_refinement_stages():
    student = build_student()
    assert 1_000_000 <= count_params(student) <= 3_000_000
    assert len(student.stages) == 2
    assert student.cfg["width"] == 16
    assert student.cfg["kind"] == "image"
    assert student.cfg["quantity"] == "amplitude"
