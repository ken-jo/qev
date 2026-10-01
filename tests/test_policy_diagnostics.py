import runpy
from pathlib import Path

import pytest

truth = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "diagnose_policy_stages.py")
)["truth"]


@pytest.mark.parametrize(
    "condition,x,y,expected",
    [
        ("x >= 10", 10, 0, True),
        ("x >= 10", 9, 0, False),
        ("x < 10", 10, 0, False),
        ("x < 10", 9, 0, True),
        ("(x >= 10) AND (y >= 5)", 10, 5, True),
        ("(x >= 10) AND (y >= 5)", 10, 4, False),
        ("(x >= 10) AND (y >= 5)", 9, 5, False),
        ("(x >= 10) OR (y >= 5)", 9, 5, True),
        ("(x >= 10) OR (y >= 5)", 10, 4, True),
        ("(x >= 10) OR (y >= 5)", 9, 4, False),
        ("5 <= x < 10", 4, 0, False),
        ("5 <= x < 10", 5, 0, True),
        ("5 <= x < 10", 9, 0, True),
        ("5 <= x < 10", 10, 0, False),
    ],
)
def test_diagnostic_oracle(condition, x, y, expected):
    assert truth(condition, {"x": x, "y": y}) == expected
