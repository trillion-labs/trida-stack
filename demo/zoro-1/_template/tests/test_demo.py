import pytest

from my_demo.main import LEVELS, rate_excuse
from my_demo.zoro import Zoro


def test_runs_in_mock_mode():
    r = rate_excuse(Zoro(mock=True), "my cat deleted my homework")
    assert 0 <= r["believable"] <= 1 and 0 <= r["funny"] <= 1 and r["reply"]


def test_request_passes_zoro_parser():
    """Run with PYTHONPATH=<repo>/inference/zoro-1 to check requests against Zoro-1's own parser."""
    dr = pytest.importorskip("decide_request")
    req, _ = dr.systemone_request({"model": "zoro-1", "state": {"excuse": "x"}, "reasoning": False, "questions": {
        "believable": {"type": "score", "instructions": "How believable?", "criteria": LEVELS},
        "funny": {"type": "noul", "instructions": "Funny?"}}})
    dr.parse(req)
