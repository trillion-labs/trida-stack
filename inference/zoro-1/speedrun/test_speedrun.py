"""The speedrun rules and leaderboard rendering — no GPU, no network."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("leaderboard", Path(__file__).parent / "leaderboard.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def run(**kw):
    s = {"task": "gsm8k", "n": 200, "client_concurrency": 1, "errors": 0, "accuracy_pct": 89.5, "wall_s": 900.0,
         "tok_per_s_per_request": 300.0, "settings": dict(lb.SETTINGS)}
    s.update(kw)
    return s


def test_a_valid_run_is_a_record():
    assert lb.check_run(run(), "h100", []) == []
    assert lb.check_run(run(n=50), "mlx", []) == []


def test_rules_reject_each_violation():
    cases = {
        "task": run(task="mmlu_pro"),
        "problems": run(n=100),
        "concurrency": run(client_concurrency=8),
        "failed": run(errors=2),
        "floor": run(accuracy_pct=lb.ACCURACY_FLOOR - 0.5),
        "settings.top_k": run(settings={**lb.SETTINGS, "top_k": 50}),
        "settings.thinking": run(settings={**lb.SETTINGS, "thinking": False}),
    }
    for why, summary in cases.items():
        errs = lb.check_run(summary, "h100", [])
        assert errs, why
    assert lb.check_run(run(accuracy_pct=lb.ACCURACY_FLOOR), "h100", []) == []   # the floor itself counts


def test_a_record_must_beat_the_current_one():
    best = [{"wall_s": "900.0"}]
    assert lb.check_run(run(wall_s=899.9), "h100", best) == []
    assert any("does not beat" in e for e in lb.check_run(run(wall_s=900.0), "h100", best))


def test_table_and_chart():
    rec = {"date": "2026-10-29", "wall_s": "812.4", "accuracy_pct": "89.5", "tok_per_s_per_request": "312.0",
           "description": "self-spec b15_g8", "log": "https://example.com/run", "contributors": "@a"}
    t = lb.table("h100", [rec])
    assert "| 1 | 812.4 s | 89.5% | 312 | self-spec b15_g8 | 2026-10-29 | [log](https://example.com/run) | @a |" in t
    assert "no record yet" in lb.table("h100", [])
    svg = lb.chart("h100", [rec, {**rec, "wall_s": "700"}])
    assert svg.startswith("<svg") and "<polyline" in svg and ">812<" in svg and ">700<" in svg
    assert "no record yet" in lb.chart("h100", [])


def test_readme_and_charts_match_the_records():
    assert lb.stale() == [], "run `python speedrun/leaderboard.py render`"
