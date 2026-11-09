"""The speedrun rules and leaderboard rendering — no GPU, no network."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = _load("leaderboard", HERE / "leaderboard.py")
ev = _load("zoro1_eval", HERE.parent / "eval.py")


def summary_from_eval(n=200, correct=179, num_problems=None, shard=None, max_workers=1, top_k=20, think=True):
    """A summary built by eval.py's own summarize(), so a renamed key there fails here."""
    args = argparse.Namespace(task="gsm8k", num_problems=n if num_problems is None else num_problems, shard=shard,
                              max_workers=max_workers, max_tokens=8192, temperature=1.0, top_p=0.95, top_k=top_k,
                              presence_penalty=0.0, think=think)
    recs = [{"correct": i < correct, "error": None, "completion_tokens": 300, "latency_s": 1.0,
             "finish_reason": "stop"} for i in range(n)]
    return ev.summarize(args, "http://127.0.0.1:30000", "trillionlabs/Zoro-1", n, recs, 900.0)


def run(**kw):
    s = summary_from_eval()
    s.update(kw)
    return s


def test_a_valid_run_is_a_record():
    assert lb.check_run(run(), "h100", []) == []
    assert lb.check_run(summary_from_eval(n=50, correct=45), "mlx", []) == []


def test_rules_reject_each_violation():
    cases = {
        "task": run(task="mmlu_pro"),
        "problems": run(n=100),
        "concurrency": summary_from_eval(max_workers=8),
        "failed": run(errors=2),
        "floor": run(accuracy_pct=lb.ACCURACY_FLOOR - 0.5),
        "settings.top_k": summary_from_eval(top_k=50),
        "settings.thinking": summary_from_eval(think=False),
        # n=200 taken from a bigger subsample's shard is a different problem set
        "shard": summary_from_eval(num_problems=800, shard="0/4"),
        "subsample": summary_from_eval(num_problems=800),
        "unrecorded subsample": {k: v for k, v in run().items() if k != "num_problems"},
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


OK_RECORD = {"date": "2026-10-29", "wall_s": "812.4", "accuracy_pct": "89.5", "tok_per_s_per_request": "312.0",
             "description": "self-spec b15_g8 (SGLang), q8 *fast*", "log": "https://github.com/o/r/pull/9#issue-1",
             "contributors": "@a, @b-c"}


def test_every_record_field_must_be_a_known_good_value():
    # each field is checked against what it may be, not against a list of characters that broke it before
    assert lb.record_problems(OK_RECORD) == []
    assert lb.record_problems({**OK_RECORD, "log": ""}) == []               # a log is optional
    bad = {
        "date html": {"date": "<img src=x>"},
        "date marker": {"date": "<!-- /speedrun:h100 -->"},
        "date pipe": {"date": "2026-10-29 | x"},
        "date link": {"date": "[x](https://e.com)"},
        "date format": {"date": "29/10/2026"},
        "date invalid": {"date": "2026-13-40"},
        "wall nan": {"wall_s": "nan"},
        "wall inf": {"wall_s": "inf"},
        "wall text": {"wall_s": "fast"},
        "wall zero": {"wall_s": "0"},
        "accuracy range": {"accuracy_pct": "101"},
        "tok nan": {"tok_per_s_per_request": "nan"},
        "log scheme": {"log": "javascript:alert(1)"},
        "log http": {"log": "http://example.com/run"},
        "log paren": {"log": "https://example.com/a)b"},
        "log space": {"log": "https://example.com/a b"},
        "description marker": {"description": "fast <!-- /speedrun:h100 --> x"},
        "description link": {"description": "see [log](https://phish.example)"},
        "description image": {"description": "![](https://tracker.example/p.gif)"},
        "description backslash": {"description": "a\\|b"},
        "description newline": {"description": "line one\nline two"},
        "description empty": {"description": ""},
        "contributors html": {"contributors": "<img src=x>"},
        "contributors link": {"contributors": "[@a](https://phish.example)"},
        "contributors free text": {"contributors": "me and my friend"},
    }
    for why, change in bad.items():
        assert lb.record_problems({**OK_RECORD, **change}), why


def test_a_nan_run_is_not_a_record():
    # NaN compares False with everything, so it would pass the floor and "beat" any record
    assert lb.check_run(run(accuracy_pct=float("nan")), "h100", [])
    assert lb.check_run(run(wall_s=float("nan")), "h100", [{"wall_s": "900.0"}])
    assert lb.check_run(run(wall_s=float("inf")), "h100", [])


def test_add_reports_rule_violations_for_a_partial_summary(monkeypatch, tmp_path, capsys):
    import json
    import pytest
    import sys
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"task": "gsm8k"}))                          # e.g. an older eval.py's summary
    monkeypatch.setattr(sys, "argv", ["leaderboard.py", "add", str(path), "--track", "h100",
                                      "--description", "x", "--contributors", "@a"])
    with pytest.raises(SystemExit) as exc:
        lb.main()
    assert "not a record" in str(exc.value)                                 # the rule list, not a KeyError


def test_check_separates_bad_rows_from_stale_files(monkeypatch):
    monkeypatch.setattr(lb, "load", lambda track: [{**OK_RECORD, "log": "http://x"}] if track == "h100" else [])
    assert any("h100.csv row 1" in p for p in lb.bad_records())
    assert not any("csv" in p for p in lb.stale())                          # stale() is only about render output


def test_table_escapes_pipes_in_free_text():
    rec = {"date": "2026-10-29", "wall_s": "812.4", "accuracy_pct": "89.5", "tok_per_s_per_request": "312.0",
           "description": "b15|g8", "log": "", "contributors": "@a | @b"}
    row = lb.table("h100", [rec]).splitlines()[2]
    assert row.count(" | ") == 7 and "b15\\|g8" in row and "@a \\| @b" in row


def test_readme_and_charts_match_the_records():
    assert lb.bad_records() == [], "fix these rows in speedrun/records/*.csv"
    assert lb.stale() == [], "run `python speedrun/leaderboard.py render`"
