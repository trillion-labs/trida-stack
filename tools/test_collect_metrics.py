"""CPU tests for the metrics collector's logic.

The live run against trida-stack returns no_data for the right reason -- there are no
external contributors yet -- which means the response-time path never executes. A metric
that is correctly empty and one that is broken look identical from the outside, so the
logic is tested here with synthetic data instead.
"""

import datetime as dt
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import collect_metrics as cm  # noqa: E402

UTC = dt.timezone.utc


def test_business_hours_skips_the_weekend():
    # Friday 09:00 -> Monday 09:00 is 72 clock hours but 24 business hours.
    fri = dt.datetime(2026, 10, 2, 9, 0, tzinfo=UTC)   # a Friday
    mon = dt.datetime(2026, 10, 5, 9, 0, tzinfo=UTC)   # the following Monday
    assert round(cm.business_hours(fri, mon)) == 24


def test_business_hours_within_one_weekday():
    a = dt.datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    b = dt.datetime(2026, 10, 1, 17, 30, tzinfo=UTC)
    assert round(cm.business_hours(a, b), 1) == 8.5


def test_business_hours_is_zero_when_end_precedes_start():
    a = dt.datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    assert cm.business_hours(a, a - dt.timedelta(hours=5)) == 0.0


def test_bot_detection():
    assert cm.is_bot({"type": "Bot", "login": "dependabot"})
    assert cm.is_bot({"type": "User", "login": "renovate[bot]"})
    assert cm.is_bot(None)
    assert not cm.is_bot({"type": "User", "login": "a-person"})


def test_parse_handles_github_z_suffix_and_none():
    assert cm.parse("2026-10-01T09:00:00Z") == dt.datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    assert cm.parse(None) is None


def test_first_human_response_ignores_bots_and_the_author(monkeypatch):
    author = "reporter"
    comments = [
        {"user": {"type": "Bot", "login": "ci[bot]"}, "created_at": "2026-10-01T09:00:00Z"},
        {"user": {"type": "User", "login": author}, "created_at": "2026-10-01T10:00:00Z"},
        {"user": {"type": "User", "login": "maintainer"}, "created_at": "2026-10-01T12:00:00Z"},
        {"user": {"type": "User", "login": "other"}, "created_at": "2026-10-01T15:00:00Z"},
    ]
    monkeypatch.setattr(cm, "paginate", lambda path, params=None: comments)
    got = cm.first_human_response("o/r", 1, author, is_pr=False)
    # The bot at 09:00 and the author's own reply at 10:00 are not responses.
    assert got == dt.datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def test_first_human_response_is_none_when_only_bots_and_author(monkeypatch):
    monkeypatch.setattr(cm, "paginate", lambda path, params=None: [
        {"user": {"type": "Bot", "login": "ci[bot]"}, "created_at": "2026-10-01T09:00:00Z"},
        {"user": {"type": "User", "login": "reporter"}, "created_at": "2026-10-01T10:00:00Z"},
    ])
    assert cm.first_human_response("o/r", 1, "reporter", is_pr=False) is None


def test_p90_is_nearest_rank():
    # Nearest-rank: p90 of 1..10 is the 9th value.
    assert cm.percentile([float(x) for x in range(1, 11)], 0.90) == 9.0


def test_p90_does_not_under_report_on_fractional_ranks():
    # Regression: the original int(n*q)-1 returned the 4th value here instead of the
    # 5th, quietly understating the tail the metric exists to expose.
    assert cm.percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.90) == 100.0
    assert cm.percentile([1.0, 2.0, 3.0], 0.90) == 3.0


def test_percentile_handles_a_single_value():
    assert cm.percentile([7.0], 0.90) == 7.0


# --- period labelling -------------------------------------------------------
# Two periods of the SAME measurement is what Lv3 asks for, so the boundary that
# decides which snapshot a run lands in is worth pinning down.

def test_period_is_the_iso_week():
    d = dt.datetime(2026, 10, 13, 12, 0, tzinfo=UTC)   # Tuesday
    assert cm.default_period(d) == "2026-W42"


def test_period_rolls_over_on_monday_not_sunday():
    sun = dt.datetime(2026, 10, 18, 23, 0, tzinfo=UTC)
    mon = dt.datetime(2026, 10, 19, 1, 0, tzinfo=UTC)
    assert cm.default_period(sun) == "2026-W42"
    assert cm.default_period(mon) == "2026-W43"


def test_period_week_is_zero_padded():
    # 2026-W5 would sort after 2026-W42 as a string; the pad is load-bearing.
    d = dt.datetime(2026, 2, 2, 12, 0, tzinfo=UTC)
    assert cm.default_period(d) == "2026-W06"
    assert "W06" in cm.default_period(d)


def test_period_uses_the_iso_year_not_the_calendar_year():
    # 2027-01-01 is a Friday that still belongs to ISO week 53 of 2026. Labelling it
    # by the calendar year gives "2027-W53", a week that does not exist, and the
    # period would sort away from the snapshot taken three days earlier.
    d = dt.datetime(2027, 1, 1, 12, 0, tzinfo=UTC)
    assert cm.default_period(d) == "2026-W53"

    # ...and 2025-12-29, a Monday, is already ISO week 1 of 2026.
    d2 = dt.datetime(2025, 12, 29, 12, 0, tzinfo=UTC)
    assert cm.default_period(d2) == "2026-W01"


# --- the internal series ----------------------------------------------------

def test_internal_series_is_never_a_rate():
    """The external no-response RATE is meaningful; an internal one is not.

    Most first-party items are a maintainer opening and merging their own work, which
    never waited for anyone. Reporting that as a percentage produces a large number
    that reads as neglect. The snapshot must carry a count instead -- this test exists
    so a future edit cannot quietly reintroduce the rate.
    """
    src = (pathlib.Path(__file__).resolve().parent / "collect_metrics.py").read_text()
    assert "closed_without_an_outside_response" in src
    assert "M-102_no_response_rate_pct" not in src
