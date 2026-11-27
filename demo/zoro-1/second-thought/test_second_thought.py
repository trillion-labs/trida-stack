"""Mock-mode tests for second_thought.ipynb: the notebook's own code, no model, no network.

    PYTHONPATH=../../../inference/zoro-1 uv run --no-project --with pytest --with httpx --with matplotlib pytest -q

PYTHONPATH puts Zoro-1's request parser on the path; without it the parser test is skipped.
"""
import json
import os
import random
from pathlib import Path

import pytest

NB = Path(__file__).with_name("second_thought.ipynb")


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["\n".join(line for line in "".join(c["source"]).splitlines() if not line.lstrip().startswith("%"))
            for c in nb["cells"] if c["cell_type"] == "code"]


def run(cells):
    os.environ.update(ZORO_MOCK="1", MPLBACKEND="Agg", ZORO_CONCURRENCY="1")
    ns = {"__name__": "second_thought"}
    for src in cells:
        exec(compile(src, str(NB), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def nb():
    """Definitions only: setup, road, client, reflex, second look, head (stops before the loop)."""
    cells = code_cells()
    loop = next(i for i, c in enumerate(cells) if "for rnd in range" in c)
    return run(cells[:loop])


def scene(nb, **kw):
    s = {"road": "two-lane city street, you drive in the right lane", "speed_limit_kmh": 50, "your_speed_kmh": 50,
         "car_ahead": "none", "left_lane": "free", "ahead": "clear road", "passenger_says": nb["CALM"][0]}
    s.update(kw)
    return s


def test_physics(nb):
    red = scene(nb, ahead={"traffic_light": "red", "distance_m": 25})
    assert nb["rollout"](red, "keep")["why"] == "runs a red light"
    assert nb["rollout"](red, "brake")["safe"]
    stopped = scene(nb, car_ahead={"distance_m": 10, "speed_kmh": 0, "brake_lights": "on"})
    assert nb["rollout"](stopped, "keep")["why"] == "hits the car ahead"
    assert nb["rollout"](scene(nb, left_lane="a car is right next to you"), "change_left")["why"] == \
        "cuts into the car next to you"
    assert nb["rollout"](scene(nb, your_speed_kmh=48), "speed_up")["why"] == "breaks the speed limit"


def test_raw_head_is_the_reflex(nb):
    r = nb["reflex"](nb["make_scene"](random.Random(0)))
    p = nb["predict"](nb["raw_head"](), nb["features"](r))
    assert all(abs(p[m] - r["move"][m]) < 1e-6 for m in nb["MOVES"])


def test_second_look_label_is_safe(nb):
    rng = random.Random(1)
    for _ in range(50):
        s = nb["make_scene"](rng)
        label = nb["second_look"](s)["label"]
        assert nb["rollout"](s, label)["safe"] or not any(nb["rollout"](s, m)["safe"] for m in nb["MOVES"])


def test_requests_pass_zoro_parser(nb, monkeypatch):
    dr = pytest.importorskip("decide_request")
    sent = []

    def checked(state, questions):
        req, _ = dr.systemone_request({"model": "zoro-1", "state": state, "questions": questions, "reasoning": False})
        dr.parse(req)
        sent.append(questions)
        return mock(state, questions)

    mock = nb["_mock_decide"]
    monkeypatch.setitem(nb, "_mock_decide", checked)
    rng = random.Random(2)
    for _ in range(30):
        s = nb["make_scene"](rng)
        nb["reflex"](s)
        nb["second_look"](s)
    assert any("plan" in q for q in sent) and any("move" in q for q in sent)


def test_whole_notebook_runs_in_mock_mode():
    ns = run(code_cells())   # includes the notebook's own check cell
    raw, learned = ns["results"]["raw Zoro-1 reflex"], ns["results"]["learned head"]
    assert learned["agrees with second look"] > raw["agrees with second look"]
    assert learned["would look again"] < raw["would look again"]
