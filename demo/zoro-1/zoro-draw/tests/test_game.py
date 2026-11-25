import base64
import random

import pytest
from fastapi.testclient import TestClient

from zoro_draw import judge as J
from zoro_draw.game import CLASSIC, DECOY, DECOYS, DRAWING, FINAL, JUDGING, REVEAL, SHARP_EYE, WIN_BONUS, GameError, Room
from zoro_draw.judge import Judge, Verdict
from zoro_draw.server import create_app

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()


def room_with(n, mode=CLASSIC):
    r = Room("TEST", rng=random.Random(0))
    ps = [r.join(f"p{i}") for i in range(n)]
    r.configure(ps[0].id, mode=mode, rounds=2)
    r.start(ps[0].id)
    return r, ps


class FixedJudge:
    """Puts `p` on the truth (spreading the rest), or on a named word."""
    def __init__(self, p_truth=None, favourite=None, text_p=0.0):
        self.p_truth, self.favourite, self.text_p = p_truth, favourite, text_p

    async def guess(self, image, cands, truth=None, glosses=None):
        fav = self.favourite or truth
        p = self.p_truth if self.p_truth is not None else 0.9
        rest = (1 - p) / (len(cands) - 1)
        return Verdict(probs={w: (p if w == fav else rest) for w in cands}, text_p=self.text_p)


def test_systemone_body_and_parse():
    body = J.systemone_body("zoro-1", PNG, ["cat", "dog"])
    assert body["images"] == [PNG]
    assert body["questions"]["guess"]["criteria"] == {"cat": "", "dog": ""}
    assert body["questions"]["has_text"]["type"] == "noul"
    assert set(body["questions"]["has_text"]["criteria"]) == {"true", "false"}   # noul criteria keys
    v = J.parse_response({"answers": {"guess": {"type": "choice", "choice": "dog", "probabilities": {"cat": 0.2, "dog": 0.6}},
                                      "has_text": {"type": "noul", "noul": 0.1}}}, ["cat", "dog"])
    assert v.probs["dog"] == pytest.approx(0.75) and v.text_p == 0.1


def test_body_passes_server_validation():
    """The body must be accepted by the real server-side parser (trida-stack/inference/zoro-1/decide_request.py)."""
    dr = pytest.importorskip("decide_request")
    req, opts = dr.systemone_request(J.systemone_body("zoro-1", PNG, ["cat", "dog"]))
    assert req["image"] == PNG and opts == {"reasoning": False}


async def test_classic_round_scoring():
    r, (a, b) = room_with(2)
    assert r.phase == DRAWING and r.topics[a.id] == r.topics[b.id]
    assert r.topics[a.id] in r.candidates and len(r.candidates) == 7
    r.submit_drawing(a.id, PNG)
    assert not r.drawing_done()
    r.submit_drawing(b.id, PNG + "x")
    assert r.drawing_done()
    r.end_drawing()
    assert r.phase == JUDGING
    await r.run_judging(FixedJudge(p_truth=0.8))
    assert r.phase == REVEAL
    assert a.score == b.score == 80 + WIN_BONUS  # tie: both get the bonus


async def test_text_voids_points():
    r, (a,) = room_with(1)
    r.submit_drawing(a.id, PNG)
    r.end_drawing()
    await r.run_judging(FixedJudge(p_truth=0.9, text_p=0.95))
    assert a.score == 0 and r.results[0]["voided"]


async def test_decoy_round():
    r, (a, b, c) = room_with(3, DECOY)
    assert len(set(r.topics.values())) == 3 and r.candidates == []
    for p in (a, b, c):
        r.submit_drawing(p.id, PNG + p.id)
    r.end_drawing()
    assert r.phase == DECOYS
    with pytest.raises(GameError):
        r.submit_decoy(a.id, a.id, "x")  # not for your own drawing
    r.submit_decoy(b.id, a.id, "  Space   Rocket ")
    r.submit_decoy(c.id, a.id, r.topics[a.id].upper())  # the real word -> sharp eye
    r.submit_decoy(a.id, b.id, "w1"); r.submit_decoy(c.id, b.id, "w2")
    assert not r.decoys_done()
    r.submit_decoy(a.id, c.id, "w3"); r.submit_decoy(b.id, c.id, "w4")
    assert r.decoys_done()
    # the drawer view never leaks other players' words during decoys
    assert all("truth" not in t for t in r.view(a.id)["tasks"])
    r.end_decoys()
    jobs = r.jobs()
    ja = next(j for j in jobs if j["drawer"] == a.id)
    words = [o["word"] for o in ja["options"]]
    assert "space rocket" in words and words.count(r.topics[a.id]) == 1 and ja["sharp"] == [c.id]
    await r.run_judging(FixedJudge(p_truth=0.5, favourite="space rocket"))
    # b's decoy took 0.5 of a's drawing
    assert r.round_points[b.id] >= 50 and r.round_points[c.id] >= SHARP_EYE


async def test_rounds_to_final_and_replay():
    r, (a, b) = room_with(2)
    for _ in range(2):
        r.submit_drawing(a.id, PNG); r.end_drawing()
        await r.run_judging(FixedJudge())
        with pytest.raises(GameError):
            r.next(b.id)  # not host
        r.next(a.id)
    assert r.phase == FINAL
    r.start(a.id)
    assert r.round == 1 and a.score == 0


def test_mock_judge_is_deterministic():
    v1, v2 = J._mock(PNG, ["a", "b", "c"], "a"), J._mock(PNG, ["a", "b", "c"], "a")
    assert v1.probs == v2.probs and sum(v1.probs.values()) == pytest.approx(1)


def test_websocket_game_flow():
    app = create_app(Judge(mock=True))
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws") as w1:
            w1.send_json({"t": "hello", "name": "Host"})
            hello = w1.receive_json()
            assert hello["t"] == "welcome"
            code = hello["room"]
            w1.receive_json()
            with tc.websocket_connect("/ws") as w2:
                w2.send_json({"t": "hello", "name": "Guest", "room": code})
                assert w2.receive_json()["t"] == "welcome"
                w1.receive_json(); w2.receive_json()
                w2.send_json({"t": "start"})
                assert w2.receive_json()["t"] == "error"
                w1.send_json({"t": "start"})
                s = w1.receive_json()["state"]; w2.receive_json()
                assert s["phase"] == "drawing" and s["topic"] in s["candidates"]
                w1.send_json({"t": "drawing", "image": PNG}); w1.receive_json(); w2.receive_json()
                w2.send_json({"t": "drawing", "image": PNG + "y"})
                # judging, then reveal
                seen = []
                while "reveal" not in seen:
                    seen.append(w1.receive_json()["state"]["phase"])
                assert "judging" in seen
                assert tc.get("/api/health").json()["model"]["mode"] == "mock"
                assert tc.get("/").status_code == 200 and tc.get("/static/app.js").status_code == 200


class RecordingJudge(FixedJudge):
    def __init__(self):
        super().__init__(p_truth=0.6)
        self.calls = []

    async def guess(self, image, cands, truth=None, glosses=None):
        self.calls.append((cands, glosses))
        return await super().guess(image, cands, truth)


async def test_proverb_deck_glosses_reach_zoro():
    from zoro_draw.cards import DECKS, PROVERBS
    r = Room("TEST", rng=random.Random(1))
    a = r.join("a")
    r.configure(a.id, deck="proverbs")
    r.start(a.id)
    assert r.topics[a.id] in PROVERBS and all(c in PROVERBS for c in r.candidates) and len(r.candidates) == 7
    assert r.view(a.id)["topic_gloss"] == PROVERBS[r.topics[a.id]]
    r.submit_drawing(a.id, PNG); r.end_drawing()
    j = RecordingJudge()
    await r.run_judging(j)
    cands, glosses = j.calls[0]
    assert set(glosses) == set(cands) and glosses[r.topics[a.id]] == PROVERBS[r.topics[a.id]]
    body = J.systemone_body("zoro-1", PNG, cands, glosses)
    assert body["questions"]["guess"]["criteria"][cands[0]] == PROVERBS[cands[0]]
    assert {d for d in DECKS} >= {"words", "proverbs", "ko_memes", "memes"}


async def test_mix_and_decoy_mode_decks():
    from zoro_draw.cards import DECK_OF
    r = Room("TEST", rng=random.Random(2))
    ps = [r.join(f"p{i}") for i in range(3)]
    r.configure(ps[0].id, deck="mix", mode=DECOY)
    r.start(ps[0].id)
    decks = {DECK_OF[w] for w in r.topics.values()}
    assert len(decks) == 1          # one deck per round
    with pytest.raises(GameError):
        r.configure(ps[0].id, deck="nope")


def test_deck_body_passes_server_validation():
    dr = pytest.importorskip("decide_request")
    from zoro_draw.cards import PROVERBS
    cands = list(PROVERBS)[:7]
    req, _ = dr.systemone_request(J.systemone_body("zoro-1", PNG, cands, {c: PROVERBS[c] for c in cands}))
    assert req["questions"]["guess"]["criteria"][cands[0]] == PROVERBS[cands[0]]


def test_pages_are_not_cached():
    with TestClient(create_app(Judge(mock=True))) as tc:
        assert tc.get("/draw").headers["cache-control"] == "no-store"
        assert tc.get("/static/app.js").headers["cache-control"] == "no-cache"
        assert 'id="deck"' in tc.get("/draw").text
