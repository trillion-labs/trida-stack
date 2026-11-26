import asyncio
import base64
import random
import struct
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient

from zoro_draw import judge as J
from zoro_draw.game import CLASSIC, DECOY, DECOYS, DRAWING, FINAL, JUDGING, REVEAL, SHARP_EYE, WIN_BONUS, GameError, Room
from zoro_draw.judge import Judge, Verdict
from zoro_draw.server import RateLimit, create_app


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(seed: int = 0, width: int = 448, height: int = 336, *, declared: tuple[int, int] | None = None) -> str:
    """A real grayscale PNG data URL (the page sends 448x336). `seed` varies one pixel so drawings differ;
    `declared` writes other dimensions into the header than the pixels have (a decompression bomb's trick)."""
    rows = b"".join(b"\x00" + bytes([seed % 256 if y == 0 else 255]) + b"\xff" * (width - 1) for y in range(height))
    w, h = declared or (width, height)
    raw = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(raw).decode()


PNG = png()


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
    dr.parse(req)   # systemone_request only maps the envelope; parse() checks the questions, criteria and image


async def test_classic_round_scoring():
    r, (a, b) = room_with(2)
    assert r.phase == DRAWING and r.topics[a.id] == r.topics[b.id]
    assert r.topics[a.id] in r.candidates and len(r.candidates) == 7
    r.submit_drawing(a.id, PNG)
    assert not r.drawing_done()
    r.submit_drawing(b.id, png(1))
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
    for i, p in enumerate((a, b, c)):
        r.submit_drawing(p.id, png(i))
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
            assert hello["t"] == "welcome" and hello["token"] and hello["token"] != hello["pid"]
            code = hello["room"]
            assert w1.receive_json()["state"]["model"] == "mock"
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
                w2.send_json({"t": "drawing", "image": png(2)})
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
    dr.parse(req)


# ---- review fixes: each test fails on the code before them -----------------------------------------------------

def test_a_public_id_does_not_take_a_seat():
    r = Room("TEST")
    host = r.join("host")
    thief = r.join("thief", host.id)           # every player sees host.id in the room state
    assert thief.id != host.id and r.host == host.id
    again = r.join("host", host.token)         # only the secret token takes the seat back
    assert again is host


def test_drawings_must_be_small_real_pngs():
    r, (a,) = room_with(1)
    r.submit_drawing(a.id, PNG)
    for bad in ("data:image/png;base64,not base64!",                       # not base64
                "data:image/png;base64," + base64.b64encode(b"GIF89a" + b"\0" * 30).decode(),   # not a PNG
                png(declared=(20000, 20000)),                              # tiny file, huge declared image
                png(width=449, height=10)):                                # over the model's cap
        with pytest.raises(GameError):
            r.submit_drawing(a.id, bad)


def test_unknown_messages_and_bad_numbers_are_errors():
    r, (a,) = room_with(1)
    with pytest.raises(GameError):
        r.handle(a.id, {"t": "x"})             # no broadcast for a message that changes nothing
    r2 = Room("TEST")
    h = r2.join("h")
    for v in (float("inf"), float("nan"), "3", True):
        with pytest.raises(GameError):
            r2.configure(h.id, rounds=v)


class SlowJudge(FixedJudge):
    timeout = 0.05

    async def guess(self, image, cands, truth=None, glosses=None):
        await asyncio.sleep(5)


async def test_a_slow_model_cannot_hold_a_round():
    r, (a,) = room_with(1)
    r.submit_drawing(a.id, PNG); r.end_drawing()
    await asyncio.wait_for(r.run_judging(SlowJudge()), 2)
    assert r.phase == REVEAL and r.results[0]["error"] == J.TOO_SLOW and a.score == 0


async def test_the_model_url_and_errors_stay_on_the_server():
    url = "http://10.9.8.7:30001"
    judge = Judge(url)
    judge.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(500, text=f"internal error at {url}")))
    v = await judge.guess(PNG, ["cat", "dog"])
    assert judge.mode == "zoro-1" and v.error == J.NOT_JUDGED and url not in v.error
    judge.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    assert url not in str(await judge.health())
    await judge.aclose()


def test_a_failed_hello_leaves_no_room_and_bad_messages_keep_the_socket():
    app = create_app(Judge(mock=True))
    with TestClient(app) as tc, tc.websocket_connect("/ws") as w:
        for _ in range(3):
            w.send_json({"t": "hello", "name": "  "})
            assert w.receive_json()["t"] == "error"
        assert app.state.hub.rooms == {}
        w.send_json({"t": "hello", "name": "Host"}); w.receive_json(); w.receive_json()
        w.send_json({"t": "settings", "rounds": 1e400})   # json: Infinity
        assert w.receive_json()["t"] == "error"
        w.send_json({"t": "settings", "rounds": 2})       # still connected
        assert w.receive_json()["state"]["settings"]["rounds"] == 2


def test_rate_limit():
    limit = RateLimit(rate=0.0, burst=2)
    assert limit.allow() and limit.allow() and not limit.allow()


def test_pages_are_not_cached():
    with TestClient(create_app(Judge(mock=True))) as tc:
        assert tc.get("/draw").headers["cache-control"] == "no-store"
        assert tc.get("/static/app.js").headers["cache-control"] == "no-cache"
        assert 'id="deck"' in tc.get("/draw").text


# ---- review follow-ups (PR #115) -------------------------------------------------------------------------------
async def test_decoys_match_ignoring_case_spaces_and_punctuation():
    from zoro_draw.game import match_key
    assert match_key("Snow man!") == match_key("snowman") and match_key("우물 안  개구리") == match_key("우물안개구리")
    r, (a, b, c) = room_with(3, DECOY)
    for i, p in enumerate((a, b, c)):
        r.submit_drawing(p.id, png(i))
    r.end_drawing()
    truth = r.topics[a.id]
    spaced = " ".join(truth.replace(" ", "")).upper() + "!"         # same letters, different spacing, case, punctuation
    r.submit_decoy(b.id, a.id, spaced)
    r.end_decoys()
    ja = next(j for j in r.jobs() if j["drawer"] == a.id)
    assert ja["sharp"] == [b.id]                                     # counts as the real word
    assert [o["word"] for o in ja["options"]].count(truth) == 1      # and is not added as a second option
    card = next(o for o in ja["options"] if o["kind"] == "card")
    r.decoys = {c.id: {a.id: card["word"].upper().replace(" ", "")}}
    jc = next(j for j in r.jobs() if j["drawer"] == a.id)
    assert c.id in next(o for o in jc["options"] if match_key(o["word"]) == match_key(card["word"]))["authors"]


def test_full_proverb_fits_in_a_decoy():
    from zoro_draw.cards import PROVERBS
    from zoro_draw.game import clean_word
    longest = max(PROVERBS, key=len)
    assert clean_word(longest) == longest.lower()


def test_player_decoys_get_no_card_description():
    from zoro_draw.cards import PROVERBS
    from zoro_draw.game import card_glosses
    proverb = next(iter(PROVERBS))
    options = [{"word": "lighthouse", "kind": "truth"}, {"word": proverb, "kind": "player"}]
    assert proverb not in card_glosses(options)
    assert card_glosses([{"word": proverb, "kind": "card"}]) == {proverb: PROVERBS[proverb]}


async def test_play_again_is_a_new_game_for_the_page():
    r, (a,) = room_with(1)
    first = r.view(a.id)["game_no"]
    for _ in range(2):
        r.submit_drawing(a.id, PNG); r.end_drawing()
        await r.run_judging(FixedJudge())
        r.next(a.id)
    r.start(a.id)
    v = r.view(a.id)
    assert v["round"] == 1 and v["game_no"] == first + 1    # the page keys its canvas on (game_no, round)


def test_cli_defaults_are_safe(monkeypatch):
    import zoro_draw.server as S
    seen = {}
    monkeypatch.setattr("sys.argv", ["zoro-draw", "--mock"])
    monkeypatch.setattr(S, "create_app", lambda judge, *a, **k: seen.setdefault("judge", judge))
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **k: seen.update(host=host, port=port))
    S.main()
    assert seen["host"] == "127.0.0.1"
    assert seen["judge"].sem._value == 1
