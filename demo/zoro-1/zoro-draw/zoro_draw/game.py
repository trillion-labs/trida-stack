"""Room state machine. No networking here: server.py feeds it messages and timer expiries.

Phases: lobby -> drawing -> (decoys, decoy mode only) -> judging -> reveal -> drawing ... -> final

Modes
  classic  everyone draws the same word; Zoro-1 chooses among the word + 6 card decoys (shown to the drawers).
           Points = 100 x P(word). Highest P in the round gets a bonus.
  decoy    everyone draws a different secret word. Then everyone sees the others' drawings and writes one
           decoy word per drawing, trying to steal Zoro-1's guess. Zoro-1 chooses among word + 3 card decoys +
           player decoys. Drawer gets 100 x P(word); each decoy author gets 100 x P(their decoy); writing the
           real word earns a "sharp eye" bonus instead (it is not added as an option).
"""
from __future__ import annotations

import asyncio
import random
import re
import secrets
import time
from dataclasses import asdict, dataclass

from .cards import DECKS, decoys as card_decoys, gloss
from .judge import Judge, Verdict

CLASSIC, DECOY = "classic", "decoy"
LOBBY, DRAWING, DECOYS, JUDGING, REVEAL, FINAL = "lobby", "drawing", "decoys", "judging", "reveal", "final"

MAX_PLAYERS = 8
MAX_IMAGE_CHARS = 600_000      # a 448x336 PNG data URL is far below this
TEXT_VOID = 0.85               # P(the answer is written out) above this voids the drawer's points
WIN_BONUS = 20
SHARP_EYE = 30
MAX_OPTIONS = 12
CLASSIC_DECOYS, DECOY_MODE_CARD_DECOYS = 6, 3


class GameError(Exception):
    pass


@dataclass
class Player:
    id: str
    name: str
    score: int = 0
    connected: bool = True


@dataclass
class Settings:
    mode: str = CLASSIC
    rounds: int = 3
    draw_secs: int = 60
    decoy_secs: int = 45
    deck: str = "words"       # cards.DECKS key, or "mix"


def clean_name(name) -> str:
    name = re.sub(r"\s+", " ", str(name or "")).strip()[:16]
    if not name:
        raise GameError("Pick a name.")
    return name


def clean_word(word) -> str:
    word = re.sub(r"\s+", " ", str(word or "")).strip().lower()[:24]
    if not word:
        raise GameError("Write a word.")
    return word


class BaseRoom:
    """What every game's room shares: players, host, timers. server.py talks to rooms only through:
    join / leave / handle(pid, msg) / advance() / expire() / grace() / needs_work() / run_work(zoro) / view(pid)."""
    kind = "base"

    def __init__(self, code: str, rng: random.Random | None = None):
        self.code = code
        self.rng = rng or random.Random()
        self.players: dict[str, Player] = {}
        self.host: str | None = None
        self.phase = LOBBY
        self.round = 0
        self.deadline: float | None = None
        self.last_active = time.time()
        self.notify = None  # async callable set by the server; background tasks call it after changing state

    def join(self, name, pid: str | None = None) -> Player:
        name = clean_name(name)
        if pid and pid in self.players:
            p = self.players[pid]
            p.connected, p.name = True, name
        else:
            if len(self.players) >= MAX_PLAYERS:
                raise GameError(f"Room {self.code} is full ({MAX_PLAYERS} players).")
            p = Player(id=secrets.token_urlsafe(9), name=name)
            self.players[p.id] = p
        if self.host is None or not self.players[self.host].connected:
            self.host = p.id
        self.touch()
        return p

    def leave(self, pid: str):
        if pid in self.players:
            self.players[pid].connected = False
            if self.host == pid:
                self.host = next((q.id for q in self.players.values() if q.connected), pid)
        self.touch()

    def touch(self):
        self.last_active = time.time()

    def _host_only(self, pid):
        if pid != self.host:
            raise GameError("Only the host can do that.")

    def _active(self, ids):
        return [i for i in ids if self.players[i].connected]

    # defaults for the server interface
    def advance(self):
        pass

    def expire(self):
        pass

    def grace(self) -> float:
        return 1.5

    def needs_work(self) -> bool:
        return False

    async def run_work(self, zoro):
        pass


class Room(BaseRoom):
    """Zoro Draw."""
    kind = "draw"

    def __init__(self, code: str, rng: random.Random | None = None):
        super().__init__(code, rng)
        self.settings = Settings()
        self.used: set[str] = set()
        self.participants: list[str] = []
        self.topics: dict[str, str] = {}             # drawer -> word
        self.candidates: list[str] = []              # classic: shared, shuffled once per round
        self.drawings: dict[str, str] = {}           # drawer -> PNG data URL
        self.decoys: dict[str, dict[str, str]] = {}  # author -> {drawer: word}
        self.results: list[dict] = []
        self.round_points: dict[str, int] = {}

    # ---- server interface ---------------------------------------------------------------------------------
    def handle(self, pid, msg: dict):
        t = msg.get("t")
        if t == "settings":
            self.configure(pid, **{k: msg.get(k) for k in ("mode", "rounds", "draw_secs", "decoy_secs", "deck")})
        elif t == "start":
            self.start(pid)
        elif t == "drawing":
            self.submit_drawing(pid, msg.get("image"))
        elif t == "decoy":
            self.submit_decoy(pid, msg.get("drawer"), msg.get("word"))
        elif t == "next":
            self.next(pid)

    def advance(self):
        if self.phase == DRAWING and self.drawing_done():
            self.end_drawing()
        if self.phase == DECOYS and self.decoys_done():
            self.end_decoys()

    def expire(self):
        self.end_drawing() if self.phase == DRAWING else self.end_decoys()

    def grace(self) -> float:
        return 2.5 if self.phase == DRAWING else 1.5

    def needs_work(self) -> bool:
        return self.phase == JUDGING

    async def run_work(self, zoro):
        await self.run_judging(zoro)

    # ---- lobby --------------------------------------------------------------------------------------------
    def configure(self, pid, mode=None, rounds=None, draw_secs=None, decoy_secs=None, deck=None):
        self._host_only(pid)
        if self.phase not in (LOBBY, FINAL):
            raise GameError("Settings can only change between games.")
        s = self.settings
        if mode is not None:
            if mode not in (CLASSIC, DECOY):
                raise GameError("Unknown mode.")
            s.mode = mode
        if rounds is not None:
            s.rounds = max(1, min(10, int(rounds)))
        if draw_secs is not None:
            s.draw_secs = max(15, min(180, int(draw_secs)))
        if decoy_secs is not None:
            s.decoy_secs = max(15, min(120, int(decoy_secs)))
        if deck is not None:
            if deck != "mix" and deck not in DECKS:
                raise GameError("Unknown deck.")
            s.deck = deck

    def start(self, pid):
        self._host_only(pid)
        if self.phase not in (LOBBY, FINAL):
            raise GameError("The game is already running.")
        for p in self.players.values():
            p.score = 0
        self.round = 0
        self.start_round()

    def _pick(self, n: int) -> list[str]:
        deck = self.settings.deck
        if deck == "mix":   # one deck per round, so a round's cards and decoys come from the same deck
            deck = self.rng.choice(list(DECKS))
        pool = list(DECKS[deck]["cards"])
        free = [w for w in pool if w not in self.used]
        if len(free) < n:
            self.used -= set(pool)
            free = pool
        words = self.rng.sample(free, n)
        self.used.update(words)
        return words

    def start_round(self):
        self.round += 1
        self.participants = self._active(self.players)
        self.drawings, self.decoys, self.results, self.round_points = {}, {}, [], {}
        if self.settings.mode == CLASSIC:
            word = self._pick(1)[0]
            self.topics = {pid: word for pid in self.participants}
            self.candidates = [word, *card_decoys(word, CLASSIC_DECOYS, self.rng)]
            self.rng.shuffle(self.candidates)
        else:
            words = self._pick(len(self.participants))
            self.topics = dict(zip(self.participants, words))
            self.candidates = []
        self.phase = DRAWING
        self.deadline = time.time() + self.settings.draw_secs

    # ---- drawing ------------------------------------------------------------------------------------------
    def submit_drawing(self, pid, image):
        if self.phase != DRAWING:
            raise GameError("Drawing time is over.")
        if pid not in self.participants:
            raise GameError("You join from the next round.")
        if not isinstance(image, str) or not image.startswith("data:image/png;base64,"):
            raise GameError("The drawing must be a PNG data URL.")
        if len(image) > MAX_IMAGE_CHARS:
            raise GameError("The drawing is too large.")
        self.drawings[pid] = image
        self.touch()

    def drawing_done(self) -> bool:
        return all(p in self.drawings for p in self._active(self.participants))

    def end_drawing(self):
        if self.phase != DRAWING:
            return
        if self.settings.mode == DECOY and len(self.drawings) >= 1 and len(self.participants) >= 2:
            self.phase = DECOYS
            self.deadline = time.time() + self.settings.decoy_secs
        else:
            self.phase, self.deadline = JUDGING, None

    # ---- decoys -------------------------------------------------------------------------------------------
    def submit_decoy(self, pid, drawer, word):
        if self.phase != DECOYS:
            raise GameError("Decoy time is over.")
        if pid not in self.participants:
            raise GameError("You join from the next round.")
        if drawer == pid or drawer not in self.drawings:
            raise GameError("Not a drawing you can write a decoy for.")
        self.decoys.setdefault(pid, {})[drawer] = clean_word(word)
        self.touch()

    def decoys_done(self) -> bool:
        for pid in self._active(self.participants):
            mine = self.decoys.get(pid, {})
            if any(d not in mine for d in self.drawings if d != pid):
                return False
        return True

    def end_decoys(self):
        if self.phase == DECOYS:
            self.phase, self.deadline = JUDGING, None

    # ---- judging ------------------------------------------------------------------------------------------
    def jobs(self) -> list[dict]:
        """One model call per drawing: {drawer, truth, options: [{word, kind, authors}], sharp: [pid]}."""
        out = []
        for drawer, image in self.drawings.items():
            truth = self.topics[drawer]
            if self.settings.mode == CLASSIC:
                options = [{"word": w, "kind": "truth" if w == truth else "card", "authors": []}
                           for w in self.candidates]
                out.append({"drawer": drawer, "image": image, "truth": truth, "options": options, "sharp": []})
                continue
            options = [{"word": truth, "kind": "truth", "authors": []}]
            options += [{"word": w, "kind": "card", "authors": []} for w in card_decoys(truth, DECOY_MODE_CARD_DECOYS, self.rng)]
            by_word = {o["word"]: o for o in options}
            sharp = []
            for author, words in self.decoys.items():
                w = words.get(drawer)
                if not w:
                    continue
                if w == truth:
                    sharp.append(author)
                elif w in by_word:
                    by_word[w]["authors"].append(author)
                elif len(options) < MAX_OPTIONS:
                    by_word[w] = {"word": w, "kind": "player", "authors": [author]}
                    options.append(by_word[w])
            self.rng.shuffle(options)
            out.append({"drawer": drawer, "image": image, "truth": truth, "options": options, "sharp": sharp})
        return out

    async def run_judging(self, judge: Judge):
        jobs = self.jobs()
        verdicts = await asyncio.gather(*(
            judge.guess(j["image"], [o["word"] for o in j["options"]], truth=j["truth"],
                        glosses={o["word"]: gloss(o["word"]) for o in j["options"] if gloss(o["word"])}) for j in jobs))
        self.apply(jobs, verdicts)

    def apply(self, jobs: list[dict], verdicts: list[Verdict]):
        points: dict[str, int] = {}

        def add(pid, n):
            points[pid] = points.get(pid, 0) + n

        results = []
        for job, v in zip(jobs, verdicts):
            voided = not v.error and v.text_p > TEXT_VOID
            opts = [{**o, "p": v.probs.get(o["word"], 0.0)} for o in job["options"]]
            opts.sort(key=lambda o: -o["p"])
            p_truth = v.probs.get(job["truth"], 0.0)
            r = {"drawer": job["drawer"], "image": job["image"], "truth": job["truth"], "options": opts,
                 "p_truth": p_truth, "top": opts[0]["word"] if opts and not v.error else None,
                 "text_p": v.text_p, "voided": voided, "error": v.error, "ms": v.ms,
                 "sharp": job["sharp"], "points": {}, "bonus": False}
            if not v.error:
                r["points"][job["drawer"]] = 0 if voided else round(100 * p_truth)
                for o in opts:
                    for a in o["authors"]:
                        r["points"][a] = r["points"].get(a, 0) + round(100 * o["p"])
            for a in job["sharp"]:
                r["points"][a] = r["points"].get(a, 0) + SHARP_EYE
            results.append(r)

        if self.settings.mode == CLASSIC:
            ok = [r for r in results if not r["error"] and not r["voided"]]
            if ok:
                best = max(r["p_truth"] for r in ok)
                for r in ok:
                    if r["p_truth"] >= best - 1e-9:
                        r["bonus"] = True
                        r["points"][r["drawer"]] += WIN_BONUS

        for r in results:
            for pid, n in r["points"].items():
                add(pid, n)
        for pid, n in points.items():
            if pid in self.players:
                self.players[pid].score += n
        results.sort(key=lambda r: -r["p_truth"])
        self.results, self.round_points = results, points
        self.phase, self.deadline = REVEAL, None

    def next(self, pid):
        self._host_only(pid)
        if self.phase != REVEAL:
            raise GameError("Nothing to advance.")
        if self.round >= self.settings.rounds:
            self.phase = FINAL
        else:
            self.start_round()

    # ---- what one player may see --------------------------------------------------------------------------
    def view(self, pid: str) -> dict:
        def done(q):
            if self.phase == DRAWING:
                return q in self.drawings
            if self.phase == DECOYS:
                mine = self.decoys.get(q, {})
                return all(d in mine for d in self.drawings if d != q)
            return False

        v = {
            "game": self.kind, "code": self.code, "phase": self.phase, "round": self.round, "me": pid, "host": self.host,
            "deadline": self.deadline, "now": time.time(), "settings": asdict(self.settings),
            "decks": [{"id": k, "label": d["label"]} for k, d in DECKS.items()],
            "players": [{"id": q.id, "name": q.name, "score": q.score, "connected": q.connected,
                         "playing": q.id in self.participants, "done": done(q.id)}
                        for q in self.players.values()],
        }
        if self.phase == DRAWING and pid in self.participants:
            v["topic"] = self.topics[pid]
            v["topic_gloss"] = gloss(self.topics[pid])
            v["candidates"] = self.candidates  # classic only; empty in decoy mode
            v["submitted"] = pid in self.drawings
        if self.phase == DECOYS and pid in self.participants:
            mine = self.decoys.get(pid, {})
            v["tasks"] = [{"drawer": d, "image": img, "word": mine.get(d)}
                          for d, img in self.drawings.items() if d != pid]
            v["topic"] = self.topics[pid]
        if self.phase == JUDGING:
            v["judging"] = len(self.drawings)
        if self.phase in (REVEAL, FINAL):
            v["results"] = self.results
            v["round_points"] = self.round_points
        return v
