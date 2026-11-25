"""Zoro Draw web server: one page + one websocket per player. The Zoro-1 server stays private behind this.

  uv run zoro-draw --mock                                              # play without a model
  uv run zoro-draw --model-url http://127.0.0.1:8080 --concurrency 1  # Zoro-1 on your Mac (MLX server)
  uv run zoro-draw --model-url http://GPU_HOST:30001                   # Zoro-1 on a GPU (serve_decide.py, SGLang)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import secrets
import string
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .game import GameError, Room
from .judge import Judge

STATIC = pathlib.Path(__file__).parent / "static"
MAX_ROOMS = 500
IDLE_ROOM_SECS = 15 * 60
CODE_LETTERS = string.ascii_uppercase.replace("O", "").replace("I", "")


class Hub:
    """All rooms, their sockets and phase timers. Rooms implement the interface documented in game.BaseRoom."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self.rooms: dict[str, Room] = {}
        self.sockets: dict[str, dict[str, WebSocket]] = {}
        self.timers: dict[str, tuple[tuple, asyncio.Task]] = {}
        self.working: set[str] = set()

    def new_room(self) -> Room:
        if len(self.rooms) >= MAX_ROOMS:
            raise GameError("The server is full, try again later.")
        while True:
            code = "".join(secrets.choice(CODE_LETTERS) for _ in range(4))
            if code not in self.rooms:
                break
        room = Room(code)
        room.zoro, room.notify = self.judge, self.drive
        self.rooms[code] = room
        self.sockets[code] = {}
        return room

    async def send(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_json(msg)
        except Exception:  # noqa: BLE001 - a dead socket is cleaned up by its own receive loop
            pass

    async def broadcast(self, room: Room):
        socks = list(self.sockets.get(room.code, {}).items())
        await asyncio.gather(*(self.send(ws, {"t": "state", "state": {**room.view(pid), "model": self.judge.mode}})
                               for pid, ws in socks))

    async def drive(self, room: Room):
        """Advance whatever can advance, start background work (judging), (re)arm the phase timer, broadcast."""
        room.advance()
        if room.needs_work() and room.code not in self.working:
            self.working.add(room.code)
            asyncio.create_task(self._work(room))
        self._arm(room)
        await self.broadcast(room)

    async def _work(self, room: Room):
        try:
            await room.run_work(self.judge)
        finally:
            self.working.discard(room.code)
        await self.drive(room)

    def _arm(self, room: Room):
        key = (room.phase, room.round, room.deadline)
        old = self.timers.get(room.code)
        if old and old[0] == key:
            return
        if old:
            old[1].cancel()
            self.timers.pop(room.code, None)
        if room.deadline:
            self.timers[room.code] = (key, asyncio.create_task(self._expire(room, key)))

    async def _expire(self, room: Room, key):
        await asyncio.sleep(max(0.0, room.deadline - time.time()) + room.grace())
        if (room.phase, room.round, room.deadline) != key:
            return
        self.timers.pop(room.code, None)
        room.expire()
        await self.drive(room)

    async def sweep(self):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            for code, room in list(self.rooms.items()):
                if not self.sockets.get(code) and now - room.last_active > IDLE_ROOM_SECS:
                    self.rooms.pop(code, None)
                    self.sockets.pop(code, None)
                    t = self.timers.pop(code, None)
                    if t:
                        t[1].cancel()


def create_app(judge: Judge) -> FastAPI:
    hub = Hub(judge)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        sweeper = asyncio.create_task(hub.sweep())
        yield
        sweeper.cancel()
        await judge.aclose()

    app = FastAPI(title="Zoro Draw", lifespan=lifespan)
    app.state.hub = hub

    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    app.add_api_route("/", index, methods=["GET", "HEAD"])
    app.add_api_route("/draw", index, methods=["GET", "HEAD"])   # old links

    @app.middleware("http")
    async def no_stale_assets(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    async def health():
        return {"rooms": len(hub.rooms), "model": await judge.health()}

    @app.get("/api/room/{code}")
    def room_info(code: str):
        room = hub.rooms.get(code.strip().upper())
        if room is None:
            raise HTTPException(404, f"No room {code.upper()}.")
        return {"code": room.code, "players": len(room.players), "phase": room.phase}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        room: Room | None = None
        pid: str | None = None
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    continue
                try:
                    if msg.get("t") == "hello":
                        if room is not None:
                            raise GameError("Already in a room.")
                        code = str(msg.get("room") or "").strip().upper()
                        if code and code not in hub.rooms:
                            raise GameError(f"No room {code}.")
                        r = hub.rooms[code] if code else hub.new_room()
                        p = r.join(msg.get("name"), msg.get("pid"))
                        room, pid = r, p.id
                        old = hub.sockets[room.code].get(pid)
                        hub.sockets[room.code][pid] = ws
                        if old is not None and old is not ws:
                            await hub.send(old, {"t": "error", "msg": "Opened in another tab."})
                            with contextlib.suppress(Exception):
                                await old.close()
                        await hub.send(ws, {"t": "welcome", "pid": pid, "room": room.code, "game": "draw"})
                    elif room is None:
                        raise GameError("Join a room first.")
                    else:
                        room.handle(pid, msg)
                except (GameError, ValueError, TypeError) as e:
                    await hub.send(ws, {"t": "error", "msg": str(e)})
                    continue
                await hub.drive(room)
        except WebSocketDisconnect:
            pass
        finally:
            if room is not None and pid is not None and hub.sockets.get(room.code, {}).get(pid) is ws:
                hub.sockets[room.code].pop(pid, None)
                room.leave(pid)
                await hub.drive(room)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-url", default=os.environ.get("ZORO_URL"),
                    help="Zoro-1 server base URL (trida_mlx.server or serve_decide.py); or set ZORO_URL")
    ap.add_argument("--model", default=os.environ.get("ZORO_MODEL", "zoro-1"), help="'model' field sent to the server")
    ap.add_argument("--mock", action="store_true", help="fake the model (deterministic random guesses)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max in-flight model calls (SGLang batches them; use 1 for the MLX server)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not a.mock and not a.model_url:
        ap.error("give --model-url (or ZORO_URL), or --mock")
    import httpx
    import uvicorn
    judge = Judge(a.model_url, a.model, a.concurrency, mock=a.mock)
    if not judge.mock:
        try:
            r = httpx.get(f"{judge.url}/v1/models", timeout=5)
            r.raise_for_status()
            print(f"Zoro-1 server OK: {[m.get('id') for m in r.json().get('data', [])]}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: no Zoro-1 server at {judge.url} ({e}). See ../README.md, 'Run Zoro-1'.")
    print(f"Zoro Draw on http://{a.host}:{a.port}  (model: {judge.mode})")
    uvicorn.run(create_app(judge), host=a.host, port=a.port, ws_max_size=1 << 20, log_level="info")


if __name__ == "__main__":
    main()
