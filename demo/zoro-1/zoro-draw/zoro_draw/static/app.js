"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// ---- session (survives reload; per tab) -------------------------------------------------------------------
const store = {
  get(k) { try { return sessionStorage.getItem("zd_" + k); } catch { return null; } },
  set(k, v) { try { v == null ? sessionStorage.removeItem("zd_" + k) : sessionStorage.setItem("zd_" + k, v); } catch {} },
};
let ws = null, S = null, offset = 0, wantRoom = null, joined = false, retry = 0;

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 3500);
}
function send(msg) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); }
function nameOf(pid) { const p = S && S.players.find((q) => q.id === pid); return p ? p.name : "?"; }
function isHost() { return S && S.host === S.me; }

function connect(room) {
  wantRoom = room;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  $("status").textContent = "connecting…";
  ws.onopen = () => {
    $("status").textContent = "";
    send({ t: "hello", game: "draw", name: $("name").value || store.get("name"), room: wantRoom, token: store.get("token") });
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.t === "welcome") {
      joined = true; retry = 0;
      // the token is secret (it takes this seat back after a reload); the pid is public, everyone sees it
      store.set("token", m.token); store.set("room", m.room); wantRoom = m.room;
      history.replaceState(null, "", "/?room=" + m.room);
    } else if (m.t === "state") {
      S = m.state; offset = S.now - Date.now() / 1000; render();
    } else if (m.t === "error") {
      toast(m.msg);
      if (m.msg.startsWith("Opened in another tab")) { wantRoom = null; $("status").textContent = "open in another tab"; }
      else if (!joined) { store.set("room", null); ws.close(); wantRoom = null; showHome(); }
    }
  };
  ws.onclose = () => {
    if (!wantRoom || !joined) return;
    $("status").textContent = "reconnecting…";
    setTimeout(() => connect(wantRoom), Math.min(5000, 500 * 2 ** retry++));
  };
}

function showHome() {
  $("home").classList.remove("hidden"); $("room").classList.add("hidden");
}

// ---- home -------------------------------------------------------------------------------------------------
$("name").value = store.get("name") || "";
const params = new URLSearchParams(location.search);
const urlRoom = params.get("room");
if (urlRoom) $("code").value = urlRoom.toUpperCase();

function go(room) {
  const name = $("name").value.trim();
  if (!name) { toast("Pick a name first."); $("name").focus(); return; }
  store.set("name", name);
  if (store.get("room") !== room) store.set("token", null);
  joined = false; connect(room);
}
$("create").onclick = () => go(null);
$("join").onclick = () => { const c = $("code").value.trim().toUpperCase(); if (c.length !== 4) return toast("Room codes have 4 letters."); go(c); };
$("code").onkeydown = (e) => { if (e.key === "Enter") $("join").click(); };
$("copy").onclick = () => {
  const link = `${location.origin}/?room=${S.code}`;
  navigator.clipboard.writeText(link).then(() => toast("Link copied"), () => toast(link));
};

// Arrived from the home page (create / join), or rejoining after a reload.
if (params.get("create") && store.get("name")) {
  history.replaceState(null, "", "/"); go(null);
} else if (store.get("room") && store.get("name") && (!urlRoom || urlRoom.toUpperCase() === store.get("room"))) {
  joined = false; connect(store.get("room"));
} else if (urlRoom && store.get("name")) {
  go(urlRoom.toUpperCase());
}

// ---- render -----------------------------------------------------------------------------------------------
const PHASES = ["lobby", "draw", "decoys", "judging", "reveal"];
function show(id) { for (const p of PHASES) $(p).classList.toggle("hidden", p !== id); }

function render() {
  $("home").classList.add("hidden"); $("room").classList.remove("hidden");
  $("roomcode").textContent = S.code;
  const m = S.settings;
  $("roundinfo").textContent = S.round ? `Round ${S.round} of ${m.rounds} · ${m.mode === "decoy" ? "Decoy" : "Classic"}` : "";
  $("model").textContent = S.model === "mock" ? "Model: mock (no Zoro-1)" : "Model: Zoro-1";
  renderPlayers();
  ({ lobby: renderLobby, final: renderLobby, drawing: renderDraw, decoys: renderDecoys,
     judging: renderJudging, reveal: renderReveal })[S.phase]();
}

function renderPlayers() {
  const ol = $("players"); ol.textContent = "";
  const sorted = [...S.players].sort((a, b) => b.score - a.score);
  for (const p of sorted) {
    const li = el("li", (p.connected ? "" : "off ") + (p.id === S.me ? "me" : ""));
    li.append(el("span", "pname", p.name + (p.id === S.host ? " ★" : "")));
    const busy = (S.phase === "drawing" || S.phase === "decoys") && p.playing;
    li.append(el("span", "pstate", busy ? (p.done ? "✓" : "…") : ""));
    li.append(el("span", "pscore", String(p.score)));
    ol.append(li);
  }
}

function renderLobby() {
  show("lobby");
  const final = S.phase === "final";
  $("lobbytitle").textContent = final ? "Final scores" : (S.players.length < 2 ? "Share the link to invite friends" : "Ready when you are");
  const f = $("final"); f.classList.toggle("hidden", !final); f.textContent = "";
  if (final) {
    const sorted = [...S.players].sort((a, b) => b.score - a.score);
    sorted.forEach((p, i) => {
      const row = el("div", "podium" + (i === 0 ? " first" : ""));
      row.append(el("span", "rank", ["🥇", "🥈", "🥉"][i] || `${i + 1}.`), el("span", "pname", p.name), el("span", "pscore", String(p.score)));
      f.append(row);
    });
  }
  const host = isHost();
  for (const inp of document.querySelectorAll("#settings input, #settings select")) inp.disabled = !host;
  if (document.activeElement?.closest?.("#settings") == null) {
    document.querySelector(`input[name=mode][value=${S.settings.mode}]`).checked = true;
    $("rounds").value = S.settings.rounds; $("draw_secs").value = S.settings.draw_secs; $("decoy_secs").value = S.settings.decoy_secs;
    const dk = $("deck");
    if (dk.options.length !== S.decks.length + 1) {
      dk.textContent = "";
      for (const d of S.decks) { const o = el("option", null, d.label); o.value = d.id; dk.append(o); }
      const mix = el("option", null, "Mix (a random deck each round)"); mix.value = "mix"; dk.append(mix);
    }
    dk.value = S.settings.deck;
  }
  $("start").classList.toggle("hidden", !host);
  $("start").textContent = final ? "Play again" : "Start";
  $("hostnote").textContent = host ? "" : `Waiting for ${nameOf(S.host)} to start…`;
}
function sendSettings() {
  send({ t: "settings", mode: document.querySelector("input[name=mode]:checked").value,
         rounds: +$("rounds").value, draw_secs: +$("draw_secs").value, decoy_secs: +$("decoy_secs").value, deck: $("deck").value });
}
for (const inp of document.querySelectorAll("#settings input, #settings select")) inp.onchange = sendSettings;
$("start").onclick = () => send({ t: "start" });

// ---- drawing ----------------------------------------------------------------------------------------------
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const COLORS = ["#1d1b16", "#e5484d", "#2f6fe4", "#2f9e5b", "#f5a524", "#8b5a2b"];
const SIZES = [4, 9, 18];
let strokes = [], cur = null, pen = { color: COLORS[0], size: SIZES[1], erase: false }, drawKey = null, autoSent = null;

function redraw() {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const s of strokes) drawStroke(s);
}
function drawStroke(s, from = 0) {
  ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.size; ctx.lineCap = "round"; ctx.lineJoin = "round";
  const pts = s.points;
  if (pts.length === 1) { ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], s.size / 2, 0, 7); ctx.fill(); return; }
  ctx.beginPath(); ctx.moveTo(...pts[Math.max(0, from - 1)]);
  for (let i = Math.max(1, from); i < pts.length; i++) ctx.lineTo(...pts[i]);
  ctx.stroke();
}
function pos(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
}
canvas.addEventListener("pointerdown", (e) => {
  if (S?.phase !== "drawing") return;
  canvas.setPointerCapture(e.pointerId);
  cur = { color: pen.erase ? "#fff" : pen.color, size: pen.erase ? pen.size * 2.5 : pen.size, points: [pos(e)] };
  strokes.push(cur); drawStroke(cur);
});
canvas.addEventListener("pointermove", (e) => {
  if (!cur) return;
  cur.points.push(pos(e)); drawStroke(cur, cur.points.length - 1);
});
const endStroke = () => { cur = null; };
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

COLORS.forEach((c, i) => {
  const b = el("button", "swatch" + (i === 0 ? " on" : "")); b.style.background = c; b.title = c;
  b.onclick = () => { pen.color = c; pen.erase = false; markTools(); };
  $("swatches").append(b);
});
SIZES.forEach((s, i) => {
  const b = el("button", "size" + (i === 1 ? " on" : "")); const d = el("i"); d.style.width = d.style.height = s + 4 + "px"; b.append(d);
  b.onclick = () => { pen.size = s; markTools(); };
  $("sizes").append(b);
});
function markTools() {
  [...$("swatches").children].forEach((b, i) => b.classList.toggle("on", !pen.erase && COLORS[i] === pen.color));
  [...$("sizes").children].forEach((b, i) => b.classList.toggle("on", SIZES[i] === pen.size));
  $("eraser").classList.toggle("on", pen.erase);
}
$("eraser").onclick = () => { pen.erase = !pen.erase; markTools(); };
$("undo").onclick = () => { strokes.pop(); redraw(); };
$("clear").onclick = () => { strokes = []; redraw(); };

function exportPNG() {
  // 448x336 = 150,528 px, under the model's 448*448 cap; white background (a transparent canvas reads as black).
  const off = document.createElement("canvas"); off.width = 448; off.height = 336;
  const o = off.getContext("2d"); o.fillStyle = "#fff"; o.fillRect(0, 0, 448, 336);
  o.imageSmoothingQuality = "high"; o.drawImage(canvas, 0, 0, 448, 336);
  return off.toDataURL("image/png");
}
function submitDrawing() {
  if (!strokes.length) { toast("Draw something first."); return; }
  send({ t: "drawing", image: exportPNG() });
}
$("submit").onclick = submitDrawing;

function renderDraw() {
  show("draw");
  const key = `${S.code}/${S.round}`;
  if (drawKey !== key) { drawKey = key; strokes = []; redraw(); }
  if (!("topic" in S)) {
    $("topic").textContent = "—"; $("cands").textContent = "";
    $("drawnote").textContent = "You joined mid-round: you'll play from the next round.";
    $("submit").disabled = true; return;
  }
  $("submit").disabled = false;
  $("topic").textContent = S.topic;
  $("gloss").textContent = S.topic_gloss ? `(${S.topic_gloss})` : "";
  const c = $("cands"); c.textContent = "";
  if (S.candidates.length) {
    c.append(el("span", "muted", "Zoro picks from: "));
    for (const w of S.candidates) c.append(el("span", "chip" + (w === S.topic ? " truth" : ""), w));
  } else {
    c.append(el("span", "muted", "Your word is secret. The others will try to write decoys for your drawing."));
  }
  $("submit").textContent = S.submitted ? "Update drawing" : "Done";
  $("drawnote").textContent = S.submitted ? "Sent. You can keep drawing and update it until time's up." : "";
}

// ---- decoys -----------------------------------------------------------------------------------------------
let tasksKey = null;
function renderDecoys() {
  show("decoys");
  const tasks = S.tasks || [];
  const key = `${S.round}/` + tasks.map((t) => t.drawer).join(",");
  const box = $("tasks");
  if (tasksKey !== key) {
    tasksKey = key; box.textContent = "";
    if (!S.tasks) box.append(el("p", "muted", "You joined mid-round: you'll play from the next round."));
    for (const t of tasks) {
      const card = el("div", "task"); card.dataset.drawer = t.drawer;
      const img = el("img"); img.src = t.image; img.alt = "drawing by " + nameOf(t.drawer);
      const form = el("form", "row");
      const inp = el("input"); inp.maxLength = 24; inp.placeholder = "a decoy word"; inp.value = t.word || "";
      const btn = el("button", "small", "Save");
      form.append(inp, btn);
      form.onsubmit = (e) => { e.preventDefault(); if (inp.value.trim()) send({ t: "decoy", drawer: t.drawer, word: inp.value }); };
      card.append(el("div", "by", "by " + nameOf(t.drawer)), img, form, el("div", "saved muted"));
      box.append(card);
    }
  }
  for (const t of tasks) {
    const card = box.querySelector(`[data-drawer="${CSS.escape(t.drawer)}"]`);
    if (card) { card.classList.toggle("done", !!t.word); card.querySelector(".saved").textContent = t.word ? `saved: ${t.word}` : ""; }
  }
}

// ---- judging / reveal -------------------------------------------------------------------------------------
function renderJudging() {
  show("judging");
  $("judgingtext").textContent = `Zoro is looking at ${S.judging} drawing${S.judging === 1 ? "" : "s"}…`;
}

function renderReveal() {
  show("reveal");
  const last = S.round >= S.settings.rounds;
  $("next").classList.toggle("hidden", !isHost());
  $("next").textContent = last ? "Final scores" : "Next round";
  $("revealtitle").textContent = `Round ${S.round} results` + (isHost() ? "" : ` · waiting for ${nameOf(S.host)}`);
  const box = $("results"); box.textContent = "";
  if (!S.results.length) box.append(el("div", "card muted", "Nobody drew anything this round."));
  for (const r of S.results) {
    const card = el("div", "card result" + (r.bonus ? " winner" : ""));
    const left = el("div", "rleft");
    const img = el("img"); img.src = r.image; img.alt = r.truth;
    left.append(img, el("div", "by", `${nameOf(r.drawer)} drew `), );
    left.lastChild.append(el("b", null, r.truth));
    if (r.bonus) left.append(el("div", "badge win", "Round winner"));
    if (r.voided) left.append(el("div", "badge bad", `Answer written out (${pct(r.text_p)}): no points`));
    if (r.error) left.append(el("div", "badge bad", "Zoro couldn't judge this one"));
    const right = el("div", "rright");
    if (!r.error) {
      right.append(el("div", "verdict", r.top === r.truth ? `Zoro guessed it: ${pct(r.p_truth)}` : `Zoro thought: ${r.top}`));
      for (const o of r.options) {
        const row = el("div", "bar k-" + o.kind);
        const fill = el("div", "fill"); fill.style.width = "0%";
        requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = Math.max(0.5, o.p * 100) + "%"; }));
        const label = el("div", "label");
        label.append(el("span", "w", o.word));
        if (o.authors.length) label.append(el("span", "a", "← " + o.authors.map(nameOf).join(", ")));
        label.append(el("span", "p", pct(o.p)));
        row.append(fill, label); right.append(row);
      }
    } else right.append(el("div", "muted", r.error));
    if (r.sharp.length) right.append(el("div", "muted", "Sharp eye: " + r.sharp.map(nameOf).join(", ")));
    const pts = Object.entries(r.points).filter(([, n]) => n > 0).map(([pid, n]) => `${nameOf(pid)} +${n}`);
    if (pts.length) right.append(el("div", "pts", pts.join(" · ")));
    card.append(left, right); box.append(card);
  }
}
const pct = (p) => (p * 100).toFixed(p < 0.1 ? 1 : 0) + "%";
$("next").onclick = () => send({ t: "next" });

// ---- timers -----------------------------------------------------------------------------------------------
setInterval(() => {
  if (!S || !S.deadline) return;
  const left = Math.max(0, S.deadline - (Date.now() / 1000 + offset));
  const txt = `${Math.ceil(left)}s`;
  const t = S.phase === "drawing" ? $("drawtimer") : $("decoytimer");
  t.textContent = txt; t.classList.toggle("hurry", left < 10);
  const key = `${S.phase}/${S.round}`;
  if (left < 0.4 && autoSent !== key) {
    autoSent = key;
    if (S.phase === "drawing" && "topic" in S && strokes.length) send({ t: "drawing", image: exportPNG() });
    if (S.phase === "decoys") {
      for (const card of document.querySelectorAll(".task")) {
        const v = card.querySelector("input").value.trim();
        if (v) send({ t: "decoy", drawer: card.dataset.drawer, word: v });
      }
    }
  }
}, 200);
