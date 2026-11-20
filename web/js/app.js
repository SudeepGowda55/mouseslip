"use strict";
/* Gardyloo's page: live games as three time series, calls as paper slips, an honest scoreboard. No board anywhere. */
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const INK = "#06120e", IVORY = "#fbf7ea", CHALK = "#ffd23f", HAIR = "rgba(251,247,234,.3)";
const SPEEDS = ["bullet", "blitz", "rapid"], FIRST_PLY = 12, BLUNDER = "0.30";

// Everything the API sends as text is escaped before it reaches innerHTML (the reason is written by a language model).
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cap = s => s[0].toUpperCase() + s.slice(1);
const clock = s => s == null ? "–" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const tc = t => { const [b, i] = t.split("+").map(Number); return `${{ 15: "¼", 30: "½", 45: "¾" }[b] ?? +(b / 60).toFixed(1)}+${i}`; };
const pct = v => v == null ? "–" : (v * 100).toFixed(v < 0.1 ? 1 : 0) + "%";
const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const moveNo = ply => Math.floor(ply / 2) + 1; // the move about to be played once `ply` plies are on the scoresheet
const wc = p => 2 / (1 + Math.exp(-0.368208 * p)) - 1; // Lichess winning chances from pawns, as the eval bar shows them
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const callState = c => c.void ? "void" : c.blunder == null ? "open" : c.blunder ? "hit" : "miss";
// The newest scored position. In fast games the current one is often still with Stockfish, so the last score stays up.
const scored = g => { const d = g.series.p.at(-1); return d && { ply: d[0], p: d[1], side: d[0] % 2 ? "Black" : "White", now: d[0] === g.ply }; };
const who = g => `${cap(g.speed)} ${tc(g.time_control)} · ${g.white.rating ?? "?"} v ${g.black.rating ?? "?"}`;

const S = { games: new Map(), board: null, model: "", trained: true, thr: null, slips: new Map(), seen: new Set(), times: {}, results: null, view: "watch", online: null, featured: null, info: {} };

/* ───────── the one chart routine: lanes of clocks, the unrolled evaluation bar, P(blunder) ───────── */
function plot(cv, s, o) {
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return null;
  const k = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(w * k) || cv.height !== Math.round(h * k)) { cv.width = Math.round(w * k); cv.height = Math.round(h * k); }
  const c = cv.getContext("2d");
  c.setTransform(k, 0, 0, k, 0, 0); c.clearRect(0, 0, w, h);
  const n = s.white.length, last = s.from_ply + n - 1, upto = o.upto ?? last, x0 = s.from_ply;
  const x1 = o.x1 ?? Math.max(x0 + 46, last + 6), padR = o.padR ?? 0, gap = o.gap ?? 6, lw = o.lw ?? 2;
  const X = p => (p - x0) / (x1 - x0) * (w - padR);
  const total = o.lanes.reduce((a, l) => a + l.h, 0), clockMax = Math.max(o.base || 0, ...s.white, ...s.black, 1);
  const label = (text, x, y, align = "left") => { c.font = "500 11px 'Chivo Mono', monospace"; c.textAlign = align; c.textBaseline = "top"; c.lineWidth = 3; c.strokeStyle = INK; c.strokeText(text, x, y); c.fillStyle = IVORY; c.fillText(text, x, y); };
  const heads = {}, boxes = [];
  c.lineJoin = c.lineCap = "round";
  let top = 0;
  c.save(); c.beginPath(); c.rect(0, 0, X(upto) + lw, h); c.clip(); // a scrubbed chart only shows what has been played
  for (const lane of o.lanes) {
    const lh = (h - gap * (o.lanes.length - 1)) * lane.h / total, m = Math.min(12, lh * .12);
    boxes.push([lane.kind, top, lh]);
    const trace = (vals, color, key) => {
      const Y = v => top + lh - m - v / clockMax * (lh - 2 * m);
      c.beginPath(); vals.forEach((v, i) => c[i ? "lineTo" : "moveTo"](X(x0 + i), Y(v)));
      if (o.fill) { c.save(); c.lineTo(X(last), top + lh); c.lineTo(X(x0), top + lh); c.globalAlpha = .14; c.fillStyle = color; c.fill(); c.restore(); c.beginPath(); vals.forEach((v, i) => c[i ? "lineTo" : "moveTo"](X(x0 + i), Y(v))); }
      c.strokeStyle = color; c.lineWidth = lw; c.stroke();
      heads[key] = [X(Math.min(last, upto)), Y(vals[Math.max(0, Math.min(n - 1, Math.round(upto) - x0))])];
    };
    if (lane.kind === "white" || lane.kind === "clocks") trace(s.white, IVORY, "white");
    if (lane.kind === "black" || lane.kind === "clocks") trace(s.black, INK, "black");
    if (lane.kind === "eval") { // the evaluation bar, unrolled through time: White's share below the line, Black's above
      const mid = top + lh / 2, Y = v => mid - wc(v) * (lh / 2 - 1), e = s.evaluation;
      for (let i = 0; i < n; i++) {
        if (e[i] == null) continue;
        let j = i; while (j + 1 < n && e[j + 1] != null) j++;
        for (const [edge, color] of [[top, INK], [top + lh, IVORY]]) {
          c.beginPath(); c.moveTo(X(x0 + i), edge);
          for (let q = i; q <= j; q++) c.lineTo(X(x0 + q), Y(e[q]));
          c.lineTo(X(x0 + j), edge); c.closePath(); c.fillStyle = color; c.fill();
        }
        heads.eval = [X(Math.min(x0 + j, upto)), Y(e[Math.max(i, Math.min(j, Math.round(upto) - x0))])];
        i = j;
      }
      c.setLineDash([4, 5]); c.strokeStyle = "rgba(128,128,128,.95)"; c.lineWidth = 1; c.beginPath(); c.moveTo(X(x0), mid); c.lineTo(X(last), mid); c.stroke(); c.setLineDash([]);
    }
    if (lane.kind === "p" && s.p.length) {
      const max = Math.max(.5, (o.thr || 0) * 1.4, ...s.p.map(d => d[1])), Y = v => top + lh - 3 - v / max * (lh - 16);
      c.beginPath(); s.p.forEach(([ply, v], i) => c[i ? "lineTo" : "moveTo"](X(ply), Y(v))); c.strokeStyle = CHALK; c.lineWidth = 2; c.stroke();
      for (const [ply, v] of s.p) { const hot = o.thr != null && v >= o.thr; c.beginPath(); c.arc(X(ply), Y(v), hot ? 3.6 : 2.2, 0, 7); c.fillStyle = CHALK; c.fill(); if (hot) { c.strokeStyle = INK; c.lineWidth = 1.5; c.stroke(); } }
      lane.thrY = o.thr != null ? Y(o.thr) : null;
    }
    top += lh + gap;
  }
  c.restore();
  for (const [kind, y, lh] of boxes) { // overlays that are not clipped by the scrub
    if (o.labels) label({ clocks: "clocks", eval: "evaluation bar", p: "P(blunder)" }[kind] || "", 5, y + 4);
    const lane = o.lanes.find(l => l.kind === kind);
    if (kind === "p" && lane.thrY != null) { c.setLineDash([5, 4]); c.strokeStyle = IVORY; c.lineWidth = 1; c.beginPath(); c.moveTo(0, lane.thrY); c.lineTo(w - padR, lane.thrY); c.stroke(); c.setLineDash([]); label(`dashes: calls at ${o.thr.toFixed(2)}`, w - padR - 4, y + 4, "right"); }
    if (gap) { c.strokeStyle = HAIR; c.lineWidth = 1; c.beginPath(); c.moveTo(0, y + lh + gap / 2); c.lineTo(w, y + lh + gap / 2); c.stroke(); }
  }
  for (const call of o.calls || []) { // a call: the line where it was made, and the three moves Stockfish will judge
    const x = X(call.ply), open = call.state === "open";
    if (open) { c.fillStyle = "rgba(251,247,234,.1)"; c.fillRect(x, 0, X(call.ply + 5) - x, h); if (o.zoneLabel) label(o.zoneLabel, x + 6, h - 16); }
    c.strokeStyle = IVORY; c.globalAlpha = open ? 1 : .45; c.lineWidth = open ? 1.6 : 1; c.beginPath(); c.moveTo(x, 12); c.lineTo(x, h); c.stroke(); c.globalAlpha = 1;
    c.beginPath(); c.lineWidth = 1.5; c.strokeStyle = INK;
    if (open) { c.moveTo(x - 5, 1); c.lineTo(x + 5, 1); c.lineTo(x, 10); c.closePath(); c.fillStyle = IVORY; c.fill(); c.stroke(); }
    else if (call.state === "void") { c.strokeStyle = IVORY; c.moveTo(x - 4, 6); c.lineTo(x + 4, 6); c.stroke(); }
    else { c.arc(x, 6, 4.2, 0, 7); c.fillStyle = call.state === "hit" ? CHALK : "rgba(0,0,0,0)"; c.fill(); c.strokeStyle = call.state === "hit" ? INK : IVORY; c.stroke(); }
  }
  if (o.tags) for (const [key, text] of Object.entries(o.tags(Math.max(0, Math.min(n - 1, Math.round(upto) - x0))))) if (heads[key]) label(text, heads[key][0] + 7, heads[key][1] - 6);
  if (o.hover != null) { c.strokeStyle = INK; c.lineWidth = 1; c.beginPath(); c.moveTo(X(o.hover), 0); c.lineTo(X(o.hover), h); c.stroke(); }
  return { heads, x0, x1, w: w - padR, last };
}

/* ───────── hero: the featured live game, one series per lane ───────── */
function drawHero() {
  let g = S.games.get(S.featured);
  if (!g || g.status !== "started") {
    g = [...S.games.values()].filter(x => x.status === "started" && x.series.white.length > 1).sort((a, b) => b.ply - a.ply)[0];
    S.featured = g?.id;
  }
  const cv = $("#heroCanvas"), pens = { white: $("#penW"), black: $("#penB"), eval: $("#penE") }, reading = $("#reading");
  if (!g) {
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    Object.values(pens).forEach(p => p.textContent = "");
    reading.textContent = S.online === false ? "The watcher is not answering. Retrying every few seconds." : S.online ? "No live game on the table yet. The watcher looks for new ones every 15 seconds." : "Connecting to the watcher…";
    return;
  }
  const wide = cv.clientWidth > 700;
  const r = plot(cv, g.series, { lanes: [{ kind: "white", h: 1 }, { kind: "black", h: 1 }, { kind: "eval", h: 1 }], gap: 0, lw: wide ? 3.5 : 2.5, fill: true, base: +g.time_control.split("+")[0], padR: wide ? 190 : 0 });
  if (!r) return;
  const lastEval = g.series.evaluation.findLast(v => v != null), laneH = cv.clientHeight / 3;
  const text = { white: `White's clock ${clock(g.clocks?.white)}`, black: `Black's clock ${clock(g.clocks?.black)}`, eval: lastEval == null ? "" : `Evaluation ${signed(lastEval)}` };
  Object.entries(pens).forEach(([key, pen], i) => {
    if (!r.heads[key] || !text[key]) return void (pen.textContent = "");
    pen.textContent = text[key]; pen.classList.toggle("flip", !wide);
    pen.style.left = r.heads[key][0] + "px"; pen.style.top = clamp(r.heads[key][1] - laneH * i, 14, laneH - 14) + "px";
  });
  const side = cap(g.side_to_move), sc = scored(g);
  reading.innerHTML = `Reading now: <b>${who(g)}</b> · ${esc(g.source)} · move ${moveNo(g.ply)} · ` + (sc ? `P(${sc.side} blunders) <b>${sc.p.toFixed(2)}</b>${sc.now ? "" : ` at move ${moveNo(sc.ply)}`}` : `${side} to move, not scored yet`);
}

/* ───────── game cards ───────── */
const calls = g => [...g.open_calls, ...g.resolved_calls].map(c => ({ ply: c.ply, state: callState(c) }));
const LANES = [{ kind: "clocks", h: 1.25 }, { kind: "eval", h: 1 }, { kind: "p", h: .8 }];
const ENDED = { mate: "checkmate", resign: "resignation", outoftime: "flag fall", draw: "draw", stalemate: "stalemate", timeout: "player left", lost: "stream lost", aborted: "aborted" };

function drawCard(e, g) {
  e._plot = plot($("canvas", e), g.series, { lanes: LANES, labels: true, thr: S.thr, base: +g.time_control.split("+")[0], calls: calls(g), hover: e._hover });
}

function card(g) {
  let e = document.getElementById("g-" + g.id);
  if (!e) {
    e = el("article", "game", `<div class="g-top"><span class="chip"></span><span class="tc"></span><span class="src"></span><span class="st"></span></div>
      <div class="players"><div class="pl w"><i></i><span class="who"></span><span class="clk"></span></div><div class="pl b"><i></i><span class="who"></span><span class="clk"></span></div></div>
      <canvas class="chart" role="img"></canvas><p class="tip"></p>
      <div class="pm"><span></span><b></b><div class="bar"><i></i><u></u></div></div>
      <div class="g-foot"><span class="gc"></span><a target="_blank" rel="noreferrer">Verify on Lichess ↗</a></div>`);
    e.id = "g-" + g.id;
    const cv = $("canvas", e);
    cv.addEventListener("pointermove", ev => { const p = e._plot, game = S.games.get(g.id); if (!p || !game) return; e._hover = clamp(Math.round(p.x0 + ev.offsetX / p.w * (p.x1 - p.x0)), p.x0, p.last); card(game); });
    cv.addEventListener("pointerleave", () => { e._hover = null; const game = S.games.get(g.id); if (game) card(game); });
    $("#games").append(e);
  }
  const live = g.status === "started", side = cap(g.side_to_move), s = g.series;
  $(".chip", e).textContent = g.speed; $(".tc", e).textContent = tc(g.time_control); $(".src", e).textContent = g.source;
  $(".st", e).innerHTML = live ? `<i class="live-dot"></i>move ${moveNo(g.ply)}` : `ended · ${esc(ENDED[g.status] || g.status)}`;
  for (const [cls, name, p] of [["w", "White", g.white], ["b", "Black", g.black]]) {
    const pl = $(".pl." + cls, e);
    $(".who", pl).innerHTML = (p.title ? `<em>${esc(p.title)}</em>` : "") + `${name} · ${p.rating ?? "?"}`;
    $(".clk", pl).textContent = clock(g.clocks?.[name.toLowerCase()]);
    pl.classList.toggle("tomove", live && g.side_to_move === name.toLowerCase());
  }
  e.style.order = SPEEDS.indexOf(g.speed) + (live ? 0 : 10); // games in progress first
  e.classList.toggle("calling", g.open_calls.length > 0); e.classList.toggle("over", !live);
  $("a", e).href = g.link;
  const open = g.open_calls.length, done = g.resolved_calls.map(callState), count = k => done.filter(d => d === k).length;
  $(".gc", e).textContent = open + done.length ? `Calls: ${open} open · ${count("hit")} hit · ${count("miss")} missed` + (count("void") ? ` · ${count("void")} void` : "") : "No calls on this game yet";
  const sc = live && scored(g);
  $(".pm span", e).textContent = !live ? "Game over" : sc ? `P(${sc.side} blunders in the next 3 moves)` + (sc.now ? "" : `, as of move ${moveNo(sc.ply)}`) : g.ply < FIRST_PLY ? `Too early: the model starts scoring at move ${moveNo(FIRST_PLY)}` : "Waiting for Stockfish to evaluate this position";
  $(".pm b", e).textContent = sc ? sc.p.toFixed(2) : "–";
  $(".bar i", e).style.width = (sc ? sc.p * 100 : 0) + "%"; $(".bar u", e).style.left = (S.thr ?? 0) * 100 + "%";
  const h = e._hover, i = h == null ? -1 : h - s.from_ply, pAt = h == null ? null : s.p.find(d => d[0] === h);
  $(".tip", e).textContent = i < 0 || i >= s.white.length ? "Ivory line: White's clock. Black line: Black's clock. Yellow: P(blunder). Marks on top: calls."
    : `after ply ${h} · W ${clock(s.white[i])} · B ${clock(s.black[i])} · eval ${s.evaluation[i] == null ? "pending" : signed(s.evaluation[i])}${pAt ? ` · P ${pAt[1].toFixed(2)}` : ""}`;
  $("canvas", e).setAttribute("aria-label", `${who(g)}: clocks ${clock(g.clocks?.white)} and ${clock(g.clocks?.black)}, ${sc ? "P(blunder) " + sc.p.toFixed(2) : "not scored"}`);
  if (S.view === "watch") drawCard(e, g);
}

/* ───────── calls: slips in the feed, and the banner ───────── */
function slipHTML(g, c, t) {
  const side = cap(c.side), state = callState(c);
  let out;
  if (state === "open") {
    const played = clamp(Math.ceil((g.ply - c.ply) / 2), 0, 3);
    out = `<span class="pips" aria-label="${played} of 3 moves played">${[0, 1, 2].map(i => `<i${i < played ? ' class="on"' : ""}></i>`).join("")}</span>Stockfish is judging ${side}'s next three moves`;
  } else if (state === "hit") out = `${side}'s move <code>${esc(c.worst_move)}</code> (move ${moveNo(c.worst_ply - 1)}) gave away ${c.drop.toFixed(2)} winning chances. A blunder is ${BLUNDER} or more.`;
  else if (state === "miss") out = `No blunder in ${side}'s next three moves. The worst one lost ${Math.max(0, c.drop).toFixed(2)} winning chances; a blunder is ${BLUNDER} or more.`;
  else out = `The game ended before ${side}'s third move. Counts as neither a hit nor a miss.`;
  const time = t ? " · " + new Date(t * 1000).toLocaleTimeString([], { hour12: false }) : "";
  return `<p class="meta">${who(g)} · move ${moveNo(c.ply)} · P ${c.p.toFixed(2)}${time}</p>
    <p class="say">${state === "open" ? `${side} is about to blunder` : `Called a ${side} blunder`}</p>
    ${c.explain ? `<p class="why">${esc(c.explain)}</p>` : `<p class="why wait">Wording its reason…</p>`}
    <p class="out">${out}</p>${state === "open" ? "" : `<span class="stamp ${state}">${state}</span>`}`;
}

function syncSlips(g, fresh) {
  const box = $("#slips");
  [...g.open_calls, ...g.resolved_calls].map((c, i) => [c, i]).reverse().forEach(([c, i]) => { // oldest first, so the newest ends on top
    const key = g.id + ":" + c.ply, state = callState(c);
    let e = S.slips.get(key);
    if (!e) {
      // on first load only the newest two judged calls per game are shown; older ones stay as marks on its chart
      if (S.seen.has(key) || (state !== "open" && !fresh && i - g.open_calls.length > 1)) return;
      S.seen.add(key);
      e = el("div", "slip" + (fresh ? " new" : "")); S.slips.set(key, e); box.prepend(e);
    }
    const html = slipHTML(g, c, S.times[key]);
    if (e._html !== html) { e.innerHTML = e._html = html; e.classList.toggle("done", state !== "open"); }
  });
  for (const [key, e] of [...S.slips].slice(0, Math.max(0, S.slips.size - 8))) { e.remove(); S.slips.delete(key); } // oldest first
  let none = $("#noCalls");
  if (!S.slips.size && !none) box.append(none = el("p", "empty", "")), none.id = "noCalls";
  if (none) S.slips.size ? none.remove() : none.textContent = `Waiting for the first call.${S.thr != null ? ` The model calls a blunder when P reaches ${S.thr.toFixed(2)}.` : ""}`;
}

let bannerTimer, bannerKey, bannerAt = 0;
function banner(g, c, t, seconds) {
  const b = $("#banner"), key = g.id + ":" + c.ply + ":" + callState(c);
  if (seconds === undefined) { if (bannerKey?.startsWith(g.id + ":" + c.ply + ":")) $(".slip", b).innerHTML = slipHTML(g, c, t); return; } // the reason arrived: fill it in, no second slam
  bannerKey = key; bannerAt = performance.now(); b.hidden = false; b.className = "banner";
  b.innerHTML = `<div class="slip${callState(c) === "open" ? "" : " done"}">${slipHTML(g, c, t)}</div>`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { b.classList.add("bye"); bannerTimer = setTimeout(() => { b.hidden = true; bannerKey = null; }, 380); }, seconds * 1000);
}

/* ───────── the scoreboard ───────── */
function tote() {
  const b = S.board?.all, box = $("#tote");
  if (!b) return void (box.innerHTML = `<div><dt>Scoreboard</dt><small>${S.online === false ? "the watcher is not answering" : "connecting…"}</small></div>`);
  const open = [...S.games.values()].reduce((a, g) => a + g.open_calls.length, 0);
  const cells = [
    ["Calls judged", b.calls, `${open} more open`],
    ["Hits", b.hits, b.calls ? `${b.calls - b.hits} missed` : ""],
    ["Precision", pct(b.precision), b.calls ? `${b.hits} of ${b.calls} calls` : "waiting for the first call"],
    ["Live base rate", pct(b.base_rate), b.resolved ? `${b.blunders} of ${b.resolved} positions` : "nothing judged yet"],
    ["Lift", b.lift == null ? "–" : "×" + b.lift.toFixed(1), "precision ÷ base rate"],
    ["Positions judged", b.resolved, `${b.void} void`],
  ];
  if (box.children.length !== cells.length) box.innerHTML = cells.map(() => "<div><dt></dt><b></b><small></small></div>").join("");
  cells.forEach(([k, v, sub], i) => {
    const d = box.children[i], val = $("b", d);
    $("dt", d).textContent = k; $("small", d).textContent = sub;
    if (val.textContent !== String(v)) { const first = !val.textContent; val.textContent = v; if (!first) { val.classList.remove("bump"); void val.offsetWidth; val.classList.add("bump"); } }
  });
  const stand = !S.trained ? ` <span class="warn">Development heuristic: no trained model is loaded, so these numbers mean nothing.</span>`
    : /^OpenTSLM/.test(S.model) ? "" : " This is the baseline, not the fine-tuned TSLM.";
  const line = `Model loaded: <b>${esc(S.model)}</b>.${stand} It calls a blunder when P reaches ${S.thr.toFixed(2)}.`;
  $("#toteNote").innerHTML = line; $("#modelLine").innerHTML = line;
  scoreView();
}

function scoreView() {
  const sb = S.board; if (!sb) return;
  const a = sb.all, bar = (cls, text, v, none) => `<div class="vs ${cls}" style="--v:${v ?? 0}"><p class="k"><span>${text}</span><b>${v == null ? none : pct(v)}</b></p><div class="track"><i></i></div></div>`;
  $("#versus").innerHTML = bar("model", "When Gardyloo calls a blunder, one follows", a.precision, "waiting for the first call")
    + bar("", "After any position at all, one follows (live base rate)", a.base_rate, "nothing judged yet");
  const cols = ["all", ...SPEEDS], rows = [
    ["Calls judged", b => b.calls], ["Hits", b => b.hits], ["Precision", b => pct(b.precision)], ["Live base rate", b => pct(b.base_rate)],
    ["Lift", b => b.lift == null ? "–" : "×" + b.lift.toFixed(1)], ["Positions judged", b => b.resolved], ["Blunders among them", b => b.blunders], ["Void", b => b.void],
  ];
  $("#board").innerHTML = `<thead><tr><th scope="col">Live</th>${cols.map(c => `<th scope="col">${cap(c)}</th>`).join("")}</tr></thead><tbody>`
    + rows.map(([k, f]) => `<tr><th scope="row">${k}</th>${cols.map(c => `<td>${sb[c] ? f(sb[c]) : "–"}</td>`).join("")}</tr>`).join("") + "</tbody>";
  const i = S.info, up = i.uptime_seconds, v = i.verifier;
  $("#liveFine").textContent = `A dash means there is nothing to compute yet. Feed: ${i.feed}. Referee: ${v.engine} at depth ${v.depth}. ${i.skipped} live positions arrived faster than they could be scored and were left out. Counts carry across restarts of the same model. This watcher has been up ${up >= 3600 ? (up / 3600).toFixed(1) + " hours" : Math.round(up / 60) + " minutes"}.`;
  $("#footFeed").textContent = `Feed: ${i.feed}.`;
}

const SETS = { priors: ["Priors only", "ratings, time control, ply, side to move"], "clock-dynamics": ["Clocks alone", "seconds left and spent per move"], clocks: ["Priors + clocks", "all a live game offers without an engine"], "clocks+engine": ["Priors + clocks + evaluation bar", "what Gardyloo reads"] };
function offline() {
  const R = S.results; if (!R) return;
  const base = R.baselines, x = v => clamp((v - .5) / .35), rows = [];
  if (base) {
    for (const key of Object.keys(SETS)) { // per feature set, the model its own validation AUC chose
      const r = base.results.filter(r => r.features === key).sort((a, b) => b.validation_auc - a.validation_auc)[0];
      if (r) rows.push({ cls: key === "clocks+engine" ? "strong" : "", name: SETS[key][0], sub: `${SETS[key][1]} · ${r.model} · top 10% of scores: ${pct(r.top_decile_precision)} right`, auc: r.auc, ci: r.auc_ci95 });
    }
    const t = base.splits.test;
    $("#offlineSub").textContent = `${base.dataset}. The label: ${base.label}. Test split: ${t.n.toLocaleString()} windows the models never trained on, where ${pct(t.base_rate)} are followed by a blunder. AUC of 0.5 is a coin flip.`;
    const ab = base.ablation_boosting, g = d => `${signed(d.auc_gain, 3)} AUC (95% interval ${signed(d.auc_gain_ci95[0], 3)} to ${signed(d.auc_gain_ci95[1], 3)})`;
    const text = `Adding both clocks to the priors moves the needle by ${g(ab.clocks_over_priors)}. Adding the evaluation bar on top adds ${g(ab.engine_over_clocks)}.`;
    $("#finding").textContent = "The clocks alone add almost nothing beyond knowing the ratings and the time control. The evaluation bar carries the signal.";
    $("#howFinding").textContent = text + ` Measured on ${t.n.toLocaleString()} held-out windows with gradient boosting.`;
    $("#offlineFine").textContent = `${text} Intervals: ${base.bootstrap}. Selection: ${base.selection}.`;
  }
  const tslm = Object.entries(R).filter(([k, r]) => k !== "baselines" && typeof r?.auc === "number");
  for (const [k, r] of tslm) rows.push({ cls: "tslm", name: r.model || k, sub: [r.mode, r.steps_trained != null && `${r.steps_trained} steps`, r.n != null && `${r.n.toLocaleString()} windows`, r.top_decile_precision != null && `top 10% of scores: ${pct(r.top_decile_precision)} right`].filter(Boolean).join(" · "), auc: r.auc, ci: r.auc_ci95 });
  $("#auc").innerHTML = rows.map(r => `<div class="auc-row ${r.cls}" style="--x:${x(r.auc)};--lo:${x(r.ci?.[0] ?? r.auc)};--hi:${x(r.ci?.[1] ?? r.auc)}">
      <div class="lab">${esc(r.name)}<small>${esc(r.sub)}</small></div><div class="trk">${r.ci ? '<span class="ci"></span>' : ""}<span class="dot"></span></div>
      <div class="val">${r.auc.toFixed(3)}<small>${r.ci ? `${r.ci[0].toFixed(3)} to ${r.ci[1].toFixed(3)}` : "AUC"}</small></div></div>`).join("")
    + (rows.length ? `<div class="auc-axis"><span></span><div><span>0.50</span><span>AUC on the test split</span><span>0.85</span></div><span></span></div>` : `<p class="empty">No offline results on disk yet.</p>`)
    + (tslm.length ? "" : `<p class="note">The fine-tuned TSLM is still training. Its held-out numbers will appear here when it finishes; until then there is no number to show.</p>`);
}

/* ───────── data in ───────── */
function setGame(g, fresh) { S.games.set(g.id, g); card(g); syncSlips(g, fresh); }

function applyState(d) {
  Object.assign(S, { board: d.scoreboard, model: d.model, trained: d.trained, thr: d.threshold, info: d });
  const ids = new Set(d.games.map(g => g.id));
  for (const id of [...S.games.keys()]) if (!ids.has(id)) { S.games.delete(id); document.getElementById("g-" + id)?.remove(); }
  d.games.forEach(g => setGame(g, false));
  syncEmpty(); tote(); drawHero();
  $("#feedNote").textContent = `Each card is one real game in progress: two clocks, the evaluation bar and the model's P(blunder). ${d.feed}.`;
}
function syncEmpty() {
  let e = $("#noGames");
  if (!S.games.size && !e) $("#games").append(e = el("p", "empty", "No live games on the table right now. The watcher looks for new ones every 15 seconds.")), e.id = "noGames";
  else if (S.games.size) e?.remove();
  if (!S.slips.size) syncSlips({ open_calls: [], resolved_calls: [] });
}

async function refresh() {
  try { const r = await fetch("/api/state"); if (!r.ok) throw new Error(r.status); S.online = true; applyState(await r.json()); }
  catch { S.online = false; tote(); drawHero(); }
  setTimeout(refresh, S.online ? 20000 : 5000); // judged positions that were not calls only show up here
}
async function loadResults() {
  try { const r = await fetch("/api/results"); if (r.ok) { S.results = await r.json(); offline(); } } catch { /* shown as absent */ }
  setTimeout(loadResults, 60000); // the TSLM's results appear when its training finishes
}
function onEvent(e) {
  if (S.thr == null) return; // nothing is drawn before the first full state; it will include whatever this event says
  const g = e.game, key = e.call && g.id + ":" + e.call.ply;
  if (e.type === "call") S.times[key] = e.t;
  setGame(g, true); syncEmpty();
  if (e.type === "call") banner(g, e.call, e.t, 7);
  if (e.type === "explain") banner(g, e.call, S.times[key]);
  if (e.type === "resolution") { S.board = e.scoreboard; if (!bannerKey || !bannerKey.endsWith(":open") || performance.now() - bannerAt > 3500) banner( // a fresh call keeps the lower third for a few seconds
    g, e.call, S.times[key], 5); }
  if (e.type === "resolution" || e.type === "call") tote();
  if (g.id === S.featured || !S.featured || e.type === "game") drawHero();
}

/* ───────── how it works: one real window, scrubbed by scroll ───────── */
let EX;
function story() {
  if (!EX || S.view !== "how") return;
  const sec = $("#story"), r = sec.getBoundingClientRect(), p = REDUCED ? 1 : clamp(-r.top / (r.height - innerHeight));
  const first = EX.first_ply, call = EX.call_ply, end = first + EX.white.length - 1, blunder = EX.drops.find(d => d.drop >= .3);
  const ease = t => 1 - (1 - t) ** 2;
  const upto = p < .25 ? first + (call - first) * ease(p / .25) : p < .74 ? call : Math.min(end, call + (end - call) * (p - .74) / .2);
  const judged = upto >= blunder.ply, calling = p >= .5;
  plot($("#storyCanvas"), { from_ply: first, white: EX.white, black: EX.black, evaluation: EX.evaluation, p: [] }, {
    lanes: [{ kind: "clocks", h: 1.1 }, { kind: "eval", h: 1 }], labels: true, upto, x1: end + 1, lw: 2.5, padR: 96,
    tags: i => ({ white: `White ${clock(EX.white[i])}`, black: `Black ${clock(EX.black[i])}`, eval: `eval ${signed(EX.evaluation[i])}` }), base: +EX.time_control.split("+")[0],
    calls: calling ? [{ ply: call, state: judged ? "hit" : "open" }] : [], zoneLabel: "not played yet",
  });
  $$(".ch").forEach((c, i) => c.classList.toggle("on", i === Math.min(3, Math.floor(p * 4))));
  $("#railFill").style.transform = `scaleX(${p})`;
  const side = cap(EX.side), typed = clamp((p - .27) / .18);
  $("#qaQ").textContent = p >= .25 ? `Asked: will ${side} blunder within their next three moves?` : "";
  $("#qaA").textContent = typed ? "Trained to write: " + EX.rationale.slice(0, Math.round(EX.rationale.length * typed)) : "";
  $("#qa").style.opacity = innerWidth <= 640 && calling ? 0 : 1;
  const slip = $("#demoSlip"), played = clamp(Math.ceil((Math.floor(upto) - call) / 2), 0, 3), state = (calling ? 1 : 0) + ":" + played + judged;
  slip.classList.toggle("on", calling); slip.classList.toggle("judged", judged);
  if (slip._s !== state) {
    slip._s = state;
    slip.innerHTML = `<p class="meta">Blitz ${tc(EX.time_control)} · ${EX.white_elo} v ${EX.black_elo} · move ${moveNo(call)}</p><p class="say">${side} is about to blunder</p>
      <p class="why">Called after ply ${call}, before move ${moveNo(call)} exists.</p>
      <p class="out">${judged ? `Move ${moveNo(blunder.ply - 1)} took the evaluation from ${signed(blunder.before)} to ${signed(blunder.after)}: ${blunder.drop.toFixed(2)} winning chances gone. A blunder is ${BLUNDER} or more.`
        : `<span class="pips">${[0, 1, 2].map(i => `<i${i < played ? ' class="on"' : ""}></i>`).join("")}</span>Stockfish is judging ${side}'s next three moves`}</p><span class="stamp hit">hit</span>`;
  }
}
async function loadExample() {
  try {
    EX = await (await fetch("assets/example.json")).json();
    $("#benchCap").textContent = `One real window from the test split · Blitz ${tc(EX.time_control)} · White ${EX.white_elo} v Black ${EX.black_elo} · label: ${EX.label ? "blunder follows" : "no blunder"}`;
    story();
  } catch { $("#benchCap").textContent = "The example game could not be loaded."; }
}

/* ───────── scroll: smooth, and scrubbing the story and the charts ───────── */
const lenis = REDUCED || !window.Lenis ? null : new Lenis({ autoRaf: true, lerp: .13 });
let ticking = false;
function onScroll() {
  if (ticking) return; ticking = true;
  requestAnimationFrame(() => {
    ticking = false; story();
    if (S.view === "score") for (const e of [$("#auc"), $("#versus")]) e.style.setProperty("--t", REDUCED ? 1 : (1 - (1 - clamp((innerHeight - e.getBoundingClientRect().top) / (innerHeight * .55))) ** 3).toFixed(3));
  });
}
addEventListener("scroll", onScroll, { passive: true });
function redraw() { if (S.view === "watch") { drawHero(); S.games.forEach(g => { const e = document.getElementById("g-" + g.id); if (e) drawCard(e, g); }); } onScroll(); }
addEventListener("resize", () => requestAnimationFrame(redraw));

/* ───────── headline: every letter swings like an evaluation bar, then settles ───────── */
$$(".hl").forEach(h => {
  if (!h.getAttribute("aria-label") && h.tagName === "H1") h.setAttribute("aria-label", h.textContent.replace(/\s+/g, " ").trim());
  let i = 0;
  $$("[data-split]", h).forEach(line => { line.setAttribute("aria-hidden", "true"); line.innerHTML = line.textContent.split(" ").map(w => `<span class="w">${[...w].map(ch => `<span class="c" style="--i:${i++}">${esc(ch)}</span>`).join("")}</span>`).join(" "); });
  h._n = i; h.classList.add("split");
});
function headlines(view) {
  $$(".hl", view).forEach(h => {
    if (REDUCED) return h.classList.add("done");
    h.classList.remove("go", "done"); void h.offsetWidth; h.classList.add("go");
    clearTimeout(h._t); h._t = setTimeout(() => h.classList.add("done"), 1200 + h._n * 36);
  });
}

/* ───────── views, changed by slamming a stamped slip onto the table ───────── */
const VIEWS = { watch: ["#/", "Watch"], score: ["#/score", "Scoreboard"], how: ["#/how", "How it works"] };
const fromHash = () => Object.keys(VIEWS).find(v => VIEWS[v][0] === location.hash) || "watch";
function show(v) {
  S.view = document.body.dataset.view = v;
  $$(".view").forEach(s => s.hidden = s.dataset.view !== v);
  $$(".nav nav a[data-go]").forEach(a => a.dataset.go === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  document.title = v === "watch" ? "Gardyloo · it calls the blunder without seeing the board" : `${VIEWS[v][1]} · Gardyloo`;
  lenis ? (lenis.resize(), lenis.scrollTo(0, { immediate: true })) : scrollTo(0, 0);
  headlines($(`[data-view="${v}"]`)); redraw();
}
let busy = false;
function go(v) {
  if (v === S.view || busy) return;
  if (REDUCED) return show(v);
  const slam = $("#slam"); busy = true;
  $("#slamText").textContent = VIEWS[v][1]; slam.className = "slam in";
  setTimeout(() => { show(v); slam.className = "slam out"; }, 520);
  setTimeout(() => { slam.className = "slam"; busy = false; }, 960);
}
document.addEventListener("click", ev => {
  const a = ev.target.closest("a[data-go], a[data-scroll]"); if (!a) return;
  ev.preventDefault();
  if (a.dataset.scroll) return lenis ? lenis.scrollTo(a.dataset.scroll, { offset: -10 }) : $(a.dataset.scroll).scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" });
  if (a.dataset.go !== S.view) history.pushState(null, "", VIEWS[a.dataset.go][0]);
  go(a.dataset.go);
});
addEventListener("popstate", () => go(fromHash()));

/* ───────── pointer life: the watching eye, magnetic controls, an ink ring on press ───────── */
if (matchMedia("(pointer: fine)").matches && !REDUCED) {
  const eye = $("#eye"), pupil = eye.firstElementChild; let ex = 0, ey = 0, px = 0, py = 0, raf = 0;
  const tick = () => {
    ex += (px - ex) * .16; ey += (py - ey) * .16;
    const dx = px - (ex + 29), dy = py - (ey + 25), d = Math.hypot(dx, dy) || 1; // the eye sits below-right of the pointer and looks back at it
    eye.style.translate = `${ex}px ${ey}px`; pupil.style.translate = `${dx / d * 6}px ${dy / d * 3.2}px`;
    raf = Math.abs(px - ex) + Math.abs(py - ey) > .4 ? requestAnimationFrame(tick) : 0;
  };
  addEventListener("pointermove", ev => {
    if (!eye.classList.contains("on")) { ex = ev.clientX; ey = ev.clientY; eye.classList.add("on"); }
    px = ev.clientX; py = ev.clientY; eye.classList.toggle("wide", !!ev.target.closest?.("a, button, canvas.chart"));
    raf ||= requestAnimationFrame(tick);
  }, { passive: true });
  addEventListener("pointerdown", () => eye.classList.add("shut")); addEventListener("pointerup", () => eye.classList.remove("shut"));
  $$(".mag").forEach(m => {
    m.addEventListener("pointermove", ev => { const r = m.getBoundingClientRect(); m.style.transform = `translate(${(ev.clientX - r.left - r.width / 2) * .2}px, ${(ev.clientY - r.top - r.height / 2) * .3}px)`; });
    m.addEventListener("pointerleave", () => m.style.transform = "");
  });
}
document.addEventListener("pointerdown", ev => {
  const b = ev.target.closest(".btn"); if (!b || REDUCED) return;
  const r = b.getBoundingClientRect(), ring = el("span", "inkring"); ring.style.left = ev.clientX - r.left + "px"; ring.style.top = ev.clientY - r.top + "px";
  b.append(ring); ring.addEventListener("animationend", () => ring.remove());
});

/* ───────── go ───────── */
show(fromHash());
tote(); refresh(); loadResults(); loadExample();
new EventSource("/api/events").onmessage = m => onEvent(JSON.parse(m.data));
document.fonts?.ready.then(redraw);
