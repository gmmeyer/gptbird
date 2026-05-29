"use strict";
/*
 * Dreaming Bird — client-side. A transformer (model.onnx) emits each frame autoregressively
 * via onnxruntime-web. This mirrors the Python decode loop exactly: per frame, append the
 * action token, then generate 4 state tokens (bird_y, pipe_dx, gap_y, status) with per-slot
 * legal masking (malformed frames impossible); the gap slot is sampled (new pipes are invented),
 * the rest are greedy. No physics engine — the network is the physics.
 */

const canvas = document.getElementById("game");
const g = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const setStatus = (h) => { statusEl.innerHTML = h; };

const TARGET_FPS = 30;
let cfg, off, bins, eng, tks, BOS, CAP, slotSpan, scale;
let session = null, backend = "—";
let state = "loading";                 // loading | title | playing | gameover
let ctx = [], bird = 0, pipeDx = 0, gapY = 0, alive = true, frame = 0, pipes = 0, prevDx = 0;
let shadow = { y: 0, vy: 0 };
let flapQueued = false, stepping = false, lastInfMs = 0, bestPipes = 0;

const clampQuant = (v, lo, hi, n) => Math.min(Math.max(Math.floor(((v - lo) / (hi - lo)) * n), 0), n - 1);
const dequant = (b, lo, hi, n) => lo + ((b + 0.5) / n) * (hi - lo);

async function init() {
  cfg = await (await fetch("config.json")).json();
  off = cfg.offsets; bins = cfg.bins; eng = cfg.engine; tks = cfg.tokens;
  BOS = cfg.specials.BOS; CAP = cfg.web_context_cap;
  slotSpan = [[off.by, off.dx], [off.dx, off.gap], [off.gap, off.status], [off.status, off.action]];

  scale = canvas.width / eng.width;
  canvas.height = Math.round(eng.height * scale);

  setStatus("loading model (~45&nbsp;MB, one-time)…");
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    try {
      session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["webgpu", "wasm"] });
      backend = "WebGPU";
    } catch (e) {
      session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
      backend = "WASM (no WebGPU — slower)";
    }
  } catch (e) {
    setStatus("failed to load model: " + e.message);
    return;
  }
  // warm up the graph so the first real frame isn't janky
  await runLast([BOS, off.by, off.dx, off.gap, tks.status_alive]);
  resetGame();
  state = "title";
  setStatus('backend: <span id="backend">' + backend + "</span> &nbsp;·&nbsp; press <b>Space</b> / tap to start");
  draw();
  loop();
}

async function runLast(window) {
  const w = window.slice(-CAP);
  const data = BigInt64Array.from(w, (v) => BigInt(v));
  const t = new ort.Tensor("int64", data, [1, w.length]);
  const out = await session.run({ idx: t });
  return out.logits.data;                 // Float32Array, length vocab
}

async function genFrame(action) {
  ctx.push(action ? tks.action_flap : tks.action_noflap);
  const st = [];
  for (let s = 0; s < 4; s++) {
    const logits = await runLast(ctx);
    const [lo, hi] = slotSpan[s];
    let tok;
    if (s === 2) {                        // sample gap_y (stochastic pipe spawn)
      let mx = -Infinity;
      for (let i = lo; i < hi; i++) if (logits[i] > mx) mx = logits[i];
      let sum = 0; const p = new Float64Array(hi - lo);
      for (let i = lo; i < hi; i++) { const e = Math.exp(logits[i] - mx); p[i - lo] = e; sum += e; }
      let r = Math.random() * sum, acc = 0; tok = hi - 1;
      for (let i = 0; i < hi - lo; i++) { acc += p[i]; if (r <= acc) { tok = lo + i; break; } }
    } else {                              // greedy bird_y / pipe_dx / status
      let mx = -Infinity; tok = lo;
      for (let i = lo; i < hi; i++) if (logits[i] > mx) { mx = logits[i]; tok = i; }
    }
    st.push(tok); ctx.push(tok);
  }
  if (ctx.length > 3 * CAP) ctx = ctx.slice(-CAP);   // bound memory
  return st;
}

function decodeState(st) {
  return {
    bird_y: dequant(st[0] - off.by, 0, eng.height, bins.bird_y),
    pipe_dx: dequant(st[1] - off.dx, 0, eng.max_dx, bins.pipe_dx),
    gap_y: dequant(st[2] - off.gap, 0, eng.height, bins.gap_y),
    alive: st[3] === tks.status_alive,
  };
}

function resetGame() {
  ctx = [BOS,
    off.by + clampQuant(eng.start_y, 0, eng.height, bins.bird_y),
    off.dx + clampQuant(eng.width - eng.bird_x, 0, eng.max_dx, bins.pipe_dx),
    off.gap + clampQuant(eng.height / 2, 0, eng.height, bins.gap_y),
    tks.status_alive];
  const s = decodeState(ctx.slice(1, 5));
  bird = s.bird_y; pipeDx = s.pipe_dx; gapY = s.gap_y;
  alive = true; prevDx = pipeDx; frame = 0; pipes = 0;
  shadow = { y: eng.start_y, vy: 0 };
}

async function step() {
  if (stepping) return;
  stepping = true;
  const action = flapQueued; flapQueued = false;
  const t0 = performance.now();
  const s = decodeState(await genFrame(action));
  lastInfMs = performance.now() - t0;
  bird = s.bird_y; pipeDx = s.pipe_dx; gapY = s.gap_y;
  // shadow ghost: true physics under the same action (no pipes)
  if (action) shadow.vy = eng.flap_impulse;
  shadow.vy += eng.gravity; shadow.y += shadow.vy;
  if (pipeDx > prevDx + eng.width * 0.3) pipes++;
  prevDx = pipeDx;
  if (!s.alive) { alive = false; state = "gameover"; bestPipes = Math.max(bestPipes, pipes); }
  frame++;
  stepping = false;
}

async function loop() {
  const t0 = performance.now();
  if (state === "playing" && alive && !stepping) await step();
  draw();
  const elapsed = performance.now() - t0;
  setTimeout(loop, Math.max(0, 1000 / TARGET_FPS - elapsed));
}

function px(v) { return Math.round(v * scale); }

function draw() {
  const W = canvas.width, H = canvas.height;
  g.fillStyle = "#11151e"; g.fillRect(0, 0, W, H);

  // next pipe (top + bottom)
  const x = px(eng.bird_x + pipeDx), pw = px(eng.pipe_width);
  const gt = px(gapY - eng.gap_height / 2), gb = px(gapY + eng.gap_height / 2);
  g.fillStyle = "#3ca85a";
  g.fillRect(x, 0, pw, gt);
  g.fillRect(x, gb, pw, H - gb);
  g.fillStyle = "#2c7d44";
  g.fillRect(x, gt - px(8), pw, px(8));
  g.fillRect(x, gb, pw, px(8));

  // shadow ghost (true physics)
  g.strokeStyle = "#7a82a0"; g.lineWidth = 2;
  g.beginPath(); g.arc(px(eng.bird_x), px(shadow.y), px(eng.bird_radius), 0, 7); g.stroke();

  // dreamed bird
  g.fillStyle = alive ? "#f0d24a" : "#e0564f";
  g.beginPath(); g.arc(px(eng.bird_x), px(bird), px(eng.bird_radius), 0, 7); g.fill();
  g.fillStyle = "#1a1d27";
  g.beginPath(); g.arc(px(eng.bird_x) + px(4), px(bird) - px(3), Math.max(2, px(2.5)), 0, 7); g.fill();

  // HUD
  g.fillStyle = "#d7dbe6"; g.font = Math.round(15 * scale) + "px ui-monospace, monospace";
  g.fillText("pipes " + pipes, 10, 24);
  g.fillStyle = "#6b7286"; g.font = Math.round(11 * scale) + "px ui-monospace, monospace";
  const fps = lastInfMs > 0 ? (1000 / Math.max(lastInfMs, 1000 / TARGET_FPS)).toFixed(0) : "—";
  g.fillText(backend + "  ·  " + fps + " fps  ·  " + lastInfMs.toFixed(1) + " ms/frame", 10, H - 12);

  if (state === "title" || state === "gameover") {
    g.fillStyle = "rgba(8,10,16,0.66)"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#f0d24a"; g.textAlign = "center";
    g.font = "bold " + Math.round(22 * scale) + "px ui-monospace, monospace";
    g.fillText(state === "gameover" ? "the dream ended" : "the dreaming bird", W / 2, H / 2 - 18);
    g.fillStyle = "#d7dbe6"; g.font = Math.round(13 * scale) + "px ui-monospace, monospace";
    if (state === "gameover")
      g.fillText("pipes: " + pipes + "   best: " + bestPipes, W / 2, H / 2 + 12);
    g.fillStyle = "#8b93a7";
    g.fillText("Space / tap to " + (state === "gameover" ? "dream again" : "begin"), W / 2, H / 2 + 40);
    g.textAlign = "left";
  }
}

function onFlap() {
  if (state === "title") { state = "playing"; flapQueued = true; }
  else if (state === "playing") { flapQueued = true; }
  else if (state === "gameover") { resetGame(); state = "playing"; flapQueued = true; }
}

addEventListener("keydown", (e) => {
  if (e.code === "Space" || e.key === " ") { e.preventDefault(); onFlap(); }
});
canvas.addEventListener("pointerdown", (e) => { e.preventDefault(); onFlap(); });

init();
