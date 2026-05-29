# Dreaming Bird — browser demo

The trained world model, running **entirely client-side**. No server, no backend: a ~11M-param
transformer (`model.onnx`) emits each frame autoregressively via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on **WebGPU** (WASM fallback), and
the page renders it to a canvas. The engine is gone — the network is the physics.

## Build & run locally

`model.onnx` and `config.json` are build artifacts (gitignored). Generate them from a trained
checkpoint, then serve the folder:

```bash
# 1. export the checkpoint to ONNX + config (writes web/model.onnx, web/config.json)
uv run --with onnx --with onnxruntime python -m dreaming_bird.export_onnx \
    --checkpoint checkpoints/small_pipes.pt

# 2. serve the static folder and open it
python -m http.server 8000 --directory web
#    -> http://localhost:8000   (use a WebGPU browser; see below)
```

(If you don't have a checkpoint yet, train one first — see the repo `IMPLEMENTATION_PLAN.md`,
Phases 2–3.)

## Deploy (shareable static site)

It's just static files — host `index.html`, `app.js`, `config.json`, and `model.onnx` on any
static host (GitHub Pages, Netlify, Cloudflare Pages, itch.io). No server or GPU on the host;
each visitor's browser runs the model. First load downloads the ~45 MB model once (cached after).

## Requirements

- **WebGPU browser for full speed** — Chrome/Edge 113+, recent Firefox/Safari. Without WebGPU it
  falls back to single-threaded WASM (still plays, just slower). The HUD shows the active backend
  and ms/frame.

## How it works

Each frame is 4 tokens — `bird_y, pipe_dx, gap_y, status` — generated with **slot-constrained
decoding**: logits are masked to the legal id range for each field, so a malformed frame is
impossible. The `gap_y` slot is **sampled** (a newly revealed pipe's gap is genuinely
unpredictable, so the dream invents one); the rest are greedy. `app.js` mirrors the Python decode
loop in `dreaming_bird/rollout.py` exactly. Velocity is never in the state — the model
reconstructs it from the sequence of positions.
