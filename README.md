# Dreaming Bird 🐦

A small transformer trained to *be* Flappy Bird. Feed it the current frame's state and the
player's action (flap / no-flap); it emits the next frame's state. Loop at 30–60 fps, render it,
read the spacebar, feed it back — and a real-time arcade game runs entirely inside the network's
autoregressive output. **No physics engine at runtime; the model *is* the physics.**

> Part of a set of "I trained a model for a thing I care about" world-model experiments, in the
> lineage of [World Models](https://arxiv.org/abs/1803.10122),
> [Othello-GPT](https://arxiv.org/abs/2210.13382),
> [GameNGen](https://arxiv.org/abs/2408.14837), and Genie. Flappy Bird is the deliberately
> *minimal* world model — the one most likely to work cleanly, and the best vehicle for the
> "does it learn hidden physics?" experiment.

## Status

**Playable.** Phases 1–4 are implemented and committed: a deterministic oracle + data engine, a
nanoGPT-style decoder, training, and a free autoregressive rollout you can play. A ~11M-param
model learns the physics (one-step bird_y 98.6% exact / 100% within ±1 bin; collisions within ±1
frame), and a controller playing *inside the dream* threads 5+ pipes with near-zero drift.
Cacheless dreaming runs at ~125 fps on an RTX 5090.

### Play it

```bash
uv sync
# desktop (pygame): spacebar to flap; the ghost is true physics under the same inputs
uv run python -m dreaming_bird.play --shadow
```

**Or just play it in your browser — no install:** **https://gmmeyer.github.io/gpt-bird/**
The model runs client-side via WebGPU, streamed from [🤗 gmmeyer/gpt-bird](https://huggingface.co/gmmeyer/gpt-bird).

The site lives in its own repo, [gmmeyer/gpt-bird](https://github.com/gmmeyer/gpt-bird), included
here as a submodule at [`web/`](./web). Run it locally with `python -m http.server 8000 --directory web`
(the model still streams from HF); re-export the model with
`uv run --with onnx --with onnxruntime python -m dreaming_bird.export_onnx`.

### Design & plan

- [`neural-flappy-bird-world-model.md`](./neural-flappy-bird-world-model.md) — the original design / idea document.
- [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) — the Phase 1–4 build plan, reviewed and revised via a four-way AI debate (Gemini / Codex / Sonnet / Opus).
- [`CLAUDE.md`](./CLAUDE.md) — locked design decisions and current status.

## The core idea

The hard problem here is **not** memory (pipes scroll away; there's nothing to remember). It's
(a) learning continuous 1-D physics — gravity + flap impulse — and (b) not drifting off the rails
during a long autoregressive rollout.

The elegant part: **velocity is never in the observed state.** If the bird flies well, the model
must have reconstructed velocity from the *sequence* of positions — exactly as integrating
physics requires. A later linear probe on the residual stream tests whether it really built that
hidden variable (the Othello-GPT result, but for a quantity that is genuinely latent).

## How it will work

```
state = initial_frame
loop at fixed fps:
    render(state)                  # draw bird + pipes
    a = flap if spacebar_down else noflap
    state = model.generate_next_frame(recent_frames, a)   # slot-constrained decode
    if state.is_gameover: break
```

## Roadmap (this repo targets Phases 1–4)

1. **Engine + logger** — deterministic, seeded Flappy Bird oracle that emits compact quantized
   frame streams; deterministic replay test.
2. **Nano model, no pipes** — does a ~0.25–2M model learn fall + jump (1-D kinematics)?
3. **Pipes + collision** — one-step accuracy + collision-frame timing vs the oracle.
4. **Free rollout + renderer** — swap the engine out; play it with the spacebar; measure drift.

Out of scope for now (seams left in place): scheduled-sampling training (5), the velocity probe
(6), and training an agent *inside the dream* (7).

## Target hardware & stack

- **GPU:** single NVIDIA RTX 5090 (32 GB).
- **Python 3.12** via [`uv`](https://github.com/astral-sh/uv); **PyTorch** with CUDA 12.8 (`cu128`)
  wheels (Blackwell / sm_120); a custom nanoGPT-style decoder; **pygame** for the live renderer.

See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for the full architecture, token grammar,
quantization scheme, drift-mitigation ladder, risks, and phase gates.
