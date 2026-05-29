# Dreaming Bird — development log

A narrative summary of building **Dreaming Bird**: a small transformer trained to *be* Flappy
Bird, where the network — not a physics engine — emits every frame. From design doc to a live,
playable website at **https://gpt-bird.com**.

---

## 1. Planning — design review via a 4-way AI debate

Started from the design doc (`neural-flappy-bird-world-model.md`) and ran a structured **4-way
debate** (Gemini, Codex, Sonnet, Opus; 2 rounds, cross-critique) to pressure-test the approach.
It converged on a plan, written to **`IMPLEMENTATION_PLAN.md`**. Locked decisions:

- **State = compact quantized scalars** (`bird_y`→128 bins, `pipe_dx`→64, `gap_y`→32), **velocity
  deliberately excluded** — the model must reconstruct it from the sequence of positions.
- **Token grammar:** fixed-length, one token per field, disjoint id ranges, **slot-constrained
  decoding** so malformed frames are impossible. No delimiter parsing.
- **Ordering** `S_t, A_t, S_{t+1}` with the loss **masked off action tokens** (they're input).
- **Stochastic pipe spawn:** a new gap is RNG-drawn → **sample** (don't argmax) the gap slot; in
  eval, score spawn frames on *validity*, not identity.
- **Drift mitigation is a measured ladder**, noise-augmentation gated off unless evidence demands.

## 2. Toolchain

`uv` + **Python 3.12**, **PyTorch 2.11 + CUDA 12.8** (Blackwell / sm_120). Verified the **RTX 5090**
runs fp32/bf16 matmuls + autograd before building on it.

## 3. Build — Phases 1–4 (each committed separately)

| Phase | What | Result |
|---|---|---|
| **1 — Oracle + data + eval** | `engine.py` (deterministic, seeded; velocity not in `Obs`; collision-after-update), `tokenizer.py` (the locked grammar, vocab 232), `policies.py`/`data.py` (blended controllers → packed `uint16`), `eval.py` (model-agnostic replay/rollout harness) | 19 tests pass; data tuned to **66% of frames from 5+ pipe runs** |
| **2 — Nano model, no pipes** | `model.py` (~1.9M nanoGPT decoder: RMSNorm, SDPA, tied embeddings), `train.py` (AdamW + cosine + bf16, masked loss, gated noise-aug) | one-step `bird_y` **96.5% exact / 99.4% within ±1**; long drift horizon → **noise-aug stays off** |
| **3 — Pipes + collision** | `small` tier (~11M, ctx 256 frames); `evaluate_pipes()` splits gap into stable (exact) vs spawn (validity) | held-out: bird_y **98.6%/100%**, pipe_dx 100%, gap stable 99.96%, **spawn validity 100%**, collisions **94.9% within ±1 frame** |
| **4 — Free rollout + renderer** | `rollout.py` (cacheless `DreamStepper`), `play.py` (pygame, spacebar, "shadow" ghost of true physics) | **125 fps** on the 5090 (no KV-cache needed); autopilot threads **5+ pipes** with **~0 bird_y drift** |

The model genuinely became the game: at play time the engine is gone.

## 4. The web port — the network runs in your browser

- **`export_onnx.py`** exports the checkpoint to ONNX. The decoder's `RMSNorm`/SDPA are swapped for
  explicit-op equivalents *at export time* so the graph uses only WebGPU-supported ops; validated
  against PyTorch (max logit diff **1.3e-5**, argmax agrees).
- **`web/index.html` + `web/app.js`** run the model client-side via **onnxruntime-web (WebGPU**,
  WASM fallback), mirroring the Python decode loop exactly. Redesigned visuals: gradient sky,
  parallax clouds, a scrolling multi-pipe world, a velocity-tilted bird, polished cards.

## 5. Deployment

- **🤗 Hugging Face** `gmmeyer/gpt-bird`: `model.onnx`, `config.json`, both checkpoints, model card.
- **GitHub Pages** `gmmeyer/gpt-bird`: the static site, which **streams the model from HF**
  (cross-origin verified). The site is included in the source repo as a **submodule at `web/`**.
- **Custom domain `gpt-bird.com`** — CNAME + DNS (apex A records) + **HTTPS enforced**.

Three repos: `gmmeyer/gptbird` (source + training), `gmmeyer/gpt-bird` (the site), HF `gmmeyer/gpt-bird` (the weights).

## 6. The collision saga (the interesting bug)

**Symptom (reported):** the bird dies when nothing is there — phantom collisions.

**Diagnosis (from death traces):** every phantom death is **one frame after passing a pipe**. A new
pipe spawns with a gap far from the bird; the model reads "bird far from gap → dead" **without
gating on the pipe actually being close** (it fired with the pipe ~119px away). The model learned a
loose collision association.

**Confirmed weight-independent:** a sweep of the DEAD-token loss weight (×2, ×4, ×20) left the
phantom rate ~1.0 — so it's not a tuning problem, it's a representation/exposure-bias problem.

**Fix that shipped — the geometry guard:** the model's `bird_y`/`pipe_dx`/`gap_y` are *accurate*;
only its binary death flag is broken. So collision is now **derived from the model's own dreamed
positions** (`engine.collides`, mirrored in `web/app.js`) instead of its status token. Audited:
**phantom rate 0.00**, anti-gap collision recall matches the real engine (**~58 frames**). Bonus:
one fewer forward pass per frame. No model re-upload needed — the deployed model was fine, we just
stopped trusting its death bit. `rollout.collision_audit` is the regression check.

## 7. Trying to fix it *in the model* (death detection)

Goal: make the model's own status token reliable so the guard becomes a safety net, not the
mechanism.

- **Attempt 1 — lazy-controller data augmentation** (`lazy_center_policy`: loiter at center until
  the pipe is close, flooding the "far from gap + far pipe + alive" regime). **Negative result:**
  raw-status phantom rate only moved 1.00 → 0.92 and collision *recall got worse*. Passive
  teacher-forced data doesn't reach the model's own failure distribution.
- **Attempt 2 — DAgger / scheduled-sampling — partially worked, then plateaued.** A **batched dream
  rollout** (`dagger.py`) generates the model's *own* visited states and relabels each frame's
  status with the geometry oracle; we fine-tune on those (`train.py --init-from`), iterating with
  `dagger_loop.py`. **Result:** RAW-status phantom rate **1.00 → ~0.45** (roughly halved) and
  collision recall fixed (anti-gap 72 → ~58 ≈ engine), one-step accuracy intact. But it **plateaus
  at ~0.45** — rounds 2–3 didn't compound (0.32 / 0.44 / 0.39, within metric noise; n=100 ≈ 0.46).
  So the model's native death flag got much better but is **still not solo-reliable** (~45% of its
  deaths would be phantom without the guard). **Conclusion: the geometry guard stays** (it gives
  0.00 phantom on any model); the deployed model is unchanged. (Key gotcha found the slow way: the
  batched rollout must run under **bf16 autocast** — fp32 at B=256 gets no tensor cores on the 5090
  and is ~15× slower; a batch dropped from minutes to ~4 s once fixed.)

  *Why it plateaus (hypotheses, untested):* fine-tuning each round from the phantom-prone deployed
  init may re-import the bad prior; and a single status bit predicted from one frame may lack the
  multi-frame "approach" context to gate cleanly on `dx`. Future angles: fine-tune from the latest
  model (not deployed) or train from scratch on the aggregate; feed `dx` more directly; or just
  keep the guard (it's correct and free).

## 8. Status & open threads

- **Live and correct:** https://gpt-bird.com plays correctly (geometry guard); model on HF; site on Pages.
- **In flight:** the DAgger fine-tune experiment for model-native death detection (uncertain payoff;
  the guard stays regardless).
- **Deferred (from the original plan):** Phase 5 full scheduled-sampling, **Phase 6 velocity
  linear-probe** (the headline "did it really learn the hidden variable?" experiment), Phase 7
  agent-trained-in-the-dream.

## 9. Honest takeaways

- The transformer learns the **world** — gravity, flap impulse, pipe scrolling, gap placement —
  remarkably well (one-step ~99%, near-zero free-rollout drift). Velocity, never observed, is
  recovered from the position sequence.
- Its weak spot is the **rare, binary collision event** in its *own* rollout distribution — a
  textbook exposure-bias failure that quantization and loss-weighting don't fix. The pragmatic
  answer (read collisions off the model's accurate positions) makes the game correct today; the
  principled answer (DAgger) is being tested.
- Engineering lesson logged: the DAgger generator wants a KV-cache + progress logging — the
  growing-context recompute is the bottleneck.
