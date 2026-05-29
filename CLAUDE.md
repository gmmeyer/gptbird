# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

**Dreaming Bird** — a decoder-only transformer trained to *be* Flappy Bird: `(state, action) →
next state`, looped autoregressively at 30–60 fps so the model replaces the physics engine at
runtime. See `README.md` for the pitch and `neural-flappy-bird-world-model.md` for the full design.

## Current status

**Phases 1–4 complete (the "through playable rollout" scope). A browser port is in progress.**
The authoritative build plan is **`IMPLEMENTATION_PLAN.md`**; read it before implementing.

Phase 1: deterministic engine (`engine.py`), locked token grammar with slot-constrained decoding
(`tokenizer.py`), blended policies + packed-`uint16` data (`policies.py`, `data.py`), eval/replay
harness (`eval.py`). Phase 2: nanoGPT decoder (`model.py`) + training (`train.py`); nano model
(no pipes) hit 96.5%/99.4% one-step bird_y with a long drift horizon — **noise-aug stays OFF**
(gated fallback). Phase 3: `small` (~11M, ctx 256 frames) trained on pipes data with the rare
DEAD status token up-weighted (`--dead-weight 20`) to fix collision recall. Held-out one-step:
bird_y 98.6% exact / 100% within-±1, pipe_dx 100%, gap_y stable 99.96%, **gap-spawn validity
100%** (scored on validity not identity, since a new gap is RNG-drawn), collisions 94.9% within
±1 frame. `evaluate_pipes()` in `eval.py` is the Phase-3 metric. 19 tests pass.

Phase 4: `rollout.py` (cacheless `DreamStepper`, fps `benchmark`, `free_rollout_drift`) and
`play.py` (pygame). Cacheless dreaming runs at **125 fps** on the 5090 (no KV cache needed).
Autopilot playing the dream threads **5+ pipes** with **~zero bird_y drift** (never exceeded 3
bins over 16 runs; mean 0.49). Play it: `uv run python -m dreaming_bird.play --shadow`
(spacebar flaps; the ghost is true physics under the same actions). Collisions fire in-dream
(real game-over), gaps are sampled (slot 2) so the dream invents its own pipes.

**Web port (done):** `export_onnx.py` exports the checkpoint to `web/model.onnx` (+ `config.json`)
— validated against PyTorch (max logit diff 1.3e-5). `web/index.html` + `web/app.js` run the model
client-side via onnxruntime-web (WebGPU, WASM fallback); `app.js` mirrors `rollout.py`'s decode
loop exactly. The model exports with an explicit-RMSNorm + explicit-attention path (see
`is_in_onnx_export()` in `model.py` and `_swap_rmsnorm` in `export_onnx.py`) so the ONNX graph uses
only WebGPU-supported ops. `web/model.onnx` and `web/config.json` are gitignored build artifacts —
regenerate with `export_onnx`. Build/deploy notes: `web/README.md`.

**Deployed:** the model is on Hugging Face (`gmmeyer/gpt-bird`) and the site is live at
https://gmmeyer.github.io/gpt-bird/ . The site is its own repo (`gmmeyer/gpt-bird`), included
here as a **submodule at `web/`** — edit the site inside `web/`, push that repo, then update the
submodule pointer here. `app.js` loads the model from the HF `MODEL_BASE` URL; HF serves the
files with permissive CORS so the Pages origin can fetch them.

## Locked decisions — do NOT re-litigate or silently "fix"

These came out of a reviewed design + a four-way AI debate. If you think one is wrong, raise it
explicitly; don't quietly change it.

- **Velocity is deliberately EXCLUDED from the observed state.** It is the hidden variable the
  model must reconstruct from the sequence of positions — that's the whole research point (the
  Phase-6 linear-probe experiment). Do not add velocity to the state to "make it learn faster."
- **State rep = §2.1 compact quantized scalars** (`bird_y` 128 bins, `pipe_dx` 64, `gap_y` 32,
  status). Not ASCII frames, not a continuous regression head.
- **Token grammar = fixed-length, one-token-per-field, disjoint per-field ranges.** No digit
  tokens, no `by:`-style text, **no `</frame>` parsing.** At inference, use **slot-constrained
  logit masking** (`pos mod K` → legal field range) so malformed frames are impossible by
  construction. Never write a delimiter-scanning parser.
- **Stream ordering = `S_t, A_t, S_{t+1}`** (the action precedes the state it causes). **Mask the
  loss off the action tokens** — actions are conditioning, not prediction targets.
- **Stochastic pipe spawn:** a new pipe's `gap_y` is RNG-drawn and unpredictable from context.
  At inference **sample (don't argmax) at gap slots**; in eval, **exclude spawn frames from exact
  one-step accuracy** and score them on validity/distribution instead.
- **Collision is derived from geometry, NOT the model's status token** (`engine.collides`, mirrored
  in `web/app.js`). The model's `bird_y`/`pipe_dx`/`gap_y` are accurate, but its alive/dead flag is
  unreliable in long free rollouts — it phantom-dies the frame after a pipe is passed (a new far
  gap appears and it reads "bird far from gap" as death even with `dx`≈119). So the playable
  rollout generates only the 3 world tokens and computes alive/dead from them; the geometry-correct
  status token is appended to keep the context on-distribution. Audited: phantom-death rate 0.00,
  anti-gap survival matches the engine (~58). A model-side fix (so the status token itself gates on
  `dx`) is a scheduled-sampling/training problem, deferred. `rollout.collision_audit` is the
  regression check (`derive_status=False` measures the model's *raw* status; `True` the guard).
  - **Tried & failed (negative result):** lazy-controller data augmentation
    (`policies.lazy_center_policy`, `data --p-lazy`) to teach dx-gated death. Raw-status phantom
    rate only moved 1.00 → 0.92 and anti-gap recall got *worse* (71 → 101). Passive teacher-forced
    data doesn't fix the exposure bias — the model fails on its *own* rollout distribution. The
    real fix is DAgger / scheduled-sampling (roll out the model, relabel status with the geometry
    oracle, fine-tune on those self-generated states), which needs a batched rollout for speed.
  - **DAgger works (`dagger.py`, batched + bf16 autocast; `train.py --init-from`).** One round
    (3k self-generated episodes relabeled by the geometry oracle, 2k fine-tune iters) cut the
    model's RAW-status phantom rate **1.00 → 0.44** and improved collision recall (anti-gap 72 →
    56 ≈ engine), with one-step accuracy intact. Iterative rounds (generate from the improved
    model, repeat) should push it lower. The geometry guard stays authoritative in the live game
    until raw phantom is low enough to drop it. (NB: batched rollout MUST run under bf16 autocast
    — fp32 at B=256 has no tensor-core path on the 5090 and is ~15x slower.)
- **Drift mitigation is a measured ladder, applied only as Phase-2 drift-horizon evidence
  demands:** quantization → random-reset data → cheap context-noise augmentation (gated, off by
  default) → short unrolls → full scheduled sampling (Phase 5). Don't jump to scheduled sampling
  preemptively.
- A **deterministic eval/replay harness is a Phase-1 deliverable**, not an afterthought.

## Environment

- **Python 3.12 via `uv`** — NOT the system Python 3.14 (no PyTorch wheels for 3.14).
- **PyTorch with CUDA 12.8 (`cu128`) wheels** — required for the RTX 5090 (Blackwell / sm_120).
  Day-1 go/no-go: confirm `torch.cuda.is_available()` and the device name inside the venv.
- **`torch.compile` is optional, not a dependency** — Triton/inductor is fragile on
  Windows + Blackwell. Make everything run without compile first; add it behind a flag.
- Renderer: **pygame**. Run inference on a background thread with a size-1 frame queue.
- OS is **Windows**; this is a **PowerShell**-first environment (use PowerShell syntax; Bash is
  available via git-bash for POSIX scripts).

## Commands

```powershell
uv sync                                              # install pinned deps (cu128 torch)
uv run pytest -q                                     # 19 tests: determinism, round-trip, alignment
uv run python -m dreaming_bird.data --episodes 4000 --out data/train   # generate a dataset
uv run python -m dreaming_bird.eval                  # eval-harness self-test (oracle vs oracle)
uv run python -m dreaming_bird.train --tier nano     # train (planned, Phase 2)
uv run python -m dreaming_bird.play                  # play the dream (planned, Phase 4)
```

## Gotchas

- If using the Octopus multi-LLM skills here: the **Gemini CLI** needs
  `GEMINI_CLI_TRUST_WORKSPACE=true` + `--skip-trust`, and **`codex exec`** must receive its prompt
  via **stdin redirect** (`codex exec ... < prompt.txt`) or it hangs waiting on stdin.
- `data/` and `checkpoints/` are gitignored — don't commit generated trajectories or weights.
