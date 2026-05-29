# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

**Dreaming Bird** — a decoder-only transformer trained to *be* Flappy Bird: `(state, action) →
next state`, looped autoregressively at 30–60 fps so the model replaces the physics engine at
runtime. See `README.md` for the pitch and `neural-flappy-bird-world-model.md` for the full design.

## Current status

**Phase 1 complete (engine + tokenizer + data + eval harness); Phase 2 (nano model) is next.**
The authoritative build plan is **`IMPLEMENTATION_PLAN.md`**; read it before implementing.

Done so far: deterministic seeded engine (`engine.py`), the locked token grammar with
slot-constrained decoding (`tokenizer.py`), blended behavior policies + packed-`uint16` data
generation (`policies.py`, `data.py`), and the model-agnostic eval/replay harness (`eval.py`).
19 tests pass; the toolchain go/no-go (torch 2.11+cu128, RTX 5090 / sm_120, bf16, autograd) is
verified. Next: `model.py` (nano GPT decoder) and `train.py`, then the Phase-2 no-pipes
drift-horizon gate.

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
