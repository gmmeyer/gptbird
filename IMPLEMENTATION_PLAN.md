# Dreaming Bird — Implementation Plan (Phases 1–4)

> A small decoder-only transformer trained to *be* Flappy Bird: feed it the current frame's
> quantized state + the player's action, it emits the next frame's state. Loop at 30–60 fps,
> render it, read the spacebar, feed it back. No physics engine at runtime — the model *is* the
> physics. Design doc: [`neural-flappy-bird-world-model.md`](./neural-flappy-bird-world-model.md).

**Scope of this plan:** Phases 1–4 (engine + data oracle → nano/small transformer → playable
autoregressive rollout). Phases 5–7 (scheduled sampling, velocity probe, agent-in-the-dream)
are out of scope but the design leaves clean seams for them.

**Status:** reviewed via a 4-way AI debate (Gemini / Codex / Sonnet / Opus, 2 rounds,
cross-critique). Synthesis at `~/.claude-octopus/debates/local/001-dreaming-bird-plan/`.

---

## 0. Environment & tech stack

| Choice | Decision | Rationale |
|---|---|---|
| Python | **3.12** via `uv` (NOT system 3.14) | PyTorch has no 3.14 wheels; 3.12 is the safe modern target |
| DL framework | **PyTorch ≥ 2.9, CUDA 12.8 (`cu128`) wheels** | Blackwell / sm_120 (RTX 5090) needs cu128 |
| Model | Custom nanoGPT-style decoder (~300 lines) | Own it fully; tiny vocab/model needs no framework |
| Engine/data | `numpy` + pure Python | Oracle is ~100 lines, CPU-bound, parallelizable |
| Renderer | `pygame` | Cross-platform window + spacebar input on Windows |
| Precision/speed | bf16 autocast + KV-cache; `torch.compile` **optional** | compile is a *bonus*, not a dependency (Windows+Triton fragile) |
| GPU (confirmed) | RTX 5090, 32 GB | Target hardware present on dev box |

**Day-1 go/no-go:** create the `uv` venv, install torch cu128, and verify
`torch.cuda.is_available()` + device name = RTX 5090 *inside the venv* before anything else.

---

## 1. Locked design decisions (from the review)

These are settled and should not be re-litigated mid-build:

### 1.1 State representation — §2.1 compact quantized scalars
Each state is a few quantized position scalars. **Velocity is deliberately excluded** — it is the
hidden variable the model must reconstruct from the sequence of positions (the Phase-6 probe
experiment). ASCII-frame (§2.2) was rejected: ~200–800 tokens/frame vs ~4, slower decode, worse
drift.

Default quantization (all are config params):
- `bird_y` → **128 bins** (start here; 64 makes gravity ≈ 0.5 bin/frame, borderline stair-stepping)
- `pipe_dx` (horizontal distance to next pipe) → **64 bins**
- `gap_y` (vertical center of next gap) → **32 bins** (needs less resolution)
- `status` ∈ {ALIVE, DEAD}

### 1.2 Token grammar — fixed-length, one-token-per-field, slot-constrained
**No digit tokens, no `by:` text, no `</frame>` parsing** (the delimiter was a hang/garbage trap).
Each field value is its own token in a **disjoint range**:

```
Vocab (~232 tokens):
  BY_0..BY_127      (128)   bird_y bins
  DX_0..DX_63       (64)    pipe_dx bins
  GAP_0..GAP_31     (32)    gap_y bins
  ST_ALIVE, ST_DEAD (2)     status
  A_FLAP, A_NOFLAP  (2)     action
  <BOS> <EOS> <SEP> <PAD>   (4)   specials
```

**Stream layout** (ordering matters — see 1.3). One frame unit = 4 generated state tokens +
1 supplied action token:

```
<BOS>
BY_64  DX_48  GAP_18  ST_ALIVE   A_NOFLAP
BY_66  DX_46  GAP_18  ST_ALIVE   A_NOFLAP     # gravity: accelerating down
BY_69  DX_44  GAP_18  ST_ALIVE   A_FLAP
BY_61  DX_42  GAP_18  ST_ALIVE   A_NOFLAP     # flap impulse: jumped up
...
BY_104 DX_02  GAP_18  ST_DEAD                 # collided with pipe
<EOS>
```

**Constrained decoding (inference):** at generation slot `i`, mask logits to that slot's legal
field range (`i mod 5` → BY / DX / GAP / ST). Malformed frames become **impossible by
construction** — strictly better than post-hoc clamping. No parser, ever; just read the last 4
generated tokens by slot. Keep `<SEP>` only as an optional human-readable anchor, never relied on
for parsing.

### 1.3 Action / state ordering (the silent killer)
The stream is **`S_t , A_t , S_{t+1} , A_{t+1} , …`**: the action that *causes* a transition
**precedes** the resulting state. The model predicts `S_{t+1}` conditioned on `(S_{≤t}, A_t)`.
**Loss is masked off the action tokens** — actions are conditioning, not prediction targets
(they're user-supplied at inference). A one-frame misalignment here makes the physics unlearnable
and looks like "the model just doesn't work" — get it right in Phase 1 and assert it in a test.

### 1.4 Stochastic pipe spawn (must handle explicitly)
A newly spawned pipe's `gap_y` is drawn from the oracle's RNG and is **not recoverable from
context**. Therefore:
- **Inference:** *sample* (do not argmax) at `GAP` slots, so the dream invents plausible pipes
  instead of a mean-blurred average. (Optional: an external seeded pipe scheduler can inject
  realized gaps when you want determinism for oracle comparison.)
- **Eval:** **exclude spawn frames from exact one-step accuracy.** Score them on *validity /
  distribution fidelity* (is the gap legal? NLL / calibration / KL vs the legal-gap distribution)
  — "judged on validity, not identity." Once a pipe exists on screen, subsequent frames are
  deterministic and ARE scored on identity.

---

## 2. Project structure

```
gptbird/
├── pyproject.toml                 # uv-managed, pinned deps
├── README.md
├── IMPLEMENTATION_PLAN.md         # this file
├── neural-flappy-bird-world-model.md
├── src/dreaming_bird/
│   ├── config.py                  # dataclass configs: engine, tokenizer, model tiers, train
│   ├── engine.py                  # P1 — deterministic Flappy Bird oracle
│   ├── policies.py                # P1 — random + scripted controller + random-reset starts
│   ├── tokenizer.py               # P1 — state ⇄ token stream, slot map, logit-mask helper
│   ├── data.py                    # P1 — trajectory gen + packed .bin shards (uint16)
│   ├── model.py                   # P2 — GPT decoder (attn/MLP/RMSNorm, tied embeddings)
│   ├── train.py                   # P2/3 — AdamW + cosine LR + bf16; noise-aug flag; loss mask
│   ├── rollout.py                 # P4 — KV-cached slot-constrained decode → state
│   ├── play.py                    # P4 — pygame loop (bg inference thread, size-1 frame queue)
│   └── eval.py                    # P1+ — replay determinism, one-step acc, drift horizon, etc.
└── tests/
    ├── test_engine.py             # replay determinism from (seed, actions)
    ├── test_tokenizer.py          # encode→decode round-trip is lossless
    └── test_alignment.py          # S_t,A_t,S_{t+1} ordering + loss-mask correctness
```
`data/` and `checkpoints/` are gitignored.

---

## 3. Phase-by-phase

### Phase 1 — Engine + tokenizer + eval harness
**Build:**
- `engine.py`: fixed timestep; bird `y, vy`; `vy += gravity`; flap ⇒ `vy = flap_impulse`
  (instantaneous set, not add); `y += vy`. Pipes scroll left at constant speed; periodic spawn
  with seeded-random gap center; fixed gap height. **Collision convention (frozen):** check
  collision *after* the position update; if bird hits pipe/floor/ceiling, emit the frame with
  `ST_DEAD` and terminate. Fully reproducible from `(seed, action_sequence)`.
- `tokenizer.py`: quantize/dequantize each field; expose the slot map and a `legal_mask(slot)`
  helper for constrained decoding.
- `policies.py`: behavior policies (see §4) including random-reset starts.
- `data.py`: generate trajectories in parallel, pack to `uint16` `.bin` shards.
- `eval.py` (harness skeleton): given `(seed, actions)`, run oracle vs (later) model and report
  **first-divergence frame, mean `bird_y` error, collision-frame delta, malformed-token count,
  realtime toks/sec**.

**Gate:**
- `test_engine.py` — `(seed, actions)` → run → serialize → deserialize → re-run from deserialized
  state → **bit-identical** output.
- `test_tokenizer.py` — encode→decode round-trip lossless to the grid.
- `test_alignment.py` — ordering + loss-mask asserts.
- Can dump and eyeball a token stream.

### Phase 2 — Nano model, no pipes
**Build:**
- No-pipes engine mode (bird + gravity + flap only).
- `model.py`: decoder-only GPT — token + positional embeddings, N×(causal self-attn + MLP),
  RMSNorm, tied embeddings. **Nano tier: ~0.25–2M params, ctx 64–128, ~4 layers, d_model 128–256.**
- `train.py`: AdamW, cosine LR + warmup, bf16 autocast, grad clip, checkpointing. Loss masked off
  action tokens. **`--noise-aug` flag (OFF by default):** perturb a fraction of *input* context
  state tokens by ±1 bin (never actions; targets stay clean).
- **Random-reset data ON unconditionally** (random `y, vy` starts mixed with on-policy).

**Gate (this is the go/no-go instrument):**
- One-step `bird_y` ≈ exact (e.g., >99% within ±1 bin) — learned fall + jump.
- **Measure drift horizon** (free rollout, fixed action seq) with noise-aug OFF. If the no-pipes
  model stays coherent past ~200 frames, keep noise-aug off. If it drifts by ~frame 60, flip the
  flag on and re-measure (expect a big jump). This decides whether drift mitigation is needed
  before Phase 3 — *don't guess, measure.*

### Phase 3 — Pipes + collision
**Build:** full engine; retrain at **Small tier: ~5–15M params, ctx 256–512, 6–8 layers,
d_model 384–512.** Carry over the noise-aug decision from Phase 2.

**Gate:**
- One-step accuracy on **deterministic** transitions (exclude spawn frames) — high.
- **Collision-frame delta within ±1 frame** vs oracle (don't require exact-frame match).
- Spawn-frame **gap validity / distribution** metric reported (not exact accuracy).
- Verify training-data coverage: histogram `pipe_dx`; ensure the scripted controller supplies
  enough multi-pipe survival data (target ~70% survival trajectories navigating 5+ pipes).

### Phase 4 — Free rollout + playable renderer
**Build:**
- `rollout.py`: KV-cached, **slot-constrained** autoregressive decode of the 4 state tokens;
  *sample* at GAP slots; append the user's action token; repeat. KV-cache = **preallocated
  circular buffer** (rolling window; no `list.pop(0)`, no growing `torch.cat`). Engine fully
  removed from the loop.
- `play.py`: pygame at fixed fps; draw bird + pipes; read spacebar → `A_FLAP`/`A_NOFLAP`; stop on
  `ST_DEAD`. **Inference runs on a background thread with a size-1 frame queue** to avoid
  `clock.tick()` jitter.
- `eval.py`: **drift horizon** (oracle as validator) + an **oracle-shadow overlay** — run the real
  engine in lockstep during eval and draw both birds, so drift is visible and quantifiable.

**Gate:** plays in real time (30–60 fps); drift horizon measured and reported; shadow overlay
works. Honest finish line if drift is still rough: "plays, with a measured drift horizon" →
Phase 5 becomes the immediate follow-up.

---

## 4. Data engine & behavior policy
The action distribution shapes what the model learns. Blend:
- **Random flaps** — covers crash/death dynamics and off-distribution states.
- **Scripted controller** — flap when `bird_y` below `gap_y` (or a small PID) — long survival
  trajectories covering steady flight and pipe-threading (target ~70% of frames).
- **Random-reset starts** — episodes beginning at random `(y, vy)` so the model sees the whole
  state space, not just on-policy regions. (Complements noise-aug: coverage ≠ local recovery.)
Frames are tiny (~4–5 tokens); millions are trivial CPU work generated in parallel with GPU
training. Data is free — over-train the small model freely.

---

## 5. Drift mitigation ladder (apply in order, only as evidence demands)
1. **Quantization** (built in) — snap-to-grid caps per-step error.
2. **Random-reset data** (Phase 2, always on) — state-space coverage.
3. **Context-frame noise augmentation** (Phase 2 flag) — teaches recovery from rollout-sized
   errors; the GameNGen trick. ~10 lines, no model-in-the-loop.
4. **[Fallback] short 8–32 frame unrolls** (Codex's Plan B) — only if 1–3 fail eval.
5. **[Phase 5] full scheduled sampling / DAgger** — only if measured rollout failure demands it.

---

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| torch.compile / Blackwell blocks day 1 | Training & inference run WITHOUT compile; compile is a flag. Verify a dummy 1M model in the first hour. |
| Model never learns pipes (sparse in random play) | Scripted controller for survival data; histogram `pipe_dx` before training. |
| Drift horizon too short for playable Phase 4 | Measured at Phase 2; noise-aug ladder (§5). Don't judge the idea on a pre-mitigation baseline. |
| Action/state off-by-one | `test_alignment.py` in Phase 1; freeze the `S_t,A_t,S_{t+1}` convention. |
| Collision timing off by a frame | Frozen "collision after update" convention; ±1 frame tolerance in the gate. |
| Quantization too coarse | 128 bins for `bird_y`; revisit if motion stair-steps. |
| pygame/inference jitter | Background inference thread + size-1 frame queue. |
| KV-cache O(n) per frame | Preallocated circular buffer rolling window. |

---

## 7. Out of scope (clean seams left for these)
- **Phase 5:** rollout-aware training (scheduled sampling / short unrolls) — `train.py` structured
  to drop it in.
- **Phase 6:** velocity linear probe on the residual stream (the Othello-GPT moment). Noise-aug is
  kept cheap specifically so it doesn't confound this.
- **Phase 7:** train a controller *inside the dream* and test transfer to the real engine.

---

## 8. First concrete steps (when you greenlight the build)
1. `uv init`; pin Python 3.12; add torch (cu128), numpy, pygame, pytest.
2. Verify `torch.cuda.is_available()` + RTX 5090 inside the venv (toolchain go/no-go).
3. Phase 1: `engine.py` + `tokenizer.py` + the three tests; dump and show a sample token stream.
