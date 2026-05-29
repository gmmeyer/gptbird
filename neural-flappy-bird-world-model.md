# Project: The Dreaming Bird — A Neural Flappy Bird World Model

> A small transformer trained to *be* Flappy Bird. Feed it the current frame's state
> and the player's action (flap / no-flap); it emits the next frame's state. Loop at
> ~30–60 fps, render it, read the spacebar, feed it back — and a real-time arcade game
> runs entirely inside the network's autoregressive output. No physics engine at
> runtime; the model *is* the physics.

**Target hardware:** single NVIDIA RTX 5090 (32 GB VRAM).
**Status:** design / idea document. Not a spec.
**Author intent:** third in a set with *The Dreaming Dungeon* (neural roguelike) and *tinyts* (TS code model). Same "I trained a model for a thing I care about" spirit as a prior from-scratch Latin LM project. Flappy Bird is the deliberately *minimal* world model — the one most likely to actually work cleanly and the best vehicle for the "does it learn hidden physics" experiment.

---

## 1. The core idea, and how it differs from the roguelike

Same lineage as the sibling roguelike doc — [World Models](https://arxiv.org/abs/1803.10122), [Othello-GPT](https://arxiv.org/abs/2210.13382), [GameNGen](https://arxiv.org/abs/2408.14837), Oasis, Genie — but Flappy Bird inverts which parts are hard:

| Axis | Roguelike (*Dreaming Dungeon*) | Flappy Bird (*Dreaming Bird*) |
|---|---|---|
| Time | Turn-based (1 action → 1 state) | Real-time, fixed timestep, dense frames |
| Dynamics | Discrete rules (can't walk through `#`) | Continuous **physics**: gravity + flap impulse |
| Action space | ~8 actions | **2**: flap / no-flap, every frame |
| Hidden state | Off-screen map (persistence problem) | **Velocity** (never observed; must be inferred) |
| The hard problem | Long-range persistence / memory | **Autoregressive drift** over a real-time rollout |
| Episode | Long, complex | Short, terminates on collision (game over) |

The headline consequence: **the roguelike's dominant challenge (persistence) is almost absent here.** Pipes scroll off-screen and are never revisited; there is no map to keep consistent. The model needs only short-horizon memory — enough to track its own velocity over a handful of frames. That makes a clean, *reliably working* world model far more achievable than the roguelike. The difficulty migrates to two new places: learning continuous dynamics, and not drifting off the rails during a long autoregressive rollout (§5).

Through-line shared with both sibling docs: **a deterministic oracle generates and validates the training data.** Here it's a ~100-line Flappy Bird physics engine — it produces ground-truth `(state, action, next_state)` frames, and at eval time it *validates* the model's rollouts (does the predicted trajectory obey gravity? are collisions called at the right moment?).

---

## 2. State representation — the key design fork

Flappy Bird's state is continuous (bird height, velocity, pipe positions), which is the central representational decision. Three options, in recommended order:

### 2.1 Compact quantized scalars (recommended for v1)
Represent each frame as a few **position** scalars, quantized to discrete tokens:
- `bird_y` — quantized to e.g. 64 rows
- `pipe_dx` — horizontal distance to next pipe, quantized
- `gap_y` — vertical center of the next pipe's gap, quantized
- (optional) a second upcoming pipe for lookahead

**Crucially, do *not* include velocity in the observed state.** Velocity is the hidden variable. If the model flies well, it must have reconstructed velocity from the *sequence* of `bird_y` values — exactly as the real physics requires integrating it. This is what makes §6's probe experiment work, and it's the most elegant thing about the whole project.

Vocabulary stays tiny (~a couple hundred position tokens + 2 action tokens + delimiters). Quantization also doubles as a **drift stabilizer** (§5): snapping to grid cells each frame prevents continuous error accumulation.

Example stream:
```
<bos>
<frame> by:31 dx:48 gap:18 </frame> <noflap>
<frame> by:33 dx:46 gap:18 </frame> <noflap>     # gravity: falling, accelerating
<frame> by:36 dx:44 gap:18 </frame> <flap>
<frame> by:33 dx:42 gap:18 </frame> <noflap>     # flap impulse: jumped up
...
<frame> by:52 dx:02 gap:18 </frame> <gameover>    # hit the pipe
<eos>
```

### 2.2 ASCII-frame (maximally consistent with the roguelike doc)
Render each frame as an ASCII grid (`@`=bird, `|`=pipe walls, spaces=gap/sky) and predict the next grid — literally the same pipeline as *Dreaming Dungeon*. Cute and drop-in, and the visual *is* the state. Costs many more tokens per frame (pipe columns are expensive), so it's slower and the long-rollout drift is worse. Good for a quick "reuse the roguelike harness" prototype; not the efficient choice.

### 2.3 Continuous regression (most faithful, messiest)
Predict next-frame scalars directly via a regression head (or digit-tokenized numbers). Most physically faithful, no quantization artifacts — but drift-prone and a worse fit for the tiny-vocab sequence-model framing. Note as an alternative, not the starting point.

---

## 3. The data engine (the oracle)

A tiny deterministic Flappy Bird in code, used only to generate ground truth. Never ships.

### 3.1 Engine spec
- Fixed timestep. Bird has `y`, `vy`. Each frame: `vy += gravity`; if action==flap, `vy = flap_impulse` (instantaneous set, not add); `y += vy`. Pipes scroll left at constant speed; new pipe with random gap center spawns periodically. Collision = bird hits pipe or floor/ceiling → terminate.
- Seeded RNG → every trajectory reproducible from `(seed, action_sequence)`.

### 3.2 Behavior policy — matters a lot
The action distribution shapes what the model learns. A purely random flap policy dies almost immediately and floods the dataset with start-of-game and crash frames, under-covering the "threading pipes" regime. Blend:
- **Random** flaps — cheap, covers crash/death dynamics and off-distribution recoveries.
- **Scripted controller** — a trivial heuristic (flap when `bird_y` is below `gap_y`, else fall) or a small PID — generates long survival trajectories that cover steady flight. Flappy Bird is famously easy to solve with simple RL/NEAT, so expert data is cheap.
- Mix so the model sees both competent flight *and* failure, including near-misses and recoveries (important for not drifting into states the data never showed — see §5).

### 3.3 Scale
Frames are tiny (~10–20 tokens each in the compact rep). Millions of frames are trivial CPU work, generated in parallel with GPU training. Chinchilla-ish ~20 tokens/param is the baseline ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)), but data is free, so over-train a small model freely.

---

## 4. Model & training

### 4.1 Architecture
Decoder-only transformer, GPT-style, next-token prediction over the frame/action stream. Base: [nanoGPT](https://github.com/karpathy/nanoGPT); speed tricks from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT). Structurally this is a [Decision/Trajectory Transformer](https://arxiv.org/abs/2106.01345) sequence (state, action, state, …) — same framing as the roguelike, just continuous-physics instead of discrete-rules.

This is the *smallest* of the three projects. Because the state is a handful of scalars and the only "rule" is 1-D kinematics, capacity needs are low:

| Tier | Params | Context | Purpose |
|---|---|---|---|
| Nano | ~0.5–2M | 64–128 frames | Does it learn gravity + flap impulse at all? |
| Small | ~5–15M | 256–512 frames | Stable real-time playable rollouts |
| Stretch | ~30M | 1024 frames | Long drift-free sessions, multi-pipe lookahead |

Context here = how many past frames the model sees, which sets how well it can estimate velocity and how stable rollouts are. Even a tiny context (a few frames) is enough to recover velocity in principle (two positions → finite difference).

### 4.2 Objective & the drift fix
- Baseline: autoregressive LM loss; optionally up-weight `<frame>` tokens over action tokens (actions are user-supplied at inference).
- **Train against your own rollouts** to fight compounding error (§5): scheduled sampling / DAgger-style — periodically feed the model its *own* predicted frames back in during training and supervise against what the real engine says *should* have happened next. This is the single most important trick for real-time stability. ([Scheduled sampling, Bengio et al., 2015](https://arxiv.org/abs/1506.03099); the compounding-error problem is well documented in model-based RL, e.g. [MBPO](https://arxiv.org/abs/1906.08253).)

### 4.3 Inference loop (the payoff)
```
state = initial_frame
loop at fixed fps:
    render(state)                       # draw bird + pipes to a canvas
    a = flap if spacebar_down else noflap
    state = model.generate_until(</frame>, prompt = recent_frames + a)
    if state == gameover: break
```
The engine is gone; the bird falls, flaps, and dies according to physics the network learned, never coded.

---

## 5. The hard problem: autoregressive drift (not persistence)

In a real-time rollout, each frame's small prediction error feeds into the next, and errors compound — the bird's trajectory slowly stops obeying gravity, velocity estimates degrade, and after N frames the sim diverges from plausible physics. This is the classic compounding-error problem of learned dynamics models. Unlike the roguelike, *memory* is not the issue (nothing to remember); *stability* is.

Mitigations, in order of impact:
1. **Quantization (§2.1).** Snapping to discrete cells each frame caps per-step error so it can't accumulate continuously. Discretization is a feature here.
2. **Rollout-aware training (§4.2).** Scheduled sampling / training on own predictions so the model learns to recover from its own small mistakes instead of being trained only on perfect engine states (which it never sees at inference).
3. **Adequate failure/recovery coverage in data (§3.2).** If the data never shows odd near-miss states, the model has no idea how to behave when its own drift produces one.

Eval signal for this: **drift horizon** — frames-until-the-rollout-violates-physics, plotted vs. model size, context length, and whether rollout-aware training was used.

---

## 6. The standout variant: train an agent *inside the dream*

This is the version that closes the loop with the foundational reference and is the coolest extension. The original [World Models](https://arxiv.org/abs/1803.10122) paper's whole thesis: once you have a learned model of the environment, you can train a *controller inside the model's hallucination*, never touching the real game. Flappy Bird is a perfect minimal demonstration:

1. Train the world model (above) to simulate Flappy Bird.
2. Train a tiny controller (even a handful of params, or a small policy net) to play — but let it interact **only with the neural world model**, not the real engine.
3. Drop the trained controller into the *real* Flappy Bird and see if skills learned in the dream transfer.

If it transfers, you've reproduced the central result of World Models on a 5090 over a weekend: *an agent that learned to play a game it only ever experienced as another network's hallucination.* (Same idea underpins [Dreamer / DreamerV3](https://arxiv.org/abs/2301.04104), which learns world models and trains policies "in imagination.")

The interpretability variant (parallel to the roguelike's linear-probe callout, but sharper): **probe hidden activations for the bird's velocity.** Velocity is never in the observed state, so a successful linear decode of `vy` from the residual stream is direct evidence the model built the hidden physical variable it needs — the Othello-GPT result, but for a quantity that is genuinely latent rather than merely unstated. Optionally probe for `gravity`/`flap_impulse` constants too.

---

## 7. Evaluation

The engine makes everything quantitatively checkable (oracle pass):
- **One-step physics accuracy:** given held-out `(frames, action)`, error of predicted next `bird_y`/`pipe` vs. engine. The core "did it learn gravity + impulse" signal.
- **Rollout drift horizon (§5):** frames until predicted trajectory violates physics (engine as validator).
- **Collision-timing accuracy:** does the model declare game-over on the same frame the real engine would? Tests whether it learned the collision geometry.
- **Velocity-probe accuracy (§6):** linear-decode R² of true `vy` from activations.
- **Dream-to-real transfer (§6):** score of a controller trained in the model when played in the real engine.
- **Vibe check:** is it fun and twitchy to play? Real-time physics is felt instantly if it's wrong.

---

## 8. Suggested phased roadmap

1. **Engine + logger.** Deterministic Flappy Bird, seeded, emits compact quantized frame streams. Verify replay determinism.
2. **Nano model, no pipes.** Just bird + gravity + flap. Does a ~0.5–2M model learn 1-D kinematics? Smallest possible win; confirm it learns to "fall and jump" correctly.
3. **Add pipes + collision.** Re-measure one-step accuracy and collision timing.
4. **Free rollout + renderer.** Swap the engine out; play it with the spacebar. Measure drift horizon.
5. **Rollout-aware training.** Add scheduled sampling; re-measure drift horizon (expect a big jump).
6. **Velocity probe.** The Othello-GPT moment, sharpened.
7. **Stretch: agent-in-the-dream.** Train a controller inside the model; test transfer to the real game.

---

## 9. Risks / things that might disappoint

- Real-time inference latency: the autoregressive loop must keep up with the frame rate. Compact rep + tiny model + KV-cache makes this easy on a 5090, but the ASCII-frame rep (§2.2) may be too token-heavy for smooth fps.
- Drift can make rollouts unstable before §4.2/§5 mitigations; don't judge the idea on a pre-rollout-aware baseline.
- Data coverage is everything: a pure-expert dataset leaves the model helpless once its own drift produces a weird state. Blend in random/failure trajectories.
- This is the *easy* world model — that's the point. Its job is to be the one that cleanly works and to host the velocity-probe and dream-agent experiments, not to be the hardest.

---

## 10. References

- Ha & Schmidhuber, *World Models*, 2018 (train controller in the dream). https://arxiv.org/abs/1803.10122
- Li et al., *Emergent World Representations* (Othello-GPT), ICLR 2023. https://arxiv.org/abs/2210.13382
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), 2023. https://arxiv.org/abs/2301.04104
- Chen et al., *Decision Transformer*, 2021. https://arxiv.org/abs/2106.01345
- Janner et al., *Trajectory Transformer*, 2021. https://arxiv.org/abs/2106.02039
- Bengio et al., *Scheduled Sampling*, 2015 (compounding-error mitigation). https://arxiv.org/abs/1506.03099
- Janner et al., *When to Trust Your Model* (MBPO; model-error compounding), 2019. https://arxiv.org/abs/1906.08253
- Valevski et al., *Diffusion Models Are Real-Time Game Engines* (GameNGen), 2024. https://arxiv.org/abs/2408.14837
- Bruce et al., *Genie: Generative Interactive Environments*, 2024 (+ Genie 2/3); Decart & Etched, *Oasis*, 2024.
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla), 2022. https://arxiv.org/abs/2203.15556
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat); Jordan, [modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT).

*(Verify arXiv IDs and author lists before citing in anything public; written from memory and may contain small errors.)*
