# Building the Dreaming Bird: a full build log

*A companion to [The Dreaming Bird](./the-dreaming-bird.md). That post is the story; this one is
the receipts — every design decision, every number, and the two bugs that taught me the most.*

I trained a ~11-million-parameter transformer to **be** Flappy Bird. Not to play it — to *be* it.
You give it the current frame and your button press; it emits the next frame. Loop that at frame
rate, render the output, feed your spacebar back in, and a real-time arcade game runs entirely
inside the network's autoregressive output. There is no physics engine at runtime. **The model is
the physics.**

It's live and playable in your browser — no install, no server — at
**[gpt-bird.com](https://gpt-bird.com)**. The weights stream from Hugging Face and run client-side
on WebGPU.

This is the whole build, in order.

---

## 0. Where it sits

This is a **world model**, in the lineage of Ha & Schmidhuber's [*World
Models*](https://arxiv.org/abs/1803.10122), the [Othello-GPT](https://arxiv.org/abs/2210.13382)
interpretability work, and Google's [GameNGen](https://arxiv.org/abs/2408.14837) / Genie. Instead of
*coding* an environment, you train a network to *be* one.

Flappy Bird is the deliberately **minimal** case, and it inverts what's hard. In a roguelike world
model, the dominant challenge is *persistence* — remembering the off-screen map. Here there's
nothing to remember: pipes scroll away and never come back. The difficulty migrates to two other
places:

1. **Learning continuous 1-D physics** — gravity plus a flap impulse — from nothing but positions.
2. **Not drifting off the rails** during a long autoregressive rollout, where each frame's small
   error feeds the next.

And there's one beautiful hook that makes the whole thing an experiment rather than a demo.

### The hidden variable

The model is **never shown velocity.** Every frame it sees is *positions only*: how high the bird
is (`bird_y`), how far the next pipe is (`pipe_dx`), where that pipe's gap sits (`gap_y`). Velocity —
the one quantity that actually governs the motion — is deliberately excluded from the observation.

So if the bird flies correctly, the model *must* have reconstructed velocity on its own, by
integrating the change in position across frames — exactly as the real physics requires. That's the
Othello-GPT question, sharpened: can a transformer infer a genuinely **latent** physical variable
just from watching positions? (Honest status: the flying is far too clean for it *not* to be using
something like velocity — but actually probing the residual stream to *prove* it is the one headline
experiment I haven't run yet. More on that at the end.)

---

## 1. Planning: a four-way AI debate

Before writing a line of code, I pressure-tested the design doc by running it through a structured
**four-way debate between four different models** — Gemini, Codex, Sonnet, and Opus — two rounds,
with cross-critique, each arguing about representation, drift, and failure modes. It's a
surprisingly good way to kill bad assumptions early. The output was
[`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md), and several of the locked decisions below came
straight out of that argument:

- **State = compact quantized scalars.** `bird_y` → 128 bins (the finest grid; it's what velocity is
  recovered from), `pipe_dx` → 64, `gap_y` → 32. Velocity excluded, on purpose.
- **Token grammar:** fixed-length frames, **one token per field**, with **disjoint id ranges** per
  field. No digit tokens, no `by:` text, no `</frame>` delimiter to parse.
- **Slot-constrained decoding:** at each slot we mask the logits down to only that field's legal
  ids, so a malformed frame is *impossible by construction*.
- **Ordering `Sₜ, Aₜ, Sₜ₊₁`** with the loss **masked off the action tokens** — the action is
  player-supplied conditioning, not something the model should predict.
- **Stochastic pipe spawn:** a new gap is RNG-drawn, so at spawn frames we **sample** the gap slot
  (don't argmax) and, in eval, score those frames on *validity*, not identity.
- **Drift mitigation is a measured ladder** — noise augmentation stays gated *off* unless the
  numbers demand it.

The physical constants ended up at a 288×512 world, `gravity = 1.2`, `flap_impulse = -11` (a flap
*sets* `vy`, it doesn't add to it), `pipe_speed = 4`, `gap_height = 130`. They're tuned so typical
per-frame motion is ~1 quantization bin — fine enough that velocity is recoverable from the
quantized position sequence.

**Toolchain:** `uv` + Python 3.12, PyTorch on CUDA 12.8, single **NVIDIA RTX 5090** (32 GB,
Blackwell / sm_120). First step was literally verifying the 5090 ran fp32/bf16 matmuls and autograd
before building anything on it.

---

## 2. Flappy Bird as a sentence

The move that makes everything downstream simple: turn the game into a **language.** Each frame is
four tokens —

```
bird_y   pipe_dx   gap_y   status
```

— and a whole game is a sentence that alternates frames and your button presses:

```
<BOS>  S₀ A₀  S₁ A₁  S₂ A₂  …  S_n(dead)  <EOS>
```

The model is a plain decoder-only transformer — a small [nanoGPT](https://github.com/karpathy/nanoGPT)
(RMSNorm, SDPA/flash attention, tied embeddings) — doing the most ordinary thing a language model
does: predict the next token. Playing the game *is* autoregressive generation. Your spacebar supplies
the action tokens; the network generates everything else. Total vocabulary: ~232 tokens.

**Why the disjoint-range + slot-mask grammar matters.** The obvious failure mode of "let the network
emit the game" is that it emits garbage — a state that doesn't parse. We sidestep it completely:
each field owns a contiguous, non-overlapping block of token ids, and when we generate the bird's
height we mask the logits to *only* the 128 legal height ids. There is no code path that produces a
malformed frame. No parsing, no error handling, no "what if it emits two heights." The structure is
enforced, not hoped for.

**The oracle that never ships.** Training needs ground truth, so there's a ~130-line deterministic,
seeded Flappy Bird engine ([`engine.py`](../src/dreaming_bird/engine.py)). It is the answer key: it
generates millions of `(state, action, next-state)` frames and, at eval time, *validates* the
model's rollouts. It never appears in the shipped artifact. Its collision check happens *after* the
position update and emits one final `alive=False` frame — a convention the tests pin down, and one
that turns out to matter a lot in §6.

**Data coverage is everything.** A pure-expert dataset leaves the model helpless the moment its own
drift produces a slightly-weird state. So the training data blends a competent gap-tracking
controller (long survival, threading pipes), random flapping (crashes constantly, covers death
dynamics), and random-start episodes (so the whole state space is seen). It's tuned so **~66% of
frames come from runs that clear 5+ pipes** — enough competent flight to fly, enough failure to know
what dying looks like.

---

## 3. The build, phase by phase

Each phase was a separate commit with its own eval gate.

| Phase | What shipped | Result |
|---|---|---|
| **1 — Oracle + data + eval** | `engine.py`, `tokenizer.py` (the locked grammar, vocab 232), `policies.py`/`data.py` (blended controllers → packed `uint16`), `eval.py` (a model-agnostic replay/rollout harness) | **19 tests pass**; data tuned to 66% of frames from 5+ pipe runs |
| **2 — Nano model, no pipes** | `model.py` (~1.9M params: 4 layers, 4 heads, d=192), `train.py` (AdamW + cosine + bf16, masked loss, gated noise-aug) | one-step `bird_y` **96.5% exact / 99.4% within ±1 bin**; drift horizon long enough that **noise-aug stays off** |
| **3 — Pipes + collision** | `small` tier (~11M: 6 layers, 6 heads, d=384, **context 256 frames**); `evaluate_pipes()` splits the gap metric into *stable* (exact) vs *spawn* (validity) | held-out: `bird_y` **98.6% / 100%**, `pipe_dx` **100%**, gap-stable **99.96%**, **spawn validity 100%**, collisions **94.9% within ±1 frame** |
| **4 — Free rollout + renderer** | `rollout.py` (a cacheless `DreamStepper`), `play.py` (pygame; spacebar; a "shadow" ghost showing where true physics *should* put the bird) | **~125 fps** on the 5090 with **no KV-cache**; an autopilot playing *inside the dream* threads **5+ pipes** with **~0 `bird_y` drift** |

Phase 2 was the real go/no-go: does a tiny model learn gravity + flap impulse *at all*, from
positions only? It did, cleanly — 96.5% exact next-frame height with a sub-2M-param model. Phase 3
added pipes and the stochastic-spawn handling and it stayed near-perfect. Phase 4 is the payoff:
**delete the oracle.** The loop becomes render → read spacebar → ask the network for the next frame →
repeat. The bird falls, flaps, and weaves through pipes according to physics that exists nowhere but
in the weights. And because it's tiny, cacheless generation already clears the real-time budget at
125 fps — the KV-cache we'd planned turned out to be unnecessary.

---

## 4. Putting it in a browser

A model that only runs on my GPU isn't a toy, so the whole thing is client-side.

- **`export_onnx.py`** exports the checkpoint to ONNX. The catch: `RMSNorm` and fused SDPA don't map
  onto ops that onnxruntime-web's WebGPU backend supports. So *at export time* the graph swaps them
  for explicit-op equivalents — attention becomes a plain `MatMul → masked Softmax → MatMul` with a
  dynamic causal mask, keeping the sequence length dynamic. The exported graph is validated against
  PyTorch: **max logit difference 1.3e-5, argmax agrees everywhere.**
- **`web/app.js`** runs the model via [onnxruntime-web](https://onnxruntime.ai) on **WebGPU** (WASM
  fallback), mirroring the Python decode loop line for line — same slot masks, same gap sampling,
  same geometry check. It got a proper visual pass too: gradient sky, parallax clouds, a scrolling
  multi-pipe world, a velocity-tilted bird.

**Deployment** is three repos and a domain:

- 🤗 **Hugging Face** [`gmmeyer/gpt-bird`](https://huggingface.co/gmmeyer/gpt-bird) — `model.onnx`,
  `config.json`, checkpoints, model card.
- **GitHub Pages** [`gmmeyer/gpt-bird`](https://github.com/gmmeyer/gpt-bird) — the static site, which
  **streams the model from HF** (cross-origin verified). It's included in the source repo as a
  submodule at [`web/`](../web).
- **Custom domain [gpt-bird.com](https://gpt-bird.com)** — CNAME + apex A records + HTTPS enforced.

Load the page and your browser downloads the model once, then *it* does all the dreaming. No server,
no inference API, nothing phoning home.

---

## 5. The bug that taught me the most: phantom deaths

The game worked — but a player noticed the flaw immediately: **you'd die when nothing was there.**
Mid-screen, no pipe near you, cruising, and suddenly: game over.

I instrumented the rollout and looked at the frames right before each bogus death. The pattern was
unmistakable — every phantom death happened **one frame after successfully passing a pipe:**

```
frame 70:  bird 250,  pipe   2px away,  gap 296   alive   ← threading the pipe, fine
frame 71:  bird 250,  pipe 124px away,  gap 232   alive   ← passed it; a NEW pipe appears
frame 72:  bird 242,  pipe 119px away,  gap 232   DEAD    ← ?!
```

The instant you clear a pipe, a new one spawns with its gap somewhere else. For that frame the bird
is "far" from the *new* gap — even though the new pipe is 119 pixels away and completely harmless.
The model had learned a sloppy shortcut: **"bird far from gap → dead,"** without gating on whether
the pipe was actually *close.* It panicked.

I tried to fix it in training. Reweighting the rare DEAD signal — ×2, ×4, ×20 — left the phantom
rate stuck near 1.0; it just traded phantom deaths for *missed* collisions. So this wasn't a tuning
problem. It's a **representation / exposure-bias** problem: the failure lives in the model's own
rollout distribution, which the clean teacher-forced data never contains.

**The fix that shipped is a piece of pragmatic honesty.** The model's *positions* — height, pipe
distance, gap location — are excellent. It's only the single binary "am I dead?" bit that's
unreliable. So instead of trusting that bit, the dream **adjudicates collisions from the model's own
dreamed positions** with four lines of geometry (`engine.collides`, mirrored exactly in `app.js`):
are you in the pipe's column *and* outside its gap? Then you're dead; otherwise you're not. The
network still dreams the entire world — motion, scrolling, pipes, gaps — we just answer the one
yes/no question it's bad at using the data it's good at.

Audited result: **phantom rate 0.00**, and an anti-gap controller (which deliberately flies into
pipes) dies at ~58 frames in the dream — matching the real engine. Bonus: it's one *fewer* forward
pass per frame, since we no longer generate the status token. No model re-upload was needed — the
deployed weights were always fine; we just stopped trusting their death bit.
`rollout.collision_audit` is the regression check that keeps it honest.

---

## 6. Trying to fix it *in the model*: DAgger

That concession bugged me. Could I make the model's own death detection good enough that the
geometry guard becomes a safety net rather than the mechanism?

The right tool is **DAgger** (dataset aggregation / scheduled sampling). The model fails only in its
*own* dream — the slightly-off, self-generated states clean data never shows. So: let the model run
free, collect the states it actually visits, relabel each one's status with the geometry oracle, and
fine-tune on *those.* You teach it precisely where it goes wrong.

**Attempt 1 — lazy-controller data augmentation (negative result).** I added a `lazy_center_policy`
that loiters at screen center until a pipe is close, to flood the "far from gap + far pipe + alive"
regime. It barely moved the raw phantom rate (1.00 → 0.92) and made collision *recall worse.*
Passive, teacher-forced data doesn't reach the model's own failure distribution. Logged and
discarded.

**Attempt 2 — real DAgger (partial win, then a plateau).** A **batched dream rollout**
(`dagger.py`) generates the model's own visited states at scale; each frame's status is relabeled by
the geometry oracle; we fine-tune on the aggregate (`train.py --init-from`) and iterate with
`dagger_loop.py`. Result:

- raw-status phantom rate **1.00 → ~0.45** (roughly halved) — the *only* intervention that moved it,
- collision recall **fixed** (anti-gap survival 72 → ~58, ≈ engine),
- one-step physics accuracy **intact.**

But it **plateaued at ~0.45.** Rounds 2–3 didn't compound (0.32 / 0.44 / 0.39 — within metric noise
at n=100). So the native death flag got a lot better but is still not solo-reliable: ~45% of the
model's *own* deaths would be phantom without the guard. **Conclusion: the geometry guard stays** (it
gives 0.00 on any model, for free), and the deployed weights are unchanged.

*Why it plateaus (hypotheses, untested):* fine-tuning each round from the phantom-prone deployed init
may keep re-importing the bad prior; and a single status bit predicted from one frame may simply lack
the multi-frame "approach" context needed to gate cleanly on distance. Future angles: train from
scratch on the aggregate rather than fine-tuning the deployed model, feed `dx` more directly, or just
keep the guard (it's correct and costs nothing).

**The debugging lesson buried in here** was almost the most valuable part. My first batched-rollout
generator was *agonizingly* slow — minutes per batch — and I nearly wrote the whole approach off as
impractical. The culprit: running inference in fp32, which gets **no tensor-core acceleration** at
batch 256 on the 5090. One line of `bf16` autocast took a batch from minutes to **~4 seconds** (~15×).
Always check your dtype.

---

## 7. Honest takeaways

- **Transformers are shockingly good at continuous dynamics.** Given nothing but quantized positions,
  a tiny model learned gravity, the flap impulse, pipe scrolling, and random-gap generation well
  enough to be playable — ~99% one-step accuracy and near-zero drift over hundreds of autoregressive
  frames. The hidden-velocity setup worked in spirit: the flying is too clean for the network not to
  have inferred motion.
- **The hard part is the rare, discrete event, in the model's own distribution.** The phantom-death
  saga is exposure bias in miniature — a failure that quantization and loss-weighting *cannot* fix,
  because it doesn't live in the training data. DAgger, which explicitly targets that distribution,
  is the only thing that helped, and even it only got halfway.
- **The pragmatic answer and the pure answer can coexist.** Reading collisions off the model's
  accurate positions makes the game correct *today*; DAgger is the path toward the model doing it
  natively, and now I know exactly how far that path goes before it plateaus.

The one experiment still on the bench is the one I'm most curious about: **linearly probing the
network's activations to prove it encodes velocity** — the Othello-GPT moment, for a variable that is
genuinely latent rather than merely unstated. Also deferred from the original plan: full
scheduled-sampling training, and the World-Models finale of training a controller *inside the dream*
and transferring it to the real game.

But the bird flies, it dreams its own pipes, it runs entirely in your browser, and it dies when — and
only when — it should.

**Go play it: [gpt-bird.com](https://gpt-bird.com).** Spacebar to flap. The faint ghost is where real
physics says the bird should be — watch how closely the dream tracks it.

*Code: [github.com/gmmeyer/gptbird](https://github.com/gmmeyer/gptbird) · Model:
[huggingface.co/gmmeyer/gpt-bird](https://huggingface.co/gmmeyer/gpt-bird) · Companion post:
[The Dreaming Bird](./the-dreaming-bird.md)*
