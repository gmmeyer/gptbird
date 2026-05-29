# The Dreaming Bird: a transformer that *is* Flappy Bird

There's no game engine behind [gpt-bird.com](https://gpt-bird.com). When you play it, every
frame — the bird arcing up after a flap, pipes sliding in from the right, the instant you die — is
being *hallucinated* by an ~11-million-parameter transformer running in your browser. Nothing
computes gravity. Nothing simulates a pipe. A neural network learned what Flappy Bird *is*, and now
it dreams the game into existence one frame at a time, reacting to your spacebar.

This is the story of building it: the idea, how it actually works, the bug that turned out to be
the most interesting part of the whole project, and an honest accounting of what worked and what
didn't.

---

## The idea: a game with no rules, only a network

This belongs to a small lineage of **world models** — the line of work (Ha & Schmidhuber's *World
Models*, Google's GameNGen and Genie, the Othello-GPT papers) where instead of *coding* an
environment, you train a network to *be* one. Feed it the current state and an action; it predicts
the next state. Loop that fast enough and you have a playable simulation that lives entirely inside
the network's weights.

Flappy Bird is the perfect minimal case. The rules are almost nothing — gravity, a flap impulse,
pipes that scroll — but they're *continuous* (real physics, not a grid), and there's a beautiful
hidden catch:

> **We never tell the model the bird's velocity.**

The model only ever sees *positions* — how high the bird is, how far the next pipe is, where its
gap sits. Velocity is the one thing that actually governs the motion, and it's deliberately absent
from everything the model is shown. So if the bird flies correctly, the model *must* have
reconstructed velocity on its own, by noticing how the position changes from frame to frame — the
same way you'd estimate speed from a flipbook. That's the elegant experiment at the heart of the
project: can a transformer infer a hidden physical variable just from watching positions? (Spoiler
on honesty: it clearly *uses* something like velocity — the flying is too good otherwise — but
actually *probing* the network's activations to prove it is a follow-up I haven't run yet.)

---

## How it works

### Flappy Bird as a sentence

The trick that makes everything else simple: turn the game into a **language**. Each frame is
written as four tokens —

```
bird_y   pipe_dx   gap_y   status
```

— where each value is quantized into discrete buckets (`bird_y` into 128 height levels, the pipe
distance into 64, the gap into 32). A whole game is then just a sentence that alternates frames and
your button presses:

```
<bos>  S₀  A₀  S₁  A₁  S₂  A₂  …  S_n(dead)  <eos>
        └── "frame" ──┘
```

`Sₜ` is the state at frame *t*; `Aₜ` is the action you took (flap / no-flap). The model is a plain
decoder-only transformer — a small [nanoGPT](https://github.com/karpathy/nanoGPT) — doing the most
ordinary thing a language model does: **predict the next token.** Playing the game is just
autoregressive generation. Your spacebar supplies the action tokens; the network generates
everything else.

The total vocabulary is about 230 tokens. The model is ~11M parameters with a context window of
256 frames — small enough to run in a browser tab.

### Making broken frames impossible

A subtle problem with "let the network emit the game": what if it emits garbage — a state that
doesn't parse? We sidestep it entirely with **slot-constrained decoding.** Each field has its own
disjoint block of token IDs, and when the model generates the bird's height we mask the logits down
to *only* the 128 legal height tokens. It is literally impossible for the model to emit a malformed
frame; the structure is enforced by construction, not hoped for. No parsing, no error handling, no
"what if it outputs `<frame>` twice."

### The oracle (which never ships)

To train the model you need ground truth, so there's a ~100-line deterministic Flappy Bird engine.
It's the **answer key** — it generates millions of `(state, action, next-state)` examples — and it
never appears in the final product. To make sure the model sees both graceful flight *and* dying,
the training data is a blend: a competent scripted bot that threads pipes, random flapping that
crashes constantly, and episodes that start the bird in random positions so it's seen the whole
state space.

### One genuinely tricky bit: the pipes are random

When a new pipe appears at the right edge, its gap is placed *randomly* — there's no way to predict
where it'll be from the past. So at the moment a new pipe is revealed, we let the model **sample**
its gap position rather than pick the single most-likely one. The dream invents its own pipes. (And
because that token is genuinely unpredictable, we score it on *validity* — "is this a legal gap?" —
rather than on matching the oracle.)

### Training, and then deleting the engine

Training is unremarkable in the best way — AdamW, a cosine schedule, bf16 — on a single RTX 5090.
The model learns fast and well:

- next-frame bird height: **98.6% exact**, **100% within one bucket**
- it gets the pipe geometry and the random-gap handling essentially perfect
- generating frames runs at **~125 fps** — no fancy caching needed

Then comes the payoff. At play time you **delete the oracle**. The loop becomes: render the current
state → read the spacebar → ask the network for the next frame → repeat. The bird falls, flaps, and
weaves through pipes according to physics that exists nowhere but in the weights.

> A small aside: before writing any code, I pressure-tested the whole design by running it through a
> **four-way debate between four different AI models** (Gemini, Codex, Sonnet, and Opus), each
> arguing about representation, drift, and failure modes. It's a surprisingly good way to catch bad
> assumptions early — several of the locked-in decisions above came straight out of that argument.

---

## Putting it in your browser

A model that only runs on my GPU isn't much of a toy, so the whole thing is **client-side.** The
trained model is exported to ONNX and runs via [onnxruntime-web](https://onnxruntime.ai) on
**WebGPU** (with a WASM fallback). The weights live on Hugging Face; the page is static and hosted
on GitHub Pages at [gpt-bird.com](https://gpt-bird.com). When you load it, your browser downloads
the model once and then *it* does all the dreaming — no server, no inference API, nothing phoning
home. The JavaScript decode loop is a line-for-line port of the Python one.

---

## The bug that taught me the most: phantom deaths

Here's where it gets interesting. The game worked — but it had a flaw a player noticed immediately:
**you'd die when nothing was there.** No pipe near you, mid-screen, cruising — and suddenly: game
over.

I instrumented the rollout and looked at the frames right before each bogus death, and the pattern
was unmistakable. Every phantom death happened **one frame after successfully passing a pipe:**

```
frame 70:  bird 250,  pipe 2px away,   gap 296   alive   ← threading the pipe, fine
frame 71:  bird 250,  pipe 124px away, gap 232   alive   ← passed it; a NEW pipe appears
frame 72:  bird 242,  pipe 119px away, gap 232   DEAD    ← ?!
```

The moment you clear a pipe, a new one spawns with its gap somewhere else. For that instant the bird
is "far" from the *new* gap — even though the new pipe is still 119 pixels away and completely
harmless. And the model had learned a sloppy rule: **"bird far from gap → dead,"** without checking
whether the pipe was actually *close*. It panicked.

I tried to fix it in training — reweighting the rare "death" signal, generating more data covering
that exact situation. The reweighting just traded phantom deaths for *missed* collisions, and the
extra data barely helped. It was a stubborn, weight-independent flaw.

So the fix that shipped is a bit of pragmatic honesty. The model's *positions* — bird height, pipe
distance, gap location — are **excellent**. It's only the single binary "am I dead?" bit that's
unreliable. So instead of trusting that bit, we **read the collision off the model's own dreamed
positions** with four lines of geometry: are you in the pipe's column and outside its gap? Then
you're dead; otherwise you're not. The network still dreams the entire world — the motion, the
scrolling, the pipes, the gaps — we just adjudicate the one yes/no question it's bad at from the
data it's good at. Phantom deaths went to **zero**, and real collisions land within a frame of the
true engine.

It's a small philosophical concession ("the model is the physics… except the collision check"), but
it makes the game *correct*, and it's all derived from the model's own output.

---

## Can the model learn to call death itself?

That concession bugged me, so I tried to make the model's own death-detection good enough to trust.

The right tool here is **DAgger** (dataset aggregation / scheduled sampling). The model fails only
in *its own* dream — the slightly-off, self-generated states that the clean training data never
contains. This is classic **exposure bias.** The fix is to let the model run free, collect the
states it actually visits, label each one with the correct answer (using that same geometry oracle),
and fine-tune on *those.* You're teaching it precisely where it goes wrong.

It worked — partially. One round of DAgger cut the model's phantom-death rate roughly **in half**
(from essentially always, to ~45%) and fixed its collision *recall* to match the real engine, all
without hurting its physics. That's the only intervention that moved the needle at all.

But then it **plateaued.** Iterating more rounds didn't compound — it hovered around ~45% phantom and
wouldn't go lower. Better, but nowhere near reliable enough to retire the geometry guard. The
model's single-frame "am I dead?" judgment seems to have a floor with this setup.

(There was also a great debugging lesson buried in here: my first batched-rollout data generator was
*agonizingly* slow — minutes per batch — and I almost concluded the approach was impractical. The
culprit was running inference in fp32, which gets no tensor-core acceleration on the GPU. One line
of `bf16` autocast took a batch from minutes to **four seconds.** Always check your dtype.)

So the honest verdict: **the model learned the world beautifully and the rare binary event
imperfectly.** Continuous physics — gravity, momentum, the rhythm of weaving through pipes — it
nailed. The split-second life-or-death call in its own hallucinated states is genuinely hard, DAgger
helps a lot but doesn't fully close it, and reading collisions off its (accurate) positions is the
correct engineering answer.

---

## What I take away

- **Transformers are shockingly good at continuous dynamics.** Given nothing but quantized positions,
  a tiny model learned gravity, the flap impulse, scrolling, and random gap generation well enough to
  be playable, with almost no drift over hundreds of autoregressive frames. The hidden-velocity setup
  worked in spirit — the flying is too clean for it not to have inferred motion.
- **The hard part is the rare, discrete event, in the model's own distribution.** That's the
  exposure-bias story in miniature, and it's where the honest engineering (and the unfinished
  science) lives.
- **The pragmatic answer and the pure answer can coexist.** The guard makes the game correct today;
  DAgger is the path toward the model doing it natively, and now I know exactly how far that path
  goes before it plateaus.

The headline experiment — actually **probing the network's activations to prove it encodes velocity**
— is still on the to-do list, and it's the thing I'm most curious about. But the bird flies, it
dreams its own pipes, it runs in your browser, and it dies when (and only when) it should.

**Go play it: [gpt-bird.com](https://gpt-bird.com).** Spacebar to flap. The faint ghost is where
real physics says the bird should be — watch how closely the dream tracks it.

*Code: [github.com/gmmeyer/gptbird](https://github.com/gmmeyer/gptbird) · Model:
[huggingface.co/gmmeyer/gpt-bird](https://huggingface.co/gmmeyer/gpt-bird)*
