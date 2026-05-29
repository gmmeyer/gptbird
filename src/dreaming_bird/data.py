"""Trajectory generation: turn the oracle + behavior policies into packed token streams.

Episodes are encoded with :class:`~dreaming_bird.tokenizer.Tokenizer`, concatenated, and
written as a flat ``uint16`` ``.bin`` (nanoGPT style). A sidecar ``.json`` records metadata.

Run as a module for a quick sample + stats:

    uv run python -m dreaming_bird.data --episodes 2000 --out data/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import EngineConfig, TokenizerConfig
from .engine import FlappyEngine, rollout
from .policies import noisy_scripted_policy, random_policy, random_start, scripted_policy
from .tokenizer import Tokenizer


def generate(
    n_episodes: int,
    seed: int = 0,
    engine_cfg: EngineConfig | None = None,
    tok_cfg: TokenizerConfig | None = None,
    max_frames: int = 2048,
    p_clean: float = 0.55,
    p_noisy: float = 0.20,
    p_random_reset: float = 0.15,
    long_pipes: int = 5,
) -> tuple[np.ndarray, dict]:
    """Generate ``n_episodes`` blended trajectories. Returns (token_array, meta).

    Per-episode regime: ``p_clean`` use the clean scripted controller (long multi-pipe
    survival — the bulk of the FRAMES), ``p_noisy`` use a lightly-noisy scripted controller
    (near-misses & recoveries), the rest use random flaps (crash/death dynamics). Independently,
    ``p_random_reset`` of episodes begin from a random (y, vy) start for state-space coverage.

    Every episode still ends in a death (that is correct and desirable — the model must learn
    game-over); what matters is that enough FRAMES come from long, pipe-threading runs. ``meta``
    reports the frame-weighted coverage (fraction of frames from episodes passing >=``long_pipes``).
    """
    engine_cfg = engine_cfg or EngineConfig()
    tok_cfg = tok_cfg or TokenizerConfig()
    tok = Tokenizer(engine_cfg, tok_cfg)
    engine = FlappyEngine(engine_cfg, seed=seed)
    meta_rng = np.random.default_rng(seed)
    frames_per_pipe = engine_cfg.pipe_spacing / engine_cfg.pipe_speed
    long_thresh = (engine_cfg.width - engine_cfg.bird_x) / engine_cfg.pipe_speed \
        + long_pipes * frames_per_pipe   # frames to reach + thread `long_pipes` pipes

    streams: list[np.ndarray] = []
    ep_lengths: list[int] = []
    n_deaths = 0
    dx_bins_hist = np.zeros(tok_cfg.pipe_dx_bins, dtype=np.int64)

    for _ in range(n_episodes):
        ep_seed = int(meta_rng.integers(0, 2**31 - 1))
        if meta_rng.random() < p_random_reset:
            sy, svy = random_start(engine_cfg, meta_rng)
            engine.reset(seed=ep_seed, start_y=sy, start_vy=svy)
        else:
            engine.reset(seed=ep_seed)

        r = meta_rng.random()
        if r < p_clean:
            policy = scripted_policy()
        elif r < p_clean + p_noisy:
            policy = noisy_scripted_policy(seed=int(meta_rng.integers(0, 2**31 - 1)))
        else:
            policy = random_policy(p_flap=float(meta_rng.uniform(0.05, 0.25)),
                                   seed=int(meta_rng.integers(0, 2**31 - 1)))

        obs_list, action_list = rollout(engine, policy, max_frames=max_frames)
        streams.append(tok.encode_episode(obs_list, action_list))
        ep_lengths.append(len(obs_list))
        n_deaths += 0 if obs_list[-1].alive else 1
        for o in obs_list:
            dx_bins_hist[tok.pipe_dx_bin(o.pipe_dx)] += 1

    tokens = np.concatenate(streams).astype(np.uint16)
    lengths = np.array(ep_lengths)
    n_frames = int(lengths.sum())
    frames_from_long = int(lengths[lengths >= long_thresh].sum())
    meta = {
        "n_episodes": n_episodes,
        "n_frames": n_frames,
        "n_tokens": int(tokens.size),
        "death_rate": round(n_deaths / n_episodes, 4),
        "mean_episode_frames": round(lengths.mean(), 2),
        "median_episode_frames": int(np.median(lengths)),
        "max_episode_frames": int(lengths.max()),
        f"frac_frames_{long_pipes}plus_pipes": round(frames_from_long / n_frames, 3),
        "vocab_size": tok.vocab_size,
        "tokens_per_frame_avg": round(tokens.size / n_frames, 2),
        "pipe_dx_bin_hist": dx_bins_hist.tolist(),
    }
    return tokens, meta


def write_dataset(tokens: np.ndarray, meta: dict, out_prefix: str) -> None:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(out.with_suffix(".bin"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate Dreaming Bird training data.")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train")
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--no-pipes", action="store_true",
                    help="Phase 2 mode: bird + gravity + flap only, no pipes.")
    args = ap.parse_args()

    engine_cfg = EngineConfig(pipes_enabled=not args.no_pipes)
    tokens, meta = generate(args.episodes, seed=args.seed, max_frames=args.max_frames,
                            engine_cfg=engine_cfg)
    write_dataset(tokens, meta, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "pipe_dx_bin_hist"}, indent=2))
    print(f"wrote {args.out}.bin ({tokens.size} tokens) and {args.out}.json")

    # Show a short human-readable sample of the first episode.
    tok = Tokenizer()
    print("\nsample tokens (first 24):", tokens[:24].tolist())
    print("fields:", [tok.field_of(int(t)) if t > 3 else f"<{['PAD','BOS','EOS','SEP'][int(t)]}>"
                      for t in tokens[:24]])


if __name__ == "__main__":
    _main()
