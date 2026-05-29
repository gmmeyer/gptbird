"""Real-time autoregressive rollout — the dream with the engine removed.

:class:`DreamStepper` is the playable core: it is given a true initial frame, then generates
every subsequent frame itself via slot-constrained decoding (sampling at the gap slot so newly
revealed pipes are plausibly invented rather than mean-blurred). It returns decoded :class:`Obs`
for rendering. Generation is currently cacheless; :func:`benchmark` measures whether that already
clears the real-time budget on this GPU before we invest in a KV cache.
"""

from __future__ import annotations

import time

import numpy as np

from .engine import ACTION_NOFLAP, Obs
from .tokenizer import Tokenizer


class DreamStepper:
    """Steps the learned world model one frame at a time, returning decoded observations."""

    def __init__(self, model, tok: Tokenizer, device: str = "cuda",
                 sample_slots: tuple[int, ...] = (2,), temperature: float = 1.0,
                 cfg=None, seed: int = 0):
        from .eval import ModelPredictor

        self.tok = tok
        self._mp = ModelPredictor(model, tok, device=device, sample_slots=sample_slots,
                                  temperature=temperature, cfg=cfg, seed=seed)

    def reset(self, seed: int = 0, start_y: float | None = None,
              start_vy: float | None = None) -> Obs:
        toks = self._mp.reset(seed=seed, start_y=start_y, start_vy=start_vy)
        return self.tok.decode_state_to_obs(toks)

    def step(self, action: int) -> Obs:
        return self.tok.decode_state_to_obs(self._mp.step(action))


def benchmark(model, tok: Tokenizer, n_frames: int = 300, device: str = "cuda",
              warmup: int = 20, cfg=None) -> dict:
    """Measure sustained frame rate of cacheless dreaming (real-time gate for Phase 4)."""
    stepper = DreamStepper(model, tok, device=device, cfg=cfg)
    stepper.reset(seed=0)
    for _ in range(warmup):
        stepper.step(ACTION_NOFLAP)
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_frames):
        stepper.step(ACTION_NOFLAP)
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "frames": n_frames,
        "seconds": round(dt, 3),
        "fps": round(n_frames / dt, 1),
        "ms_per_frame": round(1000 * dt / n_frames, 2),
        "realtime_30fps": (n_frames / dt) >= 30,
        "realtime_60fps": (n_frames / dt) >= 60,
    }


def free_rollout_drift(model, tok: Tokenizer, cfg=None, seed: int = 0, max_frames: int = 600,
                       device: str = "cuda", tol_bins: int = 3) -> dict:
    """A scripted controller plays the DREAM (reacting to the model's own frames) while a
    no-pipes oracle is stepped with the SAME actions. Reports how the dreamed bird_y tracks
    true physics (drift horizon / mean error) and how long the controller survives in the dream.
    """
    from .config import EngineConfig
    from .engine import FlappyEngine
    from .policies import scripted_policy

    cfg = cfg or tok.ecfg
    stepper = DreamStepper(model, tok, device=device, sample_slots=(2,), cfg=cfg, seed=seed)
    obs = stepper.reset(seed=seed)
    shadow = FlappyEngine(EngineConfig(pipes_enabled=False), seed=seed)
    shadow.reset(seed=seed, start_y=obs.bird_y, start_vy=0.0)
    policy = scripted_policy()

    errs: list[int] = []
    drift = None
    pipes = 0
    prev_dx = obs.pipe_dx
    survived = 0
    for f in range(max_frames):
        a = policy(obs)
        obs = stepper.step(a)
        sh = shadow.step(a)
        e = abs(tok.bird_y_bin(obs.bird_y) - tok.bird_y_bin(sh.bird_y))
        errs.append(e)
        if drift is None and e > tol_bins:
            drift = f
        if obs.pipe_dx > prev_dx + cfg.width * 0.3:
            pipes += 1
        prev_dx = obs.pipe_dx
        survived = f + 1
        if not obs.alive:
            break
    return {
        "survived_frames": survived,
        "pipes_passed": pipes,
        "drift_horizon": drift,                       # None => bird_y stayed within tol the whole run
        "mean_bird_y_bin_error": round(float(np.mean(errs)), 3),
        "max_bird_y_bin_error": int(np.max(errs)),
        "died_in_dream": not obs.alive,
    }


def _main() -> None:
    import argparse

    import torch

    from .model import DreamGPT

    ap = argparse.ArgumentParser(description="Benchmark cacheless dream rollout fps.")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/small_pipes.pt")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    tok = Tokenizer(engine_cfg=ck.get("engine_cfg"), tok_cfg=ck.get("tok_cfg"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamGPT(ck["model_cfg"], ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(benchmark(model, tok, n_frames=args.frames, device=device, cfg=ck.get("engine_cfg")))


if __name__ == "__main__":
    _main()
