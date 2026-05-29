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
    """Steps the learned world model one frame at a time, returning decoded observations.

    The model generates the 3 world tokens (bird_y, pipe_dx, gap_y); the alive/dead flag is
    derived from those via :func:`engine.collides` rather than the model's own status token,
    which is unreliable in long free rollouts (it spuriously fires the frame after a pipe is
    passed). The geometry-correct status token is appended to the context so the rollout stays
    on-distribution. This also saves one forward pass per frame.
    """

    def __init__(self, model, tok: Tokenizer, device: str = "cuda",
                 sample_slots: tuple[int, ...] = (2,), temperature: float = 1.0,
                 cfg=None, seed: int = 0, derive_status: bool = True):
        import torch

        from .engine import FlappyEngine
        from .tokenizer import BOS

        self.t = torch
        self.model = model
        self.tok = tok
        self.device = device
        self.sample_slots = sample_slots
        self.temperature = temperature
        self.cfg = cfg or tok.ecfg
        self.derive_status = derive_status     # True: geometry guard; False: use the model's status token
        self._init_engine = FlappyEngine(self.cfg, seed=seed)
        self._masks = [torch.from_numpy(tok.legal_mask(s)).to(device) for s in range(4)]
        self._bos = BOS

    def _decode3(self, s):
        T = self.tok
        return (T._dequant(s[0] - T.by_off, 0.0, self.cfg.height, T.tcfg.bird_y_bins),
                T._dequant(s[1] - T.dx_off, 0.0, self.cfg.max_dx, T.tcfg.pipe_dx_bins),
                T._dequant(s[2] - T.gap_off, 0.0, self.cfg.height, T.tcfg.gap_y_bins))

    def reset(self, seed: int = 0, start_y: float | None = None,
              start_vy: float | None = None) -> Obs:
        o = self._init_engine.reset(seed=seed, start_y=start_y, start_vy=start_vy)
        self.ctx = self.t.tensor([self._bos] + self.tok.encode_obs(o),
                                 device=self.device, dtype=self.t.long)
        return o

    def step(self, action: int) -> Obs:
        from .engine import collides

        t = self.t
        self.ctx = t.cat([self.ctx, t.tensor([self.tok.action_token(action)],
                                             device=self.device, dtype=t.long)])
        n = 3 if self.derive_status else 4
        s = self.model.generate_state(self.ctx, self.tok, temperature=self.temperature,
                                      sample_slots=self.sample_slots, legal_masks=self._masks,
                                      n_slots=n)
        by, dx, gp = self._decode3(s)
        if self.derive_status:
            dead = collides(self.cfg, by, dx, gp)          # geometry guard (default)
            s = s + [self.tok.status_token(not dead)]
        else:
            dead = (s[3] != self.tok.status_token(True))   # trust the model's own status token
        self.ctx = t.cat([self.ctx, t.tensor(s, device=self.device, dtype=t.long)])
        if self.ctx.numel() > self.model.cfg.block_size:
            self.ctx = self.ctx[-self.model.cfg.block_size:]
        return Obs(bird_y=by, pipe_dx=dx, gap_y=gp, alive=not dead)


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


def collision_audit(model, tok: Tokenizer, cfg=None, device: str = "cuda",
                    n: int = 30, max_frames: int = 500, derive_status: bool = True) -> dict:
    """Free-rollout collision calibration — measures BOTH failure modes.

    * Phantom deaths: a competent (gap-tracking) controller plays the dream; any death where
      the bird is mid-air with no pipe at it (or comfortably inside the gap) is a false positive.
    * Missed collisions: an anti-gap controller deliberately flies into pipes; in the real engine
      it dies at ~frame 58, so a long dream survival means the model is *missing* real collisions.
    A well-calibrated model has a low phantom rate AND anti-gap survival near the engine's ~58.
    """
    from .engine import ACTION_FLAP, ACTION_NOFLAP

    cfg = cfg or tok.ecfg
    r = cfg.bird_radius
    scripted = lambda o: ACTION_FLAP if o.bird_y > o.gap_y - 8 else ACTION_NOFLAP
    anti = lambda o: ACTION_FLAP if o.bird_y > (cfg.height - o.gap_y) else ACTION_NOFLAP

    def is_phantom(o):
        if o.bird_y <= r + 2 or o.bird_y >= cfg.height - r - 2:
            return False                                   # floor/ceiling: real
        if o.pipe_dx <= r + 2:
            return abs(o.bird_y - o.gap_y) <= cfg.gap_height / 2 - r   # inside gap => phantom
        return True                                        # no pipe near the bird => phantom

    def play(policy, seed):
        ds = DreamStepper(model, tok, device=device, sample_slots=(2,), cfg=cfg, seed=seed,
                          derive_status=derive_status)
        o = ds.reset(seed=seed)
        i = 0
        for i in range(max_frames):
            o = ds.step(policy(o))
            if not o.alive:
                return i + 1, o
        return max_frames, o

    phantom = legit = capped = 0
    s_surv = []
    for s in range(n):
        f, o = play(scripted, s)
        s_surv.append(f)
        if not o.alive:
            phantom += is_phantom(o); legit += (not is_phantom(o))
        else:
            capped += 1
    a_surv = [play(anti, 1000 + s)[0] for s in range(n)]
    deaths = phantom + legit
    return {
        "scripted_survival_mean": round(float(np.mean(s_surv)), 1),
        "phantom_deaths": phantom, "real_deaths": legit, "survived_to_cap": capped,
        "phantom_rate": round(phantom / deaths, 3) if deaths else 0.0,
        "antigap_survival_mean": round(float(np.mean(a_surv)), 1),   # engine ~58; high => misses
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
