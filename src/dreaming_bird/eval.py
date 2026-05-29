"""Evaluation / replay harness — the instrument that keeps Phase 4 honest.

Everything here is model-agnostic: metrics are computed against a :class:`FramePredictor`,
an object that — like the eventual trained model — produces the next frame's 4 state tokens
given the action applied. The oracle itself implements this interface
(:class:`OraclePredictor`), so the harness is fully testable *now*: oracle-vs-oracle must yield
zero divergence. In Phase 4 we pass a model-backed predictor instead.

Metrics (from the reviewed plan): first-divergence frame, mean bird_y bin error,
collision-frame delta, malformed-token count, and realtime tokens/sec.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .config import EngineConfig, TokenizerConfig
from .engine import FlappyEngine
from .tokenizer import STATE_TOKENS_PER_FRAME, Tokenizer


# --- determinism -----------------------------------------------------------------------
def replay_is_deterministic(seed: int, actions: list[int],
                            cfg: EngineConfig | None = None) -> bool:
    """Run the engine twice on the same (seed, actions); True iff the obs streams are identical."""
    cfg = cfg or EngineConfig()

    def run():
        e = FlappyEngine(cfg, seed=seed)
        out = [(e.observe().bird_y, e.observe().pipe_dx, e.observe().gap_y, e.observe().alive)]
        for a in actions:
            o = e.step(a)
            out.append((o.bird_y, o.pipe_dx, o.gap_y, o.alive))
            if not e.alive:
                break
        return out

    return run() == run()


# --- predictor interface ---------------------------------------------------------------
class FramePredictor(Protocol):
    """Produces the next frame as 4 state tokens, given the action applied this frame."""

    def reset(self, seed: int, start_y: float | None = None,
              start_vy: float | None = None) -> list[int]: ...

    def step(self, action: int) -> list[int]: ...


class OraclePredictor:
    """Reference predictor backed by the real engine — used to self-test the harness."""

    def __init__(self, tokenizer: Tokenizer, cfg: EngineConfig | None = None, seed: int = 0):
        self.tok = tokenizer
        self.engine = FlappyEngine(cfg or tokenizer.ecfg, seed=seed)

    def reset(self, seed, start_y=None, start_vy=None):
        return self.tok.encode_obs(self.engine.reset(seed=seed, start_y=start_y, start_vy=start_vy))

    def step(self, action):
        return self.tok.encode_obs(self.engine.step(action))


class ModelPredictor:
    """Predictor backed by a trained DreamGPT — drives the autoregressive dream rollout.

    The true initial frame is taken from a real engine (the model is *given* the start, then
    must roll forward on its own). Each ``step`` appends the action token and generates the next
    4 state tokens with slot-constrained decoding. Cacheless (Phase 4 adds a KV cache for
    real-time play); ``sample_slots`` defaults to greedy everywhere for clean drift measurement.
    """

    def __init__(self, model, tokenizer: Tokenizer, device: str = "cuda",
                 sample_slots: tuple[int, ...] = (), temperature: float = 1.0,
                 cfg: EngineConfig | None = None, seed: int = 0):
        import torch

        self.torch = torch
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.sample_slots = sample_slots
        self.temperature = temperature
        self.init_engine = FlappyEngine(cfg or tokenizer.ecfg, seed=seed)
        self._masks = [torch.from_numpy(tokenizer.legal_mask(s)).to(device) for s in range(4)]
        from .tokenizer import BOS
        self._bos = BOS

    def reset(self, seed, start_y=None, start_vy=None):
        s0 = self.tok.encode_obs(self.init_engine.reset(seed=seed, start_y=start_y, start_vy=start_vy))
        self.context = self.torch.tensor([self._bos] + s0, device=self.device, dtype=self.torch.long)
        return s0

    def step(self, action):
        act_tok = self.torch.tensor([self.tok.action_token(action)], device=self.device,
                                    dtype=self.torch.long)
        self.context = self.torch.cat([self.context, act_tok])
        state = self.model.generate_state(self.context, self.tok, temperature=self.temperature,
                                          sample_slots=self.sample_slots, legal_masks=self._masks)
        st = self.torch.tensor(state, device=self.device, dtype=self.torch.long)
        self.context = self.torch.cat([self.context, st])
        # keep the context within block_size
        if self.context.numel() > self.model.cfg.block_size:
            self.context = self.context[-self.model.cfg.block_size:]
        return state


# --- rollout comparison ----------------------------------------------------------------
def rollout_compare(predictor: FramePredictor, actions: list[int], seed: int,
                    tokenizer: Tokenizer | None = None, cfg: EngineConfig | None = None,
                    tol_bins: int = 3, stop_at_death: bool = True) -> dict:
    """Run ``predictor`` and a ground-truth oracle on the same (seed, actions) and compare.

    bird_y is the divergence signal (deterministic given actions). The **drift horizon** is the
    first frame where the bird_y bin error exceeds ``tol_bins`` (a visible deviation); a small
    occasional 1-bin blip is not drift. gap_y at pipe-spawn frames is intentionally not scored
    here (a new gap is RNG-drawn and unpredictable).
    """
    tok = tokenizer or Tokenizer()
    cfg = cfg or tok.ecfg
    oracle = FlappyEngine(cfg, seed=seed)

    predictor.reset(seed=seed)
    by_errors: list[int] = []
    malformed = 0
    first_divergence = None
    drift_horizon = None
    pred_death = None
    oracle_death = None

    t0 = time.perf_counter()
    n_tokens = 0
    for frame, a in enumerate(actions):
        pred = predictor.step(a)
        truth = tok.encode_obs(oracle.step(a))
        n_tokens += STATE_TOKENS_PER_FRAME

        for slot, t in enumerate(pred):                    # malformed = outside slot range
            if not tok.legal_mask(slot)[int(t)]:
                malformed += 1

        by_err = abs(tok.decode_state_bins(pred)["bird_y_bin"]
                     - tok.decode_state_bins(truth)["bird_y_bin"])
        by_errors.append(by_err)
        if first_divergence is None and by_err > 0:
            first_divergence = frame
        if drift_horizon is None and by_err > tol_bins:
            drift_horizon = frame

        if pred_death is None and not tok.decode_state_bins(pred)["alive"]:
            pred_death = frame
        if oracle_death is None and not oracle.alive:
            oracle_death = frame
        if stop_at_death and pred_death is not None and oracle_death is not None:
            break
    dt = time.perf_counter() - t0

    return {
        "frames_compared": len(by_errors),
        "first_divergence_frame": first_divergence,        # None => never diverged at all
        "drift_horizon": drift_horizon,                    # None => stayed within tol_bins throughout
        "mean_bird_y_bin_error": round(float(np.mean(by_errors)), 4) if by_errors else 0.0,
        "max_bird_y_bin_error": int(np.max(by_errors)) if by_errors else 0,
        "collision_frame_delta": (None if pred_death is None or oracle_death is None
                                  else abs(pred_death - oracle_death)),
        "malformed_token_count": malformed,
        "tokens_per_sec": round(n_tokens / dt) if dt > 0 else None,
    }


def evaluate_pipes(model, tok: Tokenizer, cfg: EngineConfig | None = None,
                   n_episodes: int = 64, seed: int = 0, max_frames: int = 250,
                   device: str = "cuda", policy_seed_base: int = 100000) -> dict:
    """Teacher-forced one-step evaluation on episodes WITH pipes.

    gap_y is split: on STABLE frames (same pipe still ahead) it must match exactly; on
    SWITCH frames (a newly-revealed pipe whose gap was RNG-drawn) it is scored only on
    VALIDITY — is the predicted gap within the legal spawn band — per the reviewed plan
    ("judged on validity, not identity"). Collision timing compares the frame the model first
    predicts DEAD against the oracle's death frame.
    """
    import torch

    from .engine import FlappyEngine, rollout
    from .policies import noisy_scripted_policy

    cfg = cfg or tok.ecfg
    band_lo = tok.gap_y_bin(cfg.gap_center_lo)
    band_hi = tok.gap_y_bin(cfg.gap_center_hi)
    a = {k: 0 for k in ("bird_t", "bird_e", "bird_w1", "dx_t", "dx_e", "gs_t", "gs_e",
                        "gw_t", "gw_v", "st_t", "st_e")}
    coll_deltas: list[int | None] = []

    was = model.training
    model.eval()
    for i in range(n_episodes):
        e = FlappyEngine(cfg, seed=seed + i)
        ol, al = rollout(e, noisy_scripted_policy(seed=policy_seed_base + i), max_frames=max_frames)
        stream = tok.encode_episode(ol, al)
        T = min(len(stream) - 1, model.cfg.block_size)
        x = torch.tensor(stream[:T].astype(np.int64), device=device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x)
        preds = logits[0].argmax(-1).to("cpu").numpy()
        targets = stream[1:T + 1]

        prev_gap = None
        frame = -1
        o_death = p_death = None
        dead_tok = tok.status_token(False)
        for p in range(len(targets)):
            tgt, pr = int(targets[p]), int(preds[p])
            f = tok.field_of(tgt)
            if f == "bird_y":
                frame += 1
                a["bird_t"] += 1
                a["bird_e"] += pr == tgt
                a["bird_w1"] += tok.by_off <= pr < tok.dx_off and abs(pr - tgt) <= 1
            elif f == "pipe_dx":
                a["dx_t"] += 1
                a["dx_e"] += pr == tgt
            elif f == "gap_y":
                if prev_gap is None or tgt != prev_gap:        # spawn switch -> validity only
                    a["gw_t"] += 1
                    a["gw_v"] += tok.gap_off <= pr < tok.st_off and band_lo <= pr - tok.gap_off <= band_hi
                else:                                          # stable -> must match
                    a["gs_t"] += 1
                    a["gs_e"] += pr == tgt
                prev_gap = tgt
            elif f == "status":
                a["st_t"] += 1
                a["st_e"] += pr == tgt
                if o_death is None and tgt == dead_tok:
                    o_death = frame
                if p_death is None and pr == dead_tok:
                    p_death = frame
        if o_death is not None:
            coll_deltas.append(abs(p_death - o_death) if p_death is not None else None)
    if was:
        model.train()

    def ratio(n, d):
        return round(a[n] / a[d], 4) if a[d] else float("nan")

    caught = [d for d in coll_deltas if d is not None]
    return {
        "episodes": n_episodes,
        "n_frames": a["bird_t"],
        "bird_y_exact": ratio("bird_e", "bird_t"),
        "bird_y_within1": ratio("bird_w1", "bird_t"),
        "pipe_dx_exact": ratio("dx_e", "dx_t"),
        "gap_y_stable_exact": ratio("gs_e", "gs_t"),
        "gap_y_switch_validity": ratio("gw_v", "gw_t"),
        "gap_switches": a["gw_t"],
        "status_exact": ratio("st_e", "st_t"),
        "collision_delta_mean": round(float(np.mean(caught)), 3) if caught else None,
        "collision_within1_rate": round(float(np.mean([d <= 1 for d in caught])), 3) if caught else None,
        "collision_missed": sum(1 for d in coll_deltas if d is None),
    }


def _self_test() -> None:
    """Oracle-vs-oracle must be a perfect rollout — validates the harness end to end."""
    tok = Tokenizer()
    rng = np.random.default_rng(0)
    actions = [int(rng.random() < 0.3) for _ in range(400)]
    assert replay_is_deterministic(0, actions), "engine replay is not deterministic!"
    m = rollout_compare(OraclePredictor(tok, seed=7), actions, seed=7, tokenizer=tok)
    assert m["first_divergence_frame"] is None, m
    assert m["mean_bird_y_bin_error"] == 0.0 and m["malformed_token_count"] == 0, m
    assert m["collision_frame_delta"] == 0, m
    print("eval self-test OK:", m)


if __name__ == "__main__":
    _self_test()
