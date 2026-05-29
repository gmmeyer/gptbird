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


# --- rollout comparison ----------------------------------------------------------------
def rollout_compare(predictor: FramePredictor, actions: list[int], seed: int,
                    tokenizer: Tokenizer | None = None,
                    cfg: EngineConfig | None = None) -> dict:
    """Run ``predictor`` and a ground-truth oracle on the same (seed, actions) and compare.

    bird_y is deterministic, so it is the divergence signal. gap_y is excluded from accuracy
    at pipe-spawn frames (a new gap is RNG-drawn and not predictable); here we simply report
    bird_y error and collision timing, which are well-defined.
    """
    tok = tokenizer or Tokenizer()
    cfg = cfg or tok.ecfg
    oracle = FlappyEngine(cfg, seed=seed)

    pred_tokens0 = predictor.reset(seed=seed)
    by_errors: list[int] = []
    malformed = 0
    first_divergence = None
    pred_death = None
    oracle_death = None

    t0 = time.perf_counter()
    n_tokens = 0
    for frame, a in enumerate(actions):
        pred = predictor.step(a)
        truth = tok.encode_obs(oracle.step(a))
        n_tokens += STATE_TOKENS_PER_FRAME

        # malformed = any predicted token outside its slot's legal range
        for slot, t in enumerate(pred):
            if not tok.legal_mask(slot)[int(t)]:
                malformed += 1

        by_err = abs(tok.decode_state_bins(pred)["bird_y_bin"]
                     - tok.decode_state_bins(truth)["bird_y_bin"])
        by_errors.append(by_err)
        if first_divergence is None and by_err > 0:
            first_divergence = frame

        if pred_death is None and not tok.decode_state_bins(pred)["alive"]:
            pred_death = frame
        if oracle_death is None and not oracle.alive:
            oracle_death = frame
        if pred_death is not None and oracle_death is not None:
            break
    dt = time.perf_counter() - t0

    return {
        "frames_compared": len(by_errors),
        "first_divergence_frame": first_divergence,        # None => never diverged
        "mean_bird_y_bin_error": round(float(np.mean(by_errors)), 4) if by_errors else 0.0,
        "max_bird_y_bin_error": int(np.max(by_errors)) if by_errors else 0,
        "collision_frame_delta": (None if pred_death is None or oracle_death is None
                                  else abs(pred_death - oracle_death)),
        "malformed_token_count": malformed,
        "tokens_per_sec": round(n_tokens / dt) if dt > 0 else None,
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
