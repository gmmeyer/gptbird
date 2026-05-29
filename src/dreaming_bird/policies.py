"""Behavior policies for data generation.

The action distribution shapes what the model learns. We blend:
  * random flaps        — covers crash/death dynamics and off-distribution states,
  * a scripted controller — long survival trajectories that thread pipes (steady flight),
  * random-reset starts  — episodes begun at random (y, vy) for state-space coverage.

Coverage (random-reset) and recovery (later: noise augmentation) are complementary; both ship.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .engine import ACTION_FLAP, ACTION_NOFLAP, Obs

Policy = Callable[[Obs], int]


def random_policy(p_flap: float = 0.1, seed: int = 0) -> Policy:
    """Flap with probability ``p_flap`` each frame (independent of state)."""
    rng = np.random.default_rng(seed)

    def policy(obs: Obs) -> int:
        return ACTION_FLAP if rng.random() < p_flap else ACTION_NOFLAP

    return policy


def scripted_policy(aim_offset: float = 8.0) -> Policy:
    """A trivial competent controller: flap when the bird has fallen below the gap center.

    ``aim_offset`` biases the target slightly above center to anticipate gravity (y grows
    downward, so "above" is a smaller y). Produces long survival / pipe-threading trajectories.
    """

    def policy(obs: Obs) -> int:
        return ACTION_FLAP if obs.bird_y > obs.gap_y - aim_offset else ACTION_NOFLAP

    return policy


def noisy_scripted_policy(aim_offset: float = 8.0, p_noise: float = 0.02,
                          seed: int = 0) -> Policy:
    """Scripted controller with occasional random action flips — yields near-misses and
    recoveries (states a pure-expert policy never visits)."""
    rng = np.random.default_rng(seed)
    base = scripted_policy(aim_offset)

    def policy(obs: Obs) -> int:
        a = base(obs)
        if rng.random() < p_noise:
            return ACTION_FLAP if a == ACTION_NOFLAP else ACTION_NOFLAP
        return a

    return policy


def random_start(cfg, rng: np.random.Generator) -> tuple[float, float]:
    """Sample a random (start_y, start_vy) for off-policy / random-reset coverage."""
    y = float(rng.uniform(cfg.bird_radius + 1, cfg.height - cfg.bird_radius - 1))
    vy = float(rng.uniform(cfg.flap_impulse, -cfg.flap_impulse))  # plausible velocity band
    return y, vy
