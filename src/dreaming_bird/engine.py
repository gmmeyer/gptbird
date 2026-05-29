"""The oracle: a deterministic, seeded Flappy Bird physics engine.

This is ground truth. It generates training data and, at eval time, validates the model's
rollouts. It never ships inside the playable artifact (the trained model replaces it).

Conventions (frozen — tests assert them):
  * y grows downward; gravity is positive, a flap SETS vy to a negative impulse.
  * Transition convention is ``S_t , A_t -> S_{t+1}``: the action passed to ``step`` is the
    action applied at the current frame, producing the next frame.
  * Collision is checked AFTER the position update; on collision the engine emits one final
    frame with ``alive=False`` and then terminates.

Velocity (``bird_vy``) is part of the engine's internal state but is deliberately NOT part of
the observation (:class:`Obs`) — recovering it from the sequence of positions is the project's
central experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ACTION_NOFLAP = 0
ACTION_FLAP = 1


@dataclass
class Obs:
    """A single observed frame, in physical units. Note: no velocity."""

    bird_y: float
    pipe_dx: float   # horizontal distance from the bird to the next pipe's left edge (>=0)
    gap_y: float     # vertical center of the next pipe's gap
    alive: bool


class FlappyEngine:
    """Stateful, seeded Flappy Bird simulation."""

    def __init__(self, cfg=None, seed: int = 0):
        from .config import EngineConfig

        self.cfg = cfg if cfg is not None else EngineConfig()
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, start_y: float | None = None,
              start_vy: float | None = None) -> Obs:
        """Reset to the start of an episode. Pass ``seed`` for reproducibility; ``start_y`` /
        ``start_vy`` enable random-reset (off-policy) trajectories."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.bird_y = cfg.start_y if start_y is None else float(start_y)
        self.bird_vy = cfg.start_vy if start_vy is None else float(start_vy)
        self.alive = True
        self.frame = 0
        self.pipes: list[list[float]] = []   # each pipe is [left_x, gap_center_y]
        if cfg.pipes_enabled:
            self._spawn_pipe(left=cfg.width)  # first pipe just off the right edge
        return self.observe()

    def _spawn_pipe(self, left: float) -> None:
        gc = float(self._rng.uniform(self.cfg.gap_center_lo, self.cfg.gap_center_hi))
        self.pipes.append([float(left), gc])

    def _next_pipe(self) -> list[float] | None:
        """The first pipe whose right edge is at or ahead of the bird (the one to thread)."""
        bx = self.cfg.bird_x
        for p in self.pipes:
            if p[0] + self.cfg.pipe_width >= bx:
                return p
        return None

    def observe(self) -> Obs:
        cfg = self.cfg
        p = self._next_pipe()
        if p is None:
            dx, gap_y = cfg.max_dx, cfg.height / 2.0
        else:
            dx = min(max(p[0] - cfg.bird_x, 0.0), cfg.max_dx)
            gap_y = p[1]
        return Obs(bird_y=self.bird_y, pipe_dx=dx, gap_y=gap_y, alive=self.alive)

    def step(self, action: int) -> Obs:
        """Apply ``action`` at the current frame and advance one frame. Returns the new Obs."""
        if not self.alive:
            return self.observe()
        cfg = self.cfg
        # 1. action -> velocity. A flap SETS vy (overrides momentum); gravity always applies.
        if action == ACTION_FLAP:
            self.bird_vy = cfg.flap_impulse
        self.bird_vy += cfg.gravity
        # 2. integrate position
        self.bird_y += self.bird_vy
        # 3. scroll pipes left, cull fully-passed ones, spawn ahead to maintain spacing
        if cfg.pipes_enabled:
            for p in self.pipes:
                p[0] -= cfg.pipe_speed
            self.pipes = [p for p in self.pipes if p[0] + cfg.pipe_width >= 0.0]
            if not self.pipes:
                self._spawn_pipe(left=cfg.width)
            else:
                rightmost = max(p[0] for p in self.pipes)
                if rightmost <= cfg.width - cfg.pipe_spacing:
                    self._spawn_pipe(left=rightmost + cfg.pipe_spacing)
        # 4. collision is checked AFTER the position update
        if self._collided():
            self.alive = False
        self.frame += 1
        return self.observe()

    def _collided(self) -> bool:
        cfg = self.cfg
        r = cfg.bird_radius
        y = self.bird_y
        if y - r <= 0.0 or y + r >= cfg.height:        # ceiling / floor
            return True
        bx = cfg.bird_x
        for left, gc in self.pipes:
            x_overlap = (bx + r > left) and (bx - r < left + cfg.pipe_width)
            if x_overlap:
                gap_top = gc - cfg.gap_height / 2.0
                gap_bot = gc + cfg.gap_height / 2.0
                if y - r < gap_top or y + r > gap_bot:
                    return True
        return False


def collides(cfg, bird_y: float, pipe_dx: float, gap_y: float) -> bool:
    """Collision test from an *observation* (bird_y, distance-to-next-pipe, gap center).

    Same geometry as the engine's internal check, but expressed over what the world model
    emits — so the playable dream can adjudicate collisions from its own dreamed positions
    (the model's learned alive/dead flag is unreliable in long free rollouts: it panics the
    frame after a pipe is passed). The bird overlaps the pipe column only when ``pipe_dx`` is
    within a bird radius; only then does being outside the gap kill it.
    """
    r = cfg.bird_radius
    if bird_y - r <= 0.0 or bird_y + r >= cfg.height:           # floor / ceiling
        return True
    if pipe_dx <= r:                                            # bird is in the pipe's column
        if bird_y - r < gap_y - cfg.gap_height / 2.0 or bird_y + r > gap_y + cfg.gap_height / 2.0:
            return True
    return False


def rollout(engine: FlappyEngine, policy: Callable[[Obs], int],
            max_frames: int = 2048) -> tuple[list[Obs], list[int]]:
    """Run an episode until death or ``max_frames``.

    Returns ``(obs_list, action_list)`` with ``len(obs_list) == len(action_list) + 1`` — the
    last obs is terminal (``alive=False``) unless ``max_frames`` was hit first.
    """
    obs = engine.observe()
    obs_list = [obs]
    action_list: list[int] = []
    for _ in range(max_frames):
        if not engine.alive:
            break
        a = policy(obs)
        nobs = engine.step(a)
        action_list.append(a)
        obs_list.append(nobs)
        obs = nobs
    return obs_list, action_list
