"""Central configuration dataclasses.

Physical units are pixels with y growing DOWNWARD (screen coordinates): y=0 is the top,
y=height is the floor. Gravity is therefore positive and a flap sets an upward (negative)
velocity. All values are tunable; the defaults are chosen so that:

  * the bird traverses most of the play area, and
  * typical per-frame motion is on the order of ~1 quantization bin, so that velocity is
    recoverable from the sequence of quantized positions (the whole point — velocity itself
    is never observed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Deterministic Flappy Bird physics parameters."""

    width: float = 288.0
    height: float = 512.0

    bird_x: float = 60.0          # fixed horizontal position of the bird
    bird_radius: float = 12.0     # half-size for collision (treated as a box)

    gravity: float = 1.2          # added to vy every frame (px / frame^2)
    flap_impulse: float = -11.0   # vy is SET to this on a flap (px / frame), not added

    pipe_speed: float = 4.0       # pipes scroll left this many px / frame
    pipe_width: float = 52.0
    gap_height: float = 130.0     # vertical opening the bird must pass through
    pipe_spacing: float = 180.0   # horizontal distance between successive pipe left edges
    spawn_margin: float = 24.0    # keep gap center this far from top/bottom

    start_y: float = 256.0        # default bird start height (screen center)
    start_vy: float = 0.0

    pipes_enabled: bool = True    # False -> Phase 2 "no pipes" mode (pure 1-D kinematics)

    @property
    def gap_center_lo(self) -> float:
        return self.spawn_margin + self.gap_height / 2.0

    @property
    def gap_center_hi(self) -> float:
        return self.height - self.spawn_margin - self.gap_height / 2.0

    @property
    def max_dx(self) -> float:
        """Largest horizontal distance to the next pipe we represent (clamp ceiling)."""
        return self.width


@dataclass(frozen=True)
class TokenizerConfig:
    """Quantization resolution per observed field.

    bird_y gets the finest grid (it is the quantity the model must track precisely and the
    one from which velocity is recovered). gap_y needs the least.
    """

    bird_y_bins: int = 128
    pipe_dx_bins: int = 64
    gap_y_bins: int = 32


@dataclass(frozen=True)
class ModelConfig:
    """nanoGPT-style decoder hyperparameters (used from Phase 2 on)."""

    n_layer: int = 4
    n_head: int = 4
    d_model: int = 192
    block_size: int = 640         # context length in TOKENS (~128 frames * 5 tokens)
    dropout: float = 0.0


# Convenient named tiers from the plan (params are approximate).
NANO = ModelConfig(n_layer=4, n_head=4, d_model=192, block_size=640)        # ~0.5-2M
SMALL = ModelConfig(n_layer=8, n_head=8, d_model=448, block_size=1536)      # ~5-15M
