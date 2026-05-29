"""Dreaming Bird — a neural Flappy Bird world model.

A decoder-only transformer trained to *be* Flappy Bird: given the recent frames and the
player's action, it emits the next frame's quantized state. Looped autoregressively, the
model replaces the physics engine at runtime.

See IMPLEMENTATION_PLAN.md for architecture and phase plan.
"""

__version__ = "0.1.0"
