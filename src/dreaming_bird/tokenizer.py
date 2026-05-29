"""State <-> token stream conversion, the slot map, and constrained-decoding helpers.

Token grammar (locked in the reviewed plan):
  * Fixed-length frames, ONE token per field value, disjoint per-field id ranges.
  * No digit tokens, no ``by:`` text, no ``</frame>`` parsing.
  * A state is 4 tokens in slot order ``[bird_y, pipe_dx, gap_y, status]``.
  * Stream ordering is ``S_t, A_t, S_{t+1}, ...`` — the action follows the state it is applied
    to. Episode = ``<BOS> s0 a0 s1 a1 ... s_{T-1} a_{T-1} sT <EOS>`` (sT terminal).

At inference, generation is constrained slot-by-slot via :meth:`legal_mask`, so malformed
frames are impossible by construction. The action token is supplied by the player, not
generated, and is excluded from the training loss (:meth:`supervise_mask`).
"""

from __future__ import annotations

import numpy as np

from .engine import ACTION_FLAP, Obs

# Special tokens occupy the first ids.
PAD, BOS, EOS, SEP = 0, 1, 2, 3
N_SPECIAL = 4

STATE_TOKENS_PER_FRAME = 4  # [bird_y, pipe_dx, gap_y, status]


class Tokenizer:
    def __init__(self, engine_cfg=None, tok_cfg=None):
        from .config import EngineConfig, TokenizerConfig

        self.ecfg = engine_cfg if engine_cfg is not None else EngineConfig()
        self.tcfg = tok_cfg if tok_cfg is not None else TokenizerConfig()
        nb, nd, ng = self.tcfg.bird_y_bins, self.tcfg.pipe_dx_bins, self.tcfg.gap_y_bins
        # disjoint contiguous id ranges, in slot order, then status, then action
        self.by_off = N_SPECIAL
        self.dx_off = self.by_off + nb
        self.gap_off = self.dx_off + nd
        self.st_off = self.gap_off + ng       # status: +0 ALIVE, +1 DEAD
        self.act_off = self.st_off + 2        # action: +0 NOFLAP, +1 FLAP
        self.vocab_size = self.act_off + 2

    # --- quantization -----------------------------------------------------------------
    @staticmethod
    def _quant(value: float, lo: float, hi: float, nbins: int) -> int:
        b = int(np.floor((value - lo) / (hi - lo) * nbins))
        return min(max(b, 0), nbins - 1)

    @staticmethod
    def _dequant(b: int, lo: float, hi: float, nbins: int) -> float:
        return lo + (b + 0.5) / nbins * (hi - lo)   # bin center

    def bird_y_bin(self, y: float) -> int:
        return self._quant(y, 0.0, self.ecfg.height, self.tcfg.bird_y_bins)

    def pipe_dx_bin(self, dx: float) -> int:
        return self._quant(dx, 0.0, self.ecfg.max_dx, self.tcfg.pipe_dx_bins)

    def gap_y_bin(self, g: float) -> int:
        return self._quant(g, 0.0, self.ecfg.height, self.tcfg.gap_y_bins)

    # --- token id constructors --------------------------------------------------------
    def by_token(self, b: int) -> int:
        return self.by_off + b

    def dx_token(self, b: int) -> int:
        return self.dx_off + b

    def gap_token(self, b: int) -> int:
        return self.gap_off + b

    def status_token(self, alive: bool) -> int:
        return self.st_off + (0 if alive else 1)

    def action_token(self, action: int) -> int:
        return self.act_off + (1 if action == ACTION_FLAP else 0)

    # --- classification ---------------------------------------------------------------
    def is_action_token(self, tok: int) -> bool:
        return self.act_off <= tok < self.act_off + 2

    def is_state_token(self, tok: int) -> bool:
        return self.by_off <= tok < self.act_off   # bird_y..status (excludes actions/specials)

    def field_of(self, tok: int) -> str:
        if tok < self.by_off:
            return "special"
        if tok < self.dx_off:
            return "bird_y"
        if tok < self.gap_off:
            return "pipe_dx"
        if tok < self.st_off:
            return "gap_y"
        if tok < self.act_off:
            return "status"
        return "action"

    # --- encode -----------------------------------------------------------------------
    def encode_obs(self, obs: Obs) -> list[int]:
        return [
            self.by_token(self.bird_y_bin(obs.bird_y)),
            self.dx_token(self.pipe_dx_bin(obs.pipe_dx)),
            self.gap_token(self.gap_y_bin(obs.gap_y)),
            self.status_token(obs.alive),
        ]

    def encode_episode(self, obs_list: list[Obs], action_list: list[int]) -> np.ndarray:
        toks: list[int] = [BOS]
        for i, obs in enumerate(obs_list):
            toks.extend(self.encode_obs(obs))
            if i < len(action_list):
                toks.append(self.action_token(action_list[i]))
        toks.append(EOS)
        return np.array(toks, dtype=np.uint16)

    # --- decode -----------------------------------------------------------------------
    def decode_state_bins(self, four_tokens) -> dict:
        by, dx, gap, st = four_tokens
        return {
            "bird_y_bin": int(by) - self.by_off,
            "pipe_dx_bin": int(dx) - self.dx_off,
            "gap_y_bin": int(gap) - self.gap_off,
            "alive": (int(st) - self.st_off) == 0,
        }

    def decode_state_to_obs(self, four_tokens) -> Obs:
        d = self.decode_state_bins(four_tokens)
        return Obs(
            bird_y=self._dequant(d["bird_y_bin"], 0.0, self.ecfg.height, self.tcfg.bird_y_bins),
            pipe_dx=self._dequant(d["pipe_dx_bin"], 0.0, self.ecfg.max_dx, self.tcfg.pipe_dx_bins),
            gap_y=self._dequant(d["gap_y_bin"], 0.0, self.ecfg.height, self.tcfg.gap_y_bins),
            alive=d["alive"],
        )

    # --- constrained decoding & loss masking ------------------------------------------
    def legal_mask(self, slot: int) -> np.ndarray:
        """Boolean (vocab_size,) array, True where a token is legal at state-slot 0..3.

        Apply at inference as ``logits[~mask] = -inf`` so generation can only emit a valid
        token for the current field — malformed frames become impossible by construction.
        """
        m = np.zeros(self.vocab_size, dtype=bool)
        spans = [
            (self.by_off, self.dx_off),
            (self.dx_off, self.gap_off),
            (self.gap_off, self.st_off),
            (self.st_off, self.act_off),
        ]
        lo, hi = spans[slot]
        m[lo:hi] = True
        return m

    def supervise_mask(self, tokens: np.ndarray) -> np.ndarray:
        """Boolean mask over TARGETS (``tokens[1:]``): True where loss should be computed.

        We supervise state tokens and ``<EOS>`` but NOT action tokens (player-supplied
        conditioning) or ``<PAD>``.
        """
        targets = np.asarray(tokens[1:])
        is_action = (targets >= self.act_off) & (targets < self.act_off + 2)
        is_pad = targets == PAD
        return ~(is_action | is_pad)
