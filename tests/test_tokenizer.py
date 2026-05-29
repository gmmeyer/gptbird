"""Tokenizer round-trip, disjoint-range, and constrained-decoding tests."""

import numpy as np

from dreaming_bird.config import EngineConfig, TokenizerConfig
from dreaming_bird.engine import ACTION_FLAP, ACTION_NOFLAP, Obs
from dreaming_bird.tokenizer import BOS, EOS, N_SPECIAL, Tokenizer


def test_vocab_layout_disjoint_and_sized():
    t = Tokenizer()
    tc = t.tcfg
    expected = N_SPECIAL + tc.bird_y_bins + tc.pipe_dx_bins + tc.gap_y_bins + 2 + 2
    assert t.vocab_size == expected
    # ranges are contiguous and ordered
    assert t.by_off == N_SPECIAL
    assert t.dx_off == t.by_off + tc.bird_y_bins
    assert t.gap_off == t.dx_off + tc.pipe_dx_bins
    assert t.st_off == t.gap_off + tc.gap_y_bins
    assert t.act_off == t.st_off + 2


def test_field_classification():
    t = Tokenizer()
    assert t.field_of(t.by_token(0)) == "bird_y"
    assert t.field_of(t.dx_token(0)) == "pipe_dx"
    assert t.field_of(t.gap_token(0)) == "gap_y"
    assert t.field_of(t.status_token(True)) == "status"
    assert t.field_of(t.action_token(ACTION_FLAP)) == "action"
    assert t.is_action_token(t.action_token(ACTION_NOFLAP))
    assert not t.is_action_token(t.by_token(0))
    assert t.is_state_token(t.status_token(False))
    assert not t.is_state_token(t.action_token(ACTION_FLAP))


def test_bin_quantization_idempotent():
    """quantize(dequantize(b)) == b for every bin of every field."""
    t = Tokenizer()
    h, mdx = t.ecfg.height, t.ecfg.max_dx
    for b in range(t.tcfg.bird_y_bins):
        assert t.bird_y_bin(t._dequant(b, 0.0, h, t.tcfg.bird_y_bins)) == b
    for b in range(t.tcfg.pipe_dx_bins):
        assert t.pipe_dx_bin(t._dequant(b, 0.0, mdx, t.tcfg.pipe_dx_bins)) == b
    for b in range(t.tcfg.gap_y_bins):
        assert t.gap_y_bin(t._dequant(b, 0.0, h, t.tcfg.gap_y_bins)) == b


def test_quantization_clamps_out_of_range():
    t = Tokenizer()
    assert t.bird_y_bin(-50.0) == 0
    assert t.bird_y_bin(t.ecfg.height + 999) == t.tcfg.bird_y_bins - 1


def test_obs_encode_decode_roundtrip_to_bins():
    t = Tokenizer()
    obs = Obs(bird_y=137.0, pipe_dx=90.0, gap_y=300.0, alive=True)
    toks = t.encode_obs(obs)
    assert len(toks) == 4
    d = t.decode_state_bins(toks)
    assert d["bird_y_bin"] == t.bird_y_bin(obs.bird_y)
    assert d["pipe_dx_bin"] == t.pipe_dx_bin(obs.pipe_dx)
    assert d["gap_y_bin"] == t.gap_y_bin(obs.gap_y)
    assert d["alive"] is True
    # decoded physical value lands within half a bin of the original
    back = t.decode_state_to_obs(toks)
    assert abs(back.bird_y - obs.bird_y) <= (t.ecfg.height / t.tcfg.bird_y_bins)


def test_legal_mask_matches_field_ranges():
    t = Tokenizer()
    for slot, field in enumerate(["bird_y", "pipe_dx", "gap_y", "status"]):
        m = t.legal_mask(slot)
        assert m.dtype == bool and m.shape == (t.vocab_size,)
        # every legal token classifies to the right field; nothing else is legal
        legal_ids = np.nonzero(m)[0]
        assert all(t.field_of(int(i)) == field for i in legal_ids)
        assert m.sum() > 0


def test_constrained_argmax_stays_in_field():
    """Masking arbitrary logits to a slot forces a token of that field."""
    t = Tokenizer()
    rng = np.random.default_rng(0)
    for slot, field in enumerate(["bird_y", "pipe_dx", "gap_y", "status"]):
        logits = rng.standard_normal(t.vocab_size)
        logits[~t.legal_mask(slot)] = -np.inf
        assert t.field_of(int(logits.argmax())) == field


def test_episode_structure_and_length():
    t = Tokenizer()
    obs_list = [Obs(100, 80, 256, True), Obs(110, 76, 256, True), Obs(480, 72, 256, False)]
    actions = [ACTION_NOFLAP, ACTION_FLAP]  # T=2 transitions, 3 states
    stream = t.encode_episode(obs_list, actions)
    assert stream[0] == BOS and stream[-1] == EOS
    # length = BOS + 4*(T+1) states + T actions + EOS
    assert len(stream) == 1 + 4 * 3 + 2 + 1
