"""Stream-alignment tests: the S_t, A_t, S_{t+1} ordering and the training loss mask.

A one-frame misalignment here makes the physics unlearnable and looks like "the model just
doesn't work", so this is asserted explicitly.
"""

import numpy as np

from dreaming_bird.engine import ACTION_FLAP, ACTION_NOFLAP, FlappyEngine, rollout
from dreaming_bird.policies import scripted_policy
from dreaming_bird.tokenizer import BOS, EOS, STATE_TOKENS_PER_FRAME, Tokenizer


def _episode_stream(seed=0, max_frames=60):
    t = Tokenizer()
    eng = FlappyEngine(seed=seed)
    obs_list, actions = rollout(eng, scripted_policy(), max_frames=max_frames)
    return t, t.encode_episode(obs_list, actions), obs_list, actions


def test_slot_ordering_state_then_action():
    """After BOS the stream is repeating [bird_y, pipe_dx, gap_y, status, action] units,
    ending in a terminal state (no trailing action) then EOS."""
    t, stream, obs_list, actions = _episode_stream()
    n_trans = len(actions)
    expected_fields = ["bird_y", "pipe_dx", "gap_y", "status"]

    assert stream[0] == BOS
    idx = 1
    for step in range(n_trans):                 # full [state + action] units
        for off, field in enumerate(expected_fields):
            assert t.field_of(int(stream[idx + off])) == field
        assert t.is_action_token(int(stream[idx + 4]))
        idx += 5
    # terminal state (4 tokens, no action), then EOS
    for off, field in enumerate(expected_fields):
        assert t.field_of(int(stream[idx + off])) == field
    idx += 4
    assert stream[idx] == EOS
    assert idx == len(stream) - 1


def test_action_tokens_at_expected_positions():
    t, stream, obs_list, actions = _episode_stream()
    # actions sit at index 5 + 5*k (0-based), i.e. right after each 4-token state following BOS
    for k in range(len(actions)):
        pos = 1 + 4 + 5 * k
        assert t.is_action_token(int(stream[pos]))
        decoded = 1 if int(stream[pos]) == t.action_token(ACTION_FLAP) else 0
        assert decoded == actions[k]


def test_supervise_mask_excludes_actions_only():
    t, stream, obs_list, actions = _episode_stream()
    mask = t.supervise_mask(stream)           # over targets = stream[1:]
    targets = stream[1:]
    assert len(mask) == len(targets)
    # exactly the action tokens are excluded (no PAD in this stream)
    n_actions = sum(1 for tok in targets if t.is_action_token(int(tok)))
    assert (~mask).sum() == n_actions
    # every supervised target is a state token or EOS, never an action
    for keep, tok in zip(mask, targets):
        if keep:
            assert t.is_state_token(int(tok)) or int(tok) == EOS


def test_action_is_conditioning_not_predicted():
    """Sanity: the positions we DON'T supervise are precisely the action tokens."""
    t, stream, obs_list, actions = _episode_stream()
    mask = t.supervise_mask(stream)
    targets = stream[1:]
    for keep, tok in zip(mask, targets):
        assert keep != t.is_action_token(int(tok))
