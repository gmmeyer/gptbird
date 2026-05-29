"""Engine determinism and physics-sanity tests."""

import numpy as np

from dreaming_bird.config import EngineConfig
from dreaming_bird.engine import ACTION_FLAP, ACTION_NOFLAP, FlappyEngine, rollout


def _fixed_actions(n, seed=123, p_flap=0.3):
    rng = np.random.default_rng(seed)
    return [ACTION_FLAP if rng.random() < p_flap else ACTION_NOFLAP for _ in range(n)]


def _run(seed, actions):
    eng = FlappyEngine(seed=seed)
    obs = [eng.observe()]
    for a in actions:
        obs.append(eng.step(a))
        if not eng.alive:
            break
    return obs


def test_determinism_same_seed_same_stream():
    actions = _fixed_actions(300)
    a = _run(7, actions)
    b = _run(7, actions)
    assert len(a) == len(b)
    for o1, o2 in zip(a, b):
        assert (o1.bird_y, o1.pipe_dx, o1.gap_y, o1.alive) == \
               (o2.bird_y, o2.pipe_dx, o2.gap_y, o2.alive)


def test_different_seed_differs():
    actions = _fixed_actions(300)
    a = _run(1, actions)
    b = _run(2, actions)
    # gap centers (and thus gap_y / death timing) should differ across seeds
    assert any(o1.gap_y != o2.gap_y for o1, o2 in zip(a, b))


def test_no_flap_falls_and_accelerates():
    eng = FlappyEngine(seed=0)
    y0 = eng.observe().bird_y
    o1 = eng.step(ACTION_NOFLAP)
    o2 = eng.step(ACTION_NOFLAP)
    o3 = eng.step(ACTION_NOFLAP)
    assert o1.bird_y > y0          # falls (y grows downward)
    d1, d2, d3 = o1.bird_y - y0, o2.bird_y - o1.bird_y, o3.bird_y - o2.bird_y
    assert d2 > d1 and d3 > d2     # accelerating under gravity


def test_flap_goes_up_relative_to_fall():
    fall = FlappyEngine(seed=0)
    flap = FlappyEngine(seed=0)
    y_fall = fall.step(ACTION_NOFLAP).bird_y
    y_flap = flap.step(ACTION_FLAP).bird_y
    assert y_flap < y_fall          # a flap moves the bird upward (smaller y)


def test_falls_to_death_and_stays_dead():
    eng = FlappyEngine(seed=0)
    obs_list, actions = rollout(eng, lambda o: ACTION_NOFLAP, max_frames=200)
    assert not obs_list[-1].alive               # never flapping -> the bird dies
    assert obs_list[-1].bird_y > obs_list[0].bird_y  # because it fell
    # once dead, stepping is a no-op and stays dead
    after = eng.step(ACTION_FLAP)
    assert not after.alive


def test_velocity_not_in_obs():
    obs = FlappyEngine(seed=0).observe()
    assert not hasattr(obs, "vy") and not hasattr(obs, "bird_vy")
    assert set(vars(obs).keys()) == {"bird_y", "pipe_dx", "gap_y", "alive"}


def test_dx_decreases_as_pipe_approaches():
    eng = FlappyEngine(seed=3)
    dxs = [eng.observe().pipe_dx]
    for _ in range(20):
        dxs.append(eng.step(ACTION_NOFLAP).pipe_dx if eng.alive else dxs[-1])
    # while alive, the next pipe scrolls toward the bird
    assert dxs[5] < dxs[0]
