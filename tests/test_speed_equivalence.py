"""Optimisations de vitesse (18/10/2026) : le code optimisé doit donner EXACTEMENT (à ~1e-6 près) les mêmes
observations et rewards que les copies figées d'avant optimisation (tests/reference/)."""

import numpy as np
import pytest

from helpers import make_state
from reference import obs_v1, rewards_v1
from rlagent.config import load_config
from rlagent.env import obs as obs_new
from rlagent.env import rewards as rewards_new


def random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = make_state(int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])), kickoff=True, seed=int(rng.integers(1000)))
        if rng.random() < 0.8:
            for c in s.cars.values():
                c.physics.position = np.array([rng.uniform(-4000, 4000), rng.uniform(-5000, 5000),
                                               rng.uniform(17, 1800)], dtype=np.float32)
                c.physics.linear_velocity = rng.normal(0, 900, 3).astype(np.float32)
                c.physics.angular_velocity = rng.normal(0, 2, 3).astype(np.float32)
                c.physics.euler_angles = rng.uniform(-np.pi, np.pi, 3).astype(np.float32)
                c.boost_amount = float(rng.uniform(0, 100))
                c.ball_touches = int(rng.random() < 0.3)
                c.boost_active_time = float(rng.random() < 0.5) * 0.1
                c.supersonic_time = float(rng.random() < 0.5) * 0.1
                c.wheels_with_contact = tuple(bool(rng.random() < 0.6) for _ in range(4))
            s.ball.position = np.array([rng.uniform(-4000, 4000), rng.uniform(-5000, 5000), rng.uniform(93, 1900)],
                                       dtype=np.float32)
            s.ball.linear_velocity = rng.normal(0, 1500, 3).astype(np.float32)
            s.ball.angular_velocity = rng.normal(0, 2, 3).astype(np.float32)
        out.append(s)
    return out


@pytest.mark.parametrize("bp", [False, True])
def test_obs_identical_to_reference(bp):
    new, ref = obs_new.AttentionObs(bp), obs_v1.AttentionObs(bp)
    for s in random_states(400):
        agents = list(s.cars)
        info = {"prev_actions": {a: np.random.default_rng(1).uniform(-1, 1, 8).astype(np.float32) for a in agents}}
        a, b = new.build_obs(agents, s, info), ref.build_obs(agents, s, info)
        for k in agents:
            np.testing.assert_allclose(a[k], b[k], atol=2e-6, rtol=1e-5)


def test_rewards_identical_to_reference():
    cfg = load_config(["laptop", "laptop_s2", "laptop_s3", "laptop_s5", "laptop_s6"])
    for k in cfg["rewards"]["weights"]:
        cfg["rewards"]["weights"][k] = cfg["rewards"]["weights"][k] or 0.5   # tous les termes actifs
    new, ref = rewards_new.make_reward(cfg), rewards_v1.make_reward(cfg)
    states = random_states(300, seed=3)
    agents = list(states[0].cars)
    new.reset(agents, states[0], {"global_step": 3e7})
    ref.reset(agents, states[0], {"global_step": 3e7})
    for s in states:
        if list(s.cars) != agents:            # nouveau format : nouvel épisode
            agents = list(s.cars)
            new.reset(agents, s, {})
            ref.reset(agents, s, {})
        a = new.get_rewards(agents, s, {}, {}, {"global_step": 3e7})
        b = ref.get_rewards(agents, s, {}, {}, {"global_step": 3e7})
        for k in agents:
            assert np.isclose(a[k], b[k], atol=1e-6, rtol=1e-5), (k, a[k], b[k])
