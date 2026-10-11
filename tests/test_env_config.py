"""S1-10/S1-11 : actions (90, répétition), tirage des formats (±3 %), config et surcharges."""
from collections import Counter

import numpy as np
from rlgym.rocket_league.api import GameState, PhysicsObject

from helpers import make_state
from rlagent.config import load_config
from rlagent.env.actions import N_ACTIONS, make_action_parser
from rlagent.env.mutators import RandomFormatMutator, StartStateMutator


def test_action_space_and_repeat():
    assert N_ACTIONS == 90
    parser = make_action_parser(8)
    s = make_state(1)
    info = {}
    parser.reset(list(s.cars), s, info)
    out = parser.parse_actions({"blue-0": np.array([5]), "orange-0": np.array([80])}, s, info)
    assert out["blue-0"].shape == (8, 8)
    assert np.allclose(info["prev_actions"]["blue-0"], out["blue-0"][0])


def _empty_state():
    s = GameState()
    s.cars = {}
    s.ball = PhysicsObject()
    return s


def test_format_distribution_within_3_percent():
    weights = {1: 0.7, 2: 0.2, 3: 0.1}
    m = RandomFormatMutator(weights, seed=0)
    counts = Counter()
    n = 3000
    for _ in range(n):
        s = _empty_state()
        info = {}
        m.apply(s, info)
        assert len(s.cars) == 2 * info["team_size"]
        counts[info["team_size"]] += 1
    for k, w in weights.items():
        assert abs(counts[k] / n - w) < 0.03, (k, counts[k] / n)


def test_random_start_state_valid():
    m = StartStateMutator({"random": 1.0}, seed=0)
    for _ in range(50):
        s = _empty_state()
        info = {}
        RandomFormatMutator({3: 1.0}, seed=1).apply(s, info)
        m.apply(s, info)
        assert info["start_kind"] == "random"
        assert abs(s.ball.position[0]) < 4096 and abs(s.ball.position[1]) < 5120
        for c in s.cars.values():
            assert c.physics.position[2] == 17.0
            assert c.physics.rotation_mtx.shape == (3, 3)


def test_config_profiles_and_overrides():
    cfg = load_config(["debug"], ["ppo.policy_loss=ppo", "env.format_weights={1: 1.0}"])
    assert cfg["ppo"]["policy_loss"] == "ppo"
    assert cfg["env"]["format_weights"] == {1: 1.0}
    assert cfg["model"]["d_model"] == 32
    assert isinstance(cfg["train"]["n_workers"], int)
    for p in ("laptop", "bigpc"):
        load_config([p])


def test_empty_goal_scenario():
    """S1-20 : 1v1, balle immobile dans la moitié adverse, bleu derrière la balle, orange loin de son but."""
    from rlagent.env.mutators import EmptyGoalMutator
    m = EmptyGoalMutator(seed=0)
    for _ in range(200):
        s = _empty_state()
        info = {}
        m.apply(s, info)
        assert info["team_size"] == 1 and len(s.cars) == 2
        b = s.ball.position
        assert 0 <= b[1] <= 3500 and abs(b[0]) <= 2500 and np.allclose(s.ball.linear_velocity, 0)
        blue = s.cars["blue-0"].physics.position
        orange = s.cars["orange-0"].physics.position
        assert blue[1] < b[1] and abs(blue[0]) <= 3800
        assert orange[1] < -4000  # garé dans le camp bleu, loin du but orange (+Y)
