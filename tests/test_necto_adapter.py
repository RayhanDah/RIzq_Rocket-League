"""Adaptateur Necto (audits) : symétrie bleu/orange de l'observation, table d'actions, parser mixte."""
import copy

import numpy as np
import pytest
import torch
from helpers import make_state, mirror_car_into
from rlgym.rocket_league.api import PhysicsObject
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

from rlagent.env.actions import make_action_parser
from rlagent.opponents.necto import KICKOFF_TICKS, MixedRepeatAction, NectoObs, parse_logits

PADS = np.array(BOOST_LOCATIONS, dtype=np.float64)


def _random_state(seed):
    rng = np.random.default_rng(seed)
    s = make_state(1, kickoff=True, seed=seed)
    for car in s.cars.values():
        car.physics.position = rng.uniform([-3000, -4000, 17], [3000, 4000, 1500]).astype(np.float32)
        car.physics.linear_velocity = rng.uniform(-1500, 1500, 3).astype(np.float32)
        car.physics.angular_velocity = rng.uniform(-5, 5, 3).astype(np.float32)
        car.physics.euler_angles = rng.uniform(-np.pi / 2, np.pi / 2, 3).astype(np.float32)
        car.boost_amount = float(rng.uniform(0, 100))
        car.on_ground = bool(rng.integers(2))
    s.ball.position = rng.uniform([-3000, -4000, 93], [3000, 4000, 1800]).astype(np.float32)
    s.ball.linear_velocity = rng.uniform(-2000, 2000, 3).astype(np.float32)
    s.ball.angular_velocity = rng.uniform(-6, 6, 3).astype(np.float32)
    s.ball.rotation_mtx = np.eye(3, dtype=np.float32)
    s.boost_pad_timers = np.where(rng.random(34) < 0.3, rng.uniform(0.1, 10, 34), 0).astype(np.float32)
    return s


def _mirrored(s):
    """Rotation de 180° autour de Z et échange des couleurs : l'orange de m est l'image du bleu de s."""
    m = copy.deepcopy(s)
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    orange = next(a for a, c in s.cars.items() if c.team_num == 1)
    mirror_car_into(s, blue, m, orange)
    mirror_car_into(s, orange, m, blue)
    inv = s.ball.inverted()
    m.ball = PhysicsObject()
    m.ball.position, m.ball.linear_velocity = inv.position.copy(), inv.linear_velocity.copy()
    m.ball.angular_velocity, m.ball.rotation_mtx = inv.angular_velocity.copy(), inv.rotation_mtx.copy()
    m.boost_pad_timers = s.boost_pad_timers[::-1].copy()      # pads symétriques : ordre inversé (rlgym_compat)
    return m, blue, orange


@pytest.mark.parametrize("seed", range(5))
def test_obs_orange_is_mirror_of_blue(seed):
    s = _random_state(seed)
    m, blue, orange = _mirrored(s)
    prev = np.array([1, -1, 1, -1, 0, 1, 1, 0], dtype=np.float64)
    ob, oo = NectoObs(PADS), NectoObs(PADS)
    for _ in range(3):                                 # minuteurs (boost, démo) inclus
        qb, kvb, _ = ob.build(blue, s, prev)
        qo, kvo, _ = oo.build(orange, m, prev)
        np.testing.assert_allclose(qb, qo, atol=1e-4)
        np.testing.assert_allclose(kvb[0, :3], kvo[0, :3], atol=1e-4)    # balle, soi, adversaire
        sort = lambda a: a[np.lexsort(a.T[::-1])]
        np.testing.assert_allclose(sort(np.round(kvb[0, 3:], 4)), sort(np.round(kvo[0, 3:], 4)), atol=1e-3)


def test_obs_layout():
    s = _random_state(0)
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    q, kv, mask = NectoObs(PADS).build(blue, s, np.zeros(8))
    assert q.shape == (1, 1, 32) and kv.shape == (1, 37, 24) and mask.shape == (1, 37) and not mask.any()
    assert kv[0, 0, 3] == 1 and kv[0, 1, 0] == 1 and kv[0, 1, 1] == 1 and kv[0, 2, 2] == 1 and kv[0, 3:, 4].all()
    np.testing.assert_allclose(kv[0, 1, 5:11], 0)                      # coordonnées relatives à soi
    np.testing.assert_allclose(kv[0, 3:, 20], 0.12 + 0.88 * (PADS[:, 2] > 72))


def test_parse_logits_like_agent_py():
    # têtes : throttle (3), steer (3), jump, boost, handbrake (2)
    out = [torch.tensor([[[0., 0, 5]]]), torch.tensor([[[5., 0, 0]]]), torch.tensor([[[0., 1]]]),
           torch.tensor([[[1., 0]]]), torch.tensor([[[0., 1]]])]
    np.testing.assert_array_equal(parse_logits(out), [1, -1, 1, 0, -1, 1, 0, 1])   # handbrake : steer -> roll
    out[4] = torch.tensor([[[1., 0]]])
    np.testing.assert_array_equal(parse_logits(out), [1, -1, 1, -1, 0, 1, 0, 0])


def test_kickoff_table():
    assert KICKOFF_TICKS.shape == (168, 8)
    np.testing.assert_allclose(KICKOFF_TICKS[72], [1, 0, -0.7, 0.8, 0, 1, 1, 0])   # 44+16+8+4 : début du flip
    assert not KICKOFF_TICKS[:, 6][-40:].any()


def test_mixed_parser_same_as_training_parser_for_indices():
    s = make_state(1)
    agents = list(s.cars)
    ref, mix = make_action_parser(8), MixedRepeatAction(8)
    si_ref, si_mix = {}, {}
    ref.reset(agents, s, si_ref)
    mix.reset(agents, s, si_mix)
    rng = np.random.default_rng(0)
    for _ in range(20):
        acts = {a: np.array([rng.integers(0, 90)]) for a in agents}
        r, m = ref.parse_actions(acts, s, si_ref), mix.parse_actions(acts, s, si_mix)
        for a in agents:
            np.testing.assert_array_equal(r[a], m[a])
            np.testing.assert_array_equal(si_ref["prev_actions"][a], si_mix["prev_actions"][a])
    rows = np.tile(np.arange(8, dtype=np.float32), (8, 1))
    out = mix.parse_actions({agents[0]: rows, agents[1]: np.array([3])}, s, si_mix)
    np.testing.assert_array_equal(out[agents[0]], rows)
    assert out[agents[1]].shape == (8, 8)
