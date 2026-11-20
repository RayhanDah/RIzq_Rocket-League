"""Adaptateur Nexto (audits) : table d'actions, symétrie bleu/orange, repère relatif, kickoffs stochastiques."""
import numpy as np
import pytest
import torch
from rlgym.rocket_league.action_parsers import LookupTableAction
from test_necto_adapter import PADS, _mirrored, _random_state

from rlagent.opponents.nexto import (LOOKUP_TABLE, NEXTO_DIR, NextoObs, NextoOpponent, bot_from_spec, is_bot_spec,
                                     parse_logits)


def test_lookup_table_is_rlgym_table():
    np.testing.assert_array_equal(LOOKUP_TABLE, LookupTableAction.make_lookup_table())


def test_parse_logits_argmax_and_temperature():
    logits = torch.zeros(1, 90)
    logits[0, 37] = 5.0
    np.testing.assert_array_equal(parse_logits(logits), LOOKUP_TABLE[37])
    np.testing.assert_array_equal(parse_logits(-logits, beta=-1), LOOKUP_TABLE[37])
    rng = np.random.default_rng(0)
    picks = {tuple(parse_logits(logits, beta=0.0, rng=rng)) for _ in range(200)}   # beta 0 : uniforme
    assert len(picks) > 50


@pytest.mark.parametrize("seed", range(5))
def test_obs_orange_is_mirror_of_blue(seed):
    s = _random_state(seed)
    m, blue, orange = _mirrored(s)
    prev = np.array([1, -1, 1, -1, 0, 1, 1, 0], dtype=np.float64)
    qb, kvb, _ = NextoObs(PADS).build(blue, s, prev)
    qo, kvo, _ = NextoObs(PADS).build(orange, m, prev)
    np.testing.assert_allclose(qb, qo, atol=1e-4)
    np.testing.assert_allclose(kvb[0, :3], kvo[0, :3], atol=1e-4)        # soi, adversaire, balle
    sort = lambda a: a[np.lexsort(a.T[::-1])]
    np.testing.assert_allclose(sort(np.round(kvb[0, 3:], 4)), sort(np.round(kvo[0, 3:], 4)), atol=1e-3)


def test_obs_layout_and_heading_frame():
    s = _random_state(1)
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    q, kv, mask = NextoObs(PADS).build(blue, s, np.zeros(8))
    assert q.shape == (1, 1, 32) and kv.shape == (1, 37, 24) and not mask.any()
    assert kv[0, 0, 0] == 1 and kv[0, 0, 1] == 1 and kv[0, 1, 2] == 1 and kv[0, 2, 3] == 1 and kv[0, 3:, 4].all()
    np.testing.assert_allclose(kv[0, 0, 5:8], 0)                         # position relative à soi
    fw = kv[0, 0, 11:14]                                                 # cap de la voiture tourné sur +y
    assert abs(fw[0]) < 1e-9 and fw[1] >= 0
    rel = s.ball.position - s.cars[blue].physics.position                # distances conservées par la rotation
    np.testing.assert_allclose(np.linalg.norm(kv[0, 2, 5:8]), np.linalg.norm(rel) / 2300, rtol=1e-6)


@pytest.mark.skipif(not (NEXTO_DIR / "nexto-model.pt").exists(), reason="clone NectoFamily absent")
def test_kickoff_phase_beta():
    bot = bot_from_spec("nexto", seed=0)
    assert isinstance(bot, NextoOpponent) and is_bot_spec("nexto:0.5") and not is_bot_spec("nextoo")
    s = _random_state(2)
    s.ball.position = np.array([0, 0, 93.15], dtype=np.float32)
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    bot.new_game(PADS)
    bot.on_reset(s)
    assert bot.step_beta(s) == 0.5
    s.ball.linear_velocity = np.array([0, 10, 0], dtype=np.float32)
    assert bot.step_beta(s) == 1.0
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)              # la phase ne revient qu'au prochain reset
    assert bot.step_beta(s) == 1.0
    assert NextoOpponent(stochastic_kickoffs=False).step_beta(s) == 1.0
