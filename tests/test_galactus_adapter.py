"""Adaptateur GalactusV2 (audits) : masques d'actions, taille et symétrie de l'observation, retard d'action."""
import numpy as np
import pytest
from test_necto_adapter import PADS, _mirrored, _random_state

from rlagent.opponents.galactus import (AIR_MASK, GALACTUS_DIR, GROUND_MASK, JUMP_MASK, OBS_SIZE, PRED_TICKS,
                                        GalactusObs, GalactusOpponent, action_mask, choose)
from rlagent.opponents.nexto import LOOKUP_TABLE, bot_from_spec, is_bot_spec

HAS_WEIGHTS = (GALACTUS_DIR / "POLICY.lt").exists() and (GALACTUS_DIR / "SHARED_HEAD.lt").exists()


class LinearPredictor:
    """Remplace l'arène RocketSim : trajectoire rectiligne (suffit pour la symétrie)."""

    def samples(self, state):
        b = state.ball
        return np.array([np.concatenate([b.position + b.linear_velocity * t / 120, b.linear_velocity])
                         for t in PRED_TICKS], dtype=np.float64)


def test_masks_match_default_action():
    # DefaultAction.cpp : 24 au sol, 47 aériens sans saut (le 1er, indice 24, oublié par « i > ») + 6 au sol
    assert GROUND_MASK.sum() == 24 and AIR_MASK.sum() == 53 and JUMP_MASK.sum() == 18
    assert not AIR_MASK[24] and not AIR_MASK[JUMP_MASK].any()


def test_action_mask_rules():
    s = _random_state(0)
    car = next(iter(s.cars.values()))
    car.on_ground, car.boost_amount = True, 0.0
    m = action_mask(car)
    assert m[JUMP_MASK].all() and not m[(LOOKUP_TABLE[:, 6] == 1) & ~JUMP_MASK].any()   # saut rétabli après le boost
    car.on_ground, car.boost_amount, car.has_flipped, car.has_double_jumped = False, 50.0, True, True
    m = action_mask(car)
    assert not m[JUMP_MASK].any() and m[AIR_MASK].all()


def test_choose_respects_mask():
    logits = np.zeros(90)
    logits[5] = 10.0
    mask = np.ones(90, dtype=bool)
    mask[5] = False
    assert choose(logits, mask, 1.0, np.random.default_rng(0)) != 5
    picks = {choose(logits, mask, 0.0, np.random.default_rng(i)) for i in range(100)}
    assert 5 not in picks and len(picks) > 30


@pytest.mark.parametrize("seed", range(4))
def test_obs_orange_is_mirror_of_blue(seed):
    s = _random_state(seed)
    m, blue, orange = _mirrored(s)
    prev = np.array([1, -1, 1, -1, 0, 1, 1, 0], dtype=np.float64)
    ob = GalactusObs(PADS, np.random.default_rng(1), LinearPredictor()).build(blue, s, prev)
    oo = GalactusObs(PADS, np.random.default_rng(1), LinearPredictor()).build(orange, m, prev)
    assert ob.shape == oo.shape == (OBS_SIZE,)
    np.testing.assert_allclose(ob, oo, atol=1e-5)


def test_obs_layout():
    s = _random_state(3)
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    o = GalactusObs(PADS, np.random.default_rng(0), LinearPredictor()).build(blue, s, np.zeros(8))
    np.testing.assert_allclose(o[:3], s.ball.position / 5000, rtol=1e-6)
    pads = o[17:51]
    assert ((pads > 0) & (pads <= 1)).all()
    self_start = 54
    np.testing.assert_allclose(o[self_start:self_start + 3], s.cars[blue].physics.position / 5000, rtol=1e-6)
    players = o[self_start:self_start + 29 * 6].reshape(6, 29)
    assert (np.abs(players).sum(1) > 0).sum() == 2                     # 1v1 : soi + 1 adversaire, 4 emplacements vides


def test_spec_dispatch():
    assert is_bot_spec("galactus") and is_bot_spec("galactus:0.5") and not is_bot_spec("galactu")


@pytest.mark.skipif(not HAS_WEIGHTS, reason="clone GalactusV2 absent")
def test_action_delay_rows():
    bot = bot_from_spec("galactus", seed=0)
    assert isinstance(bot, GalactusOpponent)
    bot.predictor = LinearPredictor()
    bot.new_game(PADS)
    s = _random_state(4)
    agent = next(iter(s.cars))
    r1 = bot.act([agent], s)[agent]
    assert r1.shape == (8, 8) and (r1[:7] == 0).all()                  # 7 ticks de l'action précédente (rien)
    assert any((r1[7] == row).all() for row in LOOKUP_TABLE)
    r2 = bot.act([agent], s)[agent]
    np.testing.assert_array_equal(r2[:7], np.repeat(r1[7:], 7, axis=0))


# --- Kira et Rain (AdvancedObs / AdvancedObsPadded) ---------------------------------------------------------------
from rlagent.opponents.gigalearn import BOTS, AdvancedObs, GigaLearnOpponent, model_dir  # noqa: E402

HAS_KIRA = (model_dir(BOTS["kira"], 1) / "POLICY.lt").exists()
HAS_RAIN = all((model_dir(BOTS["rain"], n) / "POLICY.lt").exists() for n in (1, 2, 3))


@pytest.mark.parametrize("padded,size", [(False, 109), (True, 231)])
def test_advanced_obs_size(padded, size):
    s = _random_state(5)
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    o = AdvancedObs(PADS, padded).build(blue, s, np.zeros(8))
    assert o.shape == (size,)


@pytest.mark.parametrize("seed", range(3))
def test_advanced_obs_kira_orange_is_mirror_of_blue(seed):
    s = _random_state(seed)
    m, blue, orange = _mirrored(s)
    prev = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
    np.testing.assert_allclose(AdvancedObs(PADS, False).build(blue, s, prev),
                               AdvancedObs(PADS, False).build(orange, m, prev), atol=1e-5)


def test_advanced_obs_rain_goal_vectors_keep_bot_defect():
    """Rain : en orange, « son but » est pris côté orange dans le repère retourné (défaut du bot, gardé)."""
    s = _random_state(6)
    m, blue, orange = _mirrored(s)
    ob = AdvancedObs(PADS, True).build(blue, s, np.zeros(8))
    oo = AdvancedObs(PADS, True).build(orange, m, np.zeros(8))
    np.testing.assert_allclose(ob[:9], oo[:9], atol=1e-5)
    np.testing.assert_allclose(ob[9:12], oo[12:15], atol=1e-5)          # buts échangés
    np.testing.assert_allclose(ob[15:], oo[15:], atol=1e-5)


@pytest.mark.skipif(not (HAS_KIRA and HAS_RAIN), reason="clones Kira / RainStream absents")
def test_kira_rain_rows_and_specs():
    assert is_bot_spec("kira") and is_bot_spec("rain:0.5") and not is_bot_spec("rainy")
    s = _random_state(7)
    for spec, delay in (("kira", 7), ("rain", 5)):
        bot = bot_from_spec(spec, seed=0)
        assert isinstance(bot, GigaLearnOpponent) and bot.name == spec
        bot.new_game(PADS)
        rows = bot.act(list(s.cars), s)
        for r in rows.values():
            assert r.shape == (8, 8) and (r[:delay] == 0).all()
            assert any((r[-1] == row).all() for row in LOOKUP_TABLE)
