"""Exercices dédiés, mesures de réussite, dosage automatique, rewards associés, prédiction de trajectoire (s6)."""
import numpy as np
import pytest
from rlgym.rocket_league import common_values as cv

from helpers import make_state, mirror_car_into
from rlagent.env.mutators import DRILLS, StartStateMutator
from rlagent.env.obs import (BALL_PRED_DIM, OBS_DIM, SELF_DIM, AttentionObs, ball_prediction_features, obs_dim)
from rlagent.metrics.episode import EpisodeTracker


# --------------------------------------------------------------------------- prédiction de trajectoire
def test_ball_prediction_free_fall_and_rolling():
    f = ball_prediction_features(np.array([0, 0, 1000.0]), np.zeros(3))
    t_land = np.sqrt(2 * (1000 - cv.BALL_RADIUS) / 650.0)
    assert f.shape == (BALL_PRED_DIM,)
    assert np.isclose(f[0] * 3.0, t_land, atol=1e-3)
    assert np.isclose(f[3] * cv.CEILING_Z, 1000.0, atol=1e-2)          # pas de montée : hauteur max = départ
    z_half = 1000 - 0.5 * 650 * 0.25
    assert np.isclose(f[7] * cv.CEILING_Z, z_half, atol=1e-2)          # z à 0,5 s
    roll = ball_prediction_features(np.array([100, 200, cv.BALL_RADIUS]), np.array([1000.0, 0, 0]))
    assert roll[0] == 0.0 and np.isclose(roll[5] * cv.SIDE_WALL_X, 600.0, atol=1e-2)   # x à 0,5 s
    wall = ball_prediction_features(np.array([3900, 0, cv.BALL_RADIUS]), np.array([2000.0, 0, 0]))
    assert abs(wall[5] * cv.SIDE_WALL_X) <= cv.SIDE_WALL_X - cv.BALL_RADIUS + 1     # rebond sur le mur


def test_obs_with_prediction_dims_and_base_unchanged():
    s = make_state(2, kickoff=True, seed=3)
    s.ball.position = np.array([500, -800, 900], dtype=np.float32)
    s.ball.linear_velocity = np.array([300, 600, 400], dtype=np.float32)
    base = AttentionObs().build_obs(list(s.cars), s, {})
    bp = AttentionObs(ball_prediction=True).build_obs(list(s.cars), s, {})
    assert obs_dim(True) == OBS_DIM + BALL_PRED_DIM
    for a in s.cars:
        assert bp[a].shape == (obs_dim(True),)
        stripped = np.concatenate([bp[a][:SELF_DIM], bp[a][SELF_DIM + BALL_PRED_DIM:]])
        assert np.array_equal(stripped, base[a])


def test_prediction_mirror_symmetric():
    s = make_state(1, kickoff=True, seed=1)
    s.ball.position = np.array([700, -1200, 800], dtype=np.float32)
    s.ball.linear_velocity = np.array([400, 900, 300], dtype=np.float32)
    s.cars["blue-0"].physics.position = np.array([-500, -2500, 17], dtype=np.float32)
    m = make_state(1, kickoff=True, seed=1)
    m.ball.position = s.ball.position * np.array([-1, -1, 1], dtype=np.float32)
    m.ball.linear_velocity = s.ball.linear_velocity * np.array([-1, -1, 1], dtype=np.float32)
    mirror_car_into(s, "blue-0", m, "orange-0")
    mirror_car_into(s, "orange-0", m, "blue-0")
    b = AttentionObs(ball_prediction=True)
    assert np.allclose(b.build_obs(["blue-0"], s, {})["blue-0"], b.build_obs(["orange-0"], m, {})["orange-0"],
                       atol=1e-5)


torch = pytest.importorskip("torch")


def _cfg(bp: bool):
    from rlagent.config import load_config
    cfg = load_config(["debug"], [f"env.ball_prediction={'true' if bp else 'false'}"])
    return cfg


def test_old_actor_plays_with_prediction_obs_and_warm_start_expansion_is_exact():
    """Un réseau sans prédiction joue sur des obs avec (colonnes ignorées), et un réseau élargi depuis ses poids
    (nouvelles entrées à zéro) calcule exactement la même chose au départ."""
    from rlagent.models.attention import build_models
    from rlagent.train import load_state_expanding
    s = make_state(3, kickoff=True, seed=5)
    s.ball.position = np.array([200, 300, 600], dtype=np.float32)
    s.ball.linear_velocity = np.array([500, -400, 300], dtype=np.float32)
    base = np.stack(list(AttentionObs().build_obs(list(s.cars), s, {}).values()))
    bp = np.stack(list(AttentionObs(True).build_obs(list(s.cars), s, {}).values()))
    torch.manual_seed(0)
    old_actor, old_critic = build_models(_cfg(False), 90)
    new_actor, new_critic = build_models(_cfg(True), 90)
    with torch.no_grad():
        ref = old_actor(torch.as_tensor(base))
        assert torch.allclose(old_actor(torch.as_tensor(bp)), ref)            # fit_obs
        expanded = load_state_expanding(new_actor, old_actor.state_dict())
        load_state_expanding(new_critic, old_critic.state_dict())
        assert expanded == ["encoder.self_embed.0.weight"]
        assert torch.allclose(new_actor(torch.as_tensor(bp)), ref, atol=1e-6)
        assert torch.allclose(new_critic(torch.as_tensor(bp)), old_critic(torch.as_tensor(base)), atol=1e-6)


# --------------------------------------------------------------------------- exercices
@pytest.mark.parametrize("kind", list(DRILLS))
def test_drills_valid_and_both_teams(kind):
    teams = set()
    for seed in range(40):
        size = 1 + seed % 3
        s = make_state(size, kickoff=True, seed=seed)
        info = {}
        DRILLS[kind](seed).apply(s, info)
        teams.add(info["drill_team"])
        for obj in [s.ball, *[c.physics for c in s.cars.values()]]:
            p = obj.position
            assert np.all(np.isfinite(p)) and abs(p[0]) <= cv.SIDE_WALL_X and abs(p[1]) <= cv.BACK_WALL_Y
            assert 0 < p[2] <= cv.CEILING_Z
        assert np.isfinite(s.ball.linear_velocity).all()
    assert teams == {0, 1}


def test_adaptive_weights_favor_failed_drills_and_keep_total():
    w = {"kickoff": 0.4, "random": 0.2, "drill_shot": 0.2, "drill_defense": 0.2}
    m = StartStateMutator(w, seed=0, adaptive=True, ema=0.5)
    for _ in range(20):
        m.report("drill_shot", 1.0)      # maîtrisé
        m.report("drill_defense", 0.0)   # raté
    cw = m.current_weights()
    assert cw["drill_defense"] > cw["drill_shot"] * 3
    assert np.isclose(cw["drill_defense"] + cw["drill_shot"], 0.4)
    assert cw["kickoff"] == 0.4 and cw["random"] == 0.2
    assert StartStateMutator(w, seed=0).current_weights() == w   # sans dosage : poids de la config


def _tracker_for(kind, team=0, size=1):
    s = make_state(size, kickoff=True, seed=0)
    s.tick_count = 0
    t = EpisodeTracker()
    t.reset(s, {"start_kind": "drill_" + kind, "drill_team": team, "team_size": size})
    return t, s


def test_drill_success_rules():
    t, s = _tracker_for("shot")
    s.tick_count = 240
    s.goal_scored = True
    s.ball.position = np.array([0, 5300, 300], dtype=np.float32)    # dans le but orange : le bleu marque
    t.step(s)
    assert t.finish(s, True)["drill_success"] == 1.0
    t, s = _tracker_for("defense", team=0)
    s.tick_count = 120
    s.goal_scored = True
    s.ball.position = np.array([0, -5300, 300], dtype=np.float32)   # dans le but bleu : l'orange marque
    t.step(s)
    assert t.finish(s, True)["drill_success"] == 0.0
    t, s = _tracker_for("defense", team=0)
    s.tick_count = int(6.5 * 120)
    t.step(s)
    assert t.finish(s, False)["drill_success"] == 1.0
    t, s = _tracker_for("low_boost", team=1)
    s.cars["orange-0"].boost_amount = 70.0
    s.tick_count = 300
    t.step(s)
    assert t.finish(s, False)["drill_success"] == 1.0
    t, s = _tracker_for("aerial", team=0)
    s.tick_count = 60
    t.step(s)
    out = t.finish(s, False)     # fin d'épisode avant toute touche aérienne
    assert out["drill"] == "aerial" and out["drill_success"] == 0.0
    assert {"boost_pickups_per_min", "avg_boost", "jumps_per_min", "air_frac"} <= set(out["mech"])


# --------------------------------------------------------------------------- rewards associés
def test_save_boost_waste_goalside():
    from rlagent.env.rewards import BoostWasteTerm, GoalsideTerm, SaveTerm
    s = make_state(1, kickoff=True, seed=0)
    agents = list(s.cars)
    t = SaveTerm()
    s.ball.position = np.array([0, -3000, 93], dtype=np.float32)
    s.ball.linear_velocity = np.array([0, -2500, 0], dtype=np.float32)    # tir vers le but bleu
    t.reset(agents, s, {})
    s.cars["blue-0"].ball_touches = 1
    s.ball.linear_velocity = np.array([800, 1000, 0], dtype=np.float32)   # dégagé
    r = t.get(agents, s, {})
    assert np.isclose(r["blue-0"], 2500 / 6000, atol=0.02) and r["orange-0"] == 0.0
    c = s.cars["blue-0"]
    c.boost_active_time, c.supersonic_time, c.wheels_with_contact = 0.1, 0.1, (True, True, True, True)
    assert BoostWasteTerm().get(["blue-0"], s, {})["blue-0"] == -1.0
    c.wheels_with_contact = (False, False, False, False)
    assert BoostWasteTerm().get(["blue-0"], s, {})["blue-0"] == 0.0
    s.ball.position = np.array([0, -1000, 93], dtype=np.float32)          # balle dans le camp bleu
    s.cars["blue-0"].physics.position = np.array([0, -3000, 17], dtype=np.float32)
    s.cars["orange-0"].physics.position = np.array([0, 2000, 17], dtype=np.float32)
    g = GoalsideTerm().get(agents, s, {})
    assert g["blue-0"] == 1.0 and g["orange-0"] == 0.0                   # balle dans le camp adverse pour l'orange
