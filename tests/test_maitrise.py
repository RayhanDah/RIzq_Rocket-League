"""Palier 6 « maitrise » (06/11/2026) : dépense de boost facturée au sol, sauts moins payés, flip reset, exercice de vol en
curriculum inversé, adversaires fixes tirés du pool du run, horizon de 10 s."""
import math

import numpy as np
import pytest
import torch

from helpers import make_state
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.common import is_flip_reset
from rlagent.env.mutators import DRILLS, AirDrill, ResetDrill
from rlagent.env.rewards import BoostGainTerm, BoostSpendTerm, FlipResetTerm, make_reward
from rlagent.metrics.episode import EpisodeTracker
from rlagent.selfplay.pool import OpponentPool
from rlagent.train import DrillCurriculum, summarize_episodes


def car_of(state, team):
    return next((a, c) for a, c in state.cars.items() if c.team_num == team)


def on_ball(state, roll=0.0, z=900.0):
    """Voiture posée sur une balle en l'air : roues contre la balle (roll = 0) ou toit contre la balle (roll = π)."""
    agent, car = car_of(state, 0)
    state.ball.position = np.array([0.0, 0.0, z - 95.0], dtype=np.float32)
    car.physics.position = np.array([0.0, 0.0, z], dtype=np.float32)
    car.physics.euler_angles = np.array([0.0, 0.0, roll], dtype=np.float32)
    car.on_ground = True
    car.has_jumped = False
    car.ball_touches = 1
    return agent, car


def fly(car):
    car.on_ground = False
    car.has_jumped = True
    car.ball_touches = 0


# --------------------------------------------------------------------------- profil
def test_profile_ends_with_maitrise_and_can_outlive_ten_billion_steps():
    cfg = load_config(["kaggle_tpu", "s10"])
    assert cfg["train"]["total_steps"] > 20_000_000_000      # default.yaml : 10 G, atteint en pleine session 4
    assert cfg["ppo"]["gamma"] == 0.995
    w = cfg["env"]["start_weights"]
    assert sum(w.values()) == pytest.approx(1.0) and w["kickoff"] < 0.4 and w["drill_air"] > w["drill_aerial"]
    stages = cfg["curriculum"]["stages"]
    assert stages[-1]["name"] == "maitrise" and stages[-2]["advance"] and "advance" not in stages[-1]


def test_maitrise_removes_jump_aids_and_keeps_what_works():
    cfg = load_config(["kaggle_tpu", "s10"])
    c = SkillCurriculum.from_cfg(cfg)
    prev, last = c.stage_weights[-2], c.stage_weights[-1]
    assert last["in_air"] < prev["in_air"] and last["flip_speed"] < prev["flip_speed"]
    assert last["boost_spend"] > 0.0 and last["flip_reset"] > 0.0 and last["kickoff_rush"] > 0.0
    for key in ("goal", "shot", "shot_threat", "vel_ball_to_goal", "vel_player_to_ball", "aerial_touch", "strong_touch"):
        assert last[key] == prev[key]
    assert {"boost_spend", "flip_reset", "kickoff_rush"} <= {t.name for t in make_reward(cfg).terms}


def test_run_resumed_at_finition_moves_on_to_maitrise():
    cfg = load_config(["kaggle_tpu", "s10"])
    c = SkillCurriculum.from_cfg(cfg)
    c.load_state_dict({"index": 4, "stage_start": 5_386_323_521, "prev_weights": {"goal": 10.0}, "ema": {}, "streak": 0})
    assert c.name == "finition"
    msg = None
    for i in range(30):
        msg = msg or c.update({"ladder/gap/s6_spo_kaggle": 528.0}, 8_140_223_478 + i * 400_000)
    assert c.name == "maitrise" and "maitrise" in msg and c.ppo_overrides()["ent_coef"] == 0.005   # 0,003 en session 4


# --------------------------------------------------------------------------- boost
def spend(term, s, agent, car, before, after):
    car.boost_amount = before
    term.reset(list(s.cars), s, {})
    car.boost_amount = after
    return term.get(list(s.cars), s, {})[agent]


def test_boost_spend_costs_on_the_ground_and_is_free_in_the_air():
    s = make_state()
    agent, car = car_of(s, 0)
    term = BoostSpendTerm()
    car.physics.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    on_ground = spend(term, s, agent, car, 50.0, 30.0)
    assert on_ground == pytest.approx((math.sqrt(0.3) - math.sqrt(0.5)) * (1 - 17.0 / 642.775), rel=1e-5)
    car.physics.position = np.array([0.0, 0.0, 321.0], dtype=np.float32)
    assert spend(term, s, agent, car, 50.0, 30.0) == pytest.approx(on_ground / 2, rel=0.05)
    car.physics.position = np.array([0.0, 0.0, 700.0], dtype=np.float32)
    assert spend(term, s, agent, car, 50.0, 30.0) == 0.0


def test_boost_spend_ignores_pickups_and_demolitions():
    s = make_state()
    agent, car = car_of(s, 0)
    term = BoostSpendTerm()
    car.physics.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    assert spend(term, s, agent, car, 30.0, 50.0) == 0.0
    car.demo_respawn_timer = 2.0
    assert spend(term, s, agent, car, 50.0, 0.0) == 0.0


def test_burning_boost_to_pick_it_up_again_pays_less():
    """Ce qui rendait le réservoir vide rentable : sans dépense facturée, un aller-retour à sec rapporte tout le gain."""
    s = make_state()
    agent, car = car_of(s, 0)
    car.physics.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    gain, cost = BoostGainTerm(), BoostSpendTerm()
    w = load_config(["kaggle_tpu", "s10"])
    last = SkillCurriculum.from_cfg(w).stage_weights[-1]
    car.boost_amount = 30.0
    gain.reset(list(s.cars), s, {})
    cost.reset(list(s.cars), s, {})
    total_before = total_after = 0.0
    for b in (0.0, 30.0):
        car.boost_amount = b
        g, c = gain.get(list(s.cars), s, {})[agent], cost.get(list(s.cars), s, {})[agent]
        total_before += last["boost_gain"] * g
        total_after += last["boost_gain"] * g + last["boost_spend"] * c
    assert total_after < 0.6 * total_before


# --------------------------------------------------------------------------- flip reset
def test_flip_reset_needs_wheels_on_a_ball_in_the_air():
    s = make_state()
    _, car = on_ball(s)
    ball = s.ball.position
    assert is_flip_reset(car, ball, prev_on_ground=False, prev_jumped=True)
    _, car = on_ball(s, roll=math.pi)
    assert not is_flip_reset(car, s.ball.position, False, True)               # toit contre la balle
    _, car = on_ball(s, z=200.0)
    assert not is_flip_reset(car, s.ball.position, False, True)               # près du sol : conduite de balle
    _, car = on_ball(s)
    car.ball_touches = 0
    assert not is_flip_reset(car, s.ball.position, False, True)               # sur un mur, sans balle
    car.ball_touches = 1
    assert not is_flip_reset(car, s.ball.position, True, False)               # déjà posé sur la balle au step d'avant


def test_flip_reset_seen_even_if_the_car_has_left_the_ball_at_the_end_of_the_step():
    """8 ticks par step : le contact peut être fini ; `has_jumped` retombé à faux en garde la trace."""
    s = make_state()
    _, car = on_ball(s)
    car.on_ground = False
    assert is_flip_reset(car, s.ball.position, False, True)
    car.has_jumped = True
    assert not is_flip_reset(car, s.ball.position, False, True)


def test_flip_reset_paid_once_then_again_after_the_cooldown():
    s = make_state()
    agent, car = car_of(s, 0)
    agents = list(s.cars)
    term = FlipResetTerm(cooldown_s=1.0)
    fly(car)
    term.reset(agents, s, {})
    on_ball(s)
    s.tick_count = 8
    assert term.get(agents, s, {})[agent] == 1.0
    fly(car)
    s.tick_count = 16
    assert term.get(agents, s, {})[agent] == 0.0
    on_ball(s)
    s.tick_count = 60                                                         # 0,5 s après : la balle ne se farme pas
    assert term.get(agents, s, {})[agent] == 0.0
    fly(car)
    s.tick_count = 200
    term.get(agents, s, {})
    on_ball(s)
    s.tick_count = 208
    assert term.get(agents, s, {})[agent] == 1.0


def test_flip_resets_are_counted_in_the_metrics():
    s = make_state()
    _, car = car_of(s, 0)
    fly(car)
    tracker = EpisodeTracker()
    tracker.reset(s, {"team_size": 1, "start_kind": "kickoff"})
    on_ball(s)
    s.tick_count += 8
    tracker.step(s)
    assert tracker.flip_resets == 1
    ep = tracker.finish(s, False)
    assert ep["flip_resets_per_min"] > 0.0
    assert summarize_episodes([{**ep, "kickoff_winner": None}])["flip_resets_per_min"] == ep["flip_resets_per_min"]


# --------------------------------------------------------------------------- exercice de vol
def flights(level, n=60):
    drill = AirDrill(seed=3)
    out = []
    for _ in range(n):
        s = make_state()
        info = {"drill_levels": {"air": level}}
        drill.apply(s, info)
        assert info["drill_level"] == level
        out.append((s, *car_of(s, info["drill_team"]), info["drill_team"]))
    return out


def gap(s, car):
    return float(np.linalg.norm(s.ball.position - car.physics.position))


def test_air_drill_level_zero_is_the_end_of_the_move():
    for s, _, car, _ in flights(0.0):
        to_ball = (s.ball.position - car.physics.position) / gap(s, car)
        assert not car.on_ground and car.has_jumped and not car.has_flipped
        assert 100.0 < gap(s, car) < 800.0 and car.boost_amount >= 60.0
        assert float(np.dot(car.physics.forward, to_ball)) > 0.7
        assert float(np.dot(car.physics.linear_velocity, to_ball)) > 600.0


def test_air_drill_backs_up_towards_the_takeoff_with_the_level():
    easy, hard = flights(0.0), flights(1.0)
    assert np.mean([gap(s, c) for s, _, c, _ in hard]) > 2 * np.mean([gap(s, c) for s, _, c, _ in easy])
    assert np.mean([c.physics.position[2] for _, _, c, _ in hard]) < 310.0
    assert max(abs(c.physics.position[0]) for _, _, c, _ in hard) <= 3900.0
    assert max(abs(c.physics.position[1]) for _, _, c, _ in hard) <= 4900.0


def test_air_drill_ball_never_hangs_and_the_car_comes_from_our_side():
    """Une balle posée sans vitesse reste suspendue en l'air (RocketSim, test du 06/11)."""
    behind = 0
    cases = flights(0.5)
    for s, _, car, team in cases:
        assert float(np.linalg.norm(s.ball.linear_velocity)) > 1.0
        attack = 1.0 if team == 0 else -1.0
        behind += float(s.ball.position[1] - car.physics.position[1]) * attack > 0.0
    assert behind > 0.7 * len(cases)


def test_air_drill_succeeds_on_a_touch_in_the_air():
    s = make_state()
    _, car = car_of(s, 0)
    tracker = EpisodeTracker()
    tracker.reset(s, {"team_size": 1, "start_kind": "drill_air", "drill_team": 0, "drill_level": 0.0})
    s.ball.position = np.array([0.0, 0.0, 700.0], dtype=np.float32)
    car.ball_touches = 1
    s.tick_count += 8
    tracker.step(s)
    assert tracker.finish(s, False)["drill_success"] == 1.0


def test_new_drills_do_not_shift_the_seeds_of_the_older_ones():
    assert list(DRILLS)[-2:] == ["air", "reset"] and list(DRILLS)[0] == "defense"


def resets(level, n=60):
    drill = ResetDrill(seed=5)
    out = []
    for _ in range(n):
        s = make_state()
        info = {"drill_levels": {"reset": level}}
        drill.apply(s, info)
        out.append((s, car_of(s, info["drill_team"])[1]))
    return out


def test_reset_drill_puts_the_wheels_in_front_of_the_ball_with_the_flip_spent():
    for s, car in resets(0.0):
        to_ball = (s.ball.position - car.physics.position) / gap(s, car)
        assert float(np.dot(car.physics.up, to_ball)) < -0.9                # les roues regardent la balle
        assert abs(float(np.dot(car.physics.forward, car.physics.up))) < 1e-3
        assert np.linalg.det(car.physics.rotation_mtx) == pytest.approx(1.0, abs=1e-3)
        assert 120.0 < gap(s, car) < 270.0 and car.physics.position[2] > 300.0
        assert car.has_jumped and car.has_flipped and not car.has_flip and not car.is_flipping
        assert float(np.dot(car.physics.linear_velocity - s.ball.linear_velocity, to_ball)) > 150.0


def test_reset_drill_gets_harder_with_the_level():
    easy, hard = resets(0.0), resets(1.0)
    assert np.mean([gap(s, c) for s, c in hard]) > 2 * np.mean([gap(s, c) for s, c in easy])


def test_reset_drill_succeeds_on_a_flip_reset_only():
    s = make_state()
    _, car = car_of(s, 0)
    fly(car)
    tracker = EpisodeTracker()
    tracker.reset(s, {"team_size": 1, "start_kind": "drill_reset", "drill_team": 0, "drill_level": 0.0})
    s.ball.position = np.array([0.0, 0.0, 900.0], dtype=np.float32)
    car.ball_touches = 1                                                     # touche sans reset : pas de réussite
    s.tick_count += 8
    tracker.step(s)
    assert tracker.drill_result is None
    fly(car)
    tracker.step(s)
    on_ball(s)
    s.tick_count += 8
    tracker.step(s)
    assert tracker.drill_result == 1.0


def test_each_drill_can_set_its_own_start_and_floor():
    cfg = load_config(["kaggle_tpu", "s10"])
    cfg["env"]["start_weights"]["drill_aerial"] = 0.05     # retiré du profil le 08/11 : on vérifie le mécanisme
    dc = DrillCurriculum.from_cfg(cfg)
    assert dc.levels["air"] == dc.levels["reset"] == 0.0
    assert dc.levels["aerial"] == 0.4 and dc.levels["aerial_defense"] == 0.2
    for _ in range(200):
        dc.update([{"drill": k, "drill_level": 0.0, "drill_success": 0.0} for k in ("air", "aerial")])
    assert dc.levels["air"] == 0.0 and dc.levels["aerial"] == 0.4


def test_new_drill_keeps_its_start_when_resuming_an_older_checkpoint():
    dc = DrillCurriculum.from_cfg(load_config(["kaggle_tpu", "s10"]))
    dc.load_state_dict({"levels": {"aerial": 0.4, "shot": 1.0}, "success": {"aerial": 0.04, "shot": 0.56}})
    assert dc.levels["shot"] == 1.0 and dc.levels["air"] == 0.0


# --------------------------------------------------------------------------- adversaires
def test_anchor_from_this_runs_pool_needs_no_path_to_the_code(tmp_path):
    """Sur Kaggle, runs/ n'est pas à côté du code : une ancre relative au code ne s'y résoudrait pas."""
    pool_dir = tmp_path / "ailleurs" / "s10" / "pool"
    pool_dir.mkdir(parents=True)
    torch.save({"actor": {"w": torch.zeros(1)}, "model_cfg": {"type": "mlp"}, "global_step": 6_800_183_709, "elo": 1.0},
               pool_dir / "actor_0006800183709.pt")
    pool = OpponentPool(pool_dir, recent=1, anchors=["pool/actor_0006800183709.pt"], anchor_root=tmp_path / "code")
    assert [a["name"] for a in pool.anchors] == ["actor_0006800183709"]
    assert pool.anchors[0]["step"] == 6_800_183_709 and (pool_dir / "anchors" / "actor_0006800183709.pt").exists()
    again = OpponentPool(pool_dir, recent=1, anchors=["pool/actor_0006800183709.pt"], anchor_root=tmp_path / "code")
    assert again.anchors[0]["name"] == "actor_0006800183709"


def test_opponent_settings_for_the_session():
    sp = load_config(["kaggle_tpu", "s10"])["selfplay"]
    assert sp["pfsp"] and sp["anchors"][0] == "runs/s6_spo_kaggle" and len(sp["anchors"]) >= 5
    assert all(a.startswith("pool/actor_") for a in sp["anchors"][1:])
