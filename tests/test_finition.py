"""Palier 5 « finition » (04/11/2026) : balle cadrée payée, déviation punie, frappe hors cadre moins payée, aériennes
avec boost, exercices de tir et d'aérienne, et ce qu'il ne faut pas toucher (l'ADN rush de l'agent)."""
import numpy as np
import pytest

from helpers import make_state
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.predict import goal_line_time, goal_threat
from rlagent.env.rewards import AerialTouchTerm, AirApproachTerm, ShotTerm, ShotThreatPotential, make_reward
from rlagent.metrics.episode import EpisodeTracker
from rlagent.selfplay.ladder import Ladder
from rlagent.train import summarize_episodes

ON_TARGET = ([0.0, 3000.0, 150.0], [0.0, 2000.0, 0.0])         # droit dans le but orange
WIDE = ([0.0, 3000.0, 150.0], [1500.0, 2000.0, 0.0])           # atteint la ligne hors du cadre


def set_ball(state, pos, vel):
    state.ball.position = np.array(pos, dtype=np.float32)
    state.ball.linear_velocity = np.array(vel, dtype=np.float32)


def car_of(state, team):
    return next((a, c) for a, c in state.cars.items() if c.team_num == team)


def in_air(car, boosting):
    car.wheels_with_contact = (False, False, False, False)
    car.boost_active_time = 0.1 if boosting else 0.0


def test_ball_scenarios_are_what_the_tests_assume():
    assert goal_threat(*ON_TARGET, 1.0) > 0.0
    assert goal_threat(*WIDE, 1.0) == 0.0 and goal_line_time(*WIDE, 1.0) is not None
    assert goal_line_time([0.0, 0.0, 93.0], [0.0, -500.0, 0.0], 1.0) is None


def test_goal_line_time_matches_goal_entry_for_on_target_ball():
    t = goal_line_time(*ON_TARGET, 1.0)
    assert 1.0 - t / 3.0 == pytest.approx(goal_threat(*ON_TARGET, 1.0))


# --------------------------------------------------------------------------- balle cadrée
def test_shot_threat_pays_shooter_at_once_and_is_antisymmetric():
    s = make_state()
    agents = list(s.cars)
    set_ball(s, [0.0, 3000.0, 150.0], [0.0, 0.0, 0.0])
    term = ShotThreatPotential(gamma=1.0, floor=0.5)
    term.reset(agents, s, {})
    set_ball(s, *ON_TARGET)
    r = term.get(agents, s, {})
    blue, _ = car_of(s, 0)
    orange, _ = car_of(s, 1)
    assert r[blue] >= 0.5 and r[orange] == pytest.approx(-r[blue])


def test_shot_threat_takes_back_a_spoiled_on_target_ball():
    """Le défaut vu en match : la balle allait au but, sa propre touche la fait sortir du cadre."""
    s = make_state()
    agents = list(s.cars)
    set_ball(s, *ON_TARGET)
    term = ShotThreatPotential(gamma=1.0, floor=0.5)
    term.reset(agents, s, {})
    set_ball(s, *WIDE)
    blue, _ = car_of(s, 0)
    assert term.get(agents, s, {})[blue] <= -0.5


def test_shot_threat_telescopes_so_it_cannot_be_farmed():
    """Aller-retour cadré / hors cadre : la somme revient à zéro (à γ = 1)."""
    s = make_state()
    agents = list(s.cars)
    set_ball(s, *WIDE)
    term = ShotThreatPotential(gamma=1.0, floor=0.5)
    term.reset(agents, s, {})
    blue, _ = car_of(s, 0)
    total = 0.0
    for _ in range(5):
        set_ball(s, *ON_TARGET)
        total += term.get(agents, s, {})[blue]
        set_ball(s, *WIDE)
        total += term.get(agents, s, {})[blue]
    assert total == pytest.approx(0.0, abs=1e-6)


def test_shot_pays_less_off_target_in_attack_but_not_clearances():
    s = make_state()
    agents = list(s.cars)
    blue, car = car_of(s, 0)
    car.ball_touches = 1

    def shot(factor, pos, vel):
        term = ShotTerm(off_target_factor=factor)
        set_ball(s, pos, [0.0, 0.0, 0.0])
        term.reset(agents, s, {})
        set_ball(s, pos, vel)
        return term.get(agents, s, {})[blue]

    assert shot(0.25, *WIDE) == pytest.approx(0.25 * shot(1.0, *WIDE))
    assert shot(0.25, *ON_TARGET) == pytest.approx(shot(1.0, *ON_TARGET))
    clearance = ([1500.0, -3000.0, 150.0], [1200.0, 2000.0, 0.0])
    assert shot(0.25, *clearance) == pytest.approx(shot(1.0, *clearance)) and shot(1.0, *clearance) > 0.0


# --------------------------------------------------------------------------- aériennes
def aerial_reward(boost_steps, min_boost=0.25, z=1000.0):
    s = make_state()
    agents = list(s.cars)
    blue, car = car_of(s, 0)
    term = AerialTouchTerm(min_z=500.0, min_air_boost_s=min_boost)
    term.reset(agents, s, {})
    set_ball(s, [0.0, 0.0, z], [0.0, 0.0, 0.0])
    for i in range(boost_steps + 1):
        s.tick_count += 8
        in_air(car, boosting=i < boost_steps or boost_steps > 0)
        car.ball_touches = 1 if i == boost_steps else 0
        r = term.get(agents, s, {})
    return r[blue]


def test_aerial_touch_needs_boost_in_the_air():
    assert aerial_reward(boost_steps=0) == 0.0                 # saut depuis le sol ou un mur, sans boost
    assert aerial_reward(boost_steps=2) == 0.0                 # 0,13 s de boost : pas encore une aérienne
    assert aerial_reward(boost_steps=5) > 0.5
    assert aerial_reward(boost_steps=0, min_boost=0.0) > 0.5   # sans l'option : comportement d'origine


def test_air_boost_resets_on_landing():
    s = make_state()
    agents = list(s.cars)
    blue, car = car_of(s, 0)
    term = AerialTouchTerm(min_z=500.0, min_air_boost_s=0.25)
    term.reset(agents, s, {})
    set_ball(s, [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0])
    for _ in range(6):
        s.tick_count += 8
        in_air(car, boosting=True)
        term.get(agents, s, {})
    s.tick_count += 8
    car.wheels_with_contact = (True, True, True, True)
    term.get(agents, s, {})
    s.tick_count += 8
    in_air(car, boosting=False)
    car.ball_touches = 1
    assert term.get(agents, s, {})[blue] == 0.0


def test_air_approach_boost_only_pays_flight_not_jumps():
    s = make_state()
    agents = list(s.cars)
    blue, car = car_of(s, 0)
    set_ball(s, [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0])
    car.physics.position = np.array([0.0, -800.0, 300.0], dtype=np.float32)
    car.physics.linear_velocity = np.array([0.0, 1000.0, 500.0], dtype=np.float32)
    in_air(car, boosting=False)
    assert AirApproachTerm(boost_only=False).get(agents, s, {})[blue] > 0.0
    assert AirApproachTerm(boost_only=True).get(agents, s, {})[blue] == 0.0
    in_air(car, boosting=True)
    assert AirApproachTerm(boost_only=True).get(agents, s, {})[blue] > 0.0


# --------------------------------------------------------------------------- mesures
def tracked_touch(before, after, team=0):
    s = make_state()
    tracker = EpisodeTracker()
    set_ball(s, *before)
    tracker.reset(s, {"team_size": 1, "start_kind": "kickoff"})
    _, car = car_of(s, team)
    set_ball(s, *after)
    car.ball_touches = 1
    s.tick_count += 8
    tracker.step(s)
    return tracker, s


def test_tracker_counts_spoils_and_off_target_shots():
    t, _ = tracked_touch(ON_TARGET, WIDE)
    assert (t.spoils, t.shots, t.off_target) == (1, 0, 0)
    t, _ = tracked_touch(([0.0, 3000.0, 150.0], [0.0, 0.0, 0.0]), WIDE)
    assert (t.spoils, t.shots, t.off_target) == (0, 0, 1)
    t, _ = tracked_touch(([0.0, 3000.0, 150.0], [0.0, 0.0, 0.0]), ON_TARGET)
    assert (t.spoils, t.shots, t.off_target) == (0, 1, 0)
    t, _ = tracked_touch(([0.0, 3000.0, 150.0], [0.0, 1900.0, 0.0]), WIDE)
    assert t.off_target == 0                                    # à peine déviée : pas une frappe


def test_tracker_separates_true_aerials_from_wall_jumps():
    s = make_state()
    tracker = EpisodeTracker()
    tracker.reset(s, {"team_size": 1, "start_kind": "kickoff"})
    _, car = car_of(s, 0)
    set_ball(s, [0.0, 0.0, 900.0], [0.0, 0.0, 0.0])
    car.physics.position = np.array([0.0, 0.0, 800.0], dtype=np.float32)
    car.wheels_with_contact = (True, True, True, True)          # sur un mur, à 800 uu
    s.tick_count += 8
    tracker.step(s)
    in_air(car, boosting=False)
    car.ball_touches = 1
    s.tick_count += 8
    tracker.step(s)
    assert (tracker.true_aerials, tracker.wall_aerials) == (0, 1)
    car.ball_touches = 0
    in_air(car, boosting=True)
    for _ in range(5):
        s.tick_count += 8
        tracker.step(s)
    car.ball_touches = 1
    s.tick_count += 8
    tracker.step(s)
    assert (tracker.true_aerials, tracker.wall_aerials) == (1, 1)


def test_style_metrics_reach_the_training_log():
    t, s = tracked_touch(ON_TARGET, WIDE)
    out = summarize_episodes([t.finish(s, terminated=False)])
    assert out["spoils_per_min"] > 0.0
    for key in ("off_target_per_min", "true_aerials_per_min", "wall_aerials_per_min", "mech/air_boost_frac",
                "mech/ball_dist", "mech/opp_half_frac", "mech/bumps_per_min"):
        assert key in out


# --------------------------------------------------------------------------- profil
def test_finition_stage_keeps_the_rush_dna():
    """Le PO y tient : le palier ne doit rien retirer de ce qui fait le jeu agressif de l'agent."""
    cfg = load_config(["kaggle_tpu", "s10"])
    c = SkillCurriculum.from_cfg(cfg)
    assert c.names[-2:] == ["precision", "finition"]
    w4, w5 = c.stage_weights[-2], c.stage_weights[-1]
    for term in ("vel_player_to_ball", "flip_speed", "demo", "bump", "goal", "vel_ball_to_goal", "shot"):
        assert w5[term] == w4[term], term
    assert w5["boost_keep"] == w4["boost_keep"] and cfg["rewards"]["boost_loss_coef"] == 0.0   # dépenser reste gratuit
    assert w5["shot_threat"] > 0.0 and w5["strong_touch"] < w4["strong_touch"]
    assert w5["aerial_touch"] > w4["aerial_touch"] and w5["boost_gain"] > w4["boost_gain"]


def test_finition_uses_new_terms_and_drills():
    cfg = load_config(["kaggle_tpu", "s10"])
    rf = make_reward(cfg)
    assert "shot_threat" in {t.name for t in rf.terms}
    assert next(t for t in rf.terms if isinstance(t, ShotTerm)).off_target_factor == 0.25
    assert next(t for t in rf.terms if isinstance(t, AirApproachTerm)).boost_only
    assert next(t for t in rf.terms if isinstance(t, AerialTouchTerm)).min_air_boost == pytest.approx(30.0)
    w = cfg["env"]["start_weights"]
    assert sum(w.values()) == pytest.approx(1.0) and w["drill_aerial"] > 0.0 and w["drill_shot"] > 0.0
    assert cfg["env"]["drill_curriculum"]["enabled"]


def test_aerial_drill_stays_where_flying_pays():
    """Le niveau ne doit pas redescendre sous la hauteur où les rewards aériens existent (validation du 04/11)."""
    from rlagent.env.mutators import _lerp
    from rlagent.train import DrillCurriculum
    cfg = load_config(["kaggle_tpu", "s10"])
    dc = DrillCurriculum.from_cfg(cfg)
    assert set(dc.levels) == {"aerial", "shot"}
    for _ in range(200):
        dc.update([{"drill": "aerial", "drill_level": dc.levels["aerial"], "drill_success": 0.0}])
    assert dc.levels["aerial"] == dc.min_level == 0.4
    assert _lerp(200, 1150, dc.min_level) > cfg["rewards"]["aerial_min_z"]      # balle la plus basse de l'exercice
    for _ in range(100):
        dc.update([{"drill": "aerial", "drill_level": dc.levels["aerial"], "drill_success": 1.0}])
    assert dc.levels["aerial"] > dc.min_level


def test_run_resumed_at_precision_moves_on_to_finition():
    """Reprise de s10 : le checkpoint est au palier 4, qui était le dernier et n'avait pas de critère de sortie."""
    cfg = load_config(["kaggle_tpu", "s10"])
    c = SkillCurriculum.from_cfg(cfg)
    c.load_state_dict({"index": 3, "stage_start": 2_918_000_000, "prev_weights": {"goal": 10.0}, "ema": {}, "streak": 0})
    assert c.name == "precision"
    step = 5_378_301_003
    msg = None
    for _ in range(c.patience):
        step += 400_000
        msg = c.update({"ladder/gap/s6_spo_kaggle": 465.0}, step)
    assert msg and c.name == "finition"
    assert c.ppo_overrides()["ent_coef"] == 0.005 and c.ppo_overrides()["lr_actor"] == 1.0e-4
    start, end = c.weights(step), c.weights(step + int(c.blend_steps))
    assert start["shot_threat"] == 0.0 and end["shot_threat"] == c.stage_weights[-1]["shot_threat"]


def test_ladder_metrics_use_last_rated_version_between_fits(tmp_path):
    lad = Ladder(tmp_path / "ladder.json")
    lad.add_version("a", 0)
    lad.add_version("b", 1)
    lad.record([("a", 1.0)] * 40)
    lad.fit()
    before = lad.metrics()["ladder/elo"]
    lad.add_version("c", 2)                                     # pas encore recalculée
    assert lad.metrics()["ladder/elo"] == before
