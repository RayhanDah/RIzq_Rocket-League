"""S1-13 : chaque terme de reward + esprit d'équipe + zéro-somme."""
import numpy as np

from helpers import make_state
from rlagent.config import load_config
from rlagent.env.rewards import (BoostGainTerm, DoubleCommitTerm, GoalTerm, TouchTerm, VelBallToGoalTerm,
                                 VelPlayerToBallTerm, combine_team_rewards, linear_schedule, make_reward)


def test_goal_zero_sum():
    s = make_state(2)
    s.goal_scored = True
    s.ball.position = np.array([0, 5300.0, 100], dtype=np.float32)  # dans le but orange → bleu marque
    raw = GoalTerm().get(list(s.cars), s, {})
    assert raw["blue-0"] == raw["blue-1"] == 1.0 and raw["orange-0"] == 0.0
    teams = {a: s.cars[a].team_num for a in s.cars}
    final = combine_team_rewards(raw, teams, tau=0.3, zero_sum=1.0)
    assert np.isclose(final["blue-0"], 1.0) and np.isclose(final["orange-1"], -1.0)
    assert np.isclose(sum(final.values()), 0.0)


def test_team_spirit_shares_reward():
    raw = {"b0": 1.0, "b1": 0.0, "o0": 0.0, "o1": 0.0}
    teams = {"b0": 0, "b1": 0, "o0": 1, "o1": 1}
    solo = combine_team_rewards(raw, teams, tau=0.0, zero_sum=0.0)
    shared = combine_team_rewards(raw, teams, tau=1.0, zero_sum=0.0)
    assert solo["b1"] == 0.0 and np.isclose(shared["b1"], 0.5) and np.isclose(shared["b0"], 0.5)
    zs = combine_team_rewards(raw, teams, tau=0.0, zero_sum=1.0)
    assert np.isclose(zs["o0"], -0.5)


def test_touch_aerial_bonus():
    s = make_state(1)
    s.cars["blue-0"].ball_touches = 1
    s.ball.position = np.array([0, 0, 2044.0], dtype=np.float32)
    r = TouchTerm(aerial_bonus=1.0).get(list(s.cars), s, {})
    assert np.isclose(r["blue-0"], 2.0) and r["orange-0"] == 0.0


def test_vel_player_to_ball_positive_only():
    s = make_state(1)
    car = s.cars["blue-0"]
    to_ball = s.ball.position - car.physics.position
    car.physics.linear_velocity = (to_ball / np.linalg.norm(to_ball) * 2300).astype(np.float32)
    r = VelPlayerToBallTerm().get(["blue-0"], s, {})
    assert np.isclose(r["blue-0"], 1.0, atol=1e-4)
    car.physics.linear_velocity = -car.physics.linear_velocity
    assert VelPlayerToBallTerm().get(["blue-0"], s, {})["blue-0"] == 0.0


def test_vel_ball_to_goal_is_team_relative():
    s = make_state(1)
    s.ball.linear_velocity = np.array([0, 3000.0, 0], dtype=np.float32)  # vers le but orange
    r = VelBallToGoalTerm().get(list(s.cars), s, {})
    assert r["blue-0"] > 0.3 and np.isclose(r["orange-0"], -r["blue-0"], atol=1e-5)


def test_boost_gain_sqrt():
    s = make_state(1)
    t = BoostGainTerm(loss_coef=0.0)
    s.cars["blue-0"].boost_amount = 0.0
    t.reset(list(s.cars), s, {})
    s.cars["blue-0"].boost_amount = 100.0
    assert np.isclose(t.get(["blue-0"], s, {})["blue-0"], 1.0)
    s.cars["blue-0"].boost_amount = 50.0
    assert t.get(["blue-0"], s, {})["blue-0"] == 0.0  # dépense non pénalisée par défaut


def test_double_commit_only_with_two_committed_teammates():
    s = make_state(2, kickoff=False)
    s.ball.position = np.array([0, 0, 93.0], dtype=np.float32)
    for i, cid in enumerate(["blue-0", "blue-1"]):
        c = s.cars[cid]
        c.physics.position = np.array([-300 + 600 * i, -500, 17], dtype=np.float32)
        d = s.ball.position - c.physics.position
        c.physics.linear_velocity = (d / np.linalg.norm(d) * 1500).astype(np.float32)
        c.physics.euler_angles = np.zeros(3, dtype=np.float32)
    for cid in ["orange-0", "orange-1"]:
        c = s.cars[cid]
        c.physics.position = np.array([0, 4000, 17], dtype=np.float32)
        c.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        c.physics.euler_angles = np.zeros(3, dtype=np.float32)
    r = DoubleCommitTerm().get(list(s.cars), s, {})
    assert r["blue-0"] == -1.0 and r["blue-1"] == -1.0 and r["orange-0"] == 0.0
    s.cars["blue-1"].physics.linear_velocity = np.zeros(3, dtype=np.float32)
    r = DoubleCommitTerm().get(list(s.cars), s, {})
    assert r["blue-0"] == 0.0


def test_schedules_and_full_reward_runs():
    assert linear_schedule(1.0, 0.0, 100, 50) == 0.5
    assert linear_schedule(1.0, 0.0, 100, 500) == 0.0
    cfg = load_config()
    rf = make_reward(cfg)
    s = make_state(3)
    info = {"global_step": 0}
    rf.reset(list(s.cars), s, info)
    out = rf.get_rewards(list(s.cars), s, {a: False for a in s.cars}, {a: False for a in s.cars}, info)
    assert set(out) == set(s.cars) and all(np.isfinite(v) for v in out.values())
    assert np.isclose(sum(out.values()), 0.0, atol=1e-5)  # symétrique au kickoff → zéro-somme
    assert np.isclose(rf.current_tau(10**12), cfg["rewards"]["team_spirit"]["end"])
    assert "touch" in info["reward_log"]
