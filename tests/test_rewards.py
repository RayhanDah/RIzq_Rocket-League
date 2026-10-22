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


def test_zero_sum_terms_default_unchanged_and_selective():
    """zero_sum_terms=None : comportement d'origine ; [goal] : la touche adverse n'est plus soustraite."""
    from rlagent.env.rewards import TeamReward
    s = make_state(1)
    s.cars["orange-0"].ball_touches = 1
    agents = list(s.cars)
    w = {"goal": 10.0, "touch": 0.5}
    base = TeamReward([GoalTerm(), TouchTerm(0.0)], w)
    info = {}
    base.reset(agents, s, info)
    r0 = base.get_rewards(agents, s, {}, {}, info)
    assert np.isclose(r0["orange-0"], 0.5) and np.isclose(r0["blue-0"], -0.5)  # zéro-somme sur tout
    sel = TeamReward([GoalTerm(), TouchTerm(0.0)], w, zero_sum_terms=["goal"])
    info = {}
    sel.reset(agents, s, info)
    r1 = sel.get_rewards(agents, s, {}, {}, info)
    assert np.isclose(r1["orange-0"], 0.5) and np.isclose(r1["blue-0"], 0.0)   # touche adverse non soustraite
    # le but reste à somme nulle
    s.goal_scored = True
    s.ball.position = np.array([0, 5300.0, 100], dtype=np.float32)
    s.cars["orange-0"].ball_touches = 0
    r2 = sel.get_rewards(agents, s, {}, {}, info)
    assert np.isclose(r2["blue-0"], 10.0) and np.isclose(r2["orange-0"], -10.0)


def test_combine_matches_numpy_reference():
    rng = np.random.default_rng(0)
    for size in (1, 2, 3):
        agents = [f"b{i}" for i in range(size)] + [f"o{i}" for i in range(size)]
        teams = {a: (0 if a[0] == "b" else 1) for a in agents}
        raw = {a: float(rng.normal()) for a in agents}
        tau = float(rng.uniform())
        out = combine_team_rewards(raw, teams, tau, 1.0)
        for a in agents:
            mine = np.mean([raw[b] for b in agents if teams[b] == teams[a]])
            opp = np.mean([raw[b] for b in agents if teams[b] != teams[a]])
            assert np.isclose(out[a], (1 - tau) * raw[a] + tau * mine - opp)


def test_touch_strength_only_on_touch_and_disabled_by_default():
    from rlagent.env.rewards import TouchStrengthTerm
    s = make_state(1)
    agents = list(s.cars)
    t = TouchStrengthTerm()
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    t.reset(agents, s, {})
    s.ball.linear_velocity = np.array([0, 3000.0, 0], dtype=np.float32)
    s.cars["blue-0"].ball_touches = 1
    r = t.get(agents, s, {})
    assert np.isclose(r["blue-0"], 3000.0 / 6000.0) and r["orange-0"] == 0.0
    # pas de poids dans default.yaml → terme absent de la reward (comportement d'origine)
    cfg = load_config()
    assert "touch_strength" not in cfg["rewards"]["weights"]
    assert all(term.name != "touch_strength" for term in make_reward(cfg).terms)


def test_shot_rewards_only_goal_directed_hits():
    from rlagent.env.rewards import ShotTerm
    s = make_state(1)
    agents = list(s.cars)
    t = ShotTerm()
    s.ball.position = np.array([0, 0, 93.0], dtype=np.float32)
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    t.reset(agents, s, {})
    s.cars["blue-0"].ball_touches = 1
    s.ball.linear_velocity = np.array([0, 3000.0, 0], dtype=np.float32)       # vers le but orange
    r = t.get(agents, s, {})
    assert r["blue-0"] > 0.45 and r["orange-0"] == 0.0
    s.ball.linear_velocity = np.array([3000.0, 3000.0, 0], dtype=np.float32)  # frappe vers le côté : ~0
    assert abs(t.get(agents, s, {})["blue-0"]) < 0.05
    s.cars["blue-0"].ball_touches = 0
    s.cars["orange-0"].ball_touches = 1
    s.ball.linear_velocity = np.array([3000.0, -1000.0, 0], dtype=np.float32)  # l'orange renvoie vers le but bleu
    assert t.get(agents, s, {})["orange-0"] > 0.3


def test_align_ball_goal_symmetric_and_signed():
    from rlagent.env.rewards import AlignBallGoalTerm
    s = make_state(1)
    s.ball.position = np.array([0, 0, 93.0], dtype=np.float32)
    s.cars["blue-0"].physics.position = np.array([0, -2000.0, 17], dtype=np.float32)    # derrière la balle
    s.cars["orange-0"].physics.position = np.array([0, 2000.0, 17], dtype=np.float32)   # derrière la balle aussi
    r = AlignBallGoalTerm().get(list(s.cars), s, {})
    assert r["blue-0"] > 0.95 and np.isclose(r["blue-0"], r["orange-0"], atol=1e-4)
    s.cars["blue-0"].physics.position = np.array([0, 3000.0, 17], dtype=np.float32)     # entre balle et but adverse
    assert AlignBallGoalTerm().get(["blue-0"], s, {})["blue-0"] < -0.95


def test_in_air_and_boost_keep():
    from rlagent.env.rewards import BoostKeepTerm, InAirTerm
    s = make_state(1)
    s.cars["blue-0"].on_ground = False
    s.cars["orange-0"].on_ground = True
    r = InAirTerm().get(list(s.cars), s, {})
    assert r == {"blue-0": 1.0, "orange-0": 0.0}
    s.cars["blue-0"].boost_amount = 25.0
    assert np.isclose(BoostKeepTerm().get(["blue-0"], s, {})["blue-0"], 0.5)
    cfg = load_config()  # sans poids dans default.yaml : termes inactifs
    assert not {"shot", "align_ball_goal", "in_air", "boost_keep"} & {t.name for t in make_reward(cfg).terms}
