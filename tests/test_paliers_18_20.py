"""Session 10 (22/11/2026) : paliers 18-20 (bump puis but, murs, double saut), Nexto en sparring, imitation de Nexto."""
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import make_state

from rlagent.algo.collector import EnvSlot, Rollout, Trajectory, nexto_settings
from rlagent.algo.learner import build_batch, ppo_update
from rlagent.algo.gae import RunningStats
from rlagent.algo.worker import EpisodeStart, StepResult, nexto_action_code, nexto_beta_of
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.rewards import BumpGoalTerm, FastAerialTerm, WallTouchTerm, make_reward
from rlagent.opponents.nexto import NEXTO_DIR
from rlagent.train import kick_coef_at

GAP = "ladder/gap/actor_0023037887394"
NEXT = ["bump_but", "murs", "double_saut"]
V25785 = "pool/actor_0025785348226.pt"
TICK = 8


@pytest.fixture(scope="module")
def cfg():
    return load_config(["kaggle_tpu", "s10"])


def stage(cfg, name):
    return next(s for s in cfg["curriculum"]["stages"] if s["name"] == name)


def weights(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    return {n: c.stage_weights[c.names.index(n)] for n in c.names}


# --------------------------------------------------------------------------- configuration
def test_new_stages_one_change_each_after_but_haut(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    assert names[names.index("but_haut"):] == ["but_haut"] + NEXT
    w = weights(cfg)
    chg = lambda a, b: {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)}
    assert chg(w["but_haut"], w["bump_but"]) == {"bump_goal"}
    assert chg(w["bump_but"], w["murs"]) == {"wall_touch"}
    assert chg(w["murs"], w["double_saut"]) == {"fast_aerial"}
    last = w["double_saut"]
    assert last["goal"] == 20.0 and last["demo_goal"] == 4.0
    assert (last["bump_goal"], last["wall_touch"], last["fast_aerial"]) == (3.0, 0.5, 1.0)


def test_new_stages_are_slow_gated_and_capped(cfg):
    assert stage(cfg, "bump_but")["min_steps"] >= 1_000_000_000          # PO : plus de temps pour le bump
    for name in NEXT:
        assert stage(cfg, name)["blend_steps"] >= 200_000_000
    for name in NEXT[:-1]:
        assert stage(cfg, name)["min_steps"] >= 500_000_000
        assert stage(cfg, name)["advance"] == {"goals_per_min": 2.0, GAP: 0}
    g = cfg["train"]["guard"]
    for t in ("bump_goal", "wall_touch", "fast_aerial"):
        assert g["term_max_share"][t] == 0.03 and g["fallback_weights"][t] == 0.0
        assert t in cfg["rewards"]["zero_sum_terms"]
    assert cfg["rewards"]["demo_goal_window_s"] == 5.0                    # PO : 5 s au lieu de 3
    assert cfg["env"]["start_weights"]["drill_wall"] == 0.10
    assert V25785 in cfg["selfplay"]["anchors"]


def test_nexto_sparring_and_kickstart_settings(cfg):
    nx = cfg["selfplay"]["nexto"]
    assert nx["prob"] == 0.10 and nx["beta_start"] == 0.5
    assert nexto_settings(cfg) == {"prob": 0.10, "beta": 0.5}
    assert nexto_settings({**cfg, "env": {**cfg["env"], "engine": "fake"}})["prob"] == 0.0
    k = cfg["train"]["kickstart"]
    assert k["enabled"] and k["begin"] == 27_450_134_560 and k["coef"] == 0.5
    assert kick_coef_at(cfg, k["begin"]) == 0.5
    assert kick_coef_at(cfg, k["begin"] + k["steps"] / 2) == pytest.approx(0.25)
    assert kick_coef_at(cfg, k["begin"] + k["steps"]) == 0.0 and kick_coef_at(cfg, k["begin"] - 1) == 0.0


def test_new_terms_are_built(cfg):
    names = {t.name for t in make_reward(cfg).terms}
    assert {"bump_goal", "wall_touch", "fast_aerial"} <= names


# --------------------------------------------------------------------------- bump puis but
def oriented(s):
    for c in s.cars.values():
        c.physics.euler_angles = np.zeros(3, dtype=np.float32)
        c.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        c.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    s.ball.position = np.array([0.0, 0.0, 93.0], dtype=np.float32)
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    return s


def bump_then_goal(delay_steps, demolished=False, attacker=False):
    s = oriented(make_state(1, kickoff=False))
    agents = list(s.cars)
    blue = next(a for a in agents if s.cars[a].team_num == 0)
    orange = next(a for a in agents if s.cars[a].team_num == 1)
    s.cars[blue].physics.position = np.array([0.0, 3000.0, 17.0], dtype=np.float32)
    s.cars[orange].physics.position = np.array([0.0, -500.0 if attacker else 4000.0, 17.0], dtype=np.float32)
    for c in s.cars.values():
        c.ball_touches = 0
        c.bump_victim_id = None
    t = BumpGoalTerm(window_s=5.0, full_s=0.75)
    t.reset(agents, s, {})
    s.tick_count += TICK
    s.cars[blue].bump_victim_id = orange
    if demolished:
        s.cars[orange].demo_respawn_timer = 3.0
    assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}
    s.cars[blue].bump_victim_id = None
    for _ in range(delay_steps - 1):
        s.tick_count += TICK
        assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}
    s.tick_count += TICK
    s.goal_scored = True                                         # balle dans le but orange : le bleu marque
    s.ball.position = np.array([0.0, 5300.0, 300.0], dtype=np.float32)
    return t.get(agents, s, {})[blue]


def test_bump_goal_full_then_down_to_0_at_5_s():
    for steps, expected in ((1, 1.0), (11, 1.0), (40, (5.0 - 40 / 15) / 4.25), (75, 0.0), (90, 0.0)):
        assert bump_then_goal(steps) == pytest.approx(expected, abs=1e-6), steps


def test_bump_goal_not_for_a_demo_or_an_attacker():
    assert bump_then_goal(5, demolished=True) == 0.0                    # la démo relève de demo_goal
    assert bump_then_goal(5, attacker=True) == 0.0                      # victime hors de son camp


# --------------------------------------------------------------------------- vols
def flight_state():
    s = oriented(make_state(1, kickoff=False))
    agents = list(s.cars)
    blue = next(a for a in agents if s.cars[a].team_num == 0)
    for c in s.cars.values():
        c.ball_touches = 0
    s.ball.position = np.array([3900.0, 0.0, 600.0], dtype=np.float32)
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    return s, agents, blue


def test_wall_touch_needs_a_wall_takeoff_and_pays_by_usefulness():
    s, agents, blue = flight_state()
    car = s.cars[blue]
    car.physics.euler_angles = np.array([0.0, 0.0, np.pi / 2], dtype=np.float32)   # roues contre le mur
    car.wheels_with_contact = (True, True, True, True)
    t = WallTouchTerm(min_z=300.0, floor=0.1)
    t.reset(agents, s, {})
    assert t.get(agents, s, {})[blue] == 0.0
    car.wheels_with_contact = (False, False, False, False)
    car.physics.euler_angles = np.zeros(3, dtype=np.float32)
    s.tick_count += TICK
    t.get(agents, s, {})
    car.ball_touches = 1
    s.ball.linear_velocity = np.array([0.0, 1000.0, 0.0], dtype=np.float32)       # 1 000 uu/s vers le but orange
    s.tick_count += TICK
    paid = t.get(agents, s, {})[blue]
    assert 0.8 < paid <= 1.0                                     # 1 000 uu/s presque droit vers le but orange
    s.tick_count += TICK
    assert t.get(agents, s, {})[blue] == 0.0                                       # une seule touche par vol


def test_fast_aerial_decays_with_the_time_since_takeoff():
    def touch_after(steps):
        s, agents, blue = flight_state()
        car = s.cars[blue]
        car.wheels_with_contact = (True, True, True, True)
        t = FastAerialTerm(min_z=500.0, floor=1.0, fast_s=0.8, slow_s=2.0)
        t.reset(agents, s, {})
        t.get(agents, s, {})
        car.wheels_with_contact = (False, False, False, False)
        car.has_double_jumped = True
        for _ in range(steps):
            s.tick_count += TICK
            t.get(agents, s, {})
        car.ball_touches = 1
        s.tick_count += TICK
        return t.get(agents, s, {})[blue]
    assert touch_after(5) == pytest.approx(1.0)
    assert touch_after(20) == pytest.approx((2.0 - 20 * TICK / 120) / 1.2, abs=1e-6)
    assert touch_after(40) == 0.0


# --------------------------------------------------------------------------- sparring et imitation
def test_nexto_action_code_roundtrip():
    for beta in (0.5, 1.0, 0.0):
        code = nexto_action_code(beta)
        assert code < 0 and nexto_beta_of(code) == pytest.approx(beta)


def test_rollout_leaves_nexto_cars_to_the_worker_and_aligns_teacher_labels():
    class Actor(torch.nn.Module):
        def act(self, x):
            n = x.shape[0]
            return torch.full((n,), 3, dtype=torch.long), torch.zeros(n)

    obs = np.zeros((2, 5), dtype=np.float32)
    start = EpisodeStart(["blue-0", "orange-0"], np.array([0, 1], dtype=np.int8), obs, None)

    def new_slot(st):
        slot = EnvSlot(episode=st)
        slot.hist_team, slot.hist_entry = 1, {"id": "nexto", "kind": "nexto", "beta": 0.5, "elo": 0.0}
        return slot
    r = Rollout([start], new_slot)
    a = r.act([0], Actor(), torch.device("cpu"), lambda e: None)
    assert a[0].tolist() == [3, nexto_action_code(0.5)]
    nxt = EpisodeStart(start.agents, start.teams, obs, np.array([7, 9], dtype=np.int16))
    r.record([0], [StepResult(np.zeros(2, np.float32), False, False, None, None, nxt)])
    r.act([0], Actor(), torch.device("cpu"), lambda e: None)               # 2e step : étiquette de Nexto
    done = StepResult(np.zeros(2, np.float32), True, True, obs, {"goal_team": 1}, start)
    r.record([0], [done])
    trajs, eps = r.cut()
    t = trajs[0].compact()
    assert t.teachers.tolist() == [-1, 7] and len(t) == 2
    assert eps[0]["vs_nexto"] == {"learner_team": 0, "beta": 0.5} and "vs_hist" not in eps[0]


def test_kickstart_loss_pulls_the_policy_toward_the_teacher(cfg):
    from rlagent.env.obs import OBS_DIM
    from rlagent.models.attention import build_models
    torch.manual_seed(0)
    actor, critic = build_models(load_config(["debug"]), 90)        # petit modèle, obs de base
    n = 64
    x = np.random.default_rng(0).normal(size=(n, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        a, lp = actor.act(torch.as_tensor(x))
    trajs = [Trajectory([x[i]], [int(a[i])], [float(lp[i])], [0.0], None, teachers=[42]) for i in range(n)]
    pcfg = {**cfg, "ppo": {**cfg["ppo"], "minibatch_size": n, "device_chunk": n, "epochs": 4, "kick_coef": 1.0,
                           "lr_schedule": "fixed"}}
    batch = build_batch(trajs, critic, RunningStats(), pcfg, torch.device("cpu"))
    assert batch["teacher"].tolist() == [42] * n
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    before = actor.dist(torch.as_tensor(x)).probs[:, 42].mean().item()
    stats = ppo_update(actor, critic, opt, batch, pcfg, torch.device("cpu"))
    after = actor.dist(torch.as_tensor(x)).probs[:, 42].mean().item()
    assert after > before and stats["kick_ce"] > 0.0


@pytest.mark.skipif(not (NEXTO_DIR / "nexto-model.pt").exists(), reason="clone NectoFamily absent")
def test_env_runner_plays_nexto_and_labels_teacher_actions(cfg):
    """Vrai RocketSim : Nexto pilote l'orange (commandes du worker), le professeur étiquette les voitures."""
    meshes = Path(__file__).resolve().parents[1] / "collision_meshes" / "soccar"
    if not meshes.is_dir():
        pytest.skip("collision meshes absents")
    from rlagent.algo.worker import EnvRunner
    c = {**cfg, "train": {**cfg["train"], "kickstart": {**cfg["train"]["kickstart"], "begin": 0, "frac": 1.0}}}
    r = EnvRunner(c, seed=3)
    r.set_global_step(10)
    cur = r.current
    for _ in range(40):
        acts = np.array([5 if t == 0 else nexto_action_code(1.0) for t in cur.teams])
        res = r.step(acts)
        cur = res.next
        assert cur.teacher is not None and (cur.teacher >= 0).all() and (cur.teacher < 90).all()
    assert r.nexto is not None
