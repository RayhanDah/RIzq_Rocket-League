"""Paliers 26-29 (26/11/2026, décision PO) : boost, mobilité × 5, bump du gardien, aériennes rapides ; drill_dribble,
drill_aerial à la place de drill_reset, sparring Nexto 9 % (session 13)."""
import numpy as np
import pytest
from helpers import make_state

from rlagent.algo.worker import sparring_settings
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.mutators import DRILLS, DribbleDrill
from rlagent.env.rewards import KeeperBumpTerm, attack_on_keeper, is_dribbling, keeper_of, make_reward

NEW = ["boost_plus", "mobilite_x5", "bump_gardien", "aeriennes_rapides"]
V33875 = "pool/actor_0033875014025.pt"
GAP = "ladder/gap/actor_0033875014025"
TICK = 8


@pytest.fixture(scope="module")
def cfg():
    return load_config(["kaggle_tpu", "s10"])


def weights(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    return {n: c.stage_weights[c.names.index(n)] for n in c.names}


def stage(cfg, name):
    return next(s for s in cfg["curriculum"]["stages"] if s["name"] == name)


# --------------------------------------------------------------------------- configuration
def test_stages_26_29_follow_mobilite_one_change_each(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    assert names[names.index("mobilite"):] == ["mobilite"] + NEW
    w = weights(cfg)
    chg = lambda a, b: {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)}   # noqa: E731
    seq = ["mobilite"] + NEW
    expect = [{"big_pad_low"}, {"slow_turn", "powerslide"}, {"keeper_bump"}, {"fast_aerial"}]
    for (a, b), e in zip(zip(seq, seq[1:]), expect):
        assert chg(w[a], w[b]) == e, (a, b)
    last = w["aeriennes_rapides"]
    assert (last["big_pad_low"], last["boost_gain"], last["slow_turn"], last["powerslide"]) == (2.0, 1.0, 1.5, 0.5)
    assert (last["keeper_bump"], last["fast_aerial"], last["bump_goal"]) == (1.0, 2.0, 4.5)
    assert (w["mobilite"]["slow_turn"], w["mobilite"]["powerslide"]) == (0.3, 0.1)          # × 5 exactement
    assert max(last["demo_goal"], last["bump_goal"]) + last["high_goal"] < 0.5 * last["goal"]   # règle du PO


def test_stages_26_29_are_gated_capped_and_anchored(cfg):
    assert V33875 in cfg["selfplay"]["anchors"]
    for name in ["mobilite"] + NEW[:-1]:
        s = stage(cfg, name)
        assert s["min_steps"] >= 500_000_000 and s["advance"][GAP] >= -25
    for name in NEW:
        assert stage(cfg, name)["blend_steps"] >= 200_000_000
    g = cfg["train"]["guard"]
    assert g["term_max_share"]["keeper_bump"] == 0.03 and g["fallback_weights"]["keeper_bump"] == 0.0
    assert g["term_max_share"]["big_pad_low"] == 0.05                  # renforcé : 2,30 % du signal avant apprentissage
    assert "keeper_bump" in cfg["rewards"]["zero_sum_terms"]
    assert "keeper_bump" in {t.name for t in make_reward(cfg).terms}


def test_start_states_session_13(cfg):
    sw = cfg["env"]["start_weights"]
    assert sum(sw.values()) == pytest.approx(1.0)
    assert (sw["drill_reset"], sw["drill_aerial"], sw["drill_dribble"], sw["random"]) == (0.0, 0.05, 0.04, 0.06)
    assert "dribble" in DRILLS and not DribbleDrill.leveled


def test_sparring_session_13(cfg):
    sp = sparring_settings(cfg)
    assert sp == {"nexto": {"prob": 0.09, "beta": 0.5}}                 # beta repris du checkpoint à la reprise


# --------------------------------------------------------------------------- dribble et gardien
def attack_state(ball_vy=900.0, keeper_y=4700.0, bump=False, touch="blue", tick=0, ball_dx=0.0):
    """1v1 : bleu dribble vers +y (but orange), orange dans ses cages."""
    st = make_state(1, kickoff=True)
    blue = next(a for a, c in st.cars.items() if c.team_num == 0)
    orange = next(a for a in st.cars if a != blue)
    b, o = st.cars[blue], st.cars[orange]
    b.physics.position = np.array([0.0, 3500.0, 17.0], dtype=np.float32)
    b.physics.linear_velocity = np.array([0.0, ball_vy, 0.0], dtype=np.float32)
    b.wheels_with_contact = (True,) * 4
    o.physics.position = np.array([100.0, keeper_y, 17.0], dtype=np.float32)
    o.physics.linear_velocity = np.zeros(3, dtype=np.float32)
    st.ball.position = np.array([ball_dx, 3515.0, 157.0], dtype=np.float32)
    st.ball.linear_velocity = np.array([0.0, ball_vy, 0.0], dtype=np.float32)
    b.ball_touches = 1 if touch == "blue" else 0
    o.ball_touches = 1 if touch == "orange" else 0
    b.bump_victim_id = orange if bump else None
    st.tick_count = tick
    return st, blue, orange


def test_dribble_and_keeper_detection():
    st, blue, orange = attack_state()
    pos, vel = st.ball.position.tolist(), st.ball.linear_velocity.tolist()
    assert is_dribbling(st.cars[blue], pos, vel)
    assert keeper_of(st, 1, pos) == orange
    assert attack_on_keeper(st, st.cars[blue], pos, vel) == orange
    st2, blue2, _ = attack_state(keeper_y=2500.0)                       # adversaire devant la balle, loin de son but
    assert attack_on_keeper(st2, st2.cars[blue2], st2.ball.position.tolist(), st2.ball.linear_velocity.tolist()) is None
    st3, blue3, _ = attack_state(ball_vy=-500.0)                        # balle qui revient vers nous
    assert attack_on_keeper(st3, st3.cars[blue3], st3.ball.position.tolist(), st3.ball.linear_velocity.tolist()) is None
    st4, blue4, _ = attack_state(ball_dx=600.0)                         # balle trop loin du toit
    assert not is_dribbling(st4.cars[blue4], st4.ball.position.tolist(), st4.ball.linear_velocity.tolist())


def test_keeper_bump_pays_once_per_possession():
    term = KeeperBumpTerm()
    st, blue, orange = attack_state()
    agents = list(st.cars)
    term.reset(agents, attack_state()[0], {})
    term.get(agents, attack_state(tick=8)[0], {})                       # possession bleue
    out = term.get(agents, attack_state(bump=True, tick=16)[0], {})
    assert out[blue] == 1.0 and out[orange] == 0.0
    term.get(agents, attack_state(tick=24)[0], {})                      # fin du contact
    assert term.get(agents, attack_state(bump=True, tick=24 + 12 * 120)[0], {})[blue] == 0.0   # même possession
    term.get(agents, attack_state(touch="orange", tick=24 + 13 * 120)[0], {})                  # l'orange touche
    term.get(agents, attack_state(tick=24 + 14 * 120)[0], {})                                  # balle reprise
    assert term.get(agents, attack_state(bump=True, tick=24 + 15 * 120)[0], {})[blue] == 1.0


def test_keeper_bump_ignores_other_bumps():
    for kw in ({"keeper_y": 2500.0}, {"ball_vy": -500.0}, {"touch": "orange"}):
        term = KeeperBumpTerm()
        st, blue, _ = attack_state(**kw)
        agents = list(st.cars)
        term.reset(agents, attack_state(**kw)[0], {})
        term.get(agents, attack_state(tick=8, **kw)[0], {})
        assert term.get(agents, attack_state(bump=True, tick=16, **kw)[0], {})[blue] == 0.0, kw


def test_keeper_bump_loop_pays_at_most_six_per_minute():
    """Bumper le gardien, lui rendre la balle, la reprendre, recommencer, le plus vite possible pendant une minute."""
    term = KeeperBumpTerm()
    st, blue, _ = attack_state()
    agents = list(st.cars)
    term.reset(agents, st, {})
    total, tick, k = 0.0, 0, 0
    while tick < 60 * 120:
        tick += TICK
        k += 1
        phase = k % 3
        s = attack_state(bump=phase == 1, touch="orange" if phase == 2 else "blue", tick=tick)[0]
        total += term.get(agents, s, {})[blue]
    assert total <= 6.0 and total * 1.0 < 0.5 * 20.0                    # bien sous un demi-but par minute


def test_dribble_drill_places_ball_on_roof_and_keeper_in_goal():
    st = make_state(1, kickoff=True)
    drill = DribbleDrill(seed=4)
    for _ in range(20):
        info = {}
        drill.apply(st, info)
        team = info["drill_team"]
        car = next(c for c in st.cars.values() if c.team_num == team)
        pos, vel = st.ball.position.tolist(), st.ball.linear_velocity.tolist()
        car.wheels_with_contact = (True,) * 4
        assert is_dribbling(car, pos, vel)
        assert keeper_of(st, 1 - team, pos) is not None
