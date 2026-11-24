"""Paliers 21-25 (25/11/2026, décision PO) : 1re touche du kickoff, contacts renforcés, gros pad quand on est à sec,
tir cadré, mobilité ; sparring réduit (session 12)."""
import numpy as np
import pytest
from helpers import make_state

from rlagent.algo.worker import sparring_settings
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.rewards import BigPadLowTerm, KickoffFirstTouchTerm, PowerslideTurnTerm, SlowTurnTerm, make_reward
from rlagent.train import sparring_promotion

NEW = ["kickoff_gagne", "contacts", "boost", "tir_cadre", "mobilite"]
V30160 = "pool/actor_0030160068410.pt"
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
def test_stages_21_25_one_change_each(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    assert names[names.index("double_saut"):] == ["double_saut"] + NEW
    w = weights(cfg)
    chg = lambda a, b: {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)}   # noqa: E731
    seq = ["double_saut"] + NEW
    expect = [{"kickoff_first"}, {"demo", "bump", "bump_goal"}, {"big_pad_low"}, {"shot_threat"}, {"slow_turn", "powerslide"}]
    for (a, b), e in zip(zip(seq, seq[1:]), expect):
        assert chg(w[a], w[b]) == e, (a, b)
    last = w["mobilite"]
    assert (last["kickoff_first"], last["demo"], last["bump"], last["bump_goal"]) == (2.0, 10.0, 2.5, 4.5)
    assert (last["big_pad_low"], last["shot_threat"], last["slow_turn"], last["powerslide"]) == (0.5, 5.0, 0.3, 0.1)
    goal = last["goal"]
    assert max(last["demo_goal"], last["bump_goal"]) + last["high_goal"] < 0.5 * goal


def test_stages_21_25_are_gated_capped_and_anchored(cfg):
    assert V30160 in cfg["selfplay"]["anchors"]
    for name in ["double_saut"] + NEW[:-1]:
        s = stage(cfg, name)
        assert s["min_steps"] >= 500_000_000 and s["advance"]["ladder/gap/actor_0030160068410"] >= -25
    for name in NEW:
        assert stage(cfg, name)["blend_steps"] >= 200_000_000
    g = cfg["train"]["guard"]
    for t in ("kickoff_first", "big_pad_low", "slow_turn", "powerslide"):
        assert g["term_max_share"][t] == 0.03 and g["fallback_weights"][t] == 0.0
    assert "kickoff_first" in cfg["rewards"]["zero_sum_terms"]
    assert "slow_turn" not in cfg["rewards"]["zero_sum_terms"] and "big_pad_low" not in cfg["rewards"]["zero_sum_terms"]
    names = {t.name for t in make_reward(cfg).terms}
    assert {"kickoff_first", "big_pad_low", "slow_turn", "powerslide"} <= names


def test_sparring_session_12(cfg):
    sp = sparring_settings(cfg)
    assert sp["nexto"]["prob"] == 0.04
    for b in ("galactus", "kira", "rain"):
        assert sp[b]["prob"] == 0.01 and sparring_promotion(cfg["selfplay"], b)[1] == 0.25
    assert sparring_promotion(cfg["selfplay"], "nexto")[1] == 0.35
    assert sum(b["prob"] for b in sp.values()) == pytest.approx(0.07)


# --------------------------------------------------------------------------- termes
def kickoff_state(ball_vel=(0.0, 0.0, 0.0), touches=(0, 0), tick=0):
    st = make_state(1, kickoff=True)
    st.ball.linear_velocity = np.array(ball_vel, dtype=np.float32)
    for car in st.cars.values():                              # bleu (équipe 0) puis orange, comme `touches`
        car.ball_touches = touches[car.team_num]
    st.tick_count = tick
    return st


def test_kickoff_first_pays_the_first_toucher_once():
    term = KickoffFirstTouchTerm()
    st = kickoff_state()
    agents = list(st.cars)
    term.reset(agents, st, {"start_kind": "kickoff"})
    assert set(term.get(agents, kickoff_state(tick=100), {}).values()) == {0.0}
    out = term.get(agents, kickoff_state(ball_vel=(0, 900, 0), touches=(1, 0), tick=240), {})
    blue = next(a for a in agents if st.cars[a].team_num == 0)
    orange = next(a for a in agents if a != blue)
    assert out[blue] == 1.0 and out[orange] == 0.0                           # bleu touche, balle vers l'orange
    assert set(term.get(agents, kickoff_state(ball_vel=(0, 900, 0), touches=(1, 0), tick=250), {}).values()) == {0.0}


def test_kickoff_first_ignores_back_touches_50_50_late_touches_and_non_kickoffs():
    agents = None
    for vel, touches, tick, kind in [((0, -900, 0), (1, 0), 240, "kickoff"),   # bleu renvoie vers son but
                                     ((0, 300, 0), (1, 1), 240, "kickoff"),    # 50/50
                                     ((0, 900, 0), (1, 0), 400, "kickoff"),    # après 3 s
                                     ((0, 900, 0), (1, 0), 240, "drill")]:     # pas un kickoff
        term = KickoffFirstTouchTerm()
        st = kickoff_state()
        agents = list(st.cars)
        term.reset(agents, st, {"start_kind": kind})
        assert set(term.get(agents, kickoff_state(vel, touches, tick), {}).values()) == {0.0}


def pad_state(boost, tick, demoed=False):
    st = make_state(1, kickoff=True)
    st.cars = dict(list(st.cars.items())[:1])
    car = next(iter(st.cars.values()))
    car.boost_amount = boost
    car.demo_respawn_timer = 2.0 if demoed else 0.0
    st.tick_count = tick
    return st


def test_big_pad_low_pays_low_pickups_with_a_cooldown():
    term = BigPadLowTerm()
    agents = list(pad_state(0, 0).cars)
    term.reset(agents, pad_state(5.0, 0), {})
    assert term.get(agents, pad_state(100.0, 8), {})[agents[0]] == pytest.approx(0.8)   # (25 − 5) / 25
    term.get(agents, pad_state(2.0, 16), {})                                  # brûlé jusqu'à 2
    assert term.get(agents, pad_state(100.0, 600), {})[agents[0]] == 0.0      # 5 s plus tard : délai
    term.get(agents, pad_state(0.0, 1300), {})
    assert term.get(agents, pad_state(100.0, 1310), {})[agents[0]] == pytest.approx(1.0)  # 10 s plus tard
    term.get(agents, pad_state(40.0, 3000), {})
    assert term.get(agents, pad_state(100.0, 3010), {})[agents[0]] == 0.0     # 40 de boost : pas à sec
    term.get(agents, pad_state(0.0, 4300, demoed=True), {})
    assert term.get(agents, pad_state(33.3, 4310), {})[agents[0]] == 0.0      # réapparition après démo


def test_big_pad_low_loop_pays_at_most_six_per_minute():
    """Boucle « brûler son boost, reprendre un gros pad » toutes les 2 s pendant une minute."""
    term = BigPadLowTerm()
    agents = list(pad_state(0, 0).cars)
    term.reset(agents, pad_state(0.0, 0), {})
    total, tick = 0.0, 0
    while tick < 60 * 120:
        tick += TICK
        boost = 100.0 if (tick // TICK) % 30 == 0 else 0.0
        total += term.get(agents, pad_state(boost, tick), {})[agents[0]]
    assert total <= 6.0
    assert total * 0.5 < 0.5 * 20.0                                           # bien sous un demi-but par minute


def turn_state(fwd_speed, ball_behind=True, on_ground=True, tick=0):
    st = make_state(1, kickoff=True)
    st.cars = dict(list(st.cars.items())[:1])
    car = next(iter(st.cars.values()))
    car.physics.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    car.physics.rotation_mtx = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    car.physics.linear_velocity = np.array([0.0, fwd_speed, 0.0], dtype=np.float32)
    car.wheels_with_contact = (on_ground,) * 4
    st.ball.position = np.array([0.0, -2000.0 if ball_behind else 2000.0, 93.0], dtype=np.float32)
    st.tick_count = tick
    return st


def test_slow_turn_costs_only_slow_ground_turns_with_the_ball_behind():
    term = SlowTurnTerm()
    agents = list(turn_state(0).cars)
    term.reset(agents, turn_state(0, tick=0), {})
    a = agents[0]
    assert term.get(agents, turn_state(300, tick=120), {})[a] == pytest.approx(-1.0)     # 1 s lente balle derrière
    assert term.get(agents, turn_state(1500, tick=240), {})[a] == 0.0                     # rapide
    assert term.get(agents, turn_state(300, ball_behind=False, tick=360), {})[a] == 0.0   # balle devant
    assert term.get(agents, turn_state(300, on_ground=False, tick=480), {})[a] == 0.0     # en l'air (demi-flip)


def slide_state(heading_deg, handbrake, speed=1200.0, tick=0, on_ground=True):
    """Voiture au centre, balle à 2 000 uu en +y ; heading 0° = face à la balle."""
    st = turn_state(0.0, ball_behind=False, on_ground=on_ground, tick=tick)
    car = next(iter(st.cars.values()))
    h = np.radians(90.0 + heading_deg)                                       # avant dans le plan, 90° = +y
    fwd = np.array([np.cos(h), np.sin(h), 0.0])
    right = np.array([np.sin(h), -np.cos(h), 0.0])
    car.physics.rotation_mtx = np.stack([fwd, right, [0.0, 0.0, 1.0]], axis=1).astype(np.float32)
    car.physics.linear_velocity = (fwd * speed).astype(np.float32)
    car.handbrake = 1.0 if handbrake else 0.0
    st.tick_count = tick
    return st


def test_powerslide_pays_turning_toward_the_ball_and_is_signed():
    term = PowerslideTurnTerm()
    agents = list(slide_state(0, False).cars)
    a = agents[0]
    term.reset(agents, slide_state(180, False), {})
    term.get(agents, slide_state(180, True, tick=8), {})                     # dérape, balle dans le dos
    assert term.get(agents, slide_state(0, False, tick=40), {})[a] == pytest.approx(1.0, abs=1e-3)   # face : (1+1)/2
    term.get(agents, slide_state(100, True, tick=400), {})                   # balle de côté…
    want = (-1.0 - np.cos(np.radians(100))) / 2.0
    assert term.get(agents, slide_state(180, False, tick=440), {})[a] == pytest.approx(want, abs=1e-3)  # …vers le dos
    term.get(agents, slide_state(0, True, tick=800), {})                     # balle déjà devant : rien
    assert term.get(agents, slide_state(10, False, tick=840), {})[a] == 0.0


def test_powerslide_needs_speed_duration_ground_and_cooldown():
    term = PowerslideTurnTerm()
    agents = list(slide_state(0, False).cars)
    a = agents[0]
    term.reset(agents, slide_state(180, False), {})
    term.get(agents, slide_state(180, True, speed=300, tick=8), {})          # trop lent : pas un dérapage
    assert term.get(agents, slide_state(0, False, tick=40), {})[a] == 0.0
    term.get(agents, slide_state(180, True, tick=48), {})
    assert term.get(agents, slide_state(0, False, tick=52), {})[a] == 0.0    # 4 ticks : trop court
    term.get(agents, slide_state(180, True, tick=100), {})
    assert term.get(agents, slide_state(0, False, tick=140), {})[a] > 0.9
    term.get(agents, slide_state(180, True, tick=148), {})
    assert term.get(agents, slide_state(0, False, tick=180), {})[a] == 0.0   # moins de 2 s après le précédent


def test_powerslide_loop_pays_at_most_three_per_minute_at_its_weight():
    """Se détourner sans frein puis revenir en dérapant, le plus vite possible, pendant une minute."""
    term = PowerslideTurnTerm()
    agents = list(slide_state(0, False).cars)
    a = agents[0]
    term.reset(agents, slide_state(180, False), {})
    total, tick = 0.0, 0
    while tick < 60 * 120:
        for heading, hb in ((180, False), (180, True), (0, False)):
            tick += TICK * 2
            total += term.get(agents, slide_state(heading, hb, tick=tick), {})[a]
    assert total <= 30.0 and total * 0.1 <= 3.0
