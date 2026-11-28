"""Palier 30 (28/11/2026, décision PO, dernier run avant la 1.0) : sortir le défenseur (bump ou démo) en possession
offensive au lieu de flicker dans le gardien ; drill_dribble avec un gardien qui sort ; repère v37780."""
import numpy as np
import pytest
from helpers import make_state

from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.mutators import DribbleDrill
from rlagent.env.rewards import (DEFENDER_MAX_DIST, KeeperBumpTerm, is_dribbling, keeper_of, make_reward,
                                 offense_on_defender)
from rlagent.metrics.episode import EpisodeTracker

V37780 = "pool/actor_0037780085008.pt"
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
def test_stage_30_is_last_and_changes_only_keeper_bump(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    assert names[-2:] == ["aeriennes_rapides", "sortir_defenseur"]
    w = weights(cfg)
    a, b = w["aeriennes_rapides"], w["sortir_defenseur"]
    assert {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)} == {"keeper_bump"}
    assert (a["keeper_bump"], b["keeper_bump"]) == (1.0, 3.0)
    assert (b["shot_threat"], b["demo"], b["bump"], b["bump_goal"]) == (5.0, 10.0, 2.5, 4.5)   # inchangés


def test_stage_29_now_gated_and_stage_30_blended(cfg):
    s29 = stage(cfg, "aeriennes_rapides")
    assert s29["min_steps"] == 500_000_000
    assert s29["advance"] == {"goals_per_min": 2.0, "ladder/gap/actor_0033875014025": -25}
    s30 = stage(cfg, "sortir_defenseur")
    assert s30["blend_steps"] >= 200_000_000 and s30["min_steps"] >= 500_000_000 and "advance" not in s30
    assert V37780 in cfg["selfplay"]["anchors"]


def test_reward_options_and_guard(cfg):
    rc = cfg["rewards"]
    assert (rc["keeper_bump_situation"], rc["keeper_bump_demo"], rc["keeper_bump_cooldown_s"]) == ("possession", True, 40.0)
    term = next(t for t in make_reward(cfg).terms if t.name == "keeper_bump")
    assert (term.situation, term.pay_demo, term.cooldown_s) == ("possession", True, 40.0)
    g = cfg["train"]["guard"]
    assert g["term_max_share"]["keeper_bump"] == 0.03 and g["fallback_weights"]["keeper_bump"] == 0.0
    assert "keeper_bump" in rc["zero_sum_terms"]
    # boucle « bumper le défenseur sans marquer » jouée à fond : 60 / 40 paiements par minute en moyenne, sous un demi-but
    w = weights(cfg)["sortir_defenseur"]
    assert 60.0 / rc["keeper_bump_cooldown_s"] * w["keeper_bump"] < 0.5 * w["goal"]


def test_situation_unknown_rejected():
    with pytest.raises(ValueError):
        KeeperBumpTerm(situation="flick")


# --------------------------------------------------------------------------- possession offensive
def attack_state(ball=(0.0, 3300.0), ball_vy=600.0, car=(0.0, 2600.0), defender=(200.0, 4300.0), bump=False,
                 touch="blue", tick=0, demoed=False):
    """1v1 : bleu attaque vers +y (but orange), balle devant lui au sol (pas sur le toit), orange en défense."""
    st = make_state(1, kickoff=True)
    blue = next(a for a, c in st.cars.items() if c.team_num == 0)
    orange = next(a for a in st.cars if a != blue)
    b, o = st.cars[blue], st.cars[orange]
    b.physics.position = np.array([car[0], car[1], 17.0], dtype=np.float32)
    b.physics.linear_velocity = np.array([0.0, 1200.0, 0.0], dtype=np.float32)
    b.wheels_with_contact = (True,) * 4
    o.physics.position = np.array([defender[0], defender[1], 17.0], dtype=np.float32)
    o.physics.linear_velocity = np.zeros(3, dtype=np.float32)
    o.demo_respawn_timer = 3.0 if demoed else 0.0
    st.ball.position = np.array([ball[0], ball[1], 93.0], dtype=np.float32)
    st.ball.linear_velocity = np.array([0.0, ball_vy, 0.0], dtype=np.float32)
    b.ball_touches = 1 if touch == "blue" else 0
    o.ball_touches = 1 if touch == "orange" else 0
    b.bump_victim_id = orange if bump else None
    st.tick_count = tick
    return st, blue, orange


def situation(**kw):
    st, blue, orange = attack_state(**kw)
    return offense_on_defender(st, st.cars[blue], st.ball.position.tolist(), st.ball.linear_velocity.tolist()), orange


def test_offensive_possession_detection():
    found, orange = situation()
    assert found == orange                                   # balle à 700 uu, pas sur le toit : ça compte
    st, blue, _ = attack_state()
    assert not is_dribbling(st.cars[blue], st.ball.position.tolist(), st.ball.linear_velocity.tolist())
    assert situation(ball=(0.0, -500.0), car=(0.0, -1200.0))[0] is None       # balle dans notre camp
    assert situation(ball_vy=-300.0)[0] is None                                # balle qui revient vers nous
    assert situation(ball=(0.0, 4000.0))[0] is None                            # défenseur plus près de la balle
    assert situation(car=(0.0, 1500.0), ball=(0.0, 2400.0))[0] is None         # balle à 900 uu… défenseur à 2 800 uu
    assert situation(defender=(200.0, 2900.0), ball=(0.0, 2800.0), car=(0.0, 2300.0))[0] is None   # loin de son but
    assert situation(defender=(200.0, 3000.0))[0] is None                      # devant la balle, pas entre elle et le but
    assert situation(ball=(0.0, 3700.0), car=(0.0, 2600.0))[0] is None         # balle à 1 100 uu : pas à nous


def test_keeper_bump_pays_bump_and_demo_of_the_defender():
    for demoed in (False, True):
        term = KeeperBumpTerm(cooldown_s=40.0, situation="possession", pay_demo=True)
        st, blue, orange = attack_state()
        agents = list(st.cars)
        term.reset(agents, st, {})
        term.get(agents, attack_state(tick=8)[0], {})
        out = term.get(agents, attack_state(bump=True, demoed=demoed, tick=16)[0], {})
        assert out[blue] == 1.0 and out[orange] == 0.0, demoed
    term = KeeperBumpTerm(cooldown_s=40.0, situation="possession", pay_demo=False)
    term.reset(agents, attack_state()[0], {})
    term.get(agents, attack_state(tick=8)[0], {})
    assert term.get(agents, attack_state(bump=True, demoed=True, tick=16)[0], {})[blue] == 0.0


def test_keeper_bump_possession_ignores_other_bumps():
    for kw in ({"touch": "orange"}, {"ball_vy": -300.0}, {"ball": (0.0, 4000.0)}, {"defender": (200.0, 3000.0)}):
        term = KeeperBumpTerm(cooldown_s=40.0, situation="possession", pay_demo=True)
        st, blue, _ = attack_state(**kw)
        agents = list(st.cars)
        term.reset(agents, st, {})
        term.get(agents, attack_state(tick=8, **kw)[0], {})
        assert term.get(agents, attack_state(bump=True, tick=16, **kw)[0], {})[blue] == 0.0, kw


def test_keeper_bump_possession_loop_bounded(cfg):
    """Bumper le défenseur, lui rendre la balle, la reprendre, recommencer, le plus vite possible pendant une minute."""
    rc = cfg["rewards"]
    term = KeeperBumpTerm(cooldown_s=rc["keeper_bump_cooldown_s"], situation="possession", pay_demo=True)
    st, blue, _ = attack_state()
    agents = list(st.cars)
    term.reset(agents, st, {})
    total, tick, k = 0.0, 0, 0
    while tick < 60 * 120:
        tick += TICK
        k += 1
        phase = k % 3
        total += term.get(agents, attack_state(bump=phase == 1, touch="orange" if phase == 2 else "blue",
                                               tick=tick)[0], {})[blue]
    w = weights(cfg)["sortir_defenseur"]
    assert total == 2.0                                      # t = 0 puis t = 40 s : 2 sur une minute, 1,5 en moyenne
    assert total * w["keeper_bump"] < 0.5 * w["goal"]        # 6 < 10 : sous un demi-but (but = 20 depuis « lucidite »)


# --------------------------------------------------------------------------- exercice
def test_dribble_drill_keeper_sometimes_comes_out():
    st = make_state(1, kickoff=True)
    drill = DribbleDrill(seed=7)
    out = 0
    for _ in range(60):
        info = {}
        drill.apply(st, info)
        team = info["drill_team"]
        car = next(c for c in st.cars.values() if c.team_num == team)
        pos, vel = st.ball.position.tolist(), st.ball.linear_velocity.tolist()
        car.wheels_with_contact = (True,) * 4
        assert is_dribbling(car, pos, vel)
        assert keeper_of(st, 1 - team, pos, DEFENDER_MAX_DIST) is not None
        out += keeper_of(st, 1 - team, pos) is None              # au-delà du gardien « dans ses cages » (1 800 uu)
    assert 10 <= out <= 50


def test_dribble_drill_succeeds_on_keeper_bump():
    s = make_state(1, kickoff=True, seed=0)
    s.tick_count = 0
    t = EpisodeTracker()
    t.reset(s, {"start_kind": "drill_dribble", "drill_team": 0, "team_size": 1})
    s.tick_count = 120
    t.step(s)
    s.tick_count = 240
    s.cars["blue-0"].bump_victim_id = "orange-0"
    t.step(s)
    assert t.finish(s, False)["drill_success"] == 1.0
    t.reset(s, {"start_kind": "drill_dribble", "drill_team": 1, "team_size": 1})   # le bump du bleu ne compte pas
    s.tick_count = 360
    t.step(s)
    assert t.finish(s, False)["drill_success"] == 0.0
