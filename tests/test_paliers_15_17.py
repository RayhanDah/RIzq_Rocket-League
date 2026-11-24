"""Paliers 15-17 de s10 (20/11/2026, décision PO après les BO3 dans le vrai jeu) : kickoffs disputés, démo du dernier
défenseur suivie d'un but, but sur une touche haute. Même méthode que les paliers 8-14 : un changement par palier, lent,
sans régression, plafonné par le garde-fou. PO : aucune faille, et le but / l'encaissement restent dominants."""
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import make_state
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.rewards import DemoGoalTerm, HighGoalTerm, KickoffStallTerm, TeamReward, make_reward
from rlagent.guard import TrainingGuard

ROOT = Path(__file__).resolve().parents[1]
V23037 = "checkpoints/ckpt_0023037887394.pt"
GAP = "ladder/gap/actor_0023037887394"
NEW = ["kickoff_dispute", "demo_but", "but_haut"]
NEXT = ["bump_but", "murs", "double_saut"]          # paliers 18-20 (session 10, tests/test_paliers_18_20.py)
TICK = 8                                          # une action = 8 ticks


@pytest.fixture(scope="module")
def cfg():
    return load_config(["kaggle_tpu", "s10"])


def stage(cfg, name):
    return next(s for s in cfg["curriculum"]["stages"] if s["name"] == name)


def weights(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    return {n: c.stage_weights[c.names.index(n)] for n in c.names}


def changed(a, b):
    return {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)}


def cars(s):
    blue = next(a for a, c in s.cars.items() if c.team_num == 0)
    orange = next(a for a, c in s.cars.items() if c.team_num == 1)
    return blue, orange


def place(car, pos):
    car.physics.position = np.array(pos, dtype=np.float32)
    car.physics.linear_velocity = np.zeros(3, dtype=np.float32)


def tick(s, n=1):
    s.tick_count += TICK * n


def no_touch(s):
    for c in s.cars.values():
        c.ball_touches = 0


# --------------------------------------------------------------------------- configuration
def test_stages_follow_petits_pads_one_change_each(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    i = names.index("petits_pads")
    assert names[i:i + 1 + len(NEW) + len(NEXT)] == ["petits_pads"] + NEW + NEXT
    w = weights(cfg)
    assert changed(w["petits_pads"], w["kickoff_dispute"]) == {"kickoff_stall"}
    assert changed(w["kickoff_dispute"], w["demo_but"]) == {"demo_goal"}
    assert changed(w["demo_but"], w["but_haut"]) == {"high_goal"}


def test_goal_and_concede_stay_dominant(cfg):
    """PO : le but marqué / pris doit rester haut face aux autres termes. Les poids d'avant ne bougent pas, et chaque
    nouveau terme vaut au plus un quart de but."""
    w = weights(cfg)
    last, before = w["but_haut"], w["petits_pads"]
    assert {k: v for k, v in last.items() if k in before} == before          # rien d'autre ne change
    goal = last["goal"]
    assert goal == 20.0 and last["concede"] == 2.0
    assert last["kickoff_stall"] == 2.0 and last["kickoff_stall"] <= 0.1 * goal
    assert last["demo_goal"] == 4.0 and last["demo_goal"] <= 0.2 * goal
    assert last["high_goal"] == 5.0 and last["high_goal"] <= 0.25 * goal


def test_bonuses_are_zero_sum_and_the_stall_hits_both(cfg):
    rc = cfg["rewards"]
    assert {"demo_goal", "high_goal"} <= set(rc["zero_sum_terms"])
    assert "kickoff_stall" not in rc["zero_sum_terms"]          # sinon les deux pénalités s'annuleraient
    assert (rc["kickoff_stall_margin_s"], rc["kickoff_stall_ramp_s"], rc["kickoff_stall_loser"]) == (0.1, 1.0, 0.125)
    # 22/11 (PO, session 10) : fenêtre portée de 3 à 5 s
    assert (rc["demo_goal_full_s"], rc["demo_goal_window_s"]) == (0.75, 5.0) and rc["demo_goal_own_half"] is True
    assert rc["demo_goal_last_defender"] is True
    assert 500.0 < rc["high_goal_min_z"] < 642.775               # au-dessus d'un double saut, sous la barre


def test_new_stages_are_slow_and_gated_against_v23037(cfg):
    pp = stage(cfg, "petits_pads")
    assert pp["advance"] == {"goals_per_min": 2.0, GAP: -25} and pp["min_steps"] >= 500_000_000
    for name in NEW:
        s = stage(cfg, name)
        assert s["blend_steps"] >= 200_000_000
        assert s["min_steps"] >= 500_000_000
        # session 10 : but_haut n'est plus le dernier palier, il passe la main s'il tient face à v23037
        assert s["advance"] == {"goals_per_min": 2.0, GAP: 0 if name == NEW[-1] else -25}


def test_session_9_continues_from_v23037_and_measures_against_it(cfg):
    assert cfg["train"]["rollback_to"] is None
    assert V23037 in cfg["selfplay"]["anchors"]


def test_guard_caps_the_new_terms(cfg):
    g = cfg["train"]["guard"]
    for t in ("kickoff_stall", "demo_goal", "high_goal"):
        assert g["term_max_share"][t] == 0.03
    assert g["fallback_weights"]["kickoff_stall"] == g["fallback_weights"]["high_goal"] == 0.0
    assert g["fallback_weights"]["demo_goal"] == 4.0             # session 10 : retour au poids de la session 9
    assert g["enabled"] and g["max_rollbacks"] == 3 and g["min_goals_ratio"] == 0.6


def test_resume_at_petits_pads_hands_over_to_kickoff_dispute(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    c.load_state_dict({"index": c.names.index("petits_pads"), "stage_start": 20_174_387_960, "prev_weights": None,
                       "ema": {}, "streak": 0})
    msgs = [c.update({"goals_per_min": 2.45, GAP: 0.0}, 23_037_887_394 + i * 400_000) for i in range(c.patience)]
    assert msgs[-1] and c.name == "kickoff_dispute"
    assert c.weights(23_045_887_394 + 100_000_000)["kickoff_stall"] == pytest.approx(1.0, rel=0.05)   # mi-transition
    c2 = SkillCurriculum.from_cfg(cfg)
    c2.load_state_dict({"index": c2.names.index("petits_pads"), "stage_start": 20_174_387_960})
    assert all(c2.update({"goals_per_min": 2.45, GAP: -40.0}, 23_037_887_394 + i * 400_000) is None
               for i in range(3 * c2.patience))                 # 40 Elo sous v23037 : on reste


def test_every_new_term_is_built_and_reported(cfg):
    rf = make_reward(cfg)
    names = {t.name for t in rf.terms}
    assert {"kickoff_stall", "demo_goal", "high_goal"} <= names


# --------------------------------------------------------------------------- kickoff_stall
def kickoff_state(seed=0):
    s = make_state(kickoff=True, seed=seed)
    no_touch(s)
    return s


def run_kickoff(t, s, touch_at=None, teams=(1,)):
    """Rejoue 6 s de kickoff ; `touch_at` (s) : la voiture de chaque équipe de `teams` touche la balle à ce moment."""
    agents = list(s.cars)
    t.reset(agents, s, {"start_kind": "kickoff"})
    total = {a: 0.0 for a in agents}
    when = None
    for i in range(6 * 15):
        tick(s)
        now = (i + 1) * TICK / 120.0
        no_touch(s)
        if touch_at is not None and abs(now - touch_at) < 1e-6:
            for a, c in s.cars.items():
                c.ball_touches = 1 if c.team_num in teams else 0
        r = t.get(agents, s, {})
        if any(r.values()) and when is None:
            when = now
        for a in agents:
            total[a] += r[a]
    return total, when


def test_kickoff_rush_time_by_spawn():
    """t0 = kickoff foncé + 0,1 s : 2,17 s (diagonale), 2,43 s (décalée), 2,70 s (centre), choix du PO."""
    seen = {}
    for seed in range(40):
        s = kickoff_state(seed)
        t = KickoffStallTerm(margin_s=0.1)
        t.reset(list(s.cars), s, {"start_kind": "kickoff"})
        blue = next(c for c in s.cars.values() if c.team_num == 0)
        x = abs(float(blue.physics.position[0]))
        kind = "diagonale" if x > 1000 else ("decalee" if x > 100 else "centre")
        seen[kind] = t.t0
    assert seen == {"diagonale": pytest.approx(2.167, abs=0.01), "decalee": pytest.approx(2.433, abs=0.01),
                    "centre": pytest.approx(2.700, abs=0.01)}


def test_kickoff_stall_full_penalty_once_when_nobody_touches():
    s = kickoff_state()
    t = KickoffStallTerm(margin_s=0.1, ramp_s=1.0, loser=0.125)
    total, when = run_kickoff(t, s)
    assert total == {a: -1.0 for a in s.cars}
    assert when == pytest.approx(t.t1, abs=TICK / 120.0) and t.t1 == pytest.approx(t.t0 + 1.0)


def test_kickoff_stall_ramp_and_lost_race():
    for frac, expected in ((-0.3, 0.0), (0.0, 0.0), (0.5, 0.5), (0.9, 0.9)):
        s = kickoff_state()
        t = KickoffStallTerm(margin_s=0.1, ramp_s=1.0, loser=0.125)
        t.reset(list(s.cars), s, {"start_kind": "kickoff"})
        n = round((t.t0 + frac) * 120.0 / TICK)                # step le plus proche de t0 + frac
        at = n * TICK / 120.0
        total, _ = run_kickoff(t, s, touch_at=at, teams=(1,))    # l'orange touche en premier
        blue, orange = cars(s)
        late = min(max((at - t.t0) / 1.0, 0.0), 1.0)
        assert total[orange] == pytest.approx(-late)
        assert total[blue] == pytest.approx(-late - 0.125)      # le bleu a perdu la course
        assert late == pytest.approx(expected, abs=TICK / 120.0)


def test_kickoff_stall_no_loser_when_both_touch_together():
    s = kickoff_state()
    t = KickoffStallTerm(margin_s=0.1, ramp_s=1.0, loser=0.125)
    t.reset(list(s.cars), s, {"start_kind": "kickoff"})
    total, _ = run_kickoff(t, s, touch_at=round(2.0 * 15) * TICK / 120.0, teams=(0, 1))
    assert total == {a: 0.0 for a in s.cars}


def test_kickoff_stall_is_never_positive_and_ignores_other_starts():
    s = kickoff_state()
    agents = list(s.cars)
    t = KickoffStallTerm()
    for kind in ("random", "drill_air", "aerial", None):
        t.reset(agents, s, {"start_kind": kind} if kind else {})
        for _ in range(60):
            tick(s)
            assert all(v == 0.0 for v in t.get(agents, s, {}).values())
    for at in (1.0, 2.5, 3.0, 4.0, None):
        total, _ = run_kickoff(KickoffStallTerm(), kickoff_state(), touch_at=at)
        assert all(-1.125 - 1e-9 <= v <= 0.0 for v in total.values())


def test_kickoff_stall_through_the_team_reward(cfg):
    """Avec les poids du palier 15 : −2 pour les deux si personne n'y va, −0,25 de plus pour celui qui perd la course."""
    cur = SkillCurriculum.from_cfg(cfg)
    rc = cfg["rewards"]
    w = cur.stage_weights[cur.names.index("kickoff_dispute")]
    for touch_at, team_loss in ((None, (-2.0, -2.0)), (2.0, (-0.25, 0.0))):
        rf = TeamReward(make_reward(cfg).terms, rc["weights"], zero_sum=rc["zero_sum"],
                        zero_sum_terms=rc["zero_sum_terms"], stage_weights=cur.stage_weights)
        rf.terms = [x for x in rf.terms if x.name == "kickoff_stall"]
        s = kickoff_state()
        agents = list(s.cars)
        info = {"start_kind": "kickoff", "reward_weights": w}
        rf.reset(agents, s, info)
        totals = {a: 0.0 for a in agents}
        for i in range(6 * 15):
            tick(s)
            no_touch(s)
            if touch_at is not None and i + 1 == round(touch_at * 15):
                for c in s.cars.values():
                    c.ball_touches = 1 if c.team_num == 1 else 0
            r = rf.get_rewards(agents, s, {a: False for a in agents}, {a: False for a in agents}, info)
            for a in agents:
                totals[a] += r[a]
        blue, orange = cars(s)
        assert (totals[blue], totals[orange]) == (pytest.approx(team_loss[0]), pytest.approx(team_loss[1]))


# --------------------------------------------------------------------------- demo_goal
def demo(s, author, victim):
    s.cars[author].bump_victim_id = victim
    s.cars[victim].demo_respawn_timer = 3.0


def respawn(s, victim):
    s.cars[victim].demo_respawn_timer = 0.0
    for c in s.cars.values():
        c.bump_victim_id = None


def goal(s, team):
    """But de `team` : scoring_team se déduit de la balle (dans le but orange → le bleu marque)."""
    s.goal_scored = True
    s.ball.position = np.array([0.0, 5300.0 if team == 0 else -5300.0, 300.0], dtype=np.float32)


def demo_then_goal(delay_steps, **kw):
    """Démo de l'orange par le bleu, puis but bleu `delay_steps` actions plus tard ; renvoie le bonus du bleu."""
    s = make_state(kickoff=True)
    no_touch(s)
    blue, orange = cars(s)
    agents = list(s.cars)
    t = DemoGoalTerm(**kw)
    t.reset(agents, s, {})
    tick(s)
    demo(s, blue, orange)
    assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}     # rien à la démo elle-même (c'est `demo`)
    for _ in range(delay_steps - 1):
        tick(s)
        assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}
    tick(s)
    goal(s, 0)
    r = t.get(agents, s, {})
    assert r[orange] == 0.0
    return r[blue]


def test_demo_goal_full_until_0_75_s_then_down_to_0_at_3_s():
    """Version du PO : 4 (plein) si le but arrive dans les 0,75 s, puis linéaire jusqu'à 0 à 3 s."""
    for steps, expected in ((1, 1.0), (11, 1.0), (12, 1.0 - (0.8 - 0.75) / 2.25), (28, (3.0 - 28 / 15) / 2.25),
                            (40, (3.0 - 40 / 15) / 2.25), (45, 0.0), (60, 0.0)):
        assert demo_then_goal(steps) == pytest.approx(expected, abs=1e-6), steps
    assert demo_then_goal(28) == pytest.approx(0.5, abs=0.03)                     # ~1,9 s : moitié


def test_demo_goal_nothing_after_the_window_or_for_the_other_team():
    for wait_s, scorer in ((3.0, 0), (1.0, 1)):
        s = make_state(kickoff=True)
        no_touch(s)
        blue, orange = cars(s)
        agents = list(s.cars)
        t = DemoGoalTerm()
        t.reset(agents, s, {})
        tick(s)
        demo(s, blue, orange)
        t.get(agents, s, {})
        for _ in range(int(wait_s * 15)):
            tick(s)
            t.get(agents, s, {})
        goal(s, scorer)
        assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}


def test_demo_goal_only_for_the_last_defender_in_2v2():
    s = make_state(team_size=2, kickoff=True)
    no_touch(s)
    blues = [a for a, c in s.cars.items() if c.team_num == 0]
    oranges = [a for a, c in s.cars.items() if c.team_num == 1]
    keeper, striker = oranges
    place(s.cars[keeper], [0.0, 4800.0, 17.0])                  # devant le but orange
    place(s.cars[striker], [0.0, 1000.0, 17.0])
    for only, victim, paid in ((True, striker, 0.0), (True, keeper, 1.0), (False, striker, 1.0)):
        for c in s.cars.values():
            c.demo_respawn_timer, c.bump_victim_id = 0.0, None
        s.goal_scored = False
        agents = list(s.cars)
        t = DemoGoalTerm(last_defender_only=only)
        t.reset(agents, s, {})
        tick(s)
        demo(s, blues[0], victim)
        t.get(agents, s, {})
        tick(s)
        goal(s, 0)
        assert t.get(agents, s, {})[blues[0]] == paid, (only, victim)


def test_demo_goal_not_for_an_attacker_demolished_in_our_half():
    """PO : démolir un attaquant puis marquer en contre ne paie pas (en 1v1, il est toujours le « dernier défenseur »)."""
    for own_half_only, paid in ((True, 0.0), (False, (3.0 - 15 / 15) / 2.25)):
        s = make_state(kickoff=True)
        no_touch(s)
        blue, orange = cars(s)
        place(s.cars[orange], [0.0, -3000.0, 17.0])              # l'orange attaque, dans le camp bleu
        agents = list(s.cars)
        t = DemoGoalTerm(last_defender_only=True, own_half_only=own_half_only)
        t.reset(agents, s, {})
        tick(s)
        demo(s, blue, orange)
        t.get(agents, s, {})
        for _ in range(15):                                      # contre-attaque éclair : but 1 s plus tard
            tick(s)
            t.get(agents, s, {})
        goal(s, 0)
        assert t.get(agents, s, {})[blue] == pytest.approx(paid), own_half_only


def test_demo_goal_is_zero_sum_through_the_team_reward(cfg):
    cur = SkillCurriculum.from_cfg(cfg)
    rc = cfg["rewards"]
    rf = TeamReward([DemoGoalTerm()], rc["weights"], zero_sum=rc["zero_sum"],
                    zero_sum_terms=rc["zero_sum_terms"], stage_weights=cur.stage_weights)
    s = make_state(kickoff=True)
    no_touch(s)
    blue, orange = cars(s)
    agents = list(s.cars)
    info = {"reward_weights": cur.stage_weights[cur.names.index("demo_but")]}
    rf.reset(agents, s, info)
    tick(s)
    demo(s, blue, orange)
    rf.get_rewards(agents, s, {a: False for a in agents}, {a: False for a in agents}, info)
    tick(s)
    goal(s, 0)
    r = rf.get_rewards(agents, s, {a: True for a in agents}, {a: False for a in agents}, info)
    assert r[blue] == pytest.approx(4.0) and r[orange] == pytest.approx(-4.0)


# --------------------------------------------------------------------------- high_goal
def touch(s, agent, z):
    no_touch(s)
    s.cars[agent].ball_touches = 1
    s.ball.position = np.array([0.0, 4000.0, z], dtype=np.float32)


def test_high_goal_pays_the_scorer_of_a_high_last_touch():
    for z, paid in ((700.0, 1.0), (600.0, 1.0), (550.0, 0.0), (300.0, 0.0)):
        s = make_state(kickoff=True)
        blue, orange = cars(s)
        agents = list(s.cars)
        t = HighGoalTerm(600.0)
        t.reset(agents, s, {})
        tick(s)
        touch(s, blue, z)
        assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}
        for _ in range(10):                                      # la balle file au but
            tick(s)
            no_touch(s)
            t.get(agents, s, {})
        goal(s, 0)
        assert t.get(agents, s, {}) == {blue: paid, orange: 0.0}, z


def test_high_goal_nothing_if_the_last_touch_is_low_or_from_the_other_team():
    cases = [((0, 700.0), (0, 100.0)),     # touche haute puis reprise au sol par le même : dernière touche basse
             ((0, 700.0), (1, 100.0)),     # touche haute puis déviation adverse : but « contre son camp »
             ((1, 900.0), None)]           # touche haute adverse, but bleu
    for first, second in cases:
        s = make_state(kickoff=True)
        blue, orange = cars(s)
        agents = list(s.cars)
        who = {0: blue, 1: orange}
        t = HighGoalTerm(600.0)
        t.reset(agents, s, {})
        for step in (first, second):
            if step is None:
                continue
            tick(s)
            touch(s, who[step[0]], step[1])
            t.get(agents, s, {})
        tick(s)
        no_touch(s)
        goal(s, 0)
        assert t.get(agents, s, {}) == {blue: 0.0, orange: 0.0}, (first, second)


def test_high_goal_ambiguous_when_both_teams_touch_together():
    s = make_state(kickoff=True)
    blue, orange = cars(s)
    agents = list(s.cars)
    t = HighGoalTerm(600.0)
    t.reset(agents, s, {})
    tick(s)
    s.cars[blue].ball_touches = s.cars[orange].ball_touches = 1
    s.ball.position = np.array([0.0, 4000.0, 800.0], dtype=np.float32)
    t.get(agents, s, {})
    tick(s)
    no_touch(s)
    goal(s, 0)
    assert t.get(agents, s, {})[blue] == 0.0


def test_high_goal_is_zero_sum_through_the_team_reward(cfg):
    cur = SkillCurriculum.from_cfg(cfg)
    rc = cfg["rewards"]
    rf = TeamReward([HighGoalTerm(600.0)], rc["weights"], zero_sum=rc["zero_sum"],
                    zero_sum_terms=rc["zero_sum_terms"], stage_weights=cur.stage_weights)
    s = make_state(kickoff=True)
    blue, orange = cars(s)
    agents = list(s.cars)
    info = {"reward_weights": cur.stage_weights[cur.names.index("but_haut")]}
    rf.reset(agents, s, info)
    tick(s)
    touch(s, blue, 800.0)
    rf.get_rewards(agents, s, {a: False for a in agents}, {a: False for a in agents}, info)
    tick(s)
    no_touch(s)
    goal(s, 0)
    r = rf.get_rewards(agents, s, {a: True for a in agents}, {a: False for a in agents}, info)
    assert r[blue] == pytest.approx(5.0) and r[orange] == pytest.approx(-5.0)


# --------------------------------------------------------------------------- métriques
def test_episode_tracker_counts_stalls_demo_goals_and_high_goals(cfg):
    from rlagent.metrics.episode import EpisodeTracker
    from rlagent.train import summarize_episodes
    rc = cfg["rewards"]
    eps = []
    # 1) kickoff bloqué, sans but
    s = kickoff_state()
    tr = EpisodeTracker(rewards_cfg=rc)
    tr.reset(s, {"start_kind": "kickoff", "team_size": 1})
    for _ in range(4 * 15):
        tick(s)
        tr.step(s)
    eps.append(tr.finish(s, False))
    # 2) kickoff touché à 2 s, démo du gardien puis but haut 1 s plus tard
    s = kickoff_state()
    blue, orange = cars(s)
    tr = EpisodeTracker(rewards_cfg=rc)
    tr.reset(s, {"start_kind": "kickoff", "team_size": 1})
    for i in range(30):
        tick(s)
        s.cars[blue].ball_touches = 1 if i == 29 else 0
        tr.step(s)
    tick(s)
    no_touch(s)
    demo(s, blue, orange)
    tr.step(s)
    tick(s)
    touch(s, blue, 750.0)
    tr.step(s)
    tick(s)
    no_touch(s)
    goal(s, 0)
    tr.step(s)
    eps.append(tr.finish(s, True))
    assert eps[0]["kickoff_stalled"] is True and eps[1]["kickoff_stalled"] is False
    assert eps[1]["kickoff_penalty"] == 0.0                     # touché à 2 s : avant t0
    assert eps[1]["kickoff_first_touch_s"] == pytest.approx(2.0, abs=0.07)
    assert eps[1]["demo_goal"] == 1 and eps[1]["high_goal"] == 1 and eps[0]["demo_goal"] == 0
    out = summarize_episodes(eps)
    assert out["kickoff/stall_rate"] == pytest.approx(0.5)
    assert out["kickoff/penalty"] == pytest.approx((eps[0]["kickoff_penalty"] + 0.0) / 2)
    assert out["goals/demo_goal_share"] == 1.0 and out["goals/high_goal_share"] == 1.0
    assert out["demo_goals_per_min"] > 0.0 and out["high_goals_per_min"] > 0.0


# --------------------------------------------------------------------------- ancre v23037
def test_checkpoint_anchor_is_named_like_a_pool_version(tmp_path):
    import torch
    from rlagent.selfplay.pool import OpponentPool
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ck = {"actor": {"w": torch.zeros(2)}, "cfg": {"model": {"type": "mlp"}}, "global_step": 123, "elo": 1500.0}
    torch.save(ck, run / "checkpoints" / "ckpt_0000000123.pt")
    pool = OpponentPool(run / "pool", anchors=["checkpoints/ckpt_0000000123.pt"], anchor_prob=0.5)
    assert [a["name"] for a in pool.anchors] == ["actor_0000000123"]
    frozen = torch.load(run / "pool" / "anchors" / "actor_0000000123.pt", weights_only=False)
    assert frozen["global_step"] == 123 and frozen["model_cfg"] == {"type": "mlp"}
    (run / "checkpoints" / "ckpt_0000000123.pt").unlink()       # rotation des checkpoints : la copie figée reste
    again = OpponentPool(run / "pool", anchors=["checkpoints/ckpt_0000000123.pt"], anchor_prob=0.5)
    assert [a["name"] for a in again.anchors] == ["actor_0000000123"]


def test_v23037_checkpoint_anchor_matches_the_bot(tmp_path):
    """Le checkpoint de 23,04 G donne bien l'acteur de bot/model.pt (v23037, choisi par le PO), si le run est là."""
    import shutil

    import torch
    from rlagent.selfplay.pool import OpponentPool
    src = ROOT / "runs/s10_ppo_kaggle" / V23037
    bot = ROOT / "runs/s10_ppo_kaggle/actor_0023037887394.pt"
    if not src.exists() or not bot.exists():
        pytest.skip("run s10_ppo_kaggle absent")
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    shutil.copy(src, run / V23037)
    pool = OpponentPool(run / "pool", anchors=[V23037], anchor_prob=0.5)
    frozen = torch.load(pool.anchors[0]["path"], weights_only=False)
    ref = torch.load(bot, weights_only=False)
    assert frozen["global_step"] == 23_037_887_394
    assert all(torch.equal(frozen["actor"][k], ref["actor"][k]) for k in ref["actor"])


# --------------------------------------------------------------------------- garde-fou rejoué
def test_guard_replayed_on_sessions_7_and_8_stays_quiet():
    """PO : le garde-fou doit rester silencieux sur une session saine. Rejoué avec la configuration de la session 9 sur
    les métriques réelles des sessions 7 et 8 (13,76 → 23,04 G), jalons tous les 250 M steps comme à l'entraînement."""
    path = ROOT / "runs/s10_ppo_kaggle/metrics.jsonl"
    if not path.exists():
        pytest.skip("run s10_ppo_kaggle absent")
    import tempfile

    import yaml
    cfg = yaml.safe_load((ROOT / "configs/s10.yaml").read_text(encoding="utf-8"))
    every = int(cfg["train"]["milestone_every_steps"])
    with tempfile.TemporaryDirectory() as d:
        g = TrainingGuard(cfg["train"]["guard"], Path(d))
        nxt, n = None, 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            step = r["global_step"]
            if step < 13.762e9:
                continue
            if step > 23.05e9:                     # sessions 7 et 8 seulement (la session 10 a dérivé pour de vrai)
                break
            n += 1
            fired = g.update(r)
            assert fired is None, (step, fired)
            if nxt is None:
                nxt = (step // every + 1) * every
            if step >= nxt:
                g.note_milestone(step)
                nxt += every
        assert n > 20_000 and len(g.state["milestones"]) >= 30
