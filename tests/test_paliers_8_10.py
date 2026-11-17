"""Paliers 8-10 de s10 (14/11/2026, décision PO) : kickoff, démos puis aériennes, introduits lentement, un par un,
et un garde-fou qui voit les dérives lentes de la session 5."""
import pytest

from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.guard import TrainingGuard


@pytest.fixture(scope="module")
def cfg():
    return load_config(["kaggle_tpu", "s10"])


def stage(cfg, name):
    return next(s for s in cfg["curriculum"]["stages"] if s["name"] == name)


def weights_of(cfg, name):
    c = SkillCurriculum.from_cfg(cfg)
    return c.stage_weights[c.names.index(name)]


# ---------------------------------------------------------------- profil
def test_one_change_per_stage_after_lucidite(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    i = names.index("lucidite")
    assert names[i:i + 4] == ["lucidite", "kickoff", "demo", "aerien"]
    luc, ko, demo, air = (weights_of(cfg, n) for n in names[i:i + 4])
    changed = lambda a, b: {k for k in set(a) | set(b) if a.get(k, 0.0) != b.get(k, 0.0)}
    assert changed(luc, ko) == {"kickoff_win", "kickoff_rush", "concede"}     # concede 5 → 2 : décision PO
    assert changed(ko, demo) == {"demo"}
    assert changed(demo, air) == {"aerial_touch"}


def test_new_weights_stay_small_next_to_a_goal(cfg):
    air = weights_of(cfg, "aerien")
    goal = air["goal"]
    assert air["kickoff_win"] <= 0.3 * goal             # ±1 une fois par kickoff
    assert air["kickoff_rush"] * 4.0 * 15 <= 0.3 * goal  # 4 s de rush à pleine vitesse au plus
    assert air["demo"] <= 0.5 * goal
    assert air["aerial_touch"] <= 0.25 * 4.0             # le quart du poids des dérives de la session 5
    assert air["concede"] == 2.0                         # un duel à 50/50 n'est plus perdant d'avance (+20 / −22)


def test_new_terms_cannot_be_farmed_by_both_sides_at_once(cfg):
    zs = set(cfg["rewards"]["zero_sum_terms"])
    assert {"demo", "aerial_touch"} <= zs                # kickoff_win est antisymétrique par construction


def test_lucidite_hands_over_and_new_stages_are_slow_and_gated(cfg):
    assert stage(cfg, "lucidite")["advance"] == {"goals_per_min": 0}
    for name in ("kickoff", "demo"):
        s = stage(cfg, name)
        assert s["blend_steps"] >= 100_000_000 and s["min_steps"] >= 500_000_000
        assert s["advance"]["goals_per_min"] >= 2.0 and "ladder/gap/actor_0013535043833" in s["advance"]
    assert stage(cfg, "kickoff")["advance"]["vs_hist/kickoff_win_rate"] >= 0.5     # kickoffs de nouveau disputés
    assert stage(cfg, "demo")["blend_steps"] >= 200_000_000
    assert stage(cfg, "aerien")["blend_steps"] >= 300_000_000


def test_stage_blend_uses_its_own_duration(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    c.load_state_dict({"index": c.names.index("kickoff"), "stage_start": 1_000_000_000,
                       "prev_weights": weights_of(cfg, "lucidite"), "ema": {}, "streak": 0})
    assert c.blend(1_050_000_000) == pytest.approx(0.5)  # 100 M de transition
    assert c.weights(1_050_000_000)["kickoff_win"] == pytest.approx(4.5)
    c.load_state_dict({"index": c.names.index("demo"), "stage_start": 1_000_000_000,
                       "prev_weights": weights_of(cfg, "kickoff"), "ema": {}, "streak": 0})
    assert c.blend(1_100_000_000) == pytest.approx(0.5)  # 200 M, pas les 100 M par défaut


def test_v13600_stays_an_anchor(cfg):
    assert "pool/actor_0013600266923.pt" in cfg["selfplay"]["anchors"]       # point de départ de la session 7


def test_guard_settings(cfg):
    g = cfg["train"]["guard"]
    assert g["min_goals_ratio"] == 0.6 and g["reenable_terms"] == ["aerial_touch"]
    assert {"kickoff_win", "kickoff_rush", "demo", "aerial_touch"} <= set(g["term_max_share"])
    assert max(g["term_max_share"].values()) <= 0.08
    fb = g["fallback_weights"]
    assert [fb[k] for k in ("kickoff_win", "kickoff_rush", "demo")] == [3.0, 0.05, 5.3]     # poids du palier 7


# ---------------------------------------------------------------- garde-fou
def guard(tmp_path, **kw):
    g = {"enabled": True, "patience": 3, "warmup_iters": 0, "max_rollbacks": 3, **kw}
    return TrainingGuard(g, tmp_path)


def metrics(step, goals=2.5, **shares):
    base = {"reward_abs/goal": 1.0, **{f"reward_abs/{k}": v for k, v in shares.items()}}
    return {"global_step": step, "goals_per_min": goals, **base}


def test_term_cap_fires_below_the_general_threshold(tmp_path):
    g = guard(tmp_path, term_max_share={"aerial_touch": 0.05}, ema=1.0)
    fired = None
    for i in range(5):                                   # ~6 % du signal : sous 25 %, au-dessus de 5 %
        fired = fired or g.update(metrics(i, aerial_touch=0.065))
    assert fired and fired["term"] == "aerial_touch"


def test_goals_drop_fires_against_the_best_recent_milestone(tmp_path):
    g = guard(tmp_path, min_goals_ratio=0.6, ema=1.0)
    for step, gpm in ((0, 2.4), (1, 2.0), (2, 1.6)):     # dérive lente : chaque jalon un peu plus bas
        g.update(metrics(step, goals=gpm))
        g.note_milestone(step)
    assert g.goals_reference() == pytest.approx(2.4)
    fired = None
    for i in range(3, 8):
        fired = fired or g.update(metrics(i, goals=1.3))  # 54 % de 2,4, mais 81 % du dernier jalon
    assert fired and "buts" in fired["reason"]
    g.restart()
    assert g.goals_ema is None


def test_no_goals_signal_without_milestone(tmp_path):
    g = guard(tmp_path, min_goals_ratio=0.6, ema=1.0)
    assert all(g.update(metrics(i, goals=0.1)) is None for i in range(6))


def test_fallback_weight_instead_of_zero(tmp_path):
    g = guard(tmp_path, fallback_weights={"demo": 5.3})
    g.record_rollback(10, 5, "test", "demo", 0)
    g.record_rollback(11, 5, "test", "aerial_touch", 0)
    w = g.apply({"demo": 8.0, "aerial_touch": 1.0, "goal": 20.0})
    assert w == {"demo": 5.3, "aerial_touch": 0.0, "goal": 20.0}
    assert g.apply({"demo": 2.0})["demo"] == 2.0        # jamais au-dessus du poids du palier


def test_reenable_happens_once_per_run(tmp_path):
    g = guard(tmp_path)
    g.record_rollback(10, 5, "session 5", "aerial_touch", 0)
    g2 = guard(tmp_path, reenable_terms=["aerial_touch"])
    assert g2.reenabled_now == ["aerial_touch"] and "aerial_touch" not in g2.disabled_terms
    g2.record_rollback(20, 15, "session 7", "aerial_touch", 0)   # recoupé par une nouvelle dérive
    g3 = guard(tmp_path, reenable_terms=["aerial_touch"])
    assert g3.reenabled_now == [] and "aerial_touch" in g3.disabled_terms


def test_culprit_looks_at_new_terms_first(tmp_path):
    g = guard(tmp_path, term_max_share={"demo": 0.08}, ema=1.0)
    g.update(metrics(0, demo=0.03, strong_touch=0.30))   # demo 2,3 % (×2,3), strong_touch 23 % (×1,2)
    ref = {"demo": 0.01, "strong_touch": 0.19}
    assert g.culprit(ref) == "demo"
