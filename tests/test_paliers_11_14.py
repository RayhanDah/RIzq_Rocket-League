"""Paliers 11-14 de s10 (17/11/2026, décision PO après 4 matchs contre Necto) : duels aériens, arrêts, bumps, boost.
Même méthode que les paliers 8-10 : un changement par palier, lent, sans régression, plafonné par le garde-fou.
Poids volontairement modestes (PO : pas de boucle qui ferait perdre la session)."""
import pytest

from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum

V18160 = "pool/actor_0018160005774.pt"
NEW = ["duel_aerien", "arrets", "bump", "boost"]


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


def test_stages_follow_aerien_one_change_each(cfg):
    names = [s["name"] for s in cfg["curriculum"]["stages"]]
    assert names[names.index("aerien"):] == ["aerien"] + NEW
    w = weights(cfg)
    assert changed(w["aerien"], w["duel_aerien"]) == {"aerial_touch"}
    assert changed(w["duel_aerien"], w["arrets"]) == {"save"}
    assert changed(w["arrets"], w["bump"]) == {"bump"}
    assert changed(w["bump"], w["boost"]) == {"boost_gain"}


def test_new_weights_stay_modest(cfg):
    w = weights(cfg)
    last, before = w["boost"], w["aerien"]
    goal = last["goal"]
    assert last["aerial_touch"] <= 2.0 * before["aerial_touch"]       # ×2 au plus, sous les 4 de la session 5
    assert last["save"] * 0.5 <= 0.1 * goal                           # un gros arrêt : 10 % d'un but au plus
    assert last["bump"] <= 1.5 * before["bump"] and last["bump"] <= 0.1 * goal
    assert last["boost_gain"] < before["boost_gain"]


def test_duels_are_zero_sum_and_saves_guarded(cfg):
    rc = cfg["rewards"]
    assert {"aerial_touch", "save", "bump"} <= set(rc["zero_sum_terms"])
    assert rc["save_opp_only"] is True


def test_aerien_hands_over_and_new_stages_are_slow_and_gated(cfg):
    assert stage(cfg, "aerien")["advance"] == {"goals_per_min": 0}
    for name in NEW:
        s = stage(cfg, name)
        assert s["blend_steps"] >= 200_000_000
        if name != NEW[-1]:
            assert s["min_steps"] >= 500_000_000 and s["advance"]["goals_per_min"] >= 2.0
            assert s["advance"]["ladder/gap/actor_0018160005774"] >= -25
    assert "advance" not in stage(cfg, NEW[-1])


def test_session_8_continues_without_rollback_and_measures_against_v18160(cfg):
    assert cfg["train"]["rollback_to"] is None
    assert V18160 in cfg["selfplay"]["anchors"]


def test_guard_caps_the_new_terms(cfg):
    g = cfg["train"]["guard"]
    assert {"aerial_touch", "save", "bump"} <= set(g["term_max_share"])
    assert max(g["term_max_share"].values()) <= 0.08
    assert g["fallback_weights"]["aerial_touch"] == 1.0 and g["fallback_weights"]["bump"] == 1.3


def test_aerial_defense_drill_doubled(cfg):
    sw = cfg["env"]["start_weights"]
    assert sw["drill_aerial_defense"] == 0.10 and sum(sw.values()) == pytest.approx(1.0)


def test_resume_from_session_7_state_lands_on_aerien_then_duel(cfg):
    c = SkillCurriculum.from_cfg(cfg)
    c.load_state_dict({"index": c.names.index("aerien"), "stage_start": 15_066_464_417, "prev_weights": None,
                       "ema": {}, "streak": 0})
    msgs = [c.update({"goals_per_min": 2.5}, 18_542_362_941 + i * 400_000) for i in range(c.patience)]
    assert msgs[-1] and c.name == "duel_aerien"
