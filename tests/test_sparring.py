"""Session 11 (23/11/2026) : sparring contre plusieurs bots (Nexto, GalactusV2, Kira, Rain), chacun avec sa part des
épisodes et son beta ; correctif du plantage après un retour en arrière du garde-fou (version du pool déplacée)."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from rlagent.algo.collector import EnvSlot, Rollout
from rlagent.algo.worker import (SPARRING_BOTS, EpisodeStart, StepResult, bot_action_code, bot_of_code,
                                 nexto_action_code, pick_sparring, sparring_entry, sparring_settings)
from rlagent.config import load_config
from rlagent.selfplay.pool import version_path
from rlagent.train import _bot_score, sparring_promotion

OPP = Path(__file__).resolve().parents[1] / "dist" / "opponents"
MESHES = Path(__file__).resolve().parents[1] / "collision_meshes" / "soccar"


@pytest.fixture(scope="module")
def cfg():
    return load_config(["kaggle_tpu", "s10"])


def test_profile_spars_against_four_bots_without_imitation(cfg):
    sp = sparring_settings(cfg)
    assert set(sp) == {"nexto", "galactus", "kira", "rain"}
    assert all(0.0 < b["prob"] and 0.0 < b["beta"] <= 1.0 for b in sp.values())
    assert 0.10 <= sum(b["prob"] for b in sp.values()) <= 0.25        # le self-play reste l'essentiel
    assert not cfg["train"]["kickstart"]["enabled"]
    assert "nexto" not in cfg["selfplay"] or not cfg["selfplay"]["nexto"]
    assert sparring_settings({**cfg, "env": {**cfg["env"], "engine": "fake"}}) == {}


def test_legacy_nexto_key_and_unknown_bot():
    legacy = {"env": {}, "selfplay": {"nexto": {"prob": 0.1, "beta_start": 0.5}}}
    assert sparring_settings(legacy) == {"nexto": {"prob": 0.1, "beta": 0.5}}
    assert sparring_promotion(legacy["selfplay"], "nexto") == (300, 0.3, 1.0)
    with pytest.raises(ValueError):
        sparring_settings({"env": {}, "selfplay": {"sparring": {"bots": {"pulsar": {"prob": 0.1}}}}})
    off = {"env": {}, "selfplay": {"sparring": {"bots": {"kira": {"prob": 0.0}}}}}
    assert sparring_settings(off) == {}


def test_promotion_settings_per_bot():
    sp = {"sparring": {"promote_window": 200, "promote_score": 0.3, "promote_step": 0.25,
                       "bots": {"nexto": {"prob": 0.05}, "kira": {"prob": 0.03, "promote_score": 0.2}}}}
    assert sparring_promotion(sp, "nexto") == (200, 0.3, 0.25)
    assert sparring_promotion(sp, "kira") == (200, 0.2, 0.25)


def test_action_codes_roundtrip_for_every_bot():
    assert nexto_action_code(0.5) == -51                  # code de la session 10 inchangé pour Nexto
    for bot in SPARRING_BOTS:
        for beta in (0.0, 0.3, 0.5, 0.75, 1.0):
            code = bot_action_code(bot, beta)
            assert code < 0 and bot_of_code(code) == (bot, pytest.approx(beta))


def test_pick_sparring_follows_each_share():
    sp = {"nexto": {"prob": 0.06, "beta": 0.5}, "galactus": {"prob": 0.04, "beta": 0.3}}
    rng = random.Random(0)
    n = 100_000
    picks = [pick_sparring(sp, rng) for _ in range(n)]
    assert picks.count("nexto") / n == pytest.approx(0.06, abs=0.005)
    assert picks.count("galactus") / n == pytest.approx(0.04, abs=0.005)
    assert picks.count(None) / n == pytest.approx(0.90, abs=0.005)
    assert sparring_entry("galactus", sp) == {"id": "galactus", "kind": "bot", "bot": "galactus", "beta": 0.3,
                                              "elo": 0.0}


def test_rollout_sends_each_bot_its_code_and_tags_episodes():
    class Actor(torch.nn.Module):
        def act(self, x):
            n = x.shape[0]
            return torch.full((n,), 3, dtype=torch.long), torch.zeros(n)

    obs = np.zeros((2, 5), dtype=np.float32)
    start = EpisodeStart(["blue-0", "orange-0"], np.array([0, 1], dtype=np.int8), obs, None)
    sp = {"kira": {"prob": 1.0, "beta": 0.4}}

    def new_slot(st):
        slot = EnvSlot(episode=st)
        slot.hist_team, slot.hist_entry = 0, sparring_entry("kira", sp)
        return slot
    r = Rollout([start], new_slot)
    a = r.act([0], Actor(), torch.device("cpu"), lambda e: None)
    assert a[0].tolist() == [bot_action_code("kira", 0.4), 3]
    r.record([0], [StepResult(np.zeros(2, np.float32), True, True, obs, {"goal_team": 1}, start)])
    _, eps = r.cut()
    assert eps[0]["vs_bot"] == {"bot": "kira", "learner_team": 1, "beta": 0.4}
    assert _bot_score(eps[0]) == 1.0 and "vs_hist" not in eps[0]


def test_version_moved_by_a_rollback_is_still_found(tmp_path):
    pool = tmp_path / "pool"
    (pool / "rejected").mkdir(parents=True)
    f = pool / "rejected" / "actor_0027505164789.pt"
    f.write_bytes(b"x")
    assert version_path(pool / "actor_0027505164789.pt") == f
    kept = pool / "actor_0027250111855.pt"
    kept.write_bytes(b"y")
    assert version_path(kept) == kept
    assert version_path(pool / "absent.pt") == pool / "absent.pt"       # introuvable partout : erreur au chargement


@pytest.mark.skipif(not MESHES.is_dir() or not (OPP / "Kira").is_dir() or not (OPP / "GalactusV2").is_dir()
                    or not (OPP / "RainStream").is_dir() or not (OPP / "NectoFamily").is_dir(),
                    reason="collision meshes ou poids des bots absents")
def test_env_runner_plays_every_sparring_bot(cfg):
    """Vrai RocketSim : chaque bot pilote l'orange quelques steps, avec son beta."""
    from rlagent.algo.worker import EnvRunner
    r = EnvRunner(cfg, seed=5)
    cur = r.current
    for bot, beta in (("nexto", 0.5), ("galactus", 0.3), ("kira", 1.0), ("rain", 0.5)):
        for _ in range(10):
            acts = np.array([5 if t == 0 else bot_action_code(bot, beta) for t in cur.teams])
            cur = r.step(acts).next
        assert r.bots[bot].beta == beta
    assert set(r.bots) == {"nexto", "galactus", "kira", "rain"}


def test_milestones_of_an_undone_drift_are_set_aside(tmp_path):
    from rlagent.train import list_milestones, pick_milestone, reject_milestones
    ms = tmp_path / "checkpoints" / "milestones"
    ms.mkdir(parents=True)
    for st in (27_000_228_446, 27_250_111_855, 27_500_304_526):
        (ms / f"ckpt_{st:013d}.pt").write_bytes(b"x")
    # session 10 : retour de 27,51 G au jalon 27,25 G, enregistré dans guard.json
    moved = reject_milestones(tmp_path, [(27_250_111_855, 27_514_914_029)])
    assert moved == ["ckpt_0027500304526.pt"] and (ms / "rejected" / "ckpt_0027500304526.pt").exists()
    assert [s for s, _ in list_milestones(tmp_path)] == [27_000_228_446, 27_250_111_855]
    assert pick_milestone(tmp_path, None, 28e9)[0] == 27_250_111_855
