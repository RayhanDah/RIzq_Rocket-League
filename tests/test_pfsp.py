"""Self-play priorisé (PFSP) et adversaires « ancres » du pool (`selfplay.pfsp`, `selfplay.anchors`)."""
import json
import math
import random
import shutil
from collections import Counter

import numpy as np
import pytest
import torch

from rlagent.config import load_config
from rlagent.env.actions import N_ACTIONS
from rlagent.env.obs import BALL_PRED_DIM, SELF_DIM, obs_dim, self_dim
from rlagent.models.attention import Actor
from rlagent.selfplay.pool import STATS_FILE, OpponentPool, anchor_name, pfsp_weight
from rlagent.train import build_actor_from_model_cfg, summarize_episodes

MODEL = {"d_model": 16, "n_heads": 2, "n_layers": 1, "ff_mult": 2, "head_hidden": [16]}


def _pool(tmp_path, n_versions=3, **kw):
    pool = OpponentPool(tmp_path / "pool", seed=0, **kw)
    for i in range(n_versions):
        pool.add({}, dict(MODEL), 1000 * (i + 1), 1000.0)
    return pool


def _old_actor(sd: int, seed: int) -> Actor:
    torch.manual_seed(seed)
    return Actor(N_ACTIONS, d_model=16, n_heads=2, n_layers=1, ff_mult=2, head_hidden=(16,), self_dim=sd)


def _save_run_checkpoint(run_dir, actor, step=1000, elo=1500.0, with_self_dim=False):
    """Checkpoint au format train.py ; les anciens runs (≤ s5) n'enregistraient pas model.self_dim."""
    model = dict(MODEL, **({"self_dim": actor.encoder.self_dim} if with_self_dim else {}))
    (run_dir / "checkpoints").mkdir(parents=True)
    name = f"ckpt_{step:013d}.pt"
    torch.save({"actor": actor.state_dict(), "cfg": {"model": model}, "global_step": step, "elo": elo},
               run_dir / "checkpoints" / name)
    (run_dir / "latest.txt").write_text(name, encoding="utf-8")


def _episode(hist_id, goal_team, learner_team=0):
    return {"length_s": 10.0, "goal_team": goal_team, "touches_per_min": 1.0, "air_touches_per_min": 0.0,
            "team_size": 1, "vs_hist": {"learner_team": learner_team, "hist_elo": 1000.0, "hist_id": hist_id}}


def test_default_pool_unchanged(tmp_path):
    """Options désactivées : même tirage uniforme qu'avant (même suite pour une graine), ni fichier ni métrique."""
    pool = _pool(tmp_path)
    ref = random.Random(0)
    assert [pool.sample()["id"] for _ in range(30)] == [ref.choice(pool.entries)["id"] for _ in range(30)]
    pool.record_results([(pool.entries[0]["id"], 0.0)])
    assert not (tmp_path / "pool" / STATS_FILE).exists() and pool.metrics() == {}


def test_pfsp_weights_favor_versions_we_lose_to(tmp_path):
    pool = _pool(tmp_path, pfsp=True, pfsp_uniform=0.3, pfsp_window=50)
    easy, hard, new = [e["id"] for e in pool.entries]
    pool.record_results([(easy, 1.0)] * 30 + [(hard, 0.0)] * 30)
    assert pool.win_prob(easy) > 0.9 and pool.win_prob(hard) < 0.1 and pool.win_prob(new) == 0.5
    w = [pfsp_weight(pool.win_prob(i)) for i in (easy, hard, new)]
    expected = [0.7 * x / sum(w) + 0.3 / 3 for x in w]          # (1 − p)², mélangé à 30 % d'uniforme
    dist = {e["id"]: p for e, p in pool.sampling_distribution()}
    assert [dist[i] for i in (easy, hard, new)] == pytest.approx(expected)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist[hard] > dist[new] > dist[easy] >= 0.1 - 1e-9    # une version battue reste tirable (part uniforme)
    counts = Counter(pool.sample()["id"] for _ in range(6000))
    for i in (easy, hard, new):
        assert counts[i] / 6000 == pytest.approx(dist[i], abs=0.03)
    # score GLISSANT : une version qu'on battait et contre laquelle on se met à perdre redevient prioritaire
    pool.record_results([(easy, 0.0)] * 60)
    assert pool.win_prob(easy) < 0.3
    # nul = ½ ; entropie de la loi de tirage
    pool.record_results([(new, 0.5)] * 10)
    assert pool.win_prob(new) == pytest.approx(0.5)
    probs = [p for _, p in pool.sampling_distribution()]
    assert pool.metrics()["vs_hist/pfsp_entropy"] == pytest.approx(-sum(p * math.log(p) for p in probs))


def test_pfsp_scores_saved_and_reloaded(tmp_path):
    pool = _pool(tmp_path, n_versions=2, pfsp=True)
    a, b = [e["id"] for e in pool.entries]
    pool.record_results([(a, 1.0), (a, 0.5), (b, 0.0)])
    path = tmp_path / "pool" / STATS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stats"][a]["games"] == 2 and data["stats"][b]["l"] == 1.0
    assert not list((tmp_path / "pool").glob("*.tmp"))          # écriture atomique : pas de reste
    again = OpponentPool(tmp_path / "pool", seed=1, pfsp=True)
    assert again.win_prob(a) == pytest.approx(pool.win_prob(a)) and again.win_prob(b) == pytest.approx(pool.win_prob(b))
    again.record_results([(a, 1.0)])
    assert json.loads(path.read_text(encoding="utf-8"))["stats"][a]["games"] == 3
    path.write_text("{abîmé", encoding="utf-8")                  # fichier corrompu : on repart de zéro sans planter
    assert OpponentPool(tmp_path / "pool", seed=1, pfsp=True).stats == {}


def test_anchors_always_drawable_frozen_and_play_via_fit_obs(tmp_path):
    s4 = _old_actor(SELF_DIM, seed=1)                            # sans prédiction de balle ni course
    s6 = _old_actor(SELF_DIM + BALL_PRED_DIM, seed=2)            # avec prédiction, sans course
    _save_run_checkpoint(tmp_path / "runs" / "s4_old", s4)
    (tmp_path / "runs" / "s6_x" / "pool").mkdir(parents=True)
    torch.save({"actor": s6.state_dict(), "model_cfg": dict(MODEL, self_dim=s6.encoder.self_dim),
                "global_step": 2000, "elo": 1600.0}, tmp_path / "runs" / "s6_x" / "pool" / "actor_0000000002000.pt")
    specs = ["runs/s4_old", "runs\\s6_x\\pool\\actor_0000000002000.pt"]
    pool = _pool(tmp_path, recent=1, anchors=specs, anchor_prob=0.4, anchor_root=tmp_path)
    names = [a["name"] for a in pool.anchors]
    assert names == ["s4_old", "s6_x_actor_0000000002000"] and len(pool) == 3
    assert all((tmp_path / "pool" / "anchors" / f"{n}.pt").exists() for n in names)
    assert pool.cache_size >= 1 + 2 + len(names)

    # ancres hors fenêtre `recent`, part anchor_prob ; les vieilles versions (hors fenêtre) ne sont jamais tirées
    dist = {e["id"]: p for e, p in pool.sampling_distribution()}
    latest = pool.entries[-1]["id"]
    assert dist == pytest.approx({latest: 0.6, "anchor:s4_old": 0.2, "anchor:s6_x_actor_0000000002000": 0.2})
    counts = Counter(pool.sample()["id"] for _ in range(4000))
    assert set(counts) == set(dist) and counts["anchor:s4_old"] / 4000 == pytest.approx(0.2, abs=0.03)
    assert anchor_name("anchor:s4_old") == "s4_old" and anchor_name(latest) is None

    # elles jouent dans l'env s7 (bloc « soi » plus long) en retrouvant exactement l'observation qu'elles connaissent
    sd7 = self_dim(True, True)
    obs = torch.randn(7, obs_dim(True, True))
    for anchor, ref in zip(pool.anchors, (s4, s6)):
        actor = pool.get_actor(anchor, build_actor_from_model_cfg)
        sd = ref.encoder.self_dim
        assert actor.encoder.self_dim == sd
        with torch.no_grad():
            assert torch.allclose(actor(obs), ref(torch.cat([obs[:, :sd], obs[:, sd7:]], dim=1)), atol=1e-6)

    # figées : le run source peut disparaître (autre machine, checkpoints tournés)
    shutil.rmtree(tmp_path / "runs")
    again = OpponentPool(tmp_path / "pool", recent=1, anchors=specs, anchor_root=tmp_path)
    with torch.no_grad():
        assert torch.allclose(again.get_actor(again.anchors[0], build_actor_from_model_cfg)(obs),
                              s4(torch.cat([obs[:, :SELF_DIM], obs[:, sd7:]], dim=1)), atol=1e-6)
    with pytest.raises(FileNotFoundError):                       # même nom, autre source introuvable
        OpponentPool(tmp_path / "pool", anchors=["ailleurs/s4_old"], anchor_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        OpponentPool(tmp_path / "pool", anchors=["runs/inexistant"], anchor_root=tmp_path)


def test_anchor_metrics_separate_from_pool_versions(tmp_path):
    eps = [_episode("actor_0000000001000", 0), _episode("actor_0000000001000", None),
           _episode("anchor:s6", 0), _episode("anchor:s6", 1), _episode("anchor:s4", 1, learner_team=1)]
    out = summarize_episodes(eps)
    assert out["vs_hist/episodes"] == 2 and out["vs_hist/win_rate"] == 0.5 and out["vs_hist/loss_rate"] == 0.0
    assert out["vs_anchor/s6/episodes"] == 2 and out["vs_anchor/s6/win_rate"] == 0.5
    assert out["vs_anchor/s6/loss_rate"] == 0.5 and out["vs_anchor/s4/win_rate"] == 1.0


def test_training_pfsp_and_anchors_on_fake_engine(tmp_path):
    """Chaîne complète (profil s7, moteur factice) : PFSP + une ancre au bloc « soi » plus court, métriques,
    scores sauvegardés, puis reprise qui recharge les scores."""
    from rlagent.train import find_latest_checkpoint, train
    old_run = tmp_path / "old_run"
    _save_run_checkpoint(old_run, _old_actor(SELF_DIM + BALL_PRED_DIM, seed=3), with_self_dim=True)
    cfg = load_config(["debug", "s7"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    cfg["ppo"]["batch_schedule"] = [[0, 2000, 1000]]
    cfg["ppo"]["device"] = "cpu"
    cfg["train"].update({"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 4000})
    cfg["selfplay"].update({"pool_every_steps": 2000, "hist_prob": 0.6, "pfsp": True, "pfsp_uniform": 0.3,
                            "anchors": [str(old_run)], "anchor_prob": 0.5})
    cfg["run_name"] = "pytest_pfsp"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(l["learn/policy_loss"]) for l in lines)
    ent = [(l["vs_hist/pfsp_entropy"], l["pool_size"]) for l in lines if "vs_hist/pfsp_entropy" in l]
    assert ent and all(0.0 < h <= math.log(n + 1) + 1e-9 for h, n in ent)   # ≤ ln(versions + 1 ancre)
    assert any("vs_anchor/old_run/win_rate" in l for l in lines)
    assert all("vs_anchor/old_run/p_win" in l for l in lines)
    assert (run_dir / "pool" / "anchors" / "old_run.pt").exists()
    stats = json.loads((run_dir / "pool" / STATS_FILE).read_text(encoding="utf-8"))["stats"]
    assert stats["anchor:old_run"]["games"] > 0 and any(k.startswith("actor_") for k in stats)
    games = sum(s["games"] for s in stats.values())
    cfg["train"]["total_steps"] = 11000
    train(cfg, run_dir, find_latest_checkpoint(run_dir))
    stats2 = json.loads((run_dir / "pool" / STATS_FILE).read_text(encoding="utf-8"))["stats"]
    assert sum(s["games"] for s in stats2.values()) > games      # rechargés puis complétés, pas remis à zéro
