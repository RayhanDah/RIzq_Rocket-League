"""Chaîne complète sur le moteur factice (sans RocketSim) : protocole worker, collecteur, entraînement court.

Ces tests ne remplacent pas scripts/check_install.py (vrai RocketSim) mais valident toute la plomberie.
"""
import multiprocessing as mp
import threading

import numpy as np
import pytest

from rlagent.algo.worker import EpisodeStart, StepResult, worker_main
from rlagent.config import load_config
from rlagent.env.obs import OBS_DIM


def _fake_cfg(**over):
    cfg = load_config(["debug"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["format_weights"] = {1: 0.5, 2: 0.3, 3: 0.2}
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def test_worker_protocol_runs_episodes():
    parent, child = mp.Pipe()
    th = threading.Thread(target=worker_main, args=(child, _fake_cfg(), 0, 0), daemon=True)
    th.start()
    status, start = parent.recv()
    assert status == "ok", start
    assert isinstance(start, EpisodeStart) and start.obs.shape[1] == OBS_DIM
    cur = start
    episodes = 0
    rng = np.random.default_rng(0)
    for _ in range(2000):
        parent.send(("step", rng.integers(0, 90, size=len(cur.agents))))
        status, res = parent.recv()
        assert status == "ok", res
        assert isinstance(res, StepResult)
        assert res.rewards.shape == (len(cur.agents),) and np.all(np.isfinite(res.rewards))
        if res.done:
            episodes += 1
            assert res.final_obs.shape == (len(cur.agents), OBS_DIM)
            assert "length_s" in res.summary and "reward_terms" in res.summary
        cur = res.next
        assert np.all(np.isfinite(cur.obs))
    parent.send(("close",))
    th.join(5)
    assert episodes > 3


torch = pytest.importorskip("torch")


def test_collector_trajectories_consistent():
    from rlagent.algo.collector import Collector
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    cfg = _fake_cfg()
    actor, _ = build_models(cfg, N_ACTIONS)
    col = Collector(cfg, 2, seed=0)
    try:
        trajs, episodes, n, _ = col.collect(actor, 3000, torch.device("cpu"))
    finally:
        col.close()
    assert n >= 3000
    assert sum(len(t) for t in trajs) == n
    for t in trajs:
        assert len(t.obs) == len(t.actions) == len(t.log_probs) == len(t.rewards) > 0


def test_collector_multiple_envs_per_worker():
    """Plusieurs arènes par worker : chaque arène est un slot indépendant, trajectoires cohérentes."""
    from rlagent.algo.collector import Collector
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    cfg = _fake_cfg()
    actor, _ = build_models(cfg, N_ACTIONS)
    col = Collector(cfg, 2, seed=0, envs_per_worker=3)
    try:
        assert len(col.slots) == 6
        trajs, episodes, n, _ = col.collect(actor, 4000, torch.device("cpu"))
    finally:
        col.close()
    assert n >= 4000 and sum(len(t) for t in trajs) == n
    for t in trajs:
        assert len(t.obs) == len(t.actions) == len(t.log_probs) == len(t.rewards) > 0
    assert episodes  # des épisodes se terminent dans plusieurs arènes


def test_short_training_run_end_to_end(tmp_path):
    """Entraînement court sur moteur factice : checkpoints, pool, reprise, métriques, pas de NaN."""
    import json
    from rlagent.train import find_latest_checkpoint, train
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 6000, "checkpoint_every_steps": 3000},
                    selfplay={"pool_every_steps": 2000, "hist_prob": 0.5})
    cfg["run_name"] = "pytest_fake"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines and lines[-1]["global_step"] >= 6000
    assert all(np.isfinite(l["learn/policy_loss"]) for l in lines)
    ck = find_latest_checkpoint(run_dir)
    assert ck is not None and len(list((run_dir / "pool").glob("actor_*.pt"))) >= 2
    assert any("vs_hist/episodes" in l for l in lines)
    # reprise ; en lr fixe, la config fait foi (un --set ppo.lr_actor=… à la reprise est appliqué)
    cfg["train"]["total_steps"] = 9000
    cfg["ppo"]["lr_actor"] = 1.0e-4
    train(cfg, run_dir, ck)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 9000
    assert lines[-1]["learn/lr_actor"] == 1.0e-4


def test_adaptive_kl_training_run_and_resume_keeps_lr(tmp_path):
    """lr piloté par le KL : il bouge, reste dans ses bornes, et la reprise repart du lr adapté."""
    import json
    from rlagent.train import find_latest_checkpoint, train
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 4000},
                    ppo={"lr_schedule": "adaptive_kl", "target_kl": 0.5, "lr_min": 1e-5, "lr_max": 1e-3})
    cfg["run_name"] = "pytest_fake_adaptive"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    lrs = [l["learn/lr_actor"] for l in lines]
    assert all(np.isfinite(l["learn/kl_final"]) for l in lines)
    assert lrs[0] > float(cfg["ppo"]["lr_actor"]) and all(1e-5 <= x <= 1e-3 for x in lrs)  # KL ≪ cible → lr ↑
    cfg["train"]["total_steps"] = 12000
    train(cfg, run_dir, find_latest_checkpoint(run_dir))
    lines2 = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines2[len(lines)]["learn/lr_actor"] >= lrs[-1]


def test_async_collect_training_run(tmp_path):
    """Collecte en parallèle de l'apprentissage : même chaîne, pas de NaN, pool, checkpoints, reprise, arrêt STOP."""
    import json
    from rlagent.train import find_latest_checkpoint, train
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 3000,
                           "async_collect": True, "envs_per_worker": 2},
                    selfplay={"pool_every_steps": 2000, "hist_prob": 0.5})
    cfg["run_name"] = "pytest_fake_async"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 8000
    assert [l["iteration"] for l in lines] == list(range(1, len(lines) + 1))
    assert all(np.isfinite(l["learn/policy_loss"]) and np.isfinite(l["learn/approx_kl"]) for l in lines)
    # les données ont au plus une itération de retard : le ratio reste proche de 1
    assert all(l["learn/approx_kl"] < 0.05 for l in lines)
    assert len(list((run_dir / "pool").glob("actor_*.pt"))) >= 2
    ck = find_latest_checkpoint(run_dir)
    assert ck is not None
    # reprise puis arrêt propre via le fichier STOP posé pendant l'entraînement
    cfg["train"]["total_steps"] = 10**9
    timer = threading.Timer(20.0, lambda: (run_dir / "STOP").touch())
    timer.start()
    try:
        train(cfg, run_dir, ck)
    finally:
        timer.cancel()
    lines2 = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(lines2) > len(lines)
    assert find_latest_checkpoint(run_dir) != ck


def test_training_with_drills_prediction_and_warm_start(tmp_path):
    """Chaîne complète avec exercices + dosage automatique + prédiction de balle, démarrée à chaud depuis un run
    SANS prédiction (couche d'entrée élargie), avec des adversaires du pool sans prédiction."""
    import json
    import shutil
    from rlagent.train import find_latest_checkpoint, train
    base = _fake_cfg(train={"n_workers": 2, "total_steps": 4000, "checkpoint_every_steps": 2000},
                     selfplay={"pool_every_steps": 2000, "hist_prob": 0.5})
    base["run_name"] = "pytest_base"
    run_a = tmp_path / "a"
    run_a.mkdir()
    train(base, run_a)

    from rlagent.config import _normalize
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 9000, "checkpoint_every_steps": 3000},
                    selfplay={"pool_every_steps": 3000, "hist_prob": 0.5})
    cfg["env"]["ball_prediction"] = True
    cfg["env"]["adaptive_drills"] = True
    cfg["env"]["start_weights"] = {"kickoff": 0.2, "random": 0.2, **{f"drill_{k}": 0.1 for k in
                                   ("defense", "shot", "aerial", "wall", "low_boost", "recovery")}}
    _normalize(cfg)
    cfg["run_name"] = "pytest_s6"
    run_b = tmp_path / "b"
    (run_b / "pool").mkdir(parents=True)
    for f in (run_a / "pool").glob("actor_*.pt"):
        shutil.copy(f, run_b / "pool" / f.name)
    train(cfg, run_b, find_latest_checkpoint(run_a))
    lines = [json.loads(x) for x in (run_b / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 9000
    assert all(np.isfinite(l["learn/policy_loss"]) for l in lines)
    keys = set().union(*lines)
    assert any(k.startswith("drill/") for k in keys) and "mech/avg_boost" in keys
    assert any("vs_hist/episodes" in l for l in lines)
