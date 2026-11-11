"""Apprentissage sur XLA (TPU de Kaggle) : vérifié avec le backend CPU de PyTorch/XLA (PJRT_DEVICE=CPU, Linux).

Sauté si torch_xla n'est pas installé (Windows, venv habituels). Sur WSL / Linux :
    PJRT_DEVICE=CPU python -m pytest tests/test_xla.py
Ce qui est vérifié : mêmes poids qu'un update sur CPU, AUCUNE recompilation d'une itération à l'autre (tailles de
lot variables, lr qui change, pas d'Adam qui avancent), entraînement complet + checkpoint relisible sur CPU + reprise.
"""
import copy
import json
import os

import numpy as np
import pytest

os.environ.setdefault("PJRT_DEVICE", "CPU")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_xla")

from rlagent import accel  # noqa: E402
from rlagent.config import load_config  # noqa: E402


def _cfg(**ppo):
    cfg = load_config(["debug"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["format_weights"] = {1: 0.5, 2: 0.3, 3: 0.2}
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    cfg["ppo"].update({"batch_steps": 600, "minibatch_size": 256, "device_chunk": 128, "epochs": 2,
                       "lr_actor": 3e-3, **ppo})
    return cfg


def _batch(cfg, actor, critic, n, seed):
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch
    from rlagent.env.obs import obs_dim
    rng = np.random.default_rng(seed)
    dim = obs_dim(cfg["env"]["ball_prediction"], cfg["env"].get("race_features", False))
    x = rng.normal(0, 0.5, size=(n, dim)).astype(np.float32)
    x[:, dim - 5:] = (np.arange(5)[None, :] < rng.integers(0, 6, n)[:, None])
    cpu_actor = copy.deepcopy(actor).cpu()
    with torch.no_grad():
        a, lp = cpu_actor.act(torch.as_tensor(x))
    trajs = [Trajectory([x[i]], [int(a[i])], [float(lp[i])], [1.0 if int(a[i]) == 7 else 0.0], None) for i in range(n)]
    return build_batch(trajs, critic, RunningStats(), cfg, next(critic.parameters()).device)


def _compiles() -> int:
    import torch_xla.debug.metrics as met
    data = met.metric_data("CompileTime")
    return int(data[0]) if data else 0


def test_xla_update_matches_cpu():
    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    cfg = _cfg()
    torch.manual_seed(0)
    actor, critic = build_models(cfg, N_ACTIONS)
    xla = accel.resolve_device("xla")
    xa, xc = copy.deepcopy(actor).to(xla), copy.deepcopy(critic).to(xla)
    accel.prepare_models(xla, xa, xc)
    batch = _batch(cfg, actor, critic, 600, 0)
    cpu = torch.device("cpu")
    torch.manual_seed(1)
    ref = ppo_update(actor, critic, make_optimizer(actor, critic, cfg), batch, cfg, cpu, decoupled=True)
    torch.manual_seed(1)
    out = ppo_update(xa, xc, make_optimizer(xa, xc, cfg), batch, cfg, xla, decoupled=True)
    for k in ("policy_loss", "value_loss", "entropy", "kl_final"):
        assert out[k] == pytest.approx(ref[k], rel=1e-3, abs=1e-5), k
    for p, q in zip(actor.parameters(), xa.parameters()):
        assert torch.allclose(p, q.cpu(), atol=1e-4)


@pytest.mark.parametrize("schedule", ["fixed", "adaptive_kl"])
def test_xla_no_recompilation_between_iterations(schedule):
    from rlagent.algo.learner import adapt_actor_lr, build_batch, make_optimizer, ppo_update  # noqa: F401
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    cfg = _cfg(lr_schedule=schedule, target_kl=1e-4)          # cible minuscule : le lr change à chaque itération
    xla = accel.resolve_device("xla")
    torch.manual_seed(0)
    actor, critic = build_models(cfg, N_ACTIONS)
    actor.to(xla)
    critic.to(xla)
    accel.prepare_models(xla, actor, critic)
    opt = make_optimizer(actor, critic, cfg)
    counts = []
    for it, n in enumerate((600, 641, 613, 667, 600)):        # tailles de lot variables, comme en vrai
        batch = _batch(cfg, actor, critic, n, it)
        stats = ppo_update(actor, critic, opt, batch, cfg, xla, decoupled=True)
        adapt_actor_lr(opt, stats["kl_final"], cfg)
        accel.step_barrier(xla)
        counts.append(_compiles())
    assert counts[-1] == counts[2], f"recompilations : {counts}"


def test_xla_training_run_checkpoint_and_resume(tmp_path):
    from rlagent.train import find_latest_checkpoint, train
    cfg = _cfg(device="xla", lr_schedule="adaptive_kl", target_kl=0.5)
    cfg["ppo"].update(batch_steps=2000, minibatch_size=1000, device_chunk=500)
    cfg["train"].update(n_workers=2, total_steps=6000, checkpoint_every_steps=3000, async_collect=True,
                        envs_per_worker=2, collect_mode="local")
    cfg["selfplay"].update(pool_every_steps=2000, hist_prob=0.5)
    cfg["run_name"] = "pytest_xla"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 6000
    assert all(np.isfinite(x["learn/policy_loss"]) for x in lines)
    ck = find_latest_checkpoint(run_dir)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in state["actor"].values())
    cfg["train"]["total_steps"] = 9000
    train(cfg, run_dir, ck)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 9000


def test_spawned_workers_do_not_open_the_xla_runtime(tmp_path, monkeypatch):
    """Les workers (spawn) héritent de PJRT_DEVICE : ils ne doivent pas initialiser XLA (session TPU 4 : 222 runtimes
    TPU ouverts pour rien). Fonction du fils dans un module à part : ce fichier-ci importe torch_xla lui-même."""
    import multiprocessing as mp
    import sys
    (tmp_path / "xla_child.py").write_text(
        "import sys\n"
        "def run(q):\n"
        "    import rlagent.accel  # comme un worker qui réimporte le module principal\n"
        "    q.put('torch_xla' in sys.modules)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import xla_child
    assert "torch_xla" in sys.modules                      # le parent, lui, a XLA
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=xla_child.run, args=(q,))
    p.start()
    loaded = q.get(timeout=120)
    p.join(30)
    assert loaded is False


def test_resident_batch_matches_host_batch():
    """Lot résident sur le device (ppo.resident_batch, défaut sur XLA) = lot copié morceau par morceau."""
    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    xla = accel.resolve_device("xla")
    outs = []
    for resident in (False, True):
        cfg = _cfg(resident_batch=resident)
        torch.manual_seed(0)
        actor, critic = build_models(cfg, N_ACTIONS)
        actor.to(xla)
        critic.to(xla)
        accel.prepare_models(xla, actor, critic)
        batch = _batch(cfg, actor, critic, 641, 3)
        torch.manual_seed(1)
        stats = ppo_update(actor, critic, make_optimizer(actor, critic, cfg), batch, cfg, xla, decoupled=True)
        outs.append((batch, stats, [q.detach().cpu() for q in actor.parameters()]))
    (b0, s0, w0), (b1, s1, w1) = outs
    assert "_obs_dev" in b1 and "_obs_dev" not in b0
    np.testing.assert_allclose(b1["values"], b0["values"], rtol=1e-5, atol=1e-6)
    for k in ("policy_loss", "value_loss", "entropy", "kl_final", "behavior_kl"):
        assert s1[k] == pytest.approx(s0[k], rel=1e-4, abs=1e-6), k
    for p, q in zip(w0, w1):
        assert torch.allclose(p, q, atol=1e-5)
