"""Garde-fou d'entraînement et retours en arrière (08/11/2026, après la session 4 : 2,2 G steps perdus sur une faille de
reward que rien n'a arrêtée)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlagent.guard import TrainingGuard
from rlagent.selfplay.ladder import CURRENT, Ladder
from rlagent.selfplay.pool import OpponentPool
from rlagent.train import list_milestones, pick_milestone

GUARD = {"enabled": True, "max_drop": 120.0, "max_share": 0.25, "min_entropy": 1.5, "patience": 3, "warmup_iters": 2,
         "max_rollbacks": 2}


def healthy(step, **over):
    m = {"global_step": step, "ladder/drop": 20.0, "learn/entropy": 2.2, "reward_abs/goal": 1.0,
         "reward_abs/vel_ball_to_goal": 1.5, "reward_abs/aerial_touch": 0.05, "reward_abs/strong_touch": 0.2}
    m.update(over)
    return m


# --------------------------------------------------------------------------- TrainingGuard
def test_guard_waits_for_warmup_and_patience(tmp_path):
    g = TrainingGuard(GUARD, tmp_path)
    fired = [g.update(healthy(i, **{"ladder/drop": 300.0})) for i in range(6)]
    assert fired[:4] == [None] * 4                      # 2 itérations de silence, puis 3 de suite exigées
    assert fired[4]["reason"].startswith("ladder/drop") and fired[4]["onset_step"] == 2


def test_guard_ignores_a_single_spike(tmp_path):
    g = TrainingGuard(GUARD, tmp_path)
    seq = [healthy(i) for i in range(4)] + [healthy(4, **{"learn/entropy": 0.5})] + [healthy(i) for i in range(5, 12)]
    assert all(g.update(m) is None for m in seq)


def test_guard_names_the_term_that_took_over_the_signal(tmp_path):
    g = TrainingGuard({**GUARD, "ema": 1.0}, tmp_path)
    fired = None
    for i in range(10):
        fired = fired or g.update(healthy(i, **{"reward_abs/aerial_touch": 8.0}))
    assert fired["term"] == "aerial_touch"


def test_guard_culprit_is_relative_to_the_milestone(tmp_path):
    g = TrainingGuard({**GUARD, "ema": 1.0}, tmp_path)
    g.update(healthy(0))
    g.note_milestone(0)
    g.update(healthy(1, **{"reward_abs/aerial_touch": 0.6}))         # 0,02 → ~0,18 : ×9 depuis le jalon
    assert g.culprit(g.reference_shares(0)) == "aerial_touch"
    assert g.culprit(g.reference_shares(12345)) is None              # jalon sans parts notées : rien n'est coupé


def test_disabled_terms_survive_a_restart_and_zero_the_weight(tmp_path):
    g = TrainingGuard(GUARD, tmp_path)
    g.record_rollback(9_000, 8_500, "ladder/drop 144 > 120", "aerial_touch", 70)
    g2 = TrainingGuard(GUARD, tmp_path)
    assert g2.disabled_terms == ["aerial_touch"]
    assert g2.apply({"goal": 10.0, "aerial_touch": 4.0}) == {"goal": 10.0, "aerial_touch": 0.0}
    assert g2.can_roll_back() and not TrainingGuard({**GUARD, "max_rollbacks": 1}, tmp_path).can_roll_back()


def test_guard_replayed_on_session_4_fires_before_9_g(tmp_path):
    """Rejoué sur les métriques réelles de la session 4 (si le run est sur la machine) : déclenché vers 8,85 G."""
    root = Path(__file__).resolve().parents[1]
    path = root / "runs/s10_ppo_kaggle/metrics.jsonl"
    if not path.exists():
        pytest.skip("run s10_ppo_kaggle absent")
    import yaml
    cfg = yaml.safe_load((root / "configs/s10.yaml").read_text(encoding="utf-8"))
    g = TrainingGuard(cfg["train"]["guard"], tmp_path)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["global_step"] > 8.1402e9 and g.update(r):
            assert 8.7e9 < r["global_step"] < 9.0e9
            return
    pytest.fail("le garde-fou ne s'est jamais déclenché sur la session 4")


# --------------------------------------------------------------------------- pool, échelle, jalons
def tiny_state():
    return {"w": torch.zeros(2)}


def test_rejected_versions_leave_the_pool_and_the_ladder(tmp_path):
    pool = OpponentPool(tmp_path / "pool", recent=20, pfsp=True)
    lad = Ladder(tmp_path / "ladder.json")
    for step in (100, 200, 300, 400):
        lad.add_version(pool.add(tiny_state(), {}, step, 1000.0), step)
        lad.record([(f"actor_{step:013d}", 1.0)])
    lad.record([("actor_0000000000400", 0.0)])
    gone = pool.reject_after(200)
    lad.remove(gone)
    assert sorted(gone) == ["actor_0000000000300", "actor_0000000000400"]
    assert [e["step"] for e in pool.entries] == [100, 200]
    assert sorted(p.name for p in (tmp_path / "pool" / "rejected").glob("*.pt")) == [f"{g}.pt" for g in sorted(gone)]
    assert set(lad.players) == {"actor_0000000000100", "actor_0000000000200"}
    assert all(CURRENT not in k and "0400" not in "".join(k) for k in lad.games)
    assert [e["step"] for e in OpponentPool(tmp_path / "pool", recent=20).entries] == [100, 200]    # après relecture


def test_pick_milestone_takes_the_best_rated_before_the_drift(tmp_path):
    d = tmp_path / "checkpoints" / "milestones"
    d.mkdir(parents=True)
    for s in (100, 250, 500, 750, 1000):
        (d / f"ckpt_{s:013d}.pt").write_bytes(b"")
    lad = Ladder(tmp_path / "ladder.json")
    lad.players = {"a": 240, "b": 490, "c": 740}
    lad.ratings = {"a": 880.0, "b": 905.0, "c": 845.0}
    assert [s for s, _ in list_milestones(tmp_path)] == [100, 250, 500, 750, 1000]
    assert pick_milestone(tmp_path, lad, 830)[0] == 500
    assert pick_milestone(tmp_path, None, 830)[0] == 750
    assert pick_milestone(tmp_path, lad, 50) is None


def test_zero_actor_lr_leaves_the_actor_untouched():
    """Recalage du critique : lr 0 pour l'acteur, Adam n'applique aucun pas."""
    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": 0.0}, {"params": critic.parameters(), "lr": 1e-2}])
    before = [p.detach().clone() for p in actor.parameters()]
    c_before = [p.detach().clone() for p in critic.parameters()]
    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(8, 3)
        (actor(x).pow(2).mean() + critic(x).pow(2).mean()).backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(c_before, critic.parameters()))


# --------------------------------------------------------------------------- entraînement factice
def _fake_cfg(**over):
    from rlagent.config import load_config
    cfg = load_config(["debug"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["format_weights"] = {1: 1.0, 2: 0.0, 3: 0.0}
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def test_rollback_to_a_pool_version_then_critic_warmup(tmp_path):
    from rlagent.train import find_latest_checkpoint, train
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 4000},
                    selfplay={"pool_every_steps": 2000, "hist_prob": 0.5, "ladder": True, "ladder_fit_every": 1})
    cfg["run_name"] = "pytest_rollback"
    run = tmp_path / "run"
    run.mkdir()
    train(cfg, run)
    versions = sorted(p.name for p in (run / "pool").glob("actor_*.pt"))
    target = next(v for v in versions if 0 < int(v[6:19]) <= 4000)
    n_after = sum(int(v[6:19]) > int(target[6:19]) for v in versions)
    assert n_after > 0
    # pool partiel comme après kaggle_sync fetch : la dernière version notée par l'échelle n'a plus son fichier
    (run / "pool" / versions[-1]).unlink()
    n_after -= 1
    old_steps = {s for s in Ladder(run / "ladder.json").players.values() if s > int(target[6:19])}
    assert int(versions[-1][6:19]) in old_steps

    cfg["train"].update({"total_steps": 14000, "rollback_to": f"pool/{target}", "rollback_critic_warmup_iters": 2,
                         "milestone_every_steps": 100_000})
    n_lines = len((run / "metrics.jsonl").read_text().splitlines())
    train(cfg, run, find_latest_checkpoint(run))
    state = json.loads((run / "guard.json").read_text(encoding="utf-8"))
    assert state["manual_rollbacks"] == [f"pool/{target}"]
    assert len(list((run / "pool" / "rejected").glob("actor_*.pt"))) == n_after
    new = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()][n_lines:]
    assert [l.get("learn/critic_warmup", 0.0) for l in new[:3]] == [1.0, 1.0, 0.0]
    assert list_milestones(run), "un jalon de départ est posé dès la fin du recalage"
    kept = set(Ladder(run / "ladder.json").players.values())
    assert not (kept & old_steps), "versions de la dérive encore dans l'échelle (même sans leur fichier)"

    # une seule fois par run : la reprise suivante ne revient pas en arrière
    cfg["train"]["total_steps"] = 16000
    train(cfg, run, find_latest_checkpoint(run))
    assert len(list((run / "pool" / "rejected").glob("actor_*.pt"))) == n_after


def test_guard_rolls_back_then_stops_on_a_persistent_drift(tmp_path):
    from rlagent.train import train
    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 200_000, "checkpoint_every_steps": 4000,
                           "milestone_every_steps": 4000, "keep_milestones": 8,
                           "guard": {"enabled": True, "min_entropy": 100.0, "patience": 1, "warmup_iters": 2,
                                     "max_rollbacks": 1}},
                    selfplay={"pool_every_steps": 2000, "hist_prob": 0.5})
    cfg["run_name"] = "pytest_guard"
    run = tmp_path / "run"
    run.mkdir()
    train(cfg, run)
    state = json.loads((run / "guard.json").read_text(encoding="utf-8"))
    lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(state["rollbacks"]) == 2 and "arrêt" in state["rollbacks"][-1]["reason"]
    assert lines[-1]["global_step"] < 50_000                          # arrêté bien avant total_steps
    assert all(r["to_step"] <= r["at_step"] for r in state["rollbacks"])
    assert sum(l.get("guard/triggered", 0.0) for l in lines) == 2
