"""Accélérations de la boucle d'apprentissage (analyse de la session TPU 4) : mêmes résultats qu'avant, plus vite.

GAE vectorisée, échelle Elo sur les seules paires jouées, choix de la collecte qui laisse des cœurs au processus
principal. Le lot résident sur TPU est vérifié dans tests/test_xla.py (backend CPU de PyTorch/XLA).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rlagent.algo.gae import RunningStats, batch_gae, compute_gae, discounted_returns, segment_discount
from rlagent.selfplay.ladder import CURRENT, Ladder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import autotune  # noqa: E402


def _segments(seed, n=200):
    rng = np.random.default_rng(seed)
    lengths = np.concatenate([[1, 1, 2], rng.integers(1, 300, n)])
    rewards = [rng.normal(size=k).astype(np.float32) for k in lengths]
    values = [rng.normal(size=k).astype(np.float32) for k in lengths]
    boot = np.where(rng.random(len(lengths)) < 0.3, 0.0, rng.normal(size=len(lengths)))
    return lengths, rewards, values, boot


@pytest.mark.parametrize("seed", [0, 1])
def test_batch_gae_matches_per_trajectory_loop(seed):
    lengths, rewards, values, boot = _segments(seed)
    ref = np.concatenate([compute_gae(r, v, float(b), 0.995, 0.95) for r, v, b in zip(rewards, values, boot)])
    out = batch_gae(np.concatenate(rewards), np.concatenate(values), lengths, boot, 0.995, 0.95)
    assert out.dtype == np.float32 and out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_segment_discount_matches_discounted_returns_and_stats():
    lengths, rewards, _, _ = _segments(2)
    ref = [discounted_returns(r, 0.995) for r in rewards]
    out = segment_discount(np.concatenate(rewards), lengths, 0.995)
    np.testing.assert_allclose(out, np.concatenate(ref), rtol=1e-5, atol=1e-5)
    a, b = RunningStats(), RunningStats()
    for r in ref:                        # ancienne mise à jour : une fusion par trajectoire
        a.update(r)
    b.update(out)                        # nouvelle : une seule fusion sur le lot
    assert b.mean == pytest.approx(a.mean, rel=1e-5) and b.var == pytest.approx(a.var, rel=1e-5)


def test_segment_discount_empty_batch():
    assert segment_discount(np.zeros(0), np.zeros(0, np.int64), 0.9).shape == (0,)


def _dense_fit(lad: Ladder, iters: int) -> dict:
    """Ancienne version de Ladder.fit (matrice pleine n × n), référence."""
    versions = lad._versions()
    names = list(lad.players) + ([CURRENT] if any(CURRENT in k for k in lad.games) else [])
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n))
    for (a, b), (s, g) in lad.games.items():
        if a in idx and b in idx:
            wins[idx[a], idx[b]] += s
            wins[idx[b], idx[a]] += g - s
    prior = np.zeros((n, n))
    prior[(wins + wins.T) > 0] = 1.0
    chain = versions + ([CURRENT] if CURRENT in idx else [])
    for a, b in zip(chain, chain[1:]):
        prior[idx[a], idx[b]] = prior[idx[b], idx[a]] = 1.0
    wins = wins + prior / 2.0
    games = wins + wins.T
    w_tot = wins.sum(axis=1)
    p = np.ones(n)
    active = w_tot > 0
    for _ in range(iters):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = np.where(active & (denom > 0), w_tot / np.maximum(denom, 1e-300), p)
        new /= math.exp(np.mean(np.log(new[active])))
        if np.max(np.abs(np.log(new[active]) - np.log(p[active]))) < 1e-7:
            p = new
            break
        p = new
    ref = p[idx[versions[0]]]
    return {name: float(400.0 * math.log10(p[i] / ref)) for name, i in idx.items()
            if active[i] and games[i].sum() > 0}


def _random_ladder(tmp_path, seed=0, n=60):
    rng = np.random.default_rng(seed)
    lad = Ladder(tmp_path / f"ladder{seed}.json")
    strength = np.cumsum(rng.normal(0.05, 0.2, n))
    for v in range(n):
        lad.players[f"v{v:03d}"] = v * 1000
    lad.players["anchor"] = -1
    for v in range(1, n):
        for o in rng.choice(v, size=min(v, 4), replace=False):
            g = float(rng.integers(1, 12))
            pw = 1 / (1 + math.exp(strength[o] - strength[v]))
            lad.games[(f"v{v:03d}", f"v{o:03d}")] = [float(rng.binomial(int(g), pw)), g]
    lad.record([("v010", 1.0), ("anchor", 0.5), ("v050", 0.0)], anchors=["anchor"])
    lad.players["unplayed"] = n * 1000 + 5        # version sans partie : hors des notes
    return lad


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_ladder_fit_matches_dense(tmp_path, seed):
    lad = _random_ladder(tmp_path, seed)
    ref = _dense_fit(lad, 5000)
    out = lad.fit(iters=5000, warm=False)
    assert out.keys() == ref.keys()
    assert max(abs(out[k] - ref[k]) for k in ref) < 1e-6


def test_warm_ladder_fit_reaches_the_same_ratings(tmp_path):
    lad = _random_ladder(tmp_path, 3)
    lad.fit(iters=5000, warm=False)
    lad.record([("v020", 1.0)] * 7)                 # nouvelles parties, puis recalcul à chaud
    truth = _dense_fit(lad, 20000)
    out = lad.fit()
    assert max(abs(out[k] - truth[k]) for k in truth) < 0.05


# --------------------------------------------------------------------------- autoréglage
def _cand(workers, sps):
    return autotune.Candidate("local", workers, 8, sps=sps)


def test_autotune_picks_fewest_workers_near_the_best_throughput():
    ok = [_cand(222, 174_917), _cand(168, 174_091), _cand(112, 127_397)]
    assert autotune.choose(ok).workers == 168
    assert autotune.choose([_cand(222, 174_917), _cand(168, 160_000)]).workers == 222


def _session4_report(tmp_path, monkeypatch):
    out = tmp_path / "autotune.yaml"
    out.write_text("ppo: {device: xla}\ntrain: {n_workers: 222}\n", encoding="utf-8")
    rep = {"date": "2026-11-07T13:44:40", "ncpu": 224, "gpus": 0, "batch": 400_000,
           "updates_s": {"xla": 1.3, "cpu": 15.9}, "learner": "xla", "best": "local 222x8", "expected_sps": 174_917,
           "candidates": [{"mode": "local", "workers": w, "envs": 8, "pipeline": 1, "device": "cpu", "threads": 1,
                           "sps": s, "error": "", "label": f"local {w}x8"}
                          for w, s in ((222, 174_917), (168, 174_091), (112, 127_397))]}
    out.with_suffix(".json").write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setattr(autotune.accel, "xla_available", lambda: True)
    return out


def test_autotune_report_from_older_rule_is_rechosen_without_measuring(tmp_path, monkeypatch):
    out = _session4_report(tmp_path, monkeypatch)
    assert autotune.reusable(out, 224, 0, 400_000, None) is None          # ancienne règle : pas gardé tel quel
    assert autotune.rechoose(out, 224, 0, 400_000) is not None
    monkeypatch.setattr(autotune.os, "cpu_count", lambda: 224)
    monkeypatch.setattr(autotune, "measure_update", lambda *a, **k: pytest.fail("mesure inutile"))
    monkeypatch.setattr(autotune, "measure_collect", lambda *a, **k: pytest.fail("mesure inutile"))
    autotune.main(["--profile", "s10", "--profile", "kaggle_tpu", "--step", "9000000000", "--out", str(out),
                   "--reuse"])
    import yaml
    prof = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert prof["train"]["n_workers"] == 168 and prof["ppo"]["device"] == "xla"
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["rule"] == autotune.PICK_RULE
    assert autotune.reusable(out, 224, 0, 400_000, None)                  # session suivante : gardé tel quel
