"""Accélération cloud : collecte en double tampon, collecte locale (workers avec leur acteur), chemin TPU (formes
fixes) simulé sur CPU, checkpoints sur CPU, autotune. Moteur factice (sans RocketSim)."""
import json
from collections import defaultdict

import numpy as np
import pytest

from rlagent.config import load_config

torch = pytest.importorskip("torch")


def _fake_cfg(**over):
    cfg = load_config(["debug"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["format_weights"] = {1: 0.5, 2: 0.3, 3: 0.2}
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    cfg["selfplay"]["hist_prob"] = 0.0
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def _actor(cfg, seed=0):
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    torch.manual_seed(seed)
    return build_models(cfg, N_ACTIONS)[0]


class RowwiseArgmax(torch.nn.Module):
    """Politique déterministe, calculée observation par observation : mêmes actions quelle que soit la composition
    des lots (sinon des écarts d'arrondi entre tailles de lot peuvent départager autrement deux actions)."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    @torch.no_grad()
    def act(self, obs):
        logits = torch.cat([self.actor(o[None]) for o in obs])
        a = logits.argmax(-1)
        return a, torch.distributions.Categorical(logits=logits).log_prob(a)


def _constant_action_actor(cfg, action=5):
    """Acteur qui joue toujours la même action (biais énorme) : déterministe dans n'importe quel processus."""
    actor = _actor(cfg)
    with torch.no_grad():
        actor.head[-1].weight.zero_()
        actor.head[-1].bias.zero_()
        actor.head[-1].bias[action] = 1e4
    return actor


def _sequences(trajs):
    """(arène, agent) → transitions concaténées dans l'ordre (obs, action, reward)."""
    seq = defaultdict(lambda: ([], [], []))
    for t in trajs:
        o, a, r = seq[(t.slot, t.agent)]
        o.extend(np.asarray(t.obs))
        a.extend(np.asarray(t.actions).tolist())
        r.extend(np.asarray(t.rewards).tolist())
    return seq


def _assert_same_prefixes(seq_a, seq_b, min_common):
    # un run qui a joué plus longtemps peut avoir atteint un épisode à plus de voitures (agents en plus)
    keys = set(seq_a) & set(seq_b)
    assert len(keys) >= 0.5 * max(len(seq_a), len(seq_b))
    common = 0
    for key in keys:
        (oa, aa, ra), (ob, ab, rb) = seq_a[key], seq_b[key]
        n = min(len(aa), len(ab))
        assert aa[:n] == ab[:n], key
        np.testing.assert_array_equal(np.array(oa[:n]), np.array(ob[:n]), err_msg=str(key))
        np.testing.assert_array_equal(np.array(ra[:n]), np.array(rb[:n]), err_msg=str(key))
        common += n
    assert common >= min_common


def _collect_twice(col, actor, n):
    trajs, episodes, total = [], [], 0
    for _ in range(2):
        t, e, k, _ = col.collect(actor, n, torch.device("cpu"))
        trajs += t
        episodes += e
        total += k
        assert sum(len(x) for x in t) == k
        for x in t:
            assert len(x.obs) == len(x.actions) == len(x.log_probs) == len(x.rewards) > 0
    return trajs, episodes, total


# --------------------------------------------------------------------------- collecte
def test_env_parts_split():
    from rlagent.algo.collector import env_parts
    assert env_parts(16, 1) == [(0, 16)]
    assert env_parts(16, 2) == [(0, 8), (8, 16)]
    assert env_parts(5, 2) == [(0, 2), (2, 5)] or env_parts(5, 2) == [(0, 3), (3, 5)]
    assert env_parts(1, 2) == [(0, 1)]                 # une seule arène : pas de double tampon possible


def test_pipelined_collection_matches_lockstep():
    """Double tampon (2 groupes d'arènes par worker) : chaque arène voit exactement la même suite d'états, d'actions
    et de rewards qu'en collecte d'origine (les arènes sont indépendantes ; seul l'ordre des calculs change)."""
    from rlagent.algo.collector import Collector
    cfg = _fake_cfg()
    policy = RowwiseArgmax(_actor(cfg))
    out = {}
    for k in (1, 2):
        col = Collector(cfg, 2, seed=0, envs_per_worker=4, pipeline=k)
        try:
            assert len(col.parts) == k
            out[k] = _collect_twice(col, policy, 1500)
        finally:
            col.close()
    _assert_same_prefixes(_sequences(out[1][0]), _sequences(out[2][0]), min_common=2500)
    assert out[1][1] and out[2][1]                     # des épisodes se terminent


def test_local_collection_matches_central():
    """Collecte locale (chaque worker joue avec sa propre copie de l'acteur, poids par mémoire partagée) : mêmes
    suites par arène que la collecte centrale avec la même politique."""
    from rlagent.algo.collector import Collector, LocalCollector
    cfg = _fake_cfg()
    actor = _constant_action_actor(cfg)
    col = Collector(cfg, 2, seed=0, envs_per_worker=3)
    try:
        central = _collect_twice(col, actor, 1500)
    finally:
        col.close()
    col = LocalCollector(cfg, 2, seed=0, envs_per_worker=3)
    try:
        assert col.weights is not None                  # mémoire partagée disponible
        local = _collect_twice(col, actor, 1500)
    finally:
        col.close()
    assert local[2] >= 3000
    assert all(int(np.asarray(t.actions).max()) == 5 for t in local[0])   # les poids envoyés sont bien utilisés
    _assert_same_prefixes(_sequences(central[0]), _sequences(local[0]), min_common=2500)


def test_local_collection_with_pool_and_new_weights(tmp_path):
    """Collecte locale contre des versions du pool : épisodes vs_hist renseignés, et les nouveaux poids sont pris."""
    from rlagent.algo.collector import LocalCollector
    from rlagent.selfplay.pool import OpponentPool
    cfg = _fake_cfg(selfplay={"hist_prob": 0.5})
    pool = OpponentPool(tmp_path / "pool", recent=5, seed=0)
    for step in (1000, 2000):
        pool.add({k: v.clone() for k, v in _actor(cfg, seed=step).state_dict().items()}, cfg["model"], step, 1000.0)
    col = LocalCollector(cfg, 2, seed=0, pool=pool, envs_per_worker=2)
    try:
        trajs, episodes, n, _ = col.collect(_constant_action_actor(cfg, 3), 3000, torch.device("cpu"))
        assert n >= 3000 and sum(len(t) for t in trajs) == n
        assert all(set(np.asarray(t.actions).tolist()) == {3} for t in trajs)
        hist = [e for e in episodes if "vs_hist" in e]
        assert hist and {e["vs_hist"]["hist_id"] for e in hist} <= {"actor_0000000001000", "actor_0000000002000"}
        trajs, _, _, _ = col.collect(_constant_action_actor(cfg, 7), 1000, torch.device("cpu"))
        assert all(set(np.asarray(t.actions).tolist()) == {7} for t in trajs)
    finally:
        col.close()


# --------------------------------------------------------------------------- apprentissage « TPU » sur CPU
def _bandit_batch(cfg, actor, critic, n=600, seed=0):
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch
    from rlagent.env.obs import obs_dim
    rng = np.random.default_rng(seed)
    dim = obs_dim(cfg["env"]["ball_prediction"], cfg["env"].get("race_features", False))
    x = rng.normal(0, 0.5, size=(n, dim)).astype(np.float32)
    x[:, dim - 5:] = (np.arange(5)[None, :] < rng.integers(0, 6, n)[:, None])
    with torch.no_grad():
        a, lp = actor.act(torch.as_tensor(x))
    trajs = [Trajectory([x[i]], [int(a[i])], [float(lp[i])], [1.0 if int(a[i]) == 7 else 0.0], None) for i in range(n)]
    return build_batch(trajs, critic, RunningStats(), cfg, torch.device("cpu"))


@pytest.mark.parametrize("lr_schedule", ["fixed", "adaptive_kl"])
def test_static_shape_learner_matches_default(monkeypatch, lr_schedule):
    """Chemin TPU (formes fixes : dernier morceau complété, pas de regroupement par nb de voitures, barrière après
    chaque pas) simulé sur CPU : mêmes valeurs, mêmes poids après l'update que le chemin normal."""
    import copy

    from rlagent import accel
    from rlagent.algo import learner
    from rlagent.algo.learner import batched_values, make_optimizer, ppo_update
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    cfg = _fake_cfg(ppo={"batch_steps": 600, "minibatch_size": 300, "device_chunk": 128, "epochs": 2,
                         "lr_actor": 3e-3, "lr_schedule": lr_schedule, "target_kl": 1.0})
    torch.manual_seed(0)
    actor, critic = build_models(cfg, N_ACTIONS)
    batch = _bandit_batch(cfg, actor, critic)
    a2, c2 = copy.deepcopy(actor), copy.deepcopy(critic)
    cpu = torch.device("cpu")

    torch.manual_seed(1)
    ref = ppo_update(actor, critic, make_optimizer(actor, critic, cfg), batch, cfg, cpu, decoupled=True)
    ref_v = batched_values(critic, batch["obs"], cpu, 128)

    barriers = []
    monkeypatch.setattr(accel, "static_shapes", lambda d: True)
    monkeypatch.setattr(accel, "step_barrier", lambda d: barriers.append(1))
    monkeypatch.setattr(learner.accel, "static_shapes", lambda d: True)
    accel.prepare_models(cpu, a2, c2)
    assert a2.encoder.skip_padding is False and c2.encoder.skip_padding is False
    torch.manual_seed(1)
    out = ppo_update(a2, c2, make_optimizer(a2, c2, cfg), batch, cfg, cpu, decoupled=True)
    out_v = batched_values(c2, batch["obs"], cpu, 128)

    assert len(barriers) >= out["n_minibatches"]
    np.testing.assert_allclose(out_v, ref_v, atol=1e-5)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "kl_final", "behavior_kl"):
        assert out[k] == pytest.approx(ref[k], rel=1e-3, abs=1e-5), k
    for p, q in zip(actor.parameters(), a2.parameters()):
        assert torch.allclose(p, q, atol=1e-5)


def test_padded_chunks_and_pad_rows():
    from rlagent.accel import pad_rows, padded_chunks
    assert list(padded_chunks(10, 4, False)) == [(0, 4, 4), (4, 8, 4), (8, 10, 2)]
    assert list(padded_chunks(10, 4, True)) == [(0, 4, 4), (4, 8, 4), (8, 10, 4)]
    x = torch.ones(3, 2)
    y = pad_rows(x, 5)
    assert y.shape == (5, 2) and y[:3].eq(1).all() and y[3:].eq(0).all()
    assert pad_rows(x, 2) is x


def test_to_cpu_and_device_resolution():
    from rlagent import accel
    state = {"a": torch.ones(2), "b": [torch.zeros(1), 3], "c": {"d": (torch.ones(1), "x")}}
    out = accel.to_cpu(state)
    assert out["b"][1] == 3 and out["c"]["d"][1] == "x" and out["a"].device.type == "cpu"
    assert accel.resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert accel.resolve_collect_device("auto", torch.device("cpu")).type == "cpu"
    with pytest.raises(ValueError):
        accel.resolve_collect_device("xla", torch.device("cpu"))


# --------------------------------------------------------------------------- entraînements complets
@pytest.mark.parametrize("mode", ["pipeline", "local"])
def test_training_run_with_cloud_collection(tmp_path, mode):
    """Entraînement asynchrone complet avec la collecte en double tampon ou la collecte locale : pas de NaN,
    pool, checkpoints sur CPU, reprise."""
    from rlagent.train import find_latest_checkpoint, train
    tr = {"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 3000, "async_collect": True,
          "envs_per_worker": 4}
    tr.update({"collect_pipeline": 2} if mode == "pipeline" else {"collect_mode": "local"})
    # collecte locale : les workers reçoivent la liste des adversaires au début de chaque collecte (une itération de
    # retard sur le pool) → pool rempli tôt et beaucoup de parties contre l'historique pour en voir finir dans ce run court
    cfg = _fake_cfg(train=tr, selfplay={"pool_every_steps": 1000, "hist_prob": 0.8})
    cfg["run_name"] = f"pytest_{mode}"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 8000
    assert all(np.isfinite(x["learn/policy_loss"]) and x["learn/approx_kl"] < 0.05 for x in lines)
    assert any("vs_hist/episodes" in x for x in lines)
    ck = find_latest_checkpoint(run_dir)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in state["actor"].values())
    cfg["train"]["total_steps"] = 11000
    train(cfg, run_dir, ck)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 11000


def test_segment_length_warning():
    from rlagent.train import check_segment_length
    cfg = _fake_cfg(train={"n_workers": 100, "envs_per_worker": 8})
    assert check_segment_length(cfg, 50_000) is not None
    cfg["train"].update(n_workers=3, envs_per_worker=16)
    assert check_segment_length(cfg, 50_000) is None
