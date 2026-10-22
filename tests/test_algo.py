"""S1-14/S1-15/S1-16 : GAE, stats courantes, Elo (numpy) ; PPO/SPO et modèle (PyTorch)."""
import numpy as np
import pytest

from rlagent.algo.gae import RunningStats, compute_gae, discounted_returns
from rlagent.selfplay.pool import elo_expected, elo_update


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    r = rng.normal(size=50).astype(np.float32)
    v = rng.normal(size=50).astype(np.float32)
    gamma, lam, boot = 0.99, 0.95, 0.7
    adv = compute_gae(r, v, boot, gamma, lam)
    # référence : somme explicite des deltas
    vs = np.append(v, boot)
    deltas = r + gamma * vs[1:] - vs[:-1]
    ref = np.array([sum((gamma * lam) ** k * deltas[t + k] for k in range(50 - t)) for t in range(50)])
    np.testing.assert_allclose(adv, ref, rtol=1e-4, atol=1e-5)
    # λ = 1 → avantage = retour actualisé - V
    adv1 = compute_gae(r, v, 0.0, gamma, 1.0)
    np.testing.assert_allclose(adv1 + v, discounted_returns(r, gamma), rtol=1e-4, atol=1e-4)


def test_running_stats():
    rng = np.random.default_rng(1)
    rs = RunningStats(eps=1e-8)
    data = rng.normal(3.0, 2.0, size=10_000)
    for chunk in np.array_split(data, 37):
        rs.update(chunk)
    assert abs(rs.mean - data.mean()) < 1e-6 and abs(rs.std - data.std()) < 1e-3


def test_elo():
    assert np.isclose(elo_expected(1000, 1000), 0.5)
    assert elo_update(1000, 1000, 1.0, 16) == 1008
    assert elo_update(1000, 1000, 0.5, 16) == 1000


torch = pytest.importorskip("torch")


def test_spo_loss_formula_and_ppo_equivalence_at_ratio_one():
    from rlagent.algo.losses import policy_loss
    old = torch.log(torch.tensor([0.2, 0.5, 0.3]))
    new = torch.log(torch.tensor([0.25, 0.4, 0.35]))
    adv = torch.tensor([1.0, -2.0, 0.5])
    eps = 0.2
    ratio = torch.exp(new - old)
    expected = -(ratio * adv - adv.abs() / (2 * eps) * (ratio - 1) ** 2).mean()
    assert torch.isclose(policy_loss(new, old, adv, eps, "spo"), expected)
    # au point r = 1 les deux pertes valent -E[A]
    assert torch.isclose(policy_loss(old, old, adv, eps, "spo"), -adv.mean())
    assert torch.isclose(policy_loss(old, old, adv, eps, "ppo"), -adv.mean())
    with pytest.raises(ValueError):
        policy_loss(new, old, adv, eps, "trpo")


def test_spo_gradient_pulls_ratio_back_outside_trust_region():
    """PPO a un gradient nul hors de la zone de clipping ; SPO ramène activement le ratio vers 1."""
    from rlagent.algo.losses import policy_loss
    old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    new = torch.tensor([float(np.log(1.5))], requires_grad=True)  # r = 1.5 > 1 + ε
    policy_loss(new, old, adv, 0.2, "ppo").backward()
    assert new.grad.abs().item() == 0.0
    new.grad = None
    policy_loss(new, old, adv, 0.2, "spo").backward()
    assert new.grad.item() > 0  # minimiser la perte → faire baisser r


def _model_cfg():
    from rlagent.config import load_config
    return load_config(["debug"])


def test_same_model_all_formats_and_permutation_invariance():
    from helpers import make_state
    from rlagent.env.obs import MAX_OTHERS, OTHER_DIM, SELF_DIM, AttentionObs
    from rlagent.models.attention import build_models
    torch.manual_seed(0)
    actor, critic = build_models(_model_cfg(), 90)
    actor.eval()
    critic.eval()
    for size in (1, 2, 3):
        s = make_state(size)
        obs = AttentionObs().build_obs(list(s.cars), s, {})
        x = torch.as_tensor(np.stack(list(obs.values())))
        assert actor(x).shape == (2 * size, 90)
        assert critic(x).shape == (2 * size,)
        assert torch.isfinite(actor(x)).all()
    # permuter les jetons "autres voitures" ne change pas la sortie
    s = make_state(3)
    o = AttentionObs().build_obs(list(s.cars), s, {})["blue-0"]
    others = o[SELF_DIM:SELF_DIM + MAX_OTHERS * OTHER_DIM].reshape(MAX_OTHERS, OTHER_DIM)
    perm = [4, 2, 0, 3, 1]
    o2 = o.copy()
    o2[SELF_DIM:SELF_DIM + MAX_OTHERS * OTHER_DIM] = others[perm].reshape(-1)
    o2[-MAX_OTHERS:] = o[-MAX_OTHERS:][perm]
    out = actor(torch.as_tensor(np.stack([o, o2])))
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_masked_tokens_are_ignored():
    from helpers import make_state
    from rlagent.env.obs import MAX_OTHERS, OTHER_DIM, SELF_DIM, AttentionObs
    from rlagent.models.attention import build_models
    actor, _ = build_models(_model_cfg(), 90)
    actor.eval()
    s = make_state(1)
    o = AttentionObs().build_obs(list(s.cars), s, {})["blue-0"]
    o2 = o.copy()
    # remplir un emplacement masqué avec du bruit : la sortie ne doit pas bouger
    start = SELF_DIM + 3 * OTHER_DIM
    o2[start:start + OTHER_DIM] = np.random.default_rng(0).normal(size=OTHER_DIM)
    out = actor(torch.as_tensor(np.stack([o, o2])))
    assert o[-MAX_OTHERS:][3] == 0
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_ppo_update_runs_and_improves_on_bandit():
    """Mini-test de bout en bout de ppo_update : un bandit où l'action 7 rapporte 1, sinon 0."""
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch, make_optimizer, ppo_update
    from rlagent.env.obs import OBS_DIM
    from rlagent.models.attention import build_models
    for kind in ("ppo", "spo"):
        cfg = _model_cfg()
        cfg["ppo"].update(policy_loss=kind, batch_steps=512, minibatch_size=256, device_chunk=128, epochs=4,
                          lr_actor=3e-3, lr_critic=3e-3, ent_coef=0.0)
        torch.manual_seed(0)
        actor, critic = build_models(cfg, 90)
        opt = make_optimizer(actor, critic, cfg)
        rs = RunningStats()
        x = np.zeros((1, OBS_DIM), dtype=np.float32)
        x[0, -5] = 1.0
        p_before = torch.softmax(actor(torch.as_tensor(x)), -1)[0, 7].item()
        for _ in range(30):
            with torch.no_grad():
                a, lp = actor.act(torch.as_tensor(np.repeat(x, 512, 0)))
            trajs = [Trajectory([x[0]], [int(ai)], [float(li)], [1.0 if int(ai) == 7 else 0.0], None)
                     for ai, li in zip(a, lp)]
            batch = build_batch(trajs, critic, rs, cfg, torch.device("cpu"))
            stats = ppo_update(actor, critic, opt, batch, cfg, torch.device("cpu"))
            assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
        p_after = torch.softmax(actor(torch.as_tensor(x)), -1)[0, 7].item()
        assert p_after > max(0.1, 4 * p_before), (kind, p_before, p_after)


def _mixed_format_obs(n: int, seed: int = 0) -> np.ndarray:
    """Observations réelles de formats mélangés (1v1/2v2/3v3), comme dans un lot d'entraînement."""
    from helpers import make_state
    from rlagent.env.obs import AttentionObs
    rng = np.random.default_rng(seed)
    rows = []
    builder = AttentionObs()
    while len(rows) < n:
        size = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
        s = make_state(size, kickoff=True, seed=int(rng.integers(1000)))
        for c in s.cars.values():
            c.physics.position = (c.physics.position + rng.normal(0, 500, 3)).astype(np.float32)
            c.physics.linear_velocity = rng.normal(0, 800, 3).astype(np.float32)
            c.physics.euler_angles = rng.uniform(-np.pi, np.pi, 3).astype(np.float32)
        s.ball.position = rng.uniform(-2000, 2000, 3).astype(np.float32)
        rows.extend(builder.build_obs(list(s.cars), s, {}).values())
    return np.stack(rows[:n]).astype(np.float32)


def test_skip_padding_matches_padded_forward_and_gradients():
    """Ne calculer que les jetons réels doit donner les MÊMES sorties et les MÊMES gradients."""
    from rlagent.algo.losses import policy_loss
    from rlagent.models.attention import build_models
    torch.manual_seed(0)
    actor, critic = build_models(load_config_default(), 90)
    x = torch.as_tensor(_mixed_format_obs(300))
    for mode in ("eval", "train"):
        getattr(actor, mode)(); getattr(critic, mode)()
        for enc_owner in (actor, critic):
            enc = enc_owner.encoder
            fast = enc(x)
            ref = enc.forward_padded(x)
            assert torch.allclose(fast, ref, atol=1e-5), (fast - ref).abs().max()
    # gradients d'une vraie perte PPO/SPO + valeur
    actor.train(); critic.train()
    a = torch.randint(0, 90, (300,))
    adv = torch.randn(300)
    grads = {}
    for skip in (True, False):
        actor.zero_grad(); critic.zero_grad()
        actor.encoder.skip_padding = critic.encoder.skip_padding = skip
        d = actor.dist(x)
        loss = policy_loss(d.log_prob(a), d.log_prob(a).detach() + 0.1, adv, 0.2, "spo") + critic(x).pow(2).mean()
        loss.backward()
        grads[skip] = [p.grad.clone() for p in list(actor.parameters()) + list(critic.parameters())]
    for g1, g0 in zip(grads[True], grads[False]):
        assert torch.allclose(g1, g0, atol=1e-5, rtol=1e-4), (g1 - g0).abs().max()


def test_skip_padding_falls_back_for_non_prefix_masks():
    from rlagent.models.attention import build_models
    actor, _ = build_models(load_config_default(), 90)
    actor.eval()
    x = torch.as_tensor(_mixed_format_obs(300))
    x[:, -5:] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])  # masque non préfixe → chemin d'origine
    assert torch.allclose(actor.encoder(x), actor.encoder.forward_padded(x))


def load_config_default():
    from rlagent.config import load_config
    return load_config(["laptop"])


def test_adaptive_kl_lr_rule_and_bounds():
    from rlagent.algo.learner import adapt_actor_lr, make_optimizer
    from rlagent.models.attention import build_models
    cfg = _model_cfg()
    actor, critic = build_models(cfg, 90)
    opt = make_optimizer(actor, critic, cfg)
    lr0 = opt.param_groups[0]["lr"]
    # lr fixe (défaut) : rien ne bouge
    assert adapt_actor_lr(opt, 1.0, cfg) == lr0
    cfg["ppo"].update(lr_schedule="adaptive_kl", target_kl=0.01, lr_min=1e-5, lr_max=1e-3)
    assert adapt_actor_lr(opt, 0.001, cfg) == pytest.approx(lr0 * 1.5)     # KL trop faible → lr ↑
    assert adapt_actor_lr(opt, 0.01, cfg) == pytest.approx(lr0 * 1.5)      # dans [cible/2, 2×cible] → inchangé
    assert adapt_actor_lr(opt, 0.05, cfg) == pytest.approx(lr0)            # KL trop fort → lr ↓
    for _ in range(50):
        adapt_actor_lr(opt, 0.0, cfg)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)                # borne haute
    for _ in range(100):
        adapt_actor_lr(opt, 1.0, cfg)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)                # borne basse
    assert opt.param_groups[1]["lr"] == pytest.approx(float(cfg["ppo"]["lr_critic"]))  # critique jamais touché
    assert adapt_actor_lr(opt, float("nan"), cfg) == pytest.approx(1e-5)


def test_final_kl_zero_before_update_and_positive_after():
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch, final_kl, make_optimizer, ppo_update
    from rlagent.env.obs import OBS_DIM
    from rlagent.models.attention import build_models
    cfg = _model_cfg()
    cfg["ppo"].update(batch_steps=512, minibatch_size=256, device_chunk=128, epochs=2, lr_actor=3e-3)
    torch.manual_seed(0)
    actor, critic = build_models(cfg, 90)
    x = np.zeros((512, OBS_DIM), dtype=np.float32)
    x[:, -5] = 1.0
    with torch.no_grad():
        a, lp = actor.act(torch.as_tensor(x))
    kl0 = final_kl(actor, torch.as_tensor(x), a, lp, torch.device("cpu"), 128)
    assert abs(kl0) < 1e-6
    trajs = [Trajectory([x[0]], [int(ai)], [float(li)], [1.0 if int(ai) == 7 else 0.0], None) for ai, li in zip(a, lp)]
    batch = build_batch(trajs, critic, RunningStats(), cfg, torch.device("cpu"))
    stats = ppo_update(actor, critic, make_optimizer(actor, critic, cfg), batch, cfg, torch.device("cpu"))
    assert stats["kl_final"] > 0 and np.isfinite(stats["kl_final"])


def test_kl_early_stop_only_in_adaptive_mode():
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch, make_optimizer, ppo_update
    from rlagent.env.obs import OBS_DIM
    from rlagent.models.attention import build_models
    x = np.zeros((512, OBS_DIM), dtype=np.float32)
    x[:, -5] = 1.0
    for mode, expected_mb in (("fixed", 8), ("adaptive_kl", 1)):
        cfg = _model_cfg()
        cfg["ppo"].update(batch_steps=512, minibatch_size=128, device_chunk=128, epochs=2, lr_actor=3e-3,
                          lr_schedule=mode, target_kl=1e-9, kl_stop=4.0)
        torch.manual_seed(0)
        actor, critic = build_models(cfg, 90)
        with torch.no_grad():
            a, lp = actor.act(torch.as_tensor(x))
        trajs = [Trajectory([x[0]], [int(ai)], [float(li)], [1.0 if int(ai) == 7 else 0.0], None)
                 for ai, li in zip(a, lp)]
        batch = build_batch(trajs, critic, RunningStats(), cfg, torch.device("cpu"))
        stats = ppo_update(actor, critic, make_optimizer(actor, critic, cfg), batch, cfg, torch.device("cpu"))
        # 1er pas toujours appliqué (KL = 0 avant), puis arrêt dès que KL > 4 × cible en mode adaptive_kl
        assert stats["n_minibatches"] == expected_mb, (mode, stats)
        assert stats["early_stop"] == float(mode == "adaptive_kl")


def test_policy_loss_weight_ones_is_identity():
    from rlagent.algo.losses import policy_loss
    torch.manual_seed(0)
    lp, old, adv = torch.randn(64) * 0.1, torch.randn(64) * 0.1, torch.randn(64)
    for kind in ("ppo", "spo"):
        assert torch.allclose(policy_loss(lp, old, adv, 0.2, kind), policy_loss(lp, old, adv, 0.2, kind, torch.ones(64)))


def _bandit_batch(actor, critic, cfg, behavior_actor=None):
    from rlagent.algo.collector import Trajectory
    from rlagent.algo.gae import RunningStats
    from rlagent.algo.learner import build_batch
    from rlagent.env.obs import OBS_DIM
    x = np.zeros((512, OBS_DIM), dtype=np.float32)
    x[:, -5] = 1.0
    with torch.no_grad():
        a, lp = (behavior_actor or actor).act(torch.as_tensor(x))
    trajs = [Trajectory([x[0]], [int(ai)], [float(li)], [1.0 if int(ai) == 7 else 0.0], None) for ai, li in zip(a, lp)]
    return build_batch(trajs, critic, RunningStats(), cfg, torch.device("cpu"))


def test_decoupled_update_equals_coupled_without_lag():
    """Sans retard de collecte (π_collecte = π_prox), l'objectif découplé est exactement l'objectif d'origine."""
    import copy
    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.models.attention import build_models
    cfg = _model_cfg()
    cfg["ppo"].update(batch_steps=512, minibatch_size=256, device_chunk=128, epochs=2, lr_actor=3e-3)
    torch.manual_seed(0)
    actor, critic = build_models(cfg, 90)
    batch = _bandit_batch(actor, critic, cfg)
    results = []
    for decoupled in (False, True):
        a, c = copy.deepcopy(actor), copy.deepcopy(critic)
        torch.manual_seed(1)
        stats = ppo_update(a, c, make_optimizer(a, c, cfg), batch, cfg, torch.device("cpu"), decoupled=decoupled)
        results.append((stats, torch.cat([p.flatten() for p in a.parameters()])))
    (s0, p0), (s1, p1) = results
    assert s1["behavior_kl"] < 1e-6
    assert abs(s0["policy_loss"] - s1["policy_loss"]) < 1e-5
    assert torch.allclose(p0, p1, atol=1e-5)


def test_decoupled_update_with_stale_behavior_policy():
    """Collecte faite par une politique plus ancienne : le KL de retard est mesuré, la zone de confiance est
    centrée sur π_prox (kl_final = déplacement de CET update), et l'apprentissage progresse."""
    import copy
    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.env.obs import OBS_DIM
    from rlagent.models.attention import build_models
    cfg = _model_cfg()
    cfg["ppo"].update(batch_steps=512, minibatch_size=256, device_chunk=128, epochs=4, lr_actor=3e-3, ent_coef=0.0)
    torch.manual_seed(0)
    actor, critic = build_models(cfg, 90)
    opt = make_optimizer(actor, critic, cfg)
    x = np.zeros((1, OBS_DIM), dtype=np.float32)
    x[0, -5] = 1.0
    p_before = torch.softmax(actor(torch.as_tensor(x)), -1)[0, 7].item()
    stale = copy.deepcopy(actor)
    for it in range(30):
        batch = _bandit_batch(actor, critic, cfg, behavior_actor=stale)   # données jouées avec 1 itération de retard
        stale = copy.deepcopy(actor)
        stats = ppo_update(actor, critic, opt, batch, cfg, torch.device("cpu"), decoupled=True)
        assert np.isfinite(stats["policy_loss"]) and stats["kl_final"] >= 0
        if it > 0:
            assert stats["behavior_kl"] > 0
    p_after = torch.softmax(actor(torch.as_tensor(x)), -1)[0, 7].item()
    assert p_after > max(0.1, 4 * p_before), (p_before, p_after)


def test_early_stop_before_any_step_logs_real_stats():
    """Arrêt dès le 1er minibatch (retard de collecte > seuil, mode couplé) : entropie journalisée réelle, pas 0."""
    import copy
    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.models.attention import build_models
    cfg = _model_cfg()
    cfg["ppo"].update(batch_steps=512, minibatch_size=256, device_chunk=128, epochs=2, lr_actor=3e-2,
                      lr_schedule="adaptive_kl", target_kl=1e-6, kl_stop=1.0)
    torch.manual_seed(0)
    actor, critic = build_models(cfg, 90)
    stale = copy.deepcopy(actor)
    opt = make_optimizer(actor, critic, cfg)
    ppo_update(actor, critic, opt, _bandit_batch(actor, critic, cfg), cfg, torch.device("cpu"))  # éloigne l'acteur
    stats = ppo_update(actor, critic, opt, _bandit_batch(actor, critic, cfg, behavior_actor=stale), cfg,
                       torch.device("cpu"))
    assert stats["n_minibatches"] == 0 and stats["early_stop"] == 1.0
    assert stats["entropy"] > 1.0 and stats["approx_kl"] > 0


def test_pool_cache_holds_all_recent_versions(tmp_path, monkeypatch):
    """Toutes les versions tirables restent en mémoire : chaque fichier n'est chargé qu'une fois."""
    from rlagent.models.attention import build_models
    from rlagent.selfplay.pool import OpponentPool
    from rlagent.train import build_actor_from_model_cfg
    cfg = _model_cfg()
    actor, _ = build_models(cfg, 90)
    pool = OpponentPool(tmp_path, recent=8, cache_size=4, seed=0)
    for i in range(8):
        pool.add(actor.state_dict(), cfg["model"], 1000 * (i + 1), 1000.0)
    loads = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (loads.append(a[0]), real_load(*a, **k))[1])
    for _ in range(3):
        for e in pool.entries:
            pool.get_actor(e, build_actor_from_model_cfg)
    assert len(loads) == 8
