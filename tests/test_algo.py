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
