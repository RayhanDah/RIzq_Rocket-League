"""Contrôle de la norme des couches invariantes d'échelle (algo/norms.py, `ppo.norm_control`)."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pipeline_fake import _fake_cfg

from rlagent.algo import norms
from rlagent.config import load_config
from rlagent.env.actions import N_ACTIONS
from rlagent.env.obs import OBS_DIM
from rlagent.models.attention import build_models

SI = ["encoder.self_embed.0", "encoder.self_embed.3", "encoder.other_embed.0", "encoder.other_embed.3",
      "head.0.0", "head.0.3"]


def _models():
    torch.manual_seed(0)
    cfg = load_config([])                                # modèle de référence (2 couches de tête)
    actor, critic = build_models(cfg, N_ACTIONS)
    return actor, critic


def _obs(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g)
    obs[:, -5:] = (torch.rand(n, 5, generator=g) > 0.5).float().sort(dim=1, descending=True).values  # masque préfixe
    return obs


def _loss(actor, critic, obs):
    d = actor.dist(obs)
    a = torch.arange(len(obs)) % N_ACTIONS
    return -(d.log_prob(a) * torch.linspace(-1, 1, len(obs))).mean() + critic(obs).pow(2).mean()


def test_scale_invariant_layers_are_the_mlp_linears():
    actor, critic = _models()
    assert [n for n, _ in norms.scale_invariant_linears(actor)] == SI
    assert [n for n, _ in norms.model_layers(actor, critic)] == [f"actor.{n}" for n in SI] + [f"critic.{n}" for n in SI]


def test_reset_keeps_policy_and_adam_step_size():
    """Après w → c·w (et m / c, v / c²), la politique est la même et le pas d'Adam garde sa taille absolue :
    il devient 1/c fois plus grand relativement aux poids (c'est le gain de lr effectif mesuré au banc)."""
    actor, critic = _models()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    obs = _obs()
    for s in range(3):                                   # moments d'Adam non triviaux
        opt.zero_grad()
        _loss(actor, critic, _obs(seed=s + 1)).backward()
        opt.step()
    a2, c2 = copy.deepcopy(actor), copy.deepcopy(critic)
    opt2 = torch.optim.Adam(list(a2.parameters()) + list(c2.parameters()), lr=1e-3)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))   # sur CPU, load_state_dict partage les tenseurs
    layers2 = norms.model_layers(a2, c2)
    targets = {n: 0.5 * v for n, v in norms.norms(layers2).items()}
    with torch.no_grad():
        lp1 = torch.log_softmax(actor(obs), -1)
    factors = norms.reset_norms(layers2, targets, opt2)
    assert all(abs(c - 0.5) < 1e-6 for c in factors.values())
    assert all(abs(v - targets[n]) < 1e-4 * targets[n] for n, v in norms.norms(layers2).items())
    with torch.no_grad():
        lp2 = torch.log_softmax(a2(obs), -1)
        v1, v2 = critic(obs), c2(obs)
    assert float((lp1 - lp2).abs().max()) < 1e-3
    assert torch.allclose(v1, v2, atol=1e-3)
    # un pas de plus sur les mêmes données : même déplacement absolu des couches invariantes
    before1 = {n: lin.weight.detach().clone() for n, lin in norms.model_layers(actor, critic)}
    before2 = {n: lin.weight.detach().clone() for n, lin in layers2}
    for o, a, c in ((opt, actor, critic), (opt2, a2, c2)):
        o.zero_grad()
        _loss(a, c, obs).backward()
        o.step()
    for (n, lin1), (_, lin2) in zip(norms.model_layers(actor, critic), layers2):
        d1, d2 = lin1.weight.detach() - before1[n], lin2.weight.detach() - before2[n]
        assert torch.allclose(d1, d2, rtol=2e-2, atol=1e-6), n


def test_projector_restores_target_norms():
    actor, critic = _models()
    layers = norms.model_layers(actor, critic)
    targets = norms.norms(layers)
    proj = norms.NormProjector(layers, targets)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-2)
    for s in range(5):
        opt.zero_grad()
        _loss(actor, critic, _obs(seed=s)).backward()
        opt.step()
        proj()
    for n, v in norms.norms(layers).items():
        assert abs(v - targets[n]) < 1e-4 * targets[n], n


def test_rotation_stats():
    actor, _ = _models()
    layers = norms.scale_invariant_linears(actor)
    before = norms.flat(layers)
    st = norms.rotation_stats(layers, before)
    assert st["si_rot_deg"] < 1e-2 and abs(st["si_norm"] - float(before.norm())) < 1e-4
    with torch.no_grad():
        for _, lin in layers:
            lin.weight.mul_(3.0)
            lin.bias.mul_(3.0)
    assert norms.rotation_stats(layers, before)["si_rot_deg"] < 1e-2      # changer d'échelle n'est pas tourner


def test_warm_start_with_norm_control_end_to_end(tmp_path):
    """Démarrage à chaud avec remise des normes + NaP : métriques, normes à la cible, cibles dans le checkpoint,
    reprise sans nouvelle remise."""
    from rlagent.train import find_latest_checkpoint, train
    base = _fake_cfg(train={"n_workers": 2, "total_steps": 4000, "checkpoint_every_steps": 2000})
    base["run_name"] = "pytest_norm_a"
    run_a = tmp_path / "a"
    run_a.mkdir()
    train(base, run_a)
    lines_a = [json.loads(x) for x in (run_a / "metrics.jsonl").read_text().splitlines()]
    assert all("learn/si_norm" in l and "learn/si_rot_deg" in l for l in lines_a)   # toujours journalisées

    cfg = _fake_cfg(train={"n_workers": 2, "total_steps": 8000, "checkpoint_every_steps": 2000},
                    ppo={"norm_control": {"reset_on_init": True, "project": True}})
    cfg["run_name"] = "pytest_norm_b"
    run_b = tmp_path / "b"
    run_b.mkdir()
    train(cfg, run_b, find_latest_checkpoint(run_a))
    ck = torch.load(find_latest_checkpoint(run_b), map_location="cpu", weights_only=False)
    targets = ck["norm_targets"]
    actor, critic = build_models(cfg, N_ACTIONS)
    assert targets and len(targets) == len(norms.model_layers(actor, critic))
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    for n, v in norms.norms(norms.model_layers(actor, critic)).items():
        assert abs(v - targets[n]) < 1e-3 * targets[n], n
    lines = [json.loads(x) for x in (run_b / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(l["learn/policy_loss"]) for l in lines)
    norm_b = [l["learn/si_norm"] for l in lines]
    assert max(norm_b) - min(norm_b) < 1e-3 * norm_b[0]                     # NaP : la norme ne bouge plus
    # reprise du même run : les cibles viennent du checkpoint (pas de nouvelle remise)
    cfg["train"]["total_steps"] = 10000
    train(cfg, run_b, find_latest_checkpoint(run_b))
    ck2 = torch.load(find_latest_checkpoint(run_b), map_location="cpu", weights_only=False)
    assert ck2["norm_targets"] == targets
