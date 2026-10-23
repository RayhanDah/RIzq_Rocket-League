"""Réseaux MLP (`model.type: mlp`, profil s10, refonte du 23/10/2026).

Acteur et critique séparés, perceptrons sur l'observation plate de env/obs.py :
    [ soi + global | 5 emplacements « autres voitures » | masque ]
En 1v1, seul le 1er emplacement est rempli (l'adversaire), les autres restent à zéro.

C'est l'architecture des bots 1v1 de référence (rlgym-ppo, GigaLearn) : simple, rapide en inférence, et sans LayerNorm,
donc sans couche invariante d'échelle (pas de dérive silencieuse du lr effectif, cf. algo/norms.py). Contrairement à
l'acteur à attention, elle n'est pas invariante à l'ordre des voitures : elle joue en 2v2 / 3v3 (même observation),
mais il faut l'y entraîner.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..env.obs import MAX_OTHERS, OTHER_DIM, SELF_DIM
from .attention import Actor, fit_obs


def relu_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, int(h)), nn.ReLU()]
        in_dim = int(h)
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def flat_obs_dim(self_dim: int) -> int:
    return self_dim + MAX_OTHERS * OTHER_DIM + MAX_OTHERS


class MLPActor(nn.Module):
    def __init__(self, n_actions: int, hidden: Sequence[int] = (512, 512, 512), self_dim: int = SELF_DIM):
        super().__init__()
        self.self_dim = int(self_dim)
        self.net = relu_mlp(flat_obs_dim(self.self_dim), hidden, n_actions)
        # politique de départ quasi uniforme
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(fit_obs(obs, self.self_dim))

    dist = Actor.dist
    act = Actor.act


class MLPCritic(nn.Module):
    def __init__(self, hidden: Sequence[int] = (1024, 1024, 512), self_dim: int = SELF_DIM):
        super().__init__()
        self.self_dim = int(self_dim)
        self.net = relu_mlp(flat_obs_dim(self.self_dim), hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(fit_obs(obs, self.self_dim)).squeeze(-1)
