"""Réseaux (ARCHITECTURE §5).

Encodeur commun : un jeton "soi" (soi + balle + global) et jusqu'à 5 jetons "autres voitures", passés dans
un petit Transformer avec masque de padding. La sortie du jeton "soi" alimente la tête.
→ invariant au nombre de joueurs : le même réseau joue en 1v1, 2v2 et 3v3.

Acteur : logits sur les 90 actions. Critique : séparé (pas de partage de poids), même architecture,
sortie scalaire. Rocket League étant entièrement observable, le critique "centralisé" de MAPPO reçoit ici
l'état global vu depuis l'agent (variante "agent-specific global state").
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from ..env.obs import MAX_OTHERS, OBS_DIM, OTHER_DIM, SELF_DIM


def mlp(sizes: Sequence[int], act=nn.GELU, last_act: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


SKIP_PADDING_MIN_BATCH = 256


class EntityEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 2):
        super().__init__()
        self.self_embed = mlp([SELF_DIM, d_model * 2, d_model])
        self.other_embed = mlp([OTHER_DIM, d_model, d_model])
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        # Ne calcule que les jetons réels (voir forward). Simple attribut : n'entre pas dans les checkpoints.
        self.skip_padding = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Sortie identique à `forward_padded`, mais sans calculer les jetons masqués.

        Les voitures présentes occupent toujours les premiers emplacements (masque en préfixe, cf. env/obs.py).
        Comme un jeton masqué n'influence aucun jeton réel (masque d'attention), on regroupe les échantillons
        par nombre de voitures et on passe seulement [soi + k voitures] dans le Transformer : 2 jetons en 1v1
        au lieu de 6. Si un masque n'est pas en préfixe (cas non produit par l'env), repli sur forward_padded.
        """
        # Petits lots (inférence pendant la collecte) : le regroupement coûte plus qu'il ne rapporte.
        if not self.skip_padding or obs.shape[0] < SKIP_PADDING_MIN_BATCH:
            return self.forward_padded(obs)
        mask = obs[:, SELF_DIM + MAX_OTHERS * OTHER_DIM:] > 0.5
        counts = mask.sum(dim=1)
        slots = torch.arange(MAX_OTHERS, device=obs.device)
        if not bool(torch.equal(mask, slots[None, :] < counts[:, None])):
            return self.forward_padded(obs)
        order, outs = [], []
        for k in counts.unique().tolist():
            sel = (counts == k).nonzero(as_tuple=True)[0]
            o = obs.index_select(0, sel)
            tokens = self.self_embed(o[:, :SELF_DIM]).unsqueeze(1)
            if k > 0:
                others = o[:, SELF_DIM:SELF_DIM + k * OTHER_DIM].reshape(-1, k, OTHER_DIM)
                tokens = torch.cat([tokens, self.other_embed(others)], dim=1)
            outs.append(self.out_norm(self.transformer(tokens)[:, 0]))
            order.append(sel)
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs).index_select(0, torch.argsort(torch.cat(order)))

    def forward_padded(self, obs: torch.Tensor) -> torch.Tensor:
        """Implémentation d'origine : les 6 jetons, avec masque de padding."""
        b = obs.shape[0]
        self_vec = obs[:, :SELF_DIM]
        others = obs[:, SELF_DIM:SELF_DIM + MAX_OTHERS * OTHER_DIM].reshape(b, MAX_OTHERS, OTHER_DIM)
        mask = obs[:, SELF_DIM + MAX_OTHERS * OTHER_DIM:]
        tokens = torch.cat([self.self_embed(self_vec).unsqueeze(1), self.other_embed(others)], dim=1)
        # True = position ignorée. Le jeton "soi" n'est jamais masqué, donc aucune ligne entièrement masquée.
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), mask < 0.5], dim=1)
        h = self.transformer(tokens, src_key_padding_mask=pad)
        return self.out_norm(h[:, 0])


class Actor(nn.Module):
    def __init__(self, n_actions: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 ff_mult: int = 2, head_hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.encoder = EntityEncoder(d_model, n_heads, n_layers, ff_mult)
        self.head = nn.Sequential(mlp([d_model, *head_hidden]), nn.Linear(head_hidden[-1], n_actions))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        d = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else d.sample()
        return action, d.log_prob(action)


class Critic(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 2,
                 head_hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.encoder = EntityEncoder(d_model, n_heads, n_layers, ff_mult)
        self.head = nn.Sequential(mlp([d_model, *head_hidden]), nn.Linear(head_hidden[-1], 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs)).squeeze(-1)


def build_models(cfg: Dict, n_actions: int):
    m = cfg["model"]
    kw = dict(d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"], ff_mult=m["ff_mult"],
              head_hidden=tuple(m["head_hidden"]))
    return Actor(n_actions, **kw), Critic(**kw)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


assert OBS_DIM == SELF_DIM + MAX_OTHERS * OTHER_DIM + MAX_OTHERS
