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

from ..env.obs import BALL_PRED_DIM, BOOST_DIM, MAX_OTHERS, OBS_DIM, OTHER_DIM, RACE_DIM, SELF_DIM


def mlp(sizes: Sequence[int], act=nn.GELU, last_act: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


SKIP_PADDING_MIN_BATCH = 256


# Tailles possibles du bloc « soi » : les options d'observation s'ajoutent TOUJOURS à la fin, dans cet ordre
# (base → + prédiction de balle → + course à la balle → + pads de boost). Un bloc plus court est donc un préfixe d'un
# bloc plus long.
SELF_DIMS = (SELF_DIM, SELF_DIM + BALL_PRED_DIM, SELF_DIM + BALL_PRED_DIM + RACE_DIM,
             SELF_DIM + BALL_PRED_DIM + RACE_DIM + BOOST_DIM)


def fit_obs(obs: torch.Tensor, self_dim: int) -> torch.Tensor:
    """Adapte une observation à un réseau qui attend un bloc « soi » plus court (agent plus ancien : sans prédiction
    de balle, sans course à la balle ou sans pads de boost) : on retire les colonnes en trop et on retrouve exactement
    l'observation qu'il connaît. Permet de faire jouer les anciens agents (pool, évaluations, RLViser) dans un env plus
    riche."""
    tail = MAX_OTHERS * OTHER_DIM + MAX_OTHERS
    expected = self_dim + tail
    if obs.shape[-1] == expected:
        return obs
    obs_self = obs.shape[-1] - tail
    if self_dim in SELF_DIMS and obs_self in SELF_DIMS and obs_self > self_dim:
        return torch.cat([obs[..., :self_dim], obs[..., obs_self:]], dim=-1)
    raise ValueError(f"observation de taille {obs.shape[-1]} incompatible avec ce réseau (attendu {expected})")


class EntityEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 2,
                 self_dim: int = SELF_DIM):
        super().__init__()
        self.self_dim = int(self_dim)
        self.self_embed = mlp([self.self_dim, d_model * 2, d_model])
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
        sd = self.self_dim
        mask = obs[:, sd + MAX_OTHERS * OTHER_DIM:] > 0.5
        counts = mask.sum(dim=1)
        slots = torch.arange(MAX_OTHERS, device=obs.device)
        if not bool(torch.equal(mask, slots[None, :] < counts[:, None])):
            return self.forward_padded(obs)
        order, outs = [], []
        for k in counts.unique().tolist():
            sel = (counts == k).nonzero(as_tuple=True)[0]
            o = obs.index_select(0, sel)
            tokens = self.self_embed(o[:, :sd]).unsqueeze(1)
            if k > 0:
                others = o[:, sd:sd + k * OTHER_DIM].reshape(-1, k, OTHER_DIM)
                tokens = torch.cat([tokens, self.other_embed(others)], dim=1)
            outs.append(self.out_norm(self.transformer(tokens)[:, 0]))
            order.append(sel)
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs).index_select(0, torch.argsort(torch.cat(order)))

    def forward_padded(self, obs: torch.Tensor) -> torch.Tensor:
        """Implémentation d'origine : les 6 jetons, avec masque de padding."""
        b = obs.shape[0]
        sd = self.self_dim
        self_vec = obs[:, :sd]
        others = obs[:, sd:sd + MAX_OTHERS * OTHER_DIM].reshape(b, MAX_OTHERS, OTHER_DIM)
        mask = obs[:, sd + MAX_OTHERS * OTHER_DIM:]
        tokens = torch.cat([self.self_embed(self_vec).unsqueeze(1), self.other_embed(others)], dim=1)
        # True = position ignorée. Le jeton "soi" n'est jamais masqué, donc aucune ligne entièrement masquée.
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), mask < 0.5], dim=1)
        h = self.transformer(tokens, src_key_padding_mask=pad)
        return self.out_norm(h[:, 0])


class Actor(nn.Module):
    def __init__(self, n_actions: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 ff_mult: int = 2, head_hidden: Sequence[int] = (256, 256), self_dim: int = SELF_DIM):
        super().__init__()
        self.encoder = EntityEncoder(d_model, n_heads, n_layers, ff_mult, self_dim)
        self.head = nn.Sequential(mlp([d_model, *head_hidden]), nn.Linear(head_hidden[-1], n_actions))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(fit_obs(obs, self.encoder.self_dim)))

    # validate_args=False : mêmes valeurs, mais sans les vérifications `if not valid.all()` qui forcent le GPU / TPU
    # à attendre le résultat à chaque appel (et coupent le graphe XLA en morceaux).
    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(obs), validate_args=False)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        d = torch.distributions.Categorical(logits=logits, validate_args=False)
        action = logits.argmax(-1) if deterministic else d.sample()
        return action, d.log_prob(action)


class Critic(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff_mult: int = 2,
                 head_hidden: Sequence[int] = (256, 256), self_dim: int = SELF_DIM):
        super().__init__()
        self.encoder = EntityEncoder(d_model, n_heads, n_layers, ff_mult, self_dim)
        self.head = nn.Sequential(mlp([d_model, *head_hidden]), nn.Linear(head_hidden[-1], 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(fit_obs(obs, self.encoder.self_dim))).squeeze(-1)


def is_mlp(model_cfg: Dict) -> bool:
    """`model.type` : attention (défaut, ARCHITECTURE §5) | mlp (models/mlp.py, profil s10)."""
    return str(model_cfg.get("type", "attention")) == "mlp"


def actor_from_model_cfg(model_cfg: Dict, n_actions: int) -> nn.Module:
    """Acteur seul à partir d'une section `model` (versions du pool, ancres, workers en collecte locale)."""
    if is_mlp(model_cfg):
        from .mlp import MLPActor
        return MLPActor(n_actions, tuple(model_cfg["actor_hidden"]), int(model_cfg.get("self_dim", SELF_DIM)))
    return Actor(n_actions, d_model=model_cfg["d_model"], n_heads=model_cfg["n_heads"],
                 n_layers=model_cfg["n_layers"], ff_mult=model_cfg["ff_mult"],
                 head_hidden=tuple(model_cfg["head_hidden"]), self_dim=int(model_cfg.get("self_dim", SELF_DIM)))


def actor_self_dim(actor: nn.Module) -> int:
    """Taille du bloc « soi » qu'attend un acteur (attention ou MLP)."""
    return int(actor.encoder.self_dim if isinstance(actor, Actor) else actor.self_dim)


def build_models(cfg: Dict, n_actions: int):
    m = cfg["model"]
    if is_mlp(m):
        from .mlp import MLPActor, MLPCritic
        sd = int(m.get("self_dim", SELF_DIM))
        return MLPActor(n_actions, tuple(m["actor_hidden"]), sd), MLPCritic(tuple(m["critic_hidden"]), sd)
    kw = dict(d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"], ff_mult=m["ff_mult"],
              head_hidden=tuple(m["head_hidden"]), self_dim=int(m.get("self_dim", SELF_DIM)))
    return Actor(n_actions, **kw), Critic(**kw)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


assert OBS_DIM == SELF_DIM + MAX_OTHERS * OTHER_DIM + MAX_OTHERS
