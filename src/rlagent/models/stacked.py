"""Inférence groupée de plusieurs versions de l'acteur (pool d'adversaires) en un seul passage.

Pendant la collecte, chaque groupe d'arènes contient des adversaires de ~10-20 versions différentes du pool : une
inférence par version = des dizaines de petits appels par step, le goulot sur GPU. `StackedActors` empile les poids
de K versions de MÊME architecture et calcule les logits de toutes en un passage : chaque couche devient un seul
produit matriciel par lot (`baddbmm`) sur des entrées [K, B, D] (B = nombre d'observations de la version la plus
représentée, les autres sont complétées).

Même calcul que `Actor.forward` (chemin `forward_padded` : 6 jetons + masque), écrit couche par couche :
`torch.func.vmap` n'aide pas ici, faute de règle de batching pour les noyaux d'attention du Transformer (PyTorch
2.14 : repli sur une boucle par version). Seules les architectures reconnues par `stack_signature` sont empilées ;
les autres (ou une seule version) passent par la boucle habituelle.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..env.obs import MAX_OTHERS, OTHER_DIM
from .attention import Actor


def _plain_mlp(seq: nn.Sequential) -> bool:
    return all(isinstance(m, (nn.Linear, nn.LayerNorm)) or (isinstance(m, nn.GELU) and m.approximate == "none")
               for m in seq)


def _mlp_signature(actor: nn.Module) -> Optional[Tuple]:
    """Acteur MLP (models/mlp.py) : Linear / ReLU en alternance, terminé par une Linear."""
    from .mlp import MLPActor
    if type(actor) is not MLPActor or actor.training:
        return None
    mods = list(actor.net)
    if not (isinstance(mods[-1], nn.Linear) and all(isinstance(m, nn.Linear) == (i % 2 == 0)
                                                     and isinstance(m, (nn.Linear, nn.ReLU))
                                                     for i, m in enumerate(mods))):
        return None
    return "mlp", actor.self_dim, tuple((n, tuple(p.shape)) for n, p in actor.named_parameters())


def stack_signature(actor: nn.Module) -> Optional[Tuple]:
    """Clé d'architecture : deux acteurs de même clé s'empilent. None = architecture non prise en charge."""
    if type(actor) is not Actor:
        return _mlp_signature(actor)
    if actor.training:
        return None
    enc = actor.encoder
    if enc.transformer.norm is not None:
        return None
    if not (_plain_mlp(enc.self_embed) and _plain_mlp(enc.other_embed) and _plain_mlp(actor.head[0])
            and isinstance(actor.head[1], nn.Linear)):
        return None
    for layer in enc.transformer.layers:
        att = layer.self_attn
        if not (layer.norm_first and layer.activation is F.gelu and att.batch_first and att._qkv_same_embed_dim
                and att.in_proj_bias is not None and att.bias_k is None and not att.add_zero_attn):
            return None
    heads = tuple(layer.self_attn.num_heads for layer in enc.transformer.layers)
    shapes = tuple((n, tuple(p.shape)) for n, p in actor.named_parameters())
    return enc.self_dim, heads, shapes


class StackedActors:
    """Poids de K acteurs de même `stack_signature`, empilés sur le device du premier. Poids figés (copie)."""

    def __init__(self, actors: Sequence[nn.Module]):
        self.template = actors[0]
        self.k = len(actors)
        self.is_mlp = type(self.template) is not Actor
        self.self_dim = int(self.template.self_dim if self.is_mlp else self.template.encoder.self_dim)
        per = [dict(a.named_parameters()) for a in actors]
        self.p: Dict[str, torch.Tensor] = {}
        for name in per[0]:
            w = torch.stack([d[name].detach() for d in per])
            # poids des Linear transposés une fois pour toutes : y = b + x @ Wᵀ (baddbmm)
            self.p[name] = w.transpose(1, 2).contiguous() if w.dim() == 3 else w.unsqueeze(1)

    def _linear(self, x: torch.Tensor, name: str, sep: str = ".") -> torch.Tensor:
        return torch.baddbmm(self.p[name + sep + "bias"], x, self.p[name + sep + "weight"])

    def _norm(self, x: torch.Tensor, name: str, m: nn.LayerNorm) -> torch.Tensor:
        return F.layer_norm(x, m.normalized_shape, eps=m.eps) * self.p[name + ".weight"] + self.p[name + ".bias"]

    def _mlp(self, x: torch.Tensor, seq: nn.Sequential, prefix: str) -> torch.Tensor:
        for i, m in enumerate(seq):
            if isinstance(m, nn.Linear):
                x = self._linear(x, f"{prefix}.{i}")
            elif isinstance(m, nn.LayerNorm):
                x = self._norm(x, f"{prefix}.{i}", m)
            else:
                x = F.gelu(x)
        return x

    @torch.no_grad()
    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [K, B, D] (déjà au format de ces acteurs, cf. `fit_obs`) → logits [K, B, n_actions]."""
        if self.is_mlp:
            x = obs
            for i, m in enumerate(self.template.net):
                x = self._linear(x, f"net.{i}") if isinstance(m, nn.Linear) else F.relu(x)
            return x
        K, B, _ = obs.shape
        enc = self.template.encoder
        sd, T, d = enc.self_dim, MAX_OTHERS + 1, enc.d_model
        self_tok = self._mlp(obs[..., :sd], enc.self_embed, "encoder.self_embed")
        others = obs[..., sd:sd + MAX_OTHERS * OTHER_DIM].reshape(K, B * MAX_OTHERS, OTHER_DIM)
        others = self._mlp(others, enc.other_embed, "encoder.other_embed")
        x = torch.cat([self_tok.unsqueeze(2), others.reshape(K, B, MAX_OTHERS, d)], dim=2).reshape(K, B * T, d)
        # True = jeton pris en compte (le jeton « soi » l'est toujours)
        keep = torch.cat([torch.ones(K, B, 1, dtype=torch.bool, device=obs.device),
                          obs[..., sd + MAX_OTHERS * OTHER_DIM:] > 0.5], dim=2).reshape(K * B, 1, 1, T)
        layers = enc.transformer.layers
        for i, layer in enumerate(layers):
            pre = f"encoder.transformer.layers.{i}"
            h = layer.self_attn.num_heads
            qkv = self._linear(self._norm(x, pre + ".norm1", layer.norm1), pre + ".self_attn.in_proj", "_")
            q, k, v = qkv.reshape(K * B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
            if i == len(layers) - 1:
                # dernière couche : seule la sortie du jeton « soi » est lue → requête et FF sur ce jeton seulement
                q, x, T = q[:, :, :1], x.reshape(K, B, T, d)[:, :, 0], 1
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
            att = att.transpose(1, 2).reshape(K, B * T, d)
            x = x + self._linear(att, pre + ".self_attn.out_proj")
            ff = F.gelu(self._linear(self._norm(x, pre + ".norm2", layer.norm2), pre + ".linear1"))
            x = x + self._linear(ff, pre + ".linear2")
        out = self._norm(x, "encoder.out_norm", enc.out_norm)
        return self._linear(self._mlp(out, self.template.head[0], "head.0"), "head.1")


def pad_stack(obs_lists: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Observations par version (K listes de B_k obs [D]) → [K, B_max, D], complétées par des zéros (sorties
    ignorées ensuite ; le jeton « soi » n'étant jamais masqué, une ligne nulle ne produit pas de NaN)."""
    b = max(len(o) for o in obs_lists)
    out = np.zeros((len(obs_lists), b, len(obs_lists[0][0])), dtype=np.float32)
    for i, o in enumerate(obs_lists):
        out[i, :len(o)] = np.stack(o)
    return out
