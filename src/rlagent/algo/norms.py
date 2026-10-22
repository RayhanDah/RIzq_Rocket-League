"""Norme des couches invariantes d'échelle (bilan du 22/10/2026, banc d'essai de la phase 0).

Une Linear suivie d'une LayerNorm calcule la même chose quelle que soit l'échelle de (W, b). Avec Adam sans weight
decay, cette norme grossit sans limite, et le même pas fait tourner les poids de moins en moins : le lr effectif
baisse en silence (Lyle et al. 2024, « Normalization and effective learning rates in reinforcement learning »).
Mesuré sur la lignée s6 : norme 22,7 → 57,8, rotation par million de steps 4,3° → 2,2° à lr constant.

Options (`ppo.norm_control`, désactivées par défaut) :
- `reset_on_init` : au démarrage à chaud, chaque couche est remise à la norme d'un réseau neuf et les moments d'Adam
  sont mis à la même échelle (m / c, v / c²). La politique ne change pas (au epsilon des LayerNorm près) ; le pas
  d'Adam garde sa taille et devient c fois plus grand relativement aux poids ;
- `project` (NaP) : après chaque pas d'optimiseur, chaque couche est ramenée à sa norme cible : le lr effectif ne
  dérive plus.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def scale_invariant_linears(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear]]:
    """Les Linear immédiatement suivies d'une LayerNorm dans un nn.Sequential (`models.attention.mlp`).

    Les couches internes du Transformer (pré-LN : leur sortie s'ajoute au flux résiduel) et la couche de sortie
    n'en font pas partie."""
    out = []
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Sequential):
            kids = list(mod.named_children())
            for (child, a), (_, b) in zip(kids, kids[1:]):
                if isinstance(a, nn.Linear) and isinstance(b, nn.LayerNorm) and a.bias is not None:
                    out.append((prefix + (f"{name}.{child}" if name else child), a))
    return out


def model_layers(actor: nn.Module, critic: nn.Module) -> list[tuple[str, nn.Linear]]:
    return scale_invariant_linears(actor, "actor.") + scale_invariant_linears(critic, "critic.")


def layer_norm(lin: nn.Linear) -> torch.Tensor:
    return torch.sqrt(lin.weight.detach().float().pow(2).sum() + lin.bias.detach().float().pow(2).sum())


def norms(layers: list[tuple[str, nn.Linear]]) -> dict[str, float]:
    return {name: float(layer_norm(lin)) for name, lin in layers}


@torch.no_grad()
def reset_norms(layers: list[tuple[str, nn.Linear]], targets: dict[str, float],
                optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    """(W, b) → c·(W, b) avec c = cible / norme ; moments d'Adam remis à la même échelle (le gradient devient g / c).
    Renvoie {couche: c}."""
    factors = {}
    for name, lin in layers:
        if name not in targets:
            continue
        c = float(targets[name]) / max(float(layer_norm(lin)), 1e-12)
        lin.weight.mul_(c)
        lin.bias.mul_(c)
        factors[name] = c
        if optimizer is not None:
            for p in (lin.weight, lin.bias):
                st = optimizer.state.get(p)
                if st and "exp_avg" in st:
                    st["exp_avg"].div_(c)
                    st["exp_avg_sq"].div_(c * c)
    return factors


class NormProjector:
    """NaP : ramène chaque couche à sa norme cible après chaque pas (tenseurs sur le device : aucune synchro)."""

    def __init__(self, layers: list[tuple[str, nn.Linear]], targets: dict[str, float]):
        self.items = [(lin, float(targets[name])) for name, lin in layers if name in targets]

    @torch.no_grad()
    def __call__(self) -> None:
        for lin, target in self.items:
            n = torch.sqrt(lin.weight.pow(2).sum() + lin.bias.pow(2).sum()).clamp_min(1e-12)
            f = target / n
            lin.weight.mul_(f)
            lin.bias.mul_(f)


def flat(layers: list[tuple[str, nn.Linear]]) -> torch.Tensor:
    return torch.cat([t.detach().float().flatten() for _, lin in layers for t in (lin.weight, lin.bias)])


def rotation_stats(layers: list[tuple[str, nn.Linear]], before: torch.Tensor) -> dict[str, float]:
    """Norme totale et rotation (degrés) des couches depuis `before` (= flat(layers) avant l'update)."""
    after = flat(layers)
    cos = float(torch.dot(before, after) / (before.norm() * after.norm()).clamp_min(1e-12))
    return {"si_norm": float(after.norm()), "si_rot_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos))))}
