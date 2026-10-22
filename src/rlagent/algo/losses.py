"""Loss de politique PPO / SPO (ARCHITECTURE §6).

PPO : -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
SPO (Xie et al., ICML 2025) : -E[r·A - |A| / (2ε) · (r - 1)²]
Seule cette fonction diffère entre les deux algorithmes : même GAE, mêmes normalisations, même critique.

Objectif « découplé » (Hilton et al. 2021, « Batch size-invariance for policy optimization ») pour la collecte
asynchrone : r = π / π_prox où π_prox = politique au début de l'update (centre de la zone de confiance), et chaque
terme est pondéré par w = π_prox / π_collecte (poids d'importance, sans gradient). Sans retard de collecte,
π_prox = π_collecte, w = 1 et on retrouve exactement la formule ci-dessus.
"""
from __future__ import annotations

from typing import Optional

import torch


def policy_loss(log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor,
                eps: float, kind: str = "ppo", weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    ratio = torch.exp(log_prob - old_log_prob)
    if kind == "ppo":
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        objective = torch.min(surr1, surr2)
    elif kind == "spo":
        objective = ratio * adv - adv.abs() / (2.0 * eps) * (ratio - 1.0) ** 2
    else:
        raise ValueError(f"policy_loss inconnu : {kind!r} (attendu 'ppo' ou 'spo')")
    if weight is not None:
        objective = objective * weight
    return -objective.mean()


@torch.no_grad()
def ratio_stats(log_prob: torch.Tensor, old_log_prob: torch.Tensor, eps: float):
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
    return approx_kl, clip_frac
