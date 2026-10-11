"""Loss de politique PPO / SPO (ARCHITECTURE §6).

PPO : -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
SPO (Xie et al., ICML 2025) : -E[r·A - |A| / (2ε) · (r - 1)²]
Seule cette fonction diffère entre les deux algorithmes : même GAE, mêmes normalisations, même critique.
"""
from __future__ import annotations

import torch


def policy_loss(log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor,
                eps: float, kind: str = "ppo") -> torch.Tensor:
    ratio = torch.exp(log_prob - old_log_prob)
    if kind == "ppo":
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        return -torch.min(surr1, surr2).mean()
    if kind == "spo":
        return -(ratio * adv - adv.abs() / (2.0 * eps) * (ratio - 1.0) ** 2).mean()
    raise ValueError(f"policy_loss inconnu : {kind!r} (attendu 'ppo' ou 'spo')")


@torch.no_grad()
def ratio_stats(log_prob: torch.Tensor, old_log_prob: torch.Tensor, eps: float):
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
    return approx_kl, clip_frac
