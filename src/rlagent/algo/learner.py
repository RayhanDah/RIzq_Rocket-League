"""Mise à jour PPO/SPO : valeurs, GAE, normalisations, epochs × minibatches (ARCHITECTURE §6).

Même calcul sur CPU, GPU (cuda / xpu) et TPU (xla). Sur TPU, toutes les formes de tenseurs restent fixes (dernier
morceau complété, cf. accel.padded_chunks) et chaque pas d'optimiseur se termine par accel.step_barrier."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .. import accel
from .collector import Trajectory
from .gae import RunningStats, compute_gae, discounted_returns
from .losses import policy_loss, ratio_stats

STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "ratio_dev")


@torch.no_grad()
def batched_values(critic: nn.Module, obs: np.ndarray, device: torch.device, chunk: int) -> np.ndarray:
    critic.eval()
    pad = accel.static_shapes(device)
    out = []
    for s, e, size in accel.padded_chunks(len(obs), chunk, pad):
        x = torch.as_tensor(obs[s:e], dtype=torch.float32)
        if pad:
            x = accel.pad_rows(x, size)
        out.append(critic(x.to(device)).float().cpu().numpy()[:e - s])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def build_batch(trajs: List[Trajectory], critic: nn.Module, ret_stats: RunningStats, cfg: Dict[str, Any],
                device: torch.device) -> Dict[str, np.ndarray]:
    p = cfg["ppo"]
    gamma, lam = float(p["gamma"]), float(p["gae_lambda"])
    # trajectoires en listes (collecte centrale) ou en tableaux (collecte locale, Trajectory.compact) : mêmes valeurs
    obs = np.concatenate([np.asarray(t.obs, dtype=np.float32).reshape(len(t), -1) for t in trajs])
    boot_idx = [i for i, t in enumerate(trajs) if t.bootstrap_obs is not None]
    boot_obs = (np.stack([trajs[i].bootstrap_obs for i in boot_idx]).astype(np.float32)
                if boot_idx else np.zeros((0, obs.shape[1]), np.float32))
    all_vals = batched_values(critic, np.concatenate([obs, boot_obs]), device, int(p["device_chunk"]))
    values, boot_vals = all_vals[:len(obs)], all_vals[len(obs):]
    boot_map = {i: float(boot_vals[k]) for k, i in enumerate(boot_idx)}

    scale = 1.0 / ret_stats.std
    advs, rets, raw_rewards = [], [], []
    pos = 0
    for i, t in enumerate(trajs):
        n = len(t)
        r = np.asarray(t.rewards, dtype=np.float32)
        v = values[pos:pos + n]
        adv = compute_gae(r * scale, v, boot_map.get(i, 0.0), gamma, lam)
        advs.append(adv)
        rets.append(adv + v)
        raw_rewards.append(r)
        pos += n
    # Mise à jour des stats de retour APRÈS usage (l'échelle de ce lot reste cohérente)
    for r in raw_rewards:
        ret_stats.update(discounted_returns(r, gamma))

    return {
        "obs": obs,
        "actions": np.concatenate([np.asarray(t.actions, dtype=np.int64) for t in trajs]),
        "log_probs": np.concatenate([np.asarray(t.log_probs, dtype=np.float32) for t in trajs]),
        "advantages": np.concatenate(advs),
        "returns": np.concatenate(rets).astype(np.float32),
        "values": values.astype(np.float32),
        "raw_rewards": np.concatenate(raw_rewards),
        "reward_scale": np.array(scale, dtype=np.float32),
    }


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var < 1e-8 else 1.0 - float(np.var(target - pred)) / var


@torch.no_grad()
def batched_log_probs(actor: nn.Module, t_obs: torch.Tensor, t_act: torch.Tensor, device: torch.device,
                      chunk: int) -> torch.Tensor:
    """log π(a|s) sur tout le lot, par morceaux ; résultat sur le device de `t_act`."""
    actor.eval()
    pad = accel.static_shapes(device)
    out = []
    for s, e, size in accel.padded_chunks(len(t_act), chunk, pad):
        o, a = t_obs[s:e], t_act[s:e]
        if pad:
            o, a = accel.pad_rows(o, size), accel.pad_rows(a, size)
        lp = actor.dist(o.to(device)).log_prob(a.to(device)).float()
        # la coupe [:e - s] se fait APRÈS le transfert : sur XLA, elle créerait une forme (donc une compilation)
        # différente à chaque taille de lot
        out.append(lp.to(t_act.device)[:e - s] if pad else lp[:e - s].to(t_act.device))
    actor.train()
    return torch.cat(out) if out else torch.zeros(0, device=t_act.device)


def batch_on_device(p: Dict[str, Any], device: torch.device) -> bool:
    """`ppo.batch_on_device` : lot entier copié une fois sur le GPU (au lieu d'un transfert par morceau).
    auto = oui sur GPU NVIDIA. Jamais sur XLA : la taille du lot varie d'une itération à l'autre (une compilation
    par taille) ; les morceaux, eux, ont une taille fixe."""
    opt = p.get("batch_on_device", "auto")
    if device.type in ("cpu", "xla"):
        return False
    return device.type == "cuda" if opt in (None, "auto") else bool(opt)


def ppo_update(actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer, batch: Dict[str, np.ndarray],
               cfg: Dict[str, Any], device: torch.device, decoupled: bool = False,
               after_step: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """decoupled=True (collecte asynchrone) : zone de confiance centrée sur la politique au début de l'update
    (π_prox) au lieu de la politique de collecte, pondérée par w = π_prox / π_collecte (cf. losses.py).
    Sinon SPO ramènerait la politique vers celle de collecte, en retard d'une itération.
    after_step : appelé après chaque pas d'optimiseur (ex. projection des normes, algo/norms.py)."""
    p = cfg["ppo"]
    device = torch.device(device)
    kind, eps = p["policy_loss"], float(p["clip_eps"])
    n = len(batch["actions"])
    mb_size = min(int(p["minibatch_size"]), n)
    chunk = int(p["device_chunk"])
    src = device if batch_on_device(p, device) else torch.device("cpu")   # où vit le lot

    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    t_obs = torch.as_tensor(batch["obs"], dtype=torch.float32).to(src)
    t_act = torch.as_tensor(batch["actions"], dtype=torch.long).to(src)
    t_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32).to(src)
    t_adv = torch.as_tensor(adv, dtype=torch.float32).to(src)
    t_ret = torch.as_tensor(batch["returns"], dtype=torch.float32).to(src)
    behavior_kl = 0.0
    if decoupled:
        t_center = batched_log_probs(actor, t_obs, t_act, device, chunk)          # log π_prox
        log_w = t_center - t_logp
        behavior_kl = float(((torch.exp(log_w) - 1.0) - log_w).mean())          # retard de la collecte
        t_w = torch.exp(log_w).clamp(max=float(p.get("is_weight_max", 2.0)))
    else:
        t_center, t_w = t_logp, None
    # Garde-fou (mode adaptive_kl) : si la politique s'est déjà trop éloignée de celle de collecte
    # (KL > kl_stop × target_kl, mesuré avant le pas), on n'applique pas ce pas et on arrête l'itération.
    stop_kl = (float(p.get("kl_stop", 4.0)) * float(p.get("target_kl", 0.01))
               if str(p.get("lr_schedule", "fixed")) == "adaptive_kl" else float("inf"))
    check_kl = stop_kl != float("inf")
    early_stop = False

    actor.train()
    critic.train()
    # Statistiques accumulées sur le device (STAT_KEYS) : une seule lecture par minibatch (garde-fou KL) ou par
    # update, au lieu de 6 attentes du GPU / TPU par morceau.
    total = torch.zeros(len(STAT_KEYS), dtype=torch.float32, device=device)
    first: List[float] = []
    n_mb = 0
    for _ in range(int(p["epochs"])):
        if early_stop:
            break
        perm = torch.randperm(n)
        for s in range(0, n - mb_size + 1, mb_size):
            mb = perm[s:s + mb_size].to(src)
            optimizer.zero_grad(set_to_none=True)
            acc = torch.zeros(len(STAT_KEYS), dtype=torch.float32, device=device)
            for c in range(0, len(mb), chunk):
                idx = mb[c:c + chunk]
                w = len(idx) / len(mb)
                o = t_obs[idx].to(device, non_blocking=True)
                dist = actor.dist(o)
                logp = dist.log_prob(t_act[idx].to(device))
                old = t_center[idx].to(device)
                a = t_adv[idx].to(device)
                pl = policy_loss(logp, old, a, eps, kind, None if t_w is None else t_w[idx].to(device))
                ent = dist.entropy().mean()
                v = critic(o)
                vl = torch.nn.functional.mse_loss(v, t_ret[idx].to(device))
                loss = (pl - float(p["ent_coef"]) * ent + float(p["vf_coef"]) * vl) * w
                loss.backward()
                lp = logp.detach()
                kl, cf = ratio_stats(lp, old, eps)
                rd = (torch.exp(lp - old) - 1).abs().mean()
                acc = acc + torch.stack([pl.detach(), vl.detach(), ent.detach(), kl, cf, rd]).float() * w
            if check_kl:
                accel.step_barrier(device)     # XLA : avant + arrière + stats en un seul graphe, avant de lire le KL
                if float(acc[STAT_KEYS.index("approx_kl")]) > stop_kl:
                    early_stop = True
                    optimizer.zero_grad(set_to_none=True)
                    if n_mb == 0:  # aucun pas appliqué : on journalise quand même ce minibatch (pas des zéros)
                        first = acc.tolist()
                    break
            nn.utils.clip_grad_norm_(actor.parameters(), float(p["max_grad_norm"]))
            nn.utils.clip_grad_norm_(critic.parameters(), float(p["max_grad_norm"]))
            optimizer.step()
            if after_step is not None:
                after_step()
            total = total + acc
            accel.step_barrier(device)
            n_mb += 1
    vals = (total / n_mb).tolist() if n_mb > 0 else (first or [0.0] * len(STAT_KEYS))
    stats: Dict[str, float] = dict(zip(STAT_KEYS, (float(x) for x in vals)))
    stats["n_minibatches"] = n_mb
    stats["early_stop"] = float(early_stop)
    stats["explained_variance"] = explained_variance(batch["values"], batch["returns"])
    stats["kl_final"] = final_kl(actor, t_obs, t_act, t_center, device, chunk)
    stats["behavior_kl"] = behavior_kl
    return stats


@torch.no_grad()
def final_kl(actor: nn.Module, t_obs: torch.Tensor, t_act: torch.Tensor, t_logp: torch.Tensor,
             device: torch.device, chunk: int, max_samples: int = 8_192) -> float:
    """KL(politique de départ ‖ politique après l'update), estimateur k3, sur un sous-échantillon du lot
    (départ = politique de collecte, ou π_prox en mode découplé).

    C'est le déplacement réel de la politique sur l'itération (`learn/approx_kl` est une moyenne sur les
    minibatchs, dont le premier vaut toujours ~0)."""
    n = len(t_act)
    idx = torch.randperm(n)[:max_samples] if n > max_samples else torch.arange(n)
    idx = idx.to(t_act.device)
    pad = accel.static_shapes(device)
    actor.eval()
    total = 0.0
    for c, e, size in accel.padded_chunks(len(idx), chunk, pad):
        sel = idx[c:e]
        o, a, lp0 = t_obs[sel], t_act[sel], t_logp[sel]
        if pad:
            o, a, lp0 = accel.pad_rows(o, size), accel.pad_rows(a, size), accel.pad_rows(lp0, size)
        logp = actor.dist(o.to(device)).log_prob(a.to(device))
        log_ratio = logp - lp0.to(device)
        k3 = (torch.exp(log_ratio) - 1.0) - log_ratio
        # XLA : morceau complet rapatrié puis coupé sur CPU (forme fixe sur le device)
        total += float(k3.cpu()[:e - c].sum()) if pad else float(k3.sum())
    actor.train()
    return total / max(1, len(idx))


def adapt_actor_lr(optimizer: torch.optim.Optimizer, kl: float, cfg: Dict[str, Any]) -> float:
    """Learning rate de l'acteur piloté par le KL (`ppo.lr_schedule: adaptive_kl`), une fois par itération.

    KL > 2 × cible → lr / 1,5 ; KL < cible / 2 → lr × 1,5 ; borné à [lr_min, lr_max]. Même règle que rsl_rl,
    appliquée au déplacement de la politique sur toute l'itération. Le lr adapté est dans l'état de l'optimiseur,
    donc sauvegardé avec les checkpoints. Renvoie le lr de l'acteur après ajustement."""
    p = cfg["ppo"]
    group = optimizer.param_groups[0]          # groupe 0 = acteur (cf. make_optimizer)
    if str(p.get("lr_schedule", "fixed")) != "adaptive_kl" or not np.isfinite(kl):
        return float(group["lr"])
    target = float(p.get("target_kl", 0.01))
    lr = float(group["lr"])
    if kl > 2.0 * target:
        lr /= 1.5
    elif kl < 0.5 * target:
        lr *= 1.5
    group["lr"] = float(min(max(lr, float(p.get("lr_min", 5e-5))), float(p.get("lr_max", 1e-3))))
    return float(group["lr"])


def make_optimizer(actor: nn.Module, critic: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    p = cfg["ppo"]
    return torch.optim.Adam([
        {"params": actor.parameters(), "lr": float(p["lr_actor"])},
        {"params": critic.parameters(), "lr": float(p["lr_critic"])},
    ])

