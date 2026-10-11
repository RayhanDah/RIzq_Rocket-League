"""Mise à jour PPO/SPO : valeurs, GAE, normalisations, epochs × minibatches (ARCHITECTURE §6)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from .collector import Trajectory
from .gae import RunningStats, compute_gae, discounted_returns
from .losses import policy_loss, ratio_stats


@torch.no_grad()
def batched_values(critic: nn.Module, obs: np.ndarray, device: torch.device, chunk: int) -> np.ndarray:
    critic.eval()
    out = []
    for s in range(0, len(obs), chunk):
        out.append(critic(torch.as_tensor(obs[s:s + chunk], dtype=torch.float32, device=device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def build_batch(trajs: List[Trajectory], critic: nn.Module, ret_stats: RunningStats, cfg: Dict[str, Any],
                device: torch.device) -> Dict[str, np.ndarray]:
    p = cfg["ppo"]
    gamma, lam = float(p["gamma"]), float(p["gae_lambda"])
    obs = np.stack([o for t in trajs for o in t.obs]).astype(np.float32)
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
        "actions": np.array([a for t in trajs for a in t.actions], dtype=np.int64),
        "log_probs": np.array([lp for t in trajs for lp in t.log_probs], dtype=np.float32),
        "advantages": np.concatenate(advs),
        "returns": np.concatenate(rets).astype(np.float32),
        "values": values.astype(np.float32),
        "raw_rewards": np.concatenate(raw_rewards),
        "reward_scale": np.array(scale, dtype=np.float32),
    }


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var < 1e-8 else 1.0 - float(np.var(target - pred)) / var


def ppo_update(actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer, batch: Dict[str, np.ndarray],
               cfg: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    p = cfg["ppo"]
    kind, eps = p["policy_loss"], float(p["clip_eps"])
    n = len(batch["actions"])
    mb_size = min(int(p["minibatch_size"]), n)
    chunk = int(p["device_chunk"])

    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    t_obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    t_act = torch.as_tensor(batch["actions"], dtype=torch.long)
    t_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32)
    t_adv = torch.as_tensor(adv, dtype=torch.float32)
    t_ret = torch.as_tensor(batch["returns"], dtype=torch.float32)

    actor.train()
    critic.train()
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
             "ratio_dev": 0.0}
    n_mb = 0
    for _ in range(int(p["epochs"])):
        perm = torch.randperm(n)
        for s in range(0, n - mb_size + 1, mb_size):
            mb = perm[s:s + mb_size]
            optimizer.zero_grad(set_to_none=True)
            acc = {k: 0.0 for k in stats}
            for c in range(0, len(mb), chunk):
                idx = mb[c:c + chunk]
                w = len(idx) / len(mb)
                o = t_obs[idx].to(device, non_blocking=True)
                dist = actor.dist(o)
                logp = dist.log_prob(t_act[idx].to(device))
                old = t_logp[idx].to(device)
                a = t_adv[idx].to(device)
                pl = policy_loss(logp, old, a, eps, kind)
                ent = dist.entropy().mean()
                v = critic(o)
                vl = torch.nn.functional.mse_loss(v, t_ret[idx].to(device))
                loss = (pl - float(p["ent_coef"]) * ent + float(p["vf_coef"]) * vl) * w
                loss.backward()
                kl, cf = ratio_stats(logp.detach(), old, eps)
                acc["policy_loss"] += pl.item() * w
                acc["value_loss"] += vl.item() * w
                acc["entropy"] += ent.item() * w
                acc["approx_kl"] += kl.item() * w
                acc["clip_frac"] += cf.item() * w
                acc["ratio_dev"] += (torch.exp(logp.detach() - old) - 1).abs().mean().item() * w
            nn.utils.clip_grad_norm_(actor.parameters(), float(p["max_grad_norm"]))
            nn.utils.clip_grad_norm_(critic.parameters(), float(p["max_grad_norm"]))
            optimizer.step()
            for k in stats:
                stats[k] += acc[k]
            n_mb += 1
    for k in stats:
        stats[k] /= max(1, n_mb)
    stats["n_minibatches"] = n_mb
    stats["explained_variance"] = explained_variance(batch["values"], batch["returns"])
    return stats


def make_optimizer(actor: nn.Module, critic: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    p = cfg["ppo"]
    return torch.optim.Adam([
        {"params": actor.parameters(), "lr": float(p["lr_actor"])},
        {"params": critic.parameters(), "lr": float(p["lr_critic"])},
    ])

