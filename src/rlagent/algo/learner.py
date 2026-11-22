"""Mise à jour PPO/SPO : valeurs, GAE, normalisations, epochs × minibatches (ARCHITECTURE §6).

Même calcul sur CPU, GPU (cuda / xpu) et TPU (xla). Sur TPU, toutes les formes de tenseurs restent fixes (dernier
morceau complété, cf. accel.padded_chunks) et chaque pas d'optimiseur se termine par accel.step_barrier."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .. import accel
from .collector import Trajectory
from .gae import RunningStats, batch_gae, segment_discount
from .losses import policy_loss, ratio_stats

STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "ratio_dev", "kick_ce", "kick_agree")


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


def resident_batch(p: Dict[str, Any], device: torch.device) -> bool:
    """`ppo.resident_batch` : sur XLA (TPU), le lot est copié UNE fois sur le device, complété à une taille fixe
    (multiple de device_chunk), et tout se fait là-bas (valeurs, log π_prox, minibatchs tirés par index_select, KL
    final). Avant : 4 copies hôte → TPU du lot entier par itération (~1,9 Go à 400 k pas) et des sélections de lignes
    sur un CPU saturé par la collecte. auto = oui sur XLA ; sans effet ailleurs."""
    if not accel.is_xla(device):
        return False
    opt = p.get("resident_batch", "auto")
    return True if opt in (None, "auto") else bool(opt)


class DeviceRows:
    """Lot de `n` lignes rangé sur le device à `cap` lignes (multiple de `chunk`, forme fixe pour XLA)."""

    def __init__(self, n: int, device: torch.device, chunk: int):
        self.n, self.chunk, self.device = int(n), int(chunk), torch.device(device)
        self.cap = max(1, -(-self.n // self.chunk)) * self.chunk

    def put(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        return accel.pad_rows(torch.as_tensor(x, dtype=dtype), self.cap).to(self.device)

    def mask(self) -> torch.Tensor:
        return self.put(np.ones(self.n, np.float32), torch.float32)

    def index(self, idx: torch.Tensor) -> torch.Tensor:
        """Index (CPU) → device. Taille fixe à l'appel : une seule forme compilée quels que soient les indices."""
        return idx.to(self.device)


@torch.no_grad()
def resident_values(critic: nn.Module, obs: torch.Tensor, rows: DeviceRows) -> np.ndarray:
    """V(s) sur tout le lot résident : un seul graphe (morceaux à bornes fixes), une seule lecture vers l'hôte."""
    critic.eval()
    v = torch.cat([critic(obs[s:s + rows.chunk]).float() for s in range(0, rows.cap, rows.chunk)])
    return v.cpu().numpy()[:rows.n]


@torch.no_grad()
def resident_log_probs(actor: nn.Module, obs: torch.Tensor, act: torch.Tensor, rows: DeviceRows) -> torch.Tensor:
    actor.eval()
    out = torch.cat([actor.dist(obs[s:s + rows.chunk]).log_prob(act[s:s + rows.chunk]).float()
                     for s in range(0, rows.cap, rows.chunk)])
    actor.train()
    return out


def shared_obs(trajs: List[Trajectory]) -> Optional[np.ndarray]:
    """Tableau commun des observations si les trajectoires en sont des vues consécutives couvrant tout le tableau
    (Trajectory.obs_base, collecte locale par mémoire partagée), sinon None."""
    if not trajs or trajs[0].obs_base is None:
        return None
    base, pos = trajs[0].obs_base[0], 0
    for t in trajs:
        if t.obs_base is None or t.obs_base[0] is not base or t.obs_base[1] != pos:
            return None
        pos += len(t)
    return base if pos == len(base) and base.dtype == np.float32 else None


def build_batch(trajs: List[Trajectory], critic: nn.Module, ret_stats: RunningStats, cfg: Dict[str, Any],
                device: torch.device) -> Dict[str, np.ndarray]:
    p = cfg["ppo"]
    gamma, lam = float(p["gamma"]), float(p["gae_lambda"])
    # trajectoires en listes (collecte centrale) ou en tableaux (collecte locale, Trajectory.compact) : mêmes valeurs ;
    # collecte locale par mémoire partagée : les obs sont déjà un seul tableau, dans l'ordre (aucune copie)
    obs = shared_obs(trajs)
    if obs is None:
        obs = np.concatenate([np.asarray(t.obs, dtype=np.float32).reshape(len(t), -1) for t in trajs])
    boot_idx = [i for i, t in enumerate(trajs) if t.bootstrap_obs is not None]
    boot_obs = (np.stack([trajs[i].bootstrap_obs for i in boot_idx]).astype(np.float32)
                if boot_idx else np.zeros((0, obs.shape[1]), np.float32))
    chunk = int(p["device_chunk"])
    resident: Dict[str, Any] = {}
    if resident_batch(p, device):
        rows = DeviceRows(len(obs), device, chunk)
        obs_dev = rows.put(obs, torch.float32)                   # la seule copie du lot vers le TPU
        values = resident_values(critic, obs_dev, rows)
        boot_vals = batched_values(critic, boot_obs, device, chunk)
        resident = {"_rows": rows, "_obs_dev": obs_dev}
    else:
        all_vals = batched_values(critic, np.concatenate([obs, boot_obs]), device, chunk)
        values, boot_vals = all_vals[:len(obs)], all_vals[len(obs):]
    boot_map = {i: float(boot_vals[k]) for k, i in enumerate(boot_idx)}

    scale = 1.0 / ret_stats.std
    lengths = np.fromiter((len(t) for t in trajs), dtype=np.int64, count=len(trajs))
    raw = np.concatenate([np.asarray(t.rewards, dtype=np.float32) for t in trajs])
    boot = np.array([boot_map.get(i, 0.0) for i in range(len(trajs))], dtype=np.float64)
    adv = batch_gae(raw * scale, values, lengths, boot, gamma, lam)     # vectorisée (gae.batch_gae)
    rets = adv + values
    # Mise à jour des stats de retour APRÈS usage (l'échelle de ce lot reste cohérente). Une seule fusion de Welford
    # sur tout le lot = la suite des fusions par trajectoire (même moyenne / variance).
    ret_stats.update(segment_discount(raw, lengths, gamma))

    teacher = None
    if any(t.teachers is not None for t in trajs):       # imitation de Nexto (train.kickstart) : −1 = pas d'étiquette
        teacher = np.concatenate([np.full(len(t), -1, np.int64) if t.teachers is None
                                  else np.asarray(t.teachers, dtype=np.int64) for t in trajs])
    return {
        "obs": obs,
        "teacher": teacher,
        "actions": np.concatenate([np.asarray(t.actions, dtype=np.int64) for t in trajs]),
        "log_probs": np.concatenate([np.asarray(t.log_probs, dtype=np.float32) for t in trajs]),
        "advantages": adv,
        "returns": rets.astype(np.float32),
        "values": values.astype(np.float32),
        "raw_rewards": raw,
        "reward_scale": np.array(scale, dtype=np.float32),
        **resident,                                              # XLA : lot déjà sur le device (resident_batch)
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

    t_start = time.perf_counter()
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    behavior_kl = 0.0
    is_max = float(p.get("is_weight_max", 2.0))
    # Imitation de Nexto (kickstarting) : + kick_coef × entropie croisée vers l'action de Nexto, sur les lignes étiquetées
    kick_coef = float(p.get("kick_coef", 0.0))
    teacher = batch.get("teacher")
    use_kick = kick_coef > 0.0 and teacher is not None and bool((teacher >= 0).any())
    rows: Optional[DeviceRows] = None
    if resident_batch(p, device):
        # Lot résident (XLA) : copié une fois, complété à rows.cap lignes ; les lignes ajoutées ne sont jamais tirées
        # dans les minibatchs (indices < n) et sont masquées dans les moyennes.
        rows = batch.get("_rows")
        t_obs = batch.get("_obs_dev")
        if rows is None or t_obs is None or rows.n != n or rows.chunk != chunk:
            rows = DeviceRows(n, device, chunk)
            t_obs = rows.put(batch["obs"], torch.float32)
        t_act = rows.put(batch["actions"], torch.long)
        t_logp = rows.put(batch["log_probs"], torch.float32)
        t_adv = rows.put(adv, torch.float32)
        t_ret = rows.put(batch["returns"], torch.float32)
        t_teach = rows.put(teacher, torch.long) if use_kick else None
        if decoupled:
            t_center = resident_log_probs(actor, t_obs, t_act, rows)                 # log π_prox
            log_w = t_center - t_logp
            behavior_kl = float((((torch.exp(log_w) - 1.0) - log_w) * rows.mask()).sum()) / n
            t_w = torch.exp(log_w).clamp(max=is_max)
            accel.step_barrier(device)
        else:
            t_center, t_w = t_logp, None
    else:
        t_obs = torch.as_tensor(batch["obs"], dtype=torch.float32).to(src)
        t_act = torch.as_tensor(batch["actions"], dtype=torch.long).to(src)
        t_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32).to(src)
        t_adv = torch.as_tensor(adv, dtype=torch.float32).to(src)
        t_ret = torch.as_tensor(batch["returns"], dtype=torch.float32).to(src)
        t_teach = torch.as_tensor(teacher, dtype=torch.long).to(src) if use_kick else None
        if decoupled:
            t_center = batched_log_probs(actor, t_obs, t_act, device, chunk)          # log π_prox
            log_w = t_center - t_logp
            behavior_kl = float(((torch.exp(log_w) - 1.0) - log_w).mean())          # retard de la collecte
            t_w = torch.exp(log_w).clamp(max=is_max)
        else:
            t_center, t_w = t_logp, None

    def fetch(idx: torch.Tensor):
        """Morceau d'indices (CPU) → (obs, actions, log π centre, avantages, retours, poids) sur le device."""
        if rows is not None:
            di = rows.index(idx)
            take = (lambda t: t.index_select(0, di))
        else:
            idx = idx.to(src)
            take = (lambda t: t[idx].to(device, non_blocking=True))
        return (take(t_obs), take(t_act), take(t_center), take(t_adv), take(t_ret),
                None if t_w is None else take(t_w), None if t_teach is None else take(t_teach))

    # Garde-fou (mode adaptive_kl) : si la politique s'est déjà trop éloignée de celle de collecte
    # (KL > kl_stop × target_kl, mesuré avant le pas), on n'applique pas ce pas et on arrête l'itération.
    stop_kl = (float(p.get("kl_stop", 4.0)) * float(p.get("target_kl", 0.01))
               if str(p.get("lr_schedule", "fixed")) == "adaptive_kl" else float("inf"))
    check_kl = stop_kl != float("inf")
    early_stop = False

    t_prepared = time.perf_counter()                 # XLA : synchronisé (lecture de behavior_kl) en mode découplé
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
            mb = perm[s:s + mb_size]
            optimizer.zero_grad(set_to_none=True)
            acc = torch.zeros(len(STAT_KEYS), dtype=torch.float32, device=device)
            for c in range(0, len(mb), chunk):
                idx = mb[c:c + chunk]
                w = len(idx) / len(mb)
                o, act, old, a, ret, wi, tch = fetch(idx)
                dist = actor.dist(o)
                logp = dist.log_prob(act)
                pl = policy_loss(logp, old, a, eps, kind, wi)
                ent = dist.entropy().mean()
                v = critic(o)
                vl = torch.nn.functional.mse_loss(v, ret)
                loss = pl - float(p["ent_coef"]) * ent + float(p["vf_coef"]) * vl
                zero = torch.zeros((), dtype=torch.float32, device=logp.device)
                kce, kag = zero, zero
                if tch is not None:
                    m = (tch >= 0).float()
                    cnt = m.sum().clamp(min=1.0)
                    kce = -(dist.log_prob(tch.clamp(min=0)) * m).sum() / cnt
                    kag = ((dist.probs.argmax(-1) == tch).float() * m).sum() / cnt
                    loss = loss + kick_coef * kce
                (loss * w).backward()
                lp = logp.detach()
                kl, cf = ratio_stats(lp, old, eps)
                rd = (torch.exp(lp - old) - 1).abs().mean()
                acc = acc + torch.stack([pl.detach(), vl.detach(), ent.detach(), kl, cf, rd,
                                         kce.detach(), kag.detach()]).float() * w
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
    stats["kl_final"] = (final_kl(actor, t_obs, t_act, t_center, device, chunk) if rows is None
                         else resident_final_kl(actor, t_obs, t_act, t_center, rows))
    stats["behavior_kl"] = behavior_kl
    # durées (train.py les range dans time/) : préparation (copies, log π_prox) puis epochs + KL final (lecture finale
    # = attente de la fin des calculs sur le device)
    stats["t_prepare_s"] = t_prepared - t_start
    stats["t_epochs_s"] = time.perf_counter() - t_prepared
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


@torch.no_grad()
def resident_final_kl(actor: nn.Module, t_obs: torch.Tensor, t_act: torch.Tensor, t_logp: torch.Tensor,
                      rows: DeviceRows, max_samples: int = 8_192) -> float:
    """final_kl sur le lot résident (même sous-échantillon, même estimateur k3) : indices complétés à un multiple de
    rows.chunk et masqués, une seule lecture vers l'hôte."""
    n = rows.n
    idx = torch.randperm(n)[:max_samples] if n > max_samples else torch.arange(n)
    m = len(idx)
    size = max(1, -(-m // rows.chunk)) * rows.chunk
    full = torch.zeros(size, dtype=torch.long)
    full[:m] = idx
    keep = torch.zeros(size, dtype=torch.float32)
    keep[:m] = 1.0
    di, dk = rows.index(full), keep.to(rows.device)
    actor.eval()
    total = None
    for c in range(0, size, rows.chunk):
        sel = di[c:c + rows.chunk]
        logp = actor.dist(t_obs.index_select(0, sel)).log_prob(t_act.index_select(0, sel))
        log_ratio = logp - t_logp.index_select(0, sel)
        part = (((torch.exp(log_ratio) - 1.0) - log_ratio) * dk[c:c + rows.chunk]).sum()
        total = part if total is None else total + part
    actor.train()
    return float(total) / max(1, m)


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

