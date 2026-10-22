"""Boucle d'entraînement principale.

Usage :
    python -m rlagent.train --profile laptop
    python -m rlagent.train --profile laptop --set ppo.policy_loss=ppo --run-name ablation_ppo
    python -m rlagent.train --resume runs/<run>          # reprend au dernier checkpoint

Arrêt propre : créer un fichier STOP dans le dossier du run (ou Ctrl+C) → checkpoint puis sortie.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from . import accel
from .algo.collector import make_collector
from .algo.gae import RunningStats
from .algo.learner import adapt_actor_lr, build_batch, make_optimizer, ppo_update
from .config import load_config
from .env.actions import N_ACTIONS
from .env.obs import env_obs_flags, obs_dim
from .models.attention import Actor, actor_from_model_cfg, build_models, count_params
from .selfplay.pool import OpponentPool, anchor_name, elo_update
from .utils.logger import MetricsLogger

REPO_ROOT = Path(__file__).resolve().parents[2]

xpu_available = accel.xpu_available


def pick_device(name: str) -> torch.device:
    """auto = GPU NVIDIA (cuda) s'il y en a un, sinon TPU (torch_xla installé), sinon CPU. L'iGPU Intel (xpu) se
    demande explicitement : son intérêt dépend de la machine (mesurer avec scripts/benchmark_update.py)."""
    return accel.resolve_device(name)


def build_actor_from_model_cfg(model_cfg: Dict[str, Any]) -> Actor:
    return actor_from_model_cfg(model_cfg, N_ACTIONS)


def check_segment_length(cfg: Dict[str, Any], batch_steps: int) -> Optional[str]:
    """Avertissement si les arènes sont si nombreuses que chaque agent ne joue que quelques steps par lot
    (trajectoires coupées très court → GAE qui repose presque entièrement sur le critique)."""
    tr = cfg["train"]
    arenas = int(tr["n_workers"]) * int(tr.get("envs_per_worker", 1) or 1)
    fw = cfg["env"]["format_weights"]
    agents = 2.0 * sum(k * w for k, w in fw.items()) / max(1e-9, sum(fw.values()))   # voitures par arène
    seg = batch_steps / max(1.0, arenas * agents)
    if seg < float(tr.get("min_segment_steps", 32)):
        return (f"[train] attention : {arenas} arènes × ~{agents:.1f} voitures pour un lot de {batch_steps:,} steps "
                f"→ ~{seg:.0f} steps par agent et par lot (< {tr.get('min_segment_steps', 32)}). Réduire "
                "n_workers × envs_per_worker ou augmenter ppo.batch_steps.")
    return None


def prevent_idle_sleep() -> None:
    """Windows : empêche la mise en veille AUTOMATIQUE tant que ce processus tourne (demande standard
    SetThreadExecutionState, libérée à la sortie ; aucun réglage système modifié). Fermer le capot endort quand même."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    except Exception:  # noqa: BLE001 — confort uniquement
        pass


def load_state_expanding(module: torch.nn.Module, state: Dict[str, torch.Tensor]) -> List[str]:
    """Charge des poids en élargissant les couches d'ENTRÉE qui ont grandi (nouvelles features d'observation) :
    colonnes ajoutées à droite, initialisées à zéro → le réseau calcule exactement la même chose qu'avant, puis
    apprend à se servir des nouvelles entrées. Renvoie les clés élargies."""
    own = module.state_dict()
    expanded, fixed = [], {}
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            new = own[k]
            if v.dim() == 2 and new.dim() == 2 and new.shape[0] == v.shape[0] and new.shape[1] > v.shape[1]:
                pad = torch.zeros(v.shape[0], new.shape[1] - v.shape[1], dtype=v.dtype, device=v.device)
                fixed[k] = torch.cat([v, pad], dim=1)
                expanded.append(k)
                continue
            raise RuntimeError(f"poids {k} incompatibles : {tuple(v.shape)} → {tuple(new.shape)}")
        fixed[k] = v
    module.load_state_dict(fixed)
    return expanded


def pad_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Après un élargissement : met les moments d'Adam des paramètres élargis à la bonne taille (zéros ajoutés)."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            st = optimizer.state.get(param, {})
            for name in ("exp_avg", "exp_avg_sq"):
                t = st.get(name)
                if t is not None and t.shape != param.shape and t.dim() == 2:
                    pad = torch.zeros(t.shape[0], param.shape[1] - t.shape[1], dtype=t.dtype, device=t.device)
                    st[name] = torch.cat([t, pad], dim=1)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(run_dir: Path, state: Dict[str, Any], keep: int) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{state['global_step']:013d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    (run_dir / "latest.txt").write_text(path.name, encoding="utf-8")
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass
    return path


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    marker = run_dir / "latest.txt"
    if marker.exists():
        p = run_dir / "checkpoints" / marker.read_text(encoding="utf-8").strip()
        if p.exists():
            return p
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


# --------------------------------------------------------------------------- curriculum des exercices
class DrillCurriculum:
    """Niveau de difficulté global des exercices « à niveau » (`env.drill_curriculum`, cf. env/mutators.py).

    Une moyenne glissante de réussite par exercice (sur les épisodes de TOUTES les arènes) ; à chaque itération, le
    niveau monte de `step` si elle dépasse `promote`, descend si elle passe sous `demote` (zone de progrès : ni
    trivial, ni impossible). Niveau 0 = version facile (ex. aérien : toucher une balle à 300 uu en sautant), 1 = cible
    (aérienne vers le plafond). Envoyé aux workers (shared_info["drill_levels"]), sauvegardé avec les checkpoints."""

    def __init__(self, kinds, start: float = 0.0, step: float = 0.01, promote: float = 0.65, demote: float = 0.35,
                 ema: float = 0.05):
        self.levels = {k: float(start) for k in kinds}
        self.success = {k: 0.5 for k in kinds}
        self.step, self.promote, self.demote, self.ema = float(step), float(promote), float(demote), float(ema)

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> Optional["DrillCurriculum"]:
        dc = cfg["env"].get("drill_curriculum") or {}
        if not dc.get("enabled", False):
            return None
        from .env.mutators import LEVELED_DRILLS
        weights = cfg["env"]["start_weights"]
        kinds = [k for k in LEVELED_DRILLS if float(weights.get("drill_" + k, 0.0)) > 0.0]
        return cls(kinds, start=dc.get("start", 0.0), step=dc.get("step", 0.01), promote=dc.get("promote", 0.65),
                   demote=dc.get("demote", 0.35), ema=dc.get("ema", 0.05))

    def update(self, episodes: List[Dict[str, Any]]) -> None:
        seen = set()
        for e in episodes:
            k = e.get("drill")
            if k in self.levels and "drill_level" in e:
                self.success[k] += self.ema * (float(e["drill_success"]) - self.success[k])
                seen.add(k)
        for k in seen:
            if self.success[k] > self.promote:
                self.levels[k] = min(1.0, self.levels[k] + self.step)
            elif self.success[k] < self.demote:
                self.levels[k] = max(0.0, self.levels[k] - self.step)

    def metrics(self) -> Dict[str, float]:
        out = {f"curriculum/{k}/level": v for k, v in self.levels.items()}
        out.update({f"curriculum/{k}/success_ema": v for k, v in self.success.items()})
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"levels": dict(self.levels), "success": dict(self.success)}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for k in self.levels:
            if k in d.get("levels", {}):
                self.levels[k] = float(d["levels"][k])
                self.success[k] = float(d.get("success", {}).get(k, self.success[k]))


def scheduled_batch(cfg: Dict[str, Any], step: int):
    """(batch_steps, minibatch_size) au step donné. `ppo.batch_schedule` : liste de [step, batch, minibatch], la
    dernière entrée dont le step est atteint s'applique (guide rlgym-ppo : lot plus gros quand le bot progresse →
    gradient moins bruité). Sans liste : valeurs fixes de la config."""
    p = cfg["ppo"]
    batch, mb = int(p["batch_steps"]), int(p["minibatch_size"])
    for entry in p.get("batch_schedule") or []:
        if step >= int(entry[0]):
            batch, mb = int(entry[1]), int(entry[2])
    return batch, min(mb, batch)


# --------------------------------------------------------------------------- épisodes → métriques
def summarize_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not episodes:
        return out
    out["episodes"] = len(episodes)
    out["ep_length_s"] = float(np.mean([e["length_s"] for e in episodes]))
    total_min = sum(e["length_s"] for e in episodes) / 60.0
    out["goals_per_min"] = sum(e["goal_team"] is not None for e in episodes) / max(total_min, 1e-6)
    out["touches_per_min"] = float(np.mean([e["touches_per_min"] for e in episodes]))
    out["air_touches_per_min"] = float(np.mean([e["air_touches_per_min"] for e in episodes]))
    for key in ("aerial_touches_per_min", "shots_per_min", "saves_per_min"):
        vals = [e[key] for e in episodes if key in e]
        if vals:
            out[key] = float(np.mean(vals))
    fmt = Counter(e["team_size"] for e in episodes)
    for k in (1, 2, 3):
        out[f"format_share/{k}v{k}"] = fmt.get(k, 0) / len(episodes)
        # par format : les moyennes globales dépendent du mélange 1v1/2v2/3v3 (plus de voitures = plus de touches)
        eps_k = [e for e in episodes if e["team_size"] == k]
        if eps_k:
            minutes_k = sum(e["length_s"] for e in eps_k) / 60.0
            out[f"by_format/{k}v{k}/goals_per_min"] = sum(e["goal_team"] is not None for e in eps_k) / max(minutes_k, 1e-6)
            out[f"by_format/{k}v{k}/touches_per_min"] = float(np.mean([e["touches_per_min"] for e in eps_k]))
    team_eps = [e for e in episodes if "double_commit_rate" in e]
    if team_eps:
        out["team/double_commit_rate"] = float(np.mean([e["double_commit_rate"] for e in team_eps]))
        out["team/teammate_bumps_per_min"] = float(np.mean([e["teammate_bumps_per_min"] for e in team_eps]))
        out["team/teammate_dist"] = float(np.mean([e["teammate_dist"] for e in team_eps]))
    mech: Dict[str, List[float]] = {}
    drills: Dict[str, List[float]] = {}
    drill_levels: Dict[str, List[float]] = {}
    for e in episodes:
        for k, v in e.get("mech", {}).items():
            mech.setdefault(k, []).append(v)
        if "drill" in e:
            drills.setdefault(e["drill"], []).append(e["drill_success"])
            if "drill_level" in e:
                drill_levels.setdefault(e["drill"], []).append(e["drill_level"])
    for k, v in mech.items():
        out[f"mech/{k}"] = float(np.mean(v))
    for k, v in drills.items():
        out[f"drill/{k}/success"] = float(np.mean(v))
        out[f"drill/{k}/episodes"] = len(v)
    for k, v in drill_levels.items():
        out[f"drill/{k}/level"] = float(np.mean(v))
    for src, prefix in (("reward_terms", "reward_term"), ("reward_abs", "reward_abs")):
        terms: Dict[str, List[float]] = {}
        for e in episodes:
            for k, v in e.get(src, {}).items():
                terms.setdefault(k, []).append(v)
        for k, v in terms.items():
            out[f"{prefix}/{k}"] = float(np.mean(v))
    # contre les versions du pool (vs_hist/*) et contre chaque ancre (vs_anchor/<nom>/*, selfplay.anchors)
    hist, by_anchor = [], {}
    for e in episodes:
        if "vs_hist" in e:
            name = anchor_name(e["vs_hist"].get("hist_id", ""))
            (hist if name is None else by_anchor.setdefault(name, [])).append(e)
    for name, eps in by_anchor.items():
        res = [_hist_score(e) for e in eps]
        out[f"vs_anchor/{name}/episodes"] = len(eps)
        out[f"vs_anchor/{name}/win_rate"] = float(np.mean([r == 1.0 for r in res]))
        out[f"vs_anchor/{name}/loss_rate"] = float(np.mean([r == 0.0 for r in res]))
    if hist:
        res = [_hist_score(e) for e in hist]
        out["vs_hist/episodes"] = len(hist)
        out["vs_hist/win_rate"] = float(np.mean([r == 1.0 for r in res]))
        out["vs_hist/loss_rate"] = float(np.mean([r == 0.0 for r in res]))
        kick = [e for e in hist if e.get("kickoff_winner") is not None]
        if kick:
            out["vs_hist/kickoff_win_rate"] = float(np.mean(
                [e["kickoff_winner"] == e["vs_hist"]["learner_team"] for e in kick]))
    return out


def _hist_score(e: Dict[str, Any]) -> float:
    g = e["goal_team"]
    if g is None:
        return 0.5
    return 1.0 if g == e["vs_hist"]["learner_team"] else 0.0


# --------------------------------------------------------------------------- main
def train(cfg: Dict[str, Any], run_dir: Path, resume_from: Optional[Path] = None) -> None:
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["train"]["torch_threads"]))
    device = pick_device(cfg["ppo"]["device"])
    if cfg["train"].get("prevent_sleep", True):
        prevent_idle_sleep()

    actor, critic = build_models(cfg, N_ACTIONS)
    actor.to(device)
    critic.to(device)
    accel.prepare_models(device, actor, critic)
    optimizer = make_optimizer(actor, critic, cfg)
    ret_stats = RunningStats()
    global_step, iteration, elo = 0, 0, 1000.0
    curriculum = DrillCurriculum.from_cfg(cfg)

    if resume_from is not None:
        # XLA : chargé sur CPU puis copié dans les paramètres (torch.load ne sait pas placer sur un TPU)
        ck = torch.load(resume_from, map_location="cpu" if accel.is_xla(device) else device, weights_only=False)
        expanded = load_state_expanding(actor, ck["actor"]) + load_state_expanding(critic, ck["critic"])
        optimizer.load_state_dict(ck["optimizer"])
        if expanded:
            pad_optimizer_state(optimizer)
            print(f"[train] couches d'entrée élargies (nouvelles features à poids nuls) : {expanded}")
        ret_stats.load_state_dict(ck["ret_stats"])
        global_step, iteration, elo = ck["global_step"], ck["iteration"], ck.get("elo", 1000.0)
        if curriculum is not None and ck.get("curriculum"):
            curriculum.load_state_dict(ck["curriculum"])
        if Path(resume_from).resolve().parent.parent != Path(run_dir).resolve():
            # démarrage à chaud (train.init_from) : rewards peut-être différents → l'échelle des retours doit
            # pouvoir s'adapter (sinon des millions d'échantillons passés la figent)
            ret_stats.count = min(ret_stats.count, 1e6)
        if str(cfg["ppo"].get("lr_schedule", "fixed")) != "adaptive_kl":
            # lr fixe : la config fait foi (sinon un --set ppo.lr_actor=… à la reprise serait ignoré)
            optimizer.param_groups[0]["lr"] = float(cfg["ppo"]["lr_actor"])
            optimizer.param_groups[1]["lr"] = float(cfg["ppo"]["lr_critic"])
        print(f"[train] reprise depuis {resume_from.name} (step {global_step:,})")

    print(f"[train] device={device} | obs={obs_dim(*env_obs_flags(cfg['env']))} | actions={N_ACTIONS} | "
          f"acteur={count_params(actor):,} params | critique={count_params(critic):,} params | "
          f"loss={cfg['ppo']['policy_loss']} | workers={cfg['train']['n_workers']} | "
          f"lr={cfg['ppo'].get('lr_schedule', 'fixed')}")

    sp_cfg = cfg["selfplay"]
    pool = OpponentPool(run_dir / "pool", recent=int(sp_cfg["pool_recent"]), seed=seed,
                        pfsp=bool(sp_cfg.get("pfsp", False)), pfsp_uniform=float(sp_cfg.get("pfsp_uniform", 0.3)),
                        pfsp_window=float(sp_cfg.get("pfsp_window", 50)), anchors=sp_cfg.get("anchors") or [],
                        anchor_prob=float(sp_cfg.get("anchor_prob", 0.2)), anchor_root=REPO_ROOT)
    for a in pool.anchors:
        # Vérifié dès le départ : un ancien agent (bloc « soi » plus court) joue via fit_obs ; un agent plus riche que
        # l'env courant, ou un autre espace d'actions, ne peut pas jouer.
        a_actor = pool.get_actor(a, build_actor_from_model_cfg)
        dummy = torch.zeros(1, obs_dim(*env_obs_flags(cfg["env"])))
        try:
            with torch.no_grad():
                n_out = a_actor(dummy).shape[-1]
        except ValueError as err:
            raise ValueError(f"selfplay.anchors : l'ancre {a['name']} ne peut pas jouer dans cet env ({err})") from err
        if n_out != N_ACTIONS:
            raise ValueError(f"selfplay.anchors : l'ancre {a['name']} a {n_out} actions (attendu {N_ACTIONS})")
        print(f"[train] ancre {a['name']} : step {a['step']:,}, bloc « soi » {a_actor.encoder.self_dim}")
    logger = MetricsLogger(run_dir, cfg)
    stop_file = run_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()

    stop_requested = {"flag": False}

    def _handle(sig, frame):  # noqa: ARG001
        print("\n[train] arrêt demandé : checkpoint en cours…")
        stop_requested["flag"] = True

    old_handlers = {}
    try:
        old_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, _handle)
        if hasattr(signal, "SIGTERM"):
            old_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _handle)
    except ValueError:  # pas dans le thread principal
        pass

    def make_state() -> Dict[str, Any]:
        # tenseurs copiés sur CPU : checkpoint relisible sur n'importe quelle machine (et torch.save ne gère pas XLA)
        return {
            "actor": accel.to_cpu(actor.state_dict()), "critic": accel.to_cpu(critic.state_dict()),
            "optimizer": accel.to_cpu(optimizer.state_dict()),
            "ret_stats": ret_stats.state_dict(), "global_step": global_step, "iteration": iteration, "elo": elo,
            "curriculum": None if curriculum is None else curriculum.state_dict(),
            "cfg": cfg, "obs_dim": obs_dim(*env_obs_flags(cfg["env"])),
            "n_actions": N_ACTIONS, "saved_at": datetime.now().isoformat(),
        }

    tr, sp = cfg["train"], cfg["selfplay"]
    next_ckpt = (global_step // int(tr["checkpoint_every_steps"]) + 1) * int(tr["checkpoint_every_steps"])
    next_pool = (global_step // int(sp["pool_every_steps"]) + 1) * int(sp["pool_every_steps"])
    poisoned = {"flag": False}
    warn = check_segment_length(cfg, scheduled_batch(cfg, global_step)[0])
    if warn:
        print(warn)
    try:
        collector = make_collector(cfg, int(tr["n_workers"]), seed, pool=pool,
                                   build_actor_fn=build_actor_from_model_cfg)
    except Exception:
        logger.close()
        raise
    # Collecte asynchrone : pendant l'update du lot k, les workers jouent le lot k+1 avec une COPIE figée de
    # l'acteur (poids d'avant l'update). Les log-probs stockés sont ceux de cette copie, donc le ratio PPO/SPO
    # reste exact ; seul effet : les données ont une itération de retard sur la politique.
    async_collect = bool(tr.get("async_collect", False))
    # Inférence de collecte : CPU par défaut (petits lots), GPU possible (`train.collect_device`, ex. cuda:1 sur
    # Kaggle T4 x2). En collecte locale, chaque worker a sa copie sur CPU : la copie ici ne sert qu'à figer les poids.
    # Copie de l'acteur rafraîchie à chaque itération.
    if str(tr.get("collect_mode", "central")) == "local":
        collect_device = torch.device("cpu")
    else:
        collect_device = accel.resolve_collect_device(str(tr.get("collect_device", "cpu")), device)
    print(f"[train] {accel.summary(cfg, device, collect_device)}")
    collect_actor = actor
    executor = None
    pending = None
    if async_collect or collect_device != device:
        # acteur neuf + poids copiés (et non deepcopy) : garde ses réglages par défaut (ex. skip_padding, coupé sur XLA)
        collect_actor = build_actor_from_model_cfg(cfg["model"]).to(collect_device)
        collect_actor.load_state_dict(actor.state_dict())
    if async_collect:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect")
    # Objectif découplé (losses.py) : par défaut dès que la collecte est asynchrone. `ppo.decoupled` pour forcer.
    decoupled = bool(async_collect if cfg["ppo"].get("decoupled") is None else cfg["ppo"]["decoupled"])

    def refresh_collect_actor() -> None:
        if collect_actor is not actor:
            collect_actor.load_state_dict(actor.state_dict())

    def sync_workers() -> None:
        collector.set_global_step(global_step)
        if curriculum is not None:
            collector.set_info({"drill_levels": dict(curriculum.levels)})

    def start_collect():
        sync_workers()
        refresh_collect_actor()
        return executor.submit(collector.collect, collect_actor, scheduled_batch(cfg, global_step)[0], collect_device)

    try:
        if async_collect:
            pending = start_collect()
        while global_step < int(tr["total_steps"]) and not stop_requested["flag"] and not stop_file.exists():
            t_iter = time.perf_counter()
            if async_collect:
                if pending is None:
                    break
                trajs, episodes, n, t_collect = pending.result()
                pending = None
                # Lancer tout de suite la collecte suivante (poids courants) avant d'apprendre sur ce lot
                if global_step + n < int(tr["total_steps"]) and not stop_requested["flag"] and not stop_file.exists():
                    pending = start_collect()
            else:
                sync_workers()
                refresh_collect_actor()
                trajs, episodes, n, t_collect = collector.collect(collect_actor, scheduled_batch(cfg, global_step)[0],
                                                                  collect_device)
            global_step += n
            iteration += 1
            if curriculum is not None:
                curriculum.update(episodes)

            t0 = time.perf_counter()
            batch = build_batch(trajs, critic, ret_stats, cfg, device)
            # minibatch du palier du lot collecté (ppo.batch_schedule), sans modifier la config sauvegardée
            upd_cfg = {**cfg, "ppo": {**cfg["ppo"], "minibatch_size": scheduled_batch(cfg, global_step - n)[1]}}
            stats = ppo_update(actor, critic, optimizer, batch, upd_cfg, device, decoupled=decoupled)
            stats["lr_actor"] = adapt_actor_lr(optimizer, stats["kl_final"], cfg)
            t_learn = time.perf_counter() - t0

            hist_results = []
            for e in episodes:
                if "vs_hist" in e:
                    hist_results.append((e["vs_hist"]["hist_id"], _hist_score(e)))
                    if anchor_name(e["vs_hist"]["hist_id"]) is None:  # l'Elo d'une ancre est celui d'un autre run
                        elo = elo_update(elo, e["vs_hist"]["hist_elo"], _hist_score(e), float(sp["elo_k"]))
            pool.record_results(hist_results)  # scores glissants PFSP / ancres (sans effet si options désactivées)

            metrics = {
                "global_step": global_step, "iteration": iteration, "elo": elo,
                "sps_collect": n / max(t_collect, 1e-6),
                "sps_total": n / max(time.perf_counter() - t_iter, 1e-6),
                "time/collect_s": t_collect, "time/learn_s": t_learn,
                "reward/mean_step": float(batch["raw_rewards"].mean()),
                "reward/scale": float(batch["reward_scale"]),
                "trajectories": len(trajs), "pool_size": len(pool), "learn/batch_steps": n,
                **({} if curriculum is None else curriculum.metrics()),
                **{f"learn/{k}": v for k, v in stats.items()},
                **summarize_episodes(episodes),
                **pool.metrics(),
            }
            if not all(math.isfinite(v) for v in (stats["policy_loss"], stats["value_loss"])):
                # Les poids viennent d'être mis à jour avec des NaN : on ne sauvegarde SURTOUT pas,
                # pour que la reprise reparte du dernier checkpoint sain.
                poisoned["flag"] = True
                logger.log(metrics)
                raise FloatingPointError("NaN/Inf dans les pertes : arrêt sans sauvegarde (reprendre au dernier checkpoint)")
            if iteration % int(tr["log_every_iters"]) == 0:
                logger.log(metrics)

            if global_step >= next_ckpt:
                save_checkpoint(run_dir, make_state(), int(tr["keep_checkpoints"]))
                next_ckpt += int(tr["checkpoint_every_steps"])
            if global_step >= next_pool:
                pool.add({k: v.cpu() for k, v in actor.state_dict().items()}, cfg["model"], global_step, elo)
                next_pool += int(sp["pool_every_steps"])
    finally:
        if pending is not None:  # lot en cours de collecte : on attend la fin des workers, puis on l'abandonne
            try:
                pending.result()
            except Exception:
                pass
        if executor is not None:
            executor.shutdown(wait=True)
        if not poisoned["flag"] and iteration > 0:
            path = save_checkpoint(run_dir, make_state(), int(tr["keep_checkpoints"]))
            print(f"[train] checkpoint final : {path}")
        collector.close()
        logger.close()
        for sig, h in old_handlers.items():
            signal.signal(sig, h)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Entraînement de l'agent Rocket League (PPO/SPO)")
    ap.add_argument("--profile", action="append", default=[], help="profil(s) de config : laptop, bigpc, debug…")
    ap.add_argument("--set", action="append", default=[], dest="overrides", help="surcharge clé=valeur")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="dossier d'un run (reprend au dernier checkpoint) ou fichier .pt")
    args = ap.parse_args(argv)

    if args.resume:
        rp = Path(args.resume)
        if rp.is_dir():
            run_dir, ckpt = rp, find_latest_checkpoint(rp)
        else:
            run_dir, ckpt = rp.parent.parent, rp
        if ckpt is None:
            sys.exit(f"Aucun checkpoint trouvé dans {rp}")
        with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from .config import apply_overrides, _normalize
        cfg = apply_overrides(cfg, args.overrides)
        _normalize(cfg)
    else:
        cfg = load_config(args.profile, args.overrides)
        name = args.run_name or cfg.get("run_name") or \
            f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['ppo']['policy_loss']}"
        cfg["run_name"] = name
        runs_dir = Path(cfg["paths"]["runs_dir"])
        if not runs_dir.is_absolute():
            runs_dir = REPO_ROOT / runs_dir
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is not None:
            print(f"[train] le run {name} existe déjà : reprise automatique.")
        elif cfg["train"].get("init_from"):
            # Démarrage à chaud : nouveau run qui part des poids, de l'optimiseur et du compteur de steps d'un autre
            # (les décroissances de rewards continuent donc là où elles en étaient).
            src = Path(cfg["train"]["init_from"])
            if not src.is_absolute():
                src = REPO_ROOT / src
            ckpt = find_latest_checkpoint(src) if src.is_dir() else src
            if ckpt is None or not ckpt.exists():
                sys.exit(f"train.init_from : aucun checkpoint dans {src}")
            print(f"[train] démarrage à chaud depuis {ckpt}")
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    train(cfg, run_dir, ckpt)


if __name__ == "__main__":
    main()
