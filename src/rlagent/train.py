"""Boucle d'entraînement principale.

Usage :
    python -m rlagent.train --profile laptop
    python -m rlagent.train --profile laptop --set ppo.policy_loss=ppo --run-name ablation_ppo
    python -m rlagent.train --resume runs/<run>          # reprend au dernier checkpoint

Arrêt propre : créer un fichier STOP dans le dossier du run (ou Ctrl+C) → checkpoint puis sortie.
"""
from __future__ import annotations

import argparse
import copy
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

from .algo.collector import Collector
from .algo.gae import RunningStats
from .algo.learner import build_batch, make_optimizer, ppo_update
from .config import load_config
from .env.actions import N_ACTIONS
from .env.obs import OBS_DIM
from .models.attention import Actor, build_models, count_params
from .selfplay.pool import OpponentPool, elo_update
from .utils.logger import MetricsLogger

REPO_ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_actor_from_model_cfg(model_cfg: Dict[str, Any]) -> Actor:
    return Actor(N_ACTIONS, d_model=model_cfg["d_model"], n_heads=model_cfg["n_heads"],
                 n_layers=model_cfg["n_layers"], ff_mult=model_cfg["ff_mult"],
                 head_hidden=tuple(model_cfg["head_hidden"]))


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
    fmt = Counter(e["team_size"] for e in episodes)
    for k in (1, 2, 3):
        out[f"format_share/{k}v{k}"] = fmt.get(k, 0) / len(episodes)
    team_eps = [e for e in episodes if "double_commit_rate" in e]
    if team_eps:
        out["team/double_commit_rate"] = float(np.mean([e["double_commit_rate"] for e in team_eps]))
        out["team/teammate_bumps_per_min"] = float(np.mean([e["teammate_bumps_per_min"] for e in team_eps]))
        out["team/teammate_dist"] = float(np.mean([e["teammate_dist"] for e in team_eps]))
    terms: Dict[str, List[float]] = {}
    for e in episodes:
        for k, v in e.get("reward_terms", {}).items():
            terms.setdefault(k, []).append(v)
    for k, v in terms.items():
        out[f"reward_term/{k}"] = float(np.mean(v))
    hist = [e for e in episodes if "vs_hist" in e]
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

    actor, critic = build_models(cfg, N_ACTIONS)
    actor.to(device)
    critic.to(device)
    optimizer = make_optimizer(actor, critic, cfg)
    ret_stats = RunningStats()
    global_step, iteration, elo = 0, 0, 1000.0

    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        optimizer.load_state_dict(ck["optimizer"])
        ret_stats.load_state_dict(ck["ret_stats"])
        global_step, iteration, elo = ck["global_step"], ck["iteration"], ck.get("elo", 1000.0)
        print(f"[train] reprise depuis {resume_from.name} (step {global_step:,})")

    print(f"[train] device={device} | obs={OBS_DIM} | actions={N_ACTIONS} | "
          f"acteur={count_params(actor):,} params | critique={count_params(critic):,} params | "
          f"loss={cfg['ppo']['policy_loss']} | workers={cfg['train']['n_workers']}")

    pool = OpponentPool(run_dir / "pool", recent=int(cfg["selfplay"]["pool_recent"]), seed=seed)
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
        return {
            "actor": actor.state_dict(), "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
            "ret_stats": ret_stats.state_dict(), "global_step": global_step, "iteration": iteration, "elo": elo,
            "cfg": cfg, "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "saved_at": datetime.now().isoformat(),
        }

    tr, sp = cfg["train"], cfg["selfplay"]
    next_ckpt = (global_step // int(tr["checkpoint_every_steps"]) + 1) * int(tr["checkpoint_every_steps"])
    next_pool = (global_step // int(sp["pool_every_steps"]) + 1) * int(sp["pool_every_steps"])
    poisoned = {"flag": False}
    try:
        collector = Collector(cfg, int(tr["n_workers"]), seed, pool=pool, build_actor_fn=build_actor_from_model_cfg)
    except Exception:
        logger.close()
        raise
    # Collecte asynchrone : pendant l'update du lot k, les workers jouent le lot k+1 avec une COPIE figée de
    # l'acteur (poids d'avant l'update). Les log-probs stockés sont ceux de cette copie, donc le ratio PPO/SPO
    # reste exact ; seul effet : les données ont une itération de retard sur la politique.
    async_collect = bool(tr.get("async_collect", False))
    collect_actor = actor
    executor = None
    pending = None
    if async_collect:
        collect_actor = copy.deepcopy(actor)
        collect_actor.load_state_dict(actor.state_dict())
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect")

    def start_collect():
        collector.set_global_step(global_step)
        if collect_actor is not actor:
            collect_actor.load_state_dict(actor.state_dict())
        return executor.submit(collector.collect, collect_actor, int(cfg["ppo"]["batch_steps"]), device)

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
                collector.set_global_step(global_step)
                trajs, episodes, n, t_collect = collector.collect(actor, int(cfg["ppo"]["batch_steps"]), device)
            global_step += n
            iteration += 1

            t0 = time.perf_counter()
            batch = build_batch(trajs, critic, ret_stats, cfg, device)
            stats = ppo_update(actor, critic, optimizer, batch, cfg, device)
            t_learn = time.perf_counter() - t0

            for e in episodes:
                if "vs_hist" in e:
                    elo = elo_update(elo, e["vs_hist"]["hist_elo"], _hist_score(e), float(sp["elo_k"]))

            metrics = {
                "global_step": global_step, "iteration": iteration, "elo": elo,
                "sps_collect": n / max(t_collect, 1e-6),
                "sps_total": n / max(time.perf_counter() - t_iter, 1e-6),
                "time/collect_s": t_collect, "time/learn_s": t_learn,
                "reward/mean_step": float(batch["raw_rewards"].mean()),
                "reward/scale": float(batch["reward_scale"]),
                "trajectories": len(trajs), "pool_size": len(pool),
                **{f"learn/{k}": v for k, v in stats.items()},
                **summarize_episodes(episodes),
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
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    train(cfg, run_dir, ckpt)


if __name__ == "__main__":
    main()
