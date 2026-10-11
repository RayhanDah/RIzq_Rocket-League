"""Processus d'environnement : fait tourner UN env RocketSim et répond aux commandes du processus principal.

Protocole (tuples via multiprocessing.Pipe) :
  → ("step", actions: np.ndarray[n_agents])        ← ("ok", StepResult)
  → ("set_step", global_step: int)                  (pas de réponse)
  → ("close",)
Au démarrage le worker envoie ("ok", EpisodeStart) ; en cas d'erreur ("error", traceback).
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeStart:
    agents: List[str]
    teams: np.ndarray        # (n,) 0 = bleu, 1 = orange
    obs: np.ndarray          # (n, OBS_DIM)


@dataclass
class StepResult:
    rewards: np.ndarray                  # (n,) pour les agents de l'épisode qui vient de jouer
    done: bool
    terminated: bool                     # True = but (pas de bootstrap) ; False + done = troncature
    final_obs: Optional[np.ndarray]      # obs après le dernier step si done (pour le bootstrap)
    summary: Optional[Dict[str, Any]]    # stats d'épisode si done
    next: EpisodeStart                   # obs courantes (nouvel épisode si done)


def worker_main(conn, cfg: Dict[str, Any], seed: int, worker_id: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:  # Ctrl+C est géré par le processus principal (checkpoint propre), pas par les workers
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)

        from ..metrics.episode import EpisodeTracker

        rc = cfg["rewards"]
        if cfg["env"].get("engine", "rocketsim") == "fake":   # tests uniquement (voir env/fake_engine.py)
            from ..env.fake_engine import build_fake_env as factory
        else:
            from ..env.factory import build_env as factory
        env = factory(cfg, seed=seed)
        tracker = EpisodeTracker(rc.get("double_commit_radius", 900), rc.get("double_commit_min_speed", 400))

        def reset() -> EpisodeStart:
            obs_dict = env.reset()
            agents = list(obs_dict.keys())
            tracker.reset(env.state, env.shared_info)
            return EpisodeStart(
                agents=agents,
                teams=np.array([env.state.cars[a].team_num for a in agents], dtype=np.int8),
                obs=np.stack([obs_dict[a] for a in agents]).astype(np.float32),
            )

        current = reset()
        conn.send(("ok", current))

        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "step":
                actions = msg[1]
                action_dict = {a: np.array([int(actions[i])], dtype=np.int64) for i, a in enumerate(current.agents)}
                obs_dict, rew_dict, term_dict, trunc_dict = env.step(action_dict)
                tracker.step(env.state)
                rewards = np.array([rew_dict[a] for a in current.agents], dtype=np.float32)
                terminated = any(term_dict.values())
                done = terminated or any(trunc_dict.values())
                if done:
                    final_obs = np.stack([obs_dict[a] for a in current.agents]).astype(np.float32)
                    summary = tracker.finish(env.state, terminated)
                    steps = max(1, tracker.steps)
                    summary["reward_terms"] = {k: v / steps for k, v in env.shared_info.get("reward_log", {}).items()}
                    current = reset()
                    conn.send(("ok", StepResult(rewards, True, terminated, final_obs, summary, current)))
                else:
                    current = EpisodeStart(
                        current.agents, current.teams,
                        np.stack([obs_dict[a] for a in current.agents]).astype(np.float32),
                    )
                    conn.send(("ok", StepResult(rewards, False, False, None, None, current)))
            elif cmd == "set_step":
                env.shared_info["global_step"] = int(msg[1])
            elif cmd == "close":
                break
            else:
                raise ValueError(f"Commande inconnue : {cmd}")
    except Exception:  # noqa: BLE001 — on renvoie tout au processus principal
        try:
            conn.send(("error", f"[worker {worker_id}]\n" + traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
