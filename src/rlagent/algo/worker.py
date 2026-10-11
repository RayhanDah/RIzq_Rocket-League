"""Processus d'environnement : fait tourner un ou plusieurs envs RocketSim et répond au processus principal.

Protocole à UN env (n_envs=None, tuples via multiprocessing.Pipe) :
  → ("step", actions: np.ndarray[n_agents])        ← ("ok", StepResult)
  → ("set_step", global_step: int)                  (pas de réponse)
  → ("close",)
Au démarrage le worker envoie ("ok", EpisodeStart) ; en cas d'erreur ("error", traceback).
Avec n_envs=k (plusieurs arènes par worker, `train.envs_per_worker`) : mêmes messages, mais les actions,
les EpisodeStart et les StepResult sont des listes (une entrée par arène).
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


class EnvRunner:
    """Un environnement + son suivi d'épisode ; produit EpisodeStart / StepResult."""

    def __init__(self, cfg: Dict[str, Any], seed: int):
        from ..metrics.episode import EpisodeTracker

        rc = cfg["rewards"]
        if cfg["env"].get("engine", "rocketsim") == "fake":   # tests uniquement (voir env/fake_engine.py)
            from ..env.fake_engine import build_fake_env as factory
        else:
            from ..env.factory import build_env as factory
        self.env = factory(cfg, seed=seed)
        self.tracker = EpisodeTracker(rc.get("double_commit_radius", 900), rc.get("double_commit_min_speed", 400))
        self.current = self.reset()

    def reset(self) -> EpisodeStart:
        env = self.env
        obs_dict = env.reset()
        agents = list(obs_dict.keys())
        self.tracker.reset(env.state, env.shared_info)
        return EpisodeStart(
            agents=agents,
            teams=np.array([env.state.cars[a].team_num for a in agents], dtype=np.int8),
            obs=np.stack([obs_dict[a] for a in agents]).astype(np.float32),
        )

    def step(self, actions: np.ndarray) -> StepResult:
        env, current = self.env, self.current
        action_dict = {a: np.array([int(actions[i])], dtype=np.int64) for i, a in enumerate(current.agents)}
        obs_dict, rew_dict, term_dict, trunc_dict = env.step(action_dict)
        self.tracker.step(env.state)
        rewards = np.array([rew_dict[a] for a in current.agents], dtype=np.float32)
        terminated = any(term_dict.values())
        done = terminated or any(trunc_dict.values())
        if done:
            final_obs = np.stack([obs_dict[a] for a in current.agents]).astype(np.float32)
            summary = self.tracker.finish(env.state, terminated)
            steps = max(1, self.tracker.steps)
            summary["reward_terms"] = {k: v / steps for k, v in env.shared_info.get("reward_log", {}).items()}
            self.current = self.reset()
            return StepResult(rewards, True, terminated, final_obs, summary, self.current)
        self.current = EpisodeStart(current.agents, current.teams,
                                    np.stack([obs_dict[a] for a in current.agents]).astype(np.float32))
        return StepResult(rewards, False, False, None, None, self.current)

    def set_global_step(self, step: int) -> None:
        self.env.shared_info["global_step"] = int(step)


def env_seed(seed: int, env_index: int) -> int:
    """Graine de la k-ième arène d'un worker (la 1re garde la graine du worker : comportement inchangé à 1 arène)."""
    return seed + 7919 * env_index


def worker_main(conn, cfg: Dict[str, Any], seed: int, worker_id: int, n_envs: Optional[int] = None) -> None:
    """n_envs=None : protocole à UN env (messages simples). n_envs=k : k arènes, messages sous forme de listes
    (→ ("step", [actions_env0, ...]) ; ← ("ok", [StepResult_env0, ...]))."""
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

        multi = n_envs is not None
        runners = [EnvRunner(cfg, env_seed(seed, k)) for k in range(n_envs if multi else 1)]
        conn.send(("ok", [r.current for r in runners] if multi else runners[0].current))

        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "step":
                if multi:
                    conn.send(("ok", [r.step(a) for r, a in zip(runners, msg[1])]))
                else:
                    conn.send(("ok", runners[0].step(msg[1])))
            elif cmd == "set_step":
                for r in runners:
                    r.set_global_step(msg[1])
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
