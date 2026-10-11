"""Collecte d'expérience multi-processus (ARCHITECTURE §1).

N workers font tourner RocketSim en parallèle ; le processus principal regroupe toutes les observations,
fait UNE inférence batchée par politique (courante / historique), renvoie les actions, et découpe
l'expérience en trajectoires par agent.

Seules les voitures pilotées par la politique COURANTE produisent des données d'apprentissage.
En self-play pur (cas majoritaire), les deux équipes sont pilotées par la politique courante et toutes
leurs transitions servent à l'apprentissage.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .worker import EpisodeStart, StepResult, worker_main


@dataclass
class Trajectory:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    bootstrap_obs: Optional[np.ndarray] = None   # None = terminée par un but (valeur finale 0)

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class EnvSlot:
    episode: EpisodeStart
    hist_team: Optional[int] = None       # équipe pilotée par une version historique (None = self-play pur)
    hist_entry: Optional[Dict] = None
    trajs: Dict[int, Trajectory] = field(default_factory=dict)   # index d'agent → trajectoire en cours

    def learning_mask(self) -> np.ndarray:
        if self.hist_team is None:
            return np.ones(len(self.episode.agents), dtype=bool)
        return self.episode.teams != self.hist_team


class WorkerError(RuntimeError):
    pass


class Collector:
    def __init__(self, cfg: Dict[str, Any], n_workers: int, seed: int, pool=None,
                 build_actor_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.pool = pool
        self.build_actor_fn = build_actor_fn
        self.hist_prob = float(cfg["selfplay"]["hist_prob"])
        self.rng = random.Random(seed)
        ctx = mp.get_context("spawn")  # identique sous Windows / Linux / macOS
        self.conns = []
        self.procs = []
        for i in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=worker_main, args=(child, cfg, seed + 1000 * (i + 1), i), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.slots: List[EnvSlot] = []
        for i, conn in enumerate(self.conns):
            start = self._recv(i, timeout=300)
            self.slots.append(self._new_slot(start))
        self.finished_episodes: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ utils
    def _recv(self, i: int, timeout: float = 120.0):
        conn = self.conns[i]
        if not conn.poll(timeout):
            alive = self.procs[i].is_alive()
            raise WorkerError(
                f"Le worker {i} ne répond pas depuis {timeout:.0f}s (vivant={alive}, exitcode={self.procs[i].exitcode}). "
                "Si exitcode != 0 juste au démarrage : vérifier les collision meshes (scripts/check_install.py)."
            )
        try:
            status, payload = conn.recv()
        except EOFError as err:
            raise WorkerError(
                f"Le worker {i} s'est arrêté brutalement (exitcode={self.procs[i].exitcode}). "
                "Cause la plus fréquente : RocketSim sans collision meshes. Lancer scripts/check_install.py."
            ) from err
        if status == "error":
            raise WorkerError(payload)
        return payload

    def _new_slot(self, start: EpisodeStart) -> EnvSlot:
        slot = EnvSlot(episode=start)
        if self.pool is not None and len(self.pool) > 0 and self.rng.random() < self.hist_prob:
            slot.hist_team = self.rng.choice([0, 1])
            slot.hist_entry = self.pool.sample()
        return slot

    def set_global_step(self, step: int) -> None:
        for conn in self.conns:
            conn.send(("set_step", int(step)))

    # ---------------------------------------------------------------- collect
    @torch.no_grad()
    def collect(self, actor: torch.nn.Module, n_steps: int, device: torch.device
                ) -> Tuple[List[Trajectory], List[Dict[str, Any]], int, float]:
        """Collecte au moins n_steps transitions apprenantes. Retourne (trajectoires, épisodes finis, n, durée)."""
        t0 = time.perf_counter()
        actor.eval()
        done_trajs: List[Trajectory] = []
        episodes: List[Dict[str, Any]] = []
        collected = 0

        while collected < n_steps:
            # 1) Regrouper les observations par politique
            cur_rows: List[Tuple[int, int]] = []
            cur_obs: List[np.ndarray] = []
            hist_groups: Dict[str, Tuple[Dict, List[Tuple[int, int]], List[np.ndarray]]] = {}
            for w, slot in enumerate(self.slots):
                learn = slot.learning_mask()
                for j in range(len(slot.episode.agents)):
                    if learn[j]:
                        cur_rows.append((w, j))
                        cur_obs.append(slot.episode.obs[j])
                    else:
                        key = slot.hist_entry["id"]
                        g = hist_groups.setdefault(key, (slot.hist_entry, [], []))
                        g[1].append((w, j))
                        g[2].append(slot.episode.obs[j])

            actions = [np.zeros(len(s.episode.agents), dtype=np.int64) for s in self.slots]

            # 2) Inférence politique courante
            obs_t = torch.as_tensor(np.stack(cur_obs), dtype=torch.float32, device=device)
            act, logp = actor.act(obs_t)
            act = act.cpu().numpy()
            logp = logp.cpu().numpy()
            for k, (w, j) in enumerate(cur_rows):
                actions[w][j] = act[k]
                traj = self.slots[w].trajs.setdefault(j, Trajectory())
                traj.obs.append(self.slots[w].episode.obs[j])
                traj.actions.append(int(act[k]))
                traj.log_probs.append(float(logp[k]))

            # 3) Inférence versions historiques (CPU)
            for entry, rows, obs_list in hist_groups.values():
                hist_actor = self.pool.get_actor(entry, self.build_actor_fn)
                h_act, _ = hist_actor.act(torch.as_tensor(np.stack(obs_list), dtype=torch.float32))
                for k, (w, j) in enumerate(rows):
                    actions[w][j] = int(h_act[k])

            # 4) Step en parallèle
            for w, conn in enumerate(self.conns):
                conn.send(("step", actions[w]))
            for w in range(len(self.conns)):
                res: StepResult = self._recv(w)
                slot = self.slots[w]
                learn = slot.learning_mask()
                for j in range(len(slot.episode.agents)):
                    if learn[j]:
                        slot.trajs[j].rewards.append(float(res.rewards[j]))
                        collected += 1
                if res.done:
                    for j, traj in slot.trajs.items():
                        traj.bootstrap_obs = None if res.terminated else res.final_obs[j]
                        done_trajs.append(traj)
                    summary = dict(res.summary)
                    if slot.hist_team is not None:
                        summary["vs_hist"] = {
                            "learner_team": 1 - slot.hist_team,
                            "hist_elo": slot.hist_entry["elo"],
                            "hist_id": slot.hist_entry["id"],
                        }
                    episodes.append(summary)
                    self.slots[w] = self._new_slot(res.next)
                else:
                    slot.episode = res.next

        # 5) Couper les trajectoires en cours (bootstrap sur l'obs courante) ; elles reprennent à vide.
        for slot in self.slots:
            for j, traj in slot.trajs.items():
                if len(traj) > 0:
                    traj.bootstrap_obs = slot.episode.obs[j]
                    done_trajs.append(traj)
            slot.trajs = {}

        return done_trajs, episodes, collected, time.perf_counter() - t0

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
