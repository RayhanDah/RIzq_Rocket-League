"""Collecte d'expérience multi-processus (ARCHITECTURE §1).

N workers font tourner RocketSim en parallèle. Deux modes (`train.collect_mode`) :

- `central` (défaut) : le processus principal regroupe toutes les observations, fait UNE inférence batchée par
  politique (courante / historique), renvoie les actions. L'inférence peut tourner sur GPU (`train.collect_device`,
  y compris les versions du pool). Avec `train.collect_pipeline: 2`, les arènes de chaque worker sont coupées en
  deux moitiés jouées en alternance : pendant que les workers simulent une moitié, le processus principal calcule
  les actions de l'autre (« double tampon ») → ni les workers ni l'inférence n'attendent.
- `local` : chaque worker a sa propre copie de l'acteur (CPU, 1 thread) et joue seul jusqu'à son quota de steps.
  Plus de calcul par step (l'inférence est faite par chaque worker) mais aucun aller-retour par step : passe à
  l'échelle sur les machines à beaucoup de cœurs (session TPU de Kaggle). Poids diffusés par mémoire partagée.

Dans les deux modes, la même classe `Rollout` découpe l'expérience en trajectoires par agent : seules les voitures
pilotées par la politique COURANTE produisent des données d'apprentissage. En self-play pur (cas majoritaire), les
deux équipes sont pilotées par la politique courante et toutes leurs transitions servent à l'apprentissage.
"""
from __future__ import annotations

import copy
import math
import multiprocessing as mp
import random
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.attention import fit_obs
from ..models.stacked import StackedActors, pad_stack, stack_signature
from .worker import EpisodeStart, StepResult, worker_main


@dataclass
class Trajectory:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    bootstrap_obs: Optional[np.ndarray] = None   # None = terminée par un but (valeur finale 0)
    slot: Optional[int] = None                   # arène (w * envs_per_worker + e) et index d'agent : diagnostic/tests
    agent: Optional[int] = None
    # (tableau, début) : obs = vue de ce tableau qui regroupe tout le lot (collecte locale par mémoire partagée) ;
    # build_batch le reprend tel quel au lieu de recopier les observations
    obs_base: Optional[Tuple[np.ndarray, int]] = None

    def __len__(self) -> int:
        return len(self.actions)

    def compact(self) -> "Trajectory":
        """Listes → tableaux numpy (envoi entre processus bien plus rapide ; mêmes valeurs)."""
        obs = None if self.obs is None else np.stack(self.obs).astype(np.float32, copy=False)
        return Trajectory(obs, np.asarray(self.actions, np.int64),
                          np.asarray(self.log_probs, np.float32), np.asarray(self.rewards, np.float32),
                          self.bootstrap_obs, self.slot, self.agent)


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


# --------------------------------------------------------------------------- suivi des arènes (commun)
class Rollout:
    """Arènes en cours (slots) : inférence groupée par politique, transitions, fin d'épisode, coupure en fin de lot.

    `act(ids)` calcule les actions des arènes `ids` et ouvre leurs transitions (obs, action, log-prob) ;
    `record(ids, résultats)` les ferme (reward) et traite les fins d'épisode. Les deux peuvent porter sur une partie
    des arènes seulement (collecte en double tampon).

    Les versions du pool de même architecture présentes dans le groupe jouent en UN passage (`StackedActors`) au
    lieu d'un appel par version (`stack_hist=False` : boucle d'origine). Mêmes lois d'action ; seul le tirage
    aléatoire est fait en un appel au lieu de K."""

    STACK_CACHE = 4     # piles de poids gardées (une par ensemble de versions ; 2 groupes en double tampon)

    def __init__(self, starts: Sequence[EpisodeStart], new_slot: Callable[[EpisodeStart], EnvSlot],
                 stack_hist: bool = True):
        self.new_slot = new_slot
        self.slots: List[EnvSlot] = [new_slot(s) for s in starts]
        self.done_trajs: List[Trajectory] = []
        self.episodes: List[Dict[str, Any]] = []
        self.stack_hist = stack_hist
        self._stacks: "OrderedDict[Tuple, StackedActors]" = OrderedDict()
        self._sigs: "weakref.WeakKeyDictionary[torch.nn.Module, Any]" = weakref.WeakKeyDictionary()
        self.stack_builds = 0   # diagnostic : piles construites (cache manqué)

    def _signature(self, actor: torch.nn.Module):
        if actor not in self._sigs:
            self._sigs[actor] = stack_signature(actor)
        return self._sigs[actor]

    def _stack(self, ids: Tuple[str, ...], actors: List[torch.nn.Module], device: torch.device) -> StackedActors:
        """Poids empilés des versions `ids` (cache LRU : reconstruits seulement quand l'ensemble change)."""
        key = (ids, str(device))
        stack = self._stacks.get(key)
        if stack is None:
            stack = self._stacks[key] = StackedActors(actors)
            self.stack_builds += 1
            while len(self._stacks) > self.STACK_CACHE:
                self._stacks.popitem(last=False)
        else:
            self._stacks.move_to_end(key)
        return stack

    @torch.no_grad()
    def act(self, ids: Sequence[int], actor: torch.nn.Module, device: torch.device,
            hist_actor: Callable[[Dict], torch.nn.Module]) -> Dict[int, np.ndarray]:
        cur_rows: List[Tuple[int, int]] = []
        cur_obs: List[np.ndarray] = []
        hist_groups: Dict[str, Tuple[Dict, List[Tuple[int, int]], List[np.ndarray]]] = {}
        for w in ids:
            slot = self.slots[w]
            learn = slot.learning_mask()
            for j in range(len(slot.episode.agents)):
                if learn[j]:
                    cur_rows.append((w, j))
                    cur_obs.append(slot.episode.obs[j])
                else:
                    g = hist_groups.setdefault(slot.hist_entry["id"], (slot.hist_entry, [], []))
                    g[1].append((w, j))
                    g[2].append(slot.episode.obs[j])
        actions = {w: np.zeros(len(self.slots[w].episode.agents), dtype=np.int64) for w in ids}

        # Toutes les inférences sont lancées avant de rapatrier un résultat : sur GPU, une seule attente par appel.
        cur = None
        if cur_obs:
            cur = actor.act(torch.as_tensor(np.stack(cur_obs), dtype=torch.float32, device=device))
        hist_out: List[Tuple[List[List[Tuple[int, int]]], torch.Tensor]] = []   # (lignes par version, actions)
        stackable: Dict[Tuple, List[Tuple[str, torch.nn.Module, List, List[np.ndarray]]]] = {}
        for entry, rows, obs_list in hist_groups.values():
            h_actor = hist_actor(entry)
            h_dev = next(h_actor.parameters()).device
            sig = self._signature(h_actor) if self.stack_hist else None
            if sig is not None:
                stackable.setdefault((sig, h_dev), []).append((entry["id"], h_actor, rows, obs_list))
                continue
            h_act, _ = h_actor.act(torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=h_dev))
            hist_out.append(([rows], h_act[None]))
        for (_, h_dev), members in stackable.items():
            if len(members) == 1:
                _, h_actor, rows, obs_list = members[0]
                h_act, _ = h_actor.act(torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=h_dev))
                hist_out.append(([rows], h_act[None]))
                continue
            members.sort(key=lambda m: m[0])
            stack = self._stack(tuple(m[0] for m in members), [m[1] for m in members], h_dev)
            x = torch.as_tensor(pad_stack([m[3] for m in members]), device=h_dev)
            logits = stack.logits(fit_obs(x, stack.self_dim))                        # [K, B_max, n_actions]
            h_act = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
            hist_out.append(([m[2] for m in members], h_act))

        if cur is not None:
            act = cur[0].cpu().numpy()
            logp = cur[1].cpu().numpy()
            for k, (w, j) in enumerate(cur_rows):
                actions[w][j] = act[k]
                traj = self.slots[w].trajs.get(j)
                if traj is None:
                    traj = self.slots[w].trajs[j] = Trajectory(slot=w, agent=j)
                traj.obs.append(self.slots[w].episode.obs[j])
                traj.actions.append(int(act[k]))
                traj.log_probs.append(float(logp[k]))
        for rows_per_version, h_act in hist_out:
            h_act = h_act.cpu().numpy()
            for v, rows in enumerate(rows_per_version):
                for k, (w, j) in enumerate(rows):
                    actions[w][j] = int(h_act[v, k])
        return actions

    def record(self, ids: Sequence[int], results: Sequence[StepResult]) -> int:
        collected = 0
        for w, res in zip(ids, results):
            slot = self.slots[w]
            learn = slot.learning_mask()
            for j in range(len(slot.episode.agents)):
                if learn[j]:
                    slot.trajs[j].rewards.append(float(res.rewards[j]))
                    collected += 1
            if res.done:
                for j, traj in slot.trajs.items():
                    traj.bootstrap_obs = None if res.terminated else res.final_obs[j]
                    self.done_trajs.append(traj)
                summary = dict(res.summary)
                if slot.hist_team is not None:
                    summary["vs_hist"] = {
                        "learner_team": 1 - slot.hist_team,
                        "hist_elo": slot.hist_entry["elo"],
                        "hist_id": slot.hist_entry["id"],
                    }
                self.episodes.append(summary)
                self.slots[w] = self.new_slot(res.next)
            else:
                slot.episode = res.next
        return collected

    def cut(self) -> Tuple[List[Trajectory], List[Dict[str, Any]]]:
        """Fin du lot : coupe les trajectoires en cours (bootstrap sur l'obs courante) ; elles reprennent à vide.
        Renvoie (trajectoires, épisodes finis) et vide les listes."""
        for slot in self.slots:
            for j, traj in slot.trajs.items():
                if len(traj) > 0:
                    traj.bootstrap_obs = slot.episode.obs[j]
                    self.done_trajs.append(traj)
            slot.trajs = {}
        out = (self.done_trajs, self.episodes)
        self.done_trajs, self.episodes = [], []
        return out


def env_parts(n_envs: int, pipeline: int) -> List[Tuple[int, int]]:
    """Découpe les arènes d'un worker en `pipeline` groupes contigus [début, fin) aussi égaux que possible."""
    k = max(1, min(int(pipeline or 1), n_envs))
    bounds = [round(i * n_envs / k) for i in range(k + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(k)]


def _spawn_workers(cfg: Dict[str, Any], n_workers: int, seed: int, envs_per_worker: int, local: bool):
    ctx = mp.get_context("spawn")  # identique sous Windows / Linux / macOS
    conns, procs = [], []
    for i in range(n_workers):
        parent, child = ctx.Pipe()
        kwargs = {"local": True} if local else {}
        p = ctx.Process(target=worker_main, args=(child, cfg, seed + 1000 * (i + 1), i, envs_per_worker),
                        kwargs=kwargs, daemon=True)
        p.start()
        child.close()
        conns.append(parent)
        procs.append(p)
    return conns, procs


class _WorkerPool:
    """Pipes vers les workers + messages communs aux deux modes."""

    conns: List[Any]
    procs: List[Any]

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

    def set_global_step(self, step: int) -> None:
        for conn in self.conns:
            conn.send(("set_step", int(step)))

    def set_info(self, info: Dict[str, Any]) -> None:
        """Copie `info` dans le shared_info de toutes les arènes (ex. {"drill_levels": {...}})."""
        for conn in self.conns:
            conn.send(("set_info", dict(info)))

    def _close_workers(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# --------------------------------------------------------------------------- mode central
class Collector(_WorkerPool):
    def __init__(self, cfg: Dict[str, Any], n_workers: int, seed: int, pool=None,
                 build_actor_fn: Optional[Callable] = None, envs_per_worker: Optional[int] = None,
                 pipeline: Optional[int] = None):
        self.cfg = cfg
        self.pool = pool
        self.build_actor_fn = build_actor_fn
        self.hist_prob = float(cfg["selfplay"]["hist_prob"])
        self.rng = random.Random(seed)
        # Plusieurs arènes par worker : lots d'inférence plus gros, moins d'allers-retours entre processus.
        self.envs_per_worker = int(envs_per_worker or cfg["train"].get("envs_per_worker", 1) or 1)
        self.parts = env_parts(self.envs_per_worker,
                               pipeline if pipeline is not None else cfg["train"].get("collect_pipeline", 1))
        self.conns, self.procs = _spawn_workers(cfg, n_workers, seed, self.envs_per_worker, local=False)
        # slots[w * envs_per_worker + e] = arène e du worker w
        starts: List[EpisodeStart] = []
        for i in range(len(self.conns)):
            starts.extend(self._recv(i, timeout=300))
        self.rollout = Rollout(starts, self._new_slot, stack_hist=bool(cfg["train"].get("stack_hist", True)))
        # Versions du pool copiées sur le device de l'inférence (GPU) : même taille de cache que le pool
        self._dev_actors: "OrderedDict[str, torch.nn.Module]" = OrderedDict()

    @property
    def slots(self) -> List[EnvSlot]:
        return self.rollout.slots

    def _new_slot(self, start: EpisodeStart) -> EnvSlot:
        slot = EnvSlot(episode=start)
        if self.pool is not None and len(self.pool) > 0 and self.rng.random() < self.hist_prob:
            slot.hist_team = self.rng.choice([0, 1])
            slot.hist_entry = self.pool.sample()
        return slot

    def _hist_actor(self, entry: Dict, device: torch.device) -> torch.nn.Module:
        actor = self.pool.get_actor(entry, self.build_actor_fn)
        if device.type == "cpu":
            return actor
        key = entry["id"]
        if key in self._dev_actors:
            self._dev_actors.move_to_end(key)
            return self._dev_actors[key]
        dev_actor = copy.deepcopy(actor).to(device)
        self._dev_actors[key] = dev_actor
        while len(self._dev_actors) > getattr(self.pool, "cache_size", 32):
            self._dev_actors.popitem(last=False)
        return dev_actor

    def _part_ids(self, k: int) -> List[int]:
        lo, hi = self.parts[k]
        E = self.envs_per_worker
        return [w * E + e for w in range(len(self.conns)) for e in range(lo, hi)]

    def _send(self, k: int, actor: torch.nn.Module, device: torch.device) -> None:
        actions = self.rollout.act(self._part_ids(k), actor, device, lambda e: self._hist_actor(e, device))
        lo, hi = self.parts[k]
        E = self.envs_per_worker
        for w, conn in enumerate(self.conns):
            acts = [actions[w * E + e] for e in range(lo, hi)]
            conn.send(("step", acts) if len(self.parts) == 1 else ("step_part", lo, acts))

    def _receive(self, k: int) -> int:
        results: List[StepResult] = []
        for w in range(len(self.conns)):
            results.extend(self._recv(w))
        return self.rollout.record(self._part_ids(k), results)

    @torch.no_grad()
    def collect(self, actor: torch.nn.Module, n_steps: int, device: torch.device
                ) -> Tuple[List[Trajectory], List[Dict[str, Any]], int, float]:
        """Collecte au moins n_steps transitions apprenantes. Retourne (trajectoires, épisodes finis, n, durée)."""
        t0 = time.perf_counter()
        actor.eval()
        device = torch.device(device)
        collected = 0
        in_flight: List[int] = []
        if n_steps > 0:
            for k in range(len(self.parts)):        # amorce : chaque groupe d'arènes reçoit ses actions
                self._send(k, actor, device)
                in_flight.append(k)
        while in_flight:
            # Les workers traitent leurs messages dans l'ordre : on reçoit le groupe le plus ancien, et pendant qu'on
            # calcule ses nouvelles actions, les workers simulent déjà le groupe suivant.
            k = in_flight.pop(0)
            collected += self._receive(k)
            if collected < n_steps:
                self._send(k, actor, device)
                in_flight.append(k)
        trajs, episodes = self.rollout.cut()
        return trajs, episodes, collected, time.perf_counter() - t0

    def close(self) -> None:
        self._close_workers()


# --------------------------------------------------------------------------- mode local
class SharedWeights:
    """Vecteur de poids de l'acteur en mémoire partagée (écrit par le processus principal, lu par les workers)."""

    def __init__(self, n_params: int):
        from multiprocessing import shared_memory
        self.n = int(n_params)
        self.shm = shared_memory.SharedMemory(create=True, size=max(4, 4 * self.n))
        self.array = np.ndarray((self.n,), dtype=np.float32, buffer=self.shm.buf)

    @property
    def name(self) -> str:
        return self.shm.name

    def write(self, flat: np.ndarray) -> None:
        self.array[:] = flat

    def close(self) -> None:
        self.array = None
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


class SharedBlock:
    """Bloc de mémoire partagée (contrôle de la collecte, tampon d'observations). make() renvoie None s'il est
    indisponible ou si /dev/shm est trop petit (conteneur Docker : 64 Mo par défaut ; écrire au-delà tuerait le worker
    par SIGBUS)."""

    def __init__(self, nbytes: int):
        from multiprocessing import shared_memory
        self.shm = shared_memory.SharedMemory(create=True, size=max(8, int(nbytes)))

    @staticmethod
    def make(nbytes: int) -> Optional["SharedBlock"]:
        import os
        try:
            if os.path.isdir("/dev/shm"):
                st = os.statvfs("/dev/shm")
                if st.f_bavail * st.f_frsize < 1.25 * nbytes + (64 << 20):
                    return None
            return SharedBlock(nbytes)
        except Exception:  # noqa: BLE001
            return None

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self) -> None:
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


def flat_params(actor: torch.nn.Module) -> np.ndarray:
    return torch.nn.utils.parameters_to_vector(actor.parameters()).detach().float().cpu().numpy()


class LocalCollector(_WorkerPool):
    """Chaque worker joue avec sa propre copie de l'acteur (voir le docstring du module).

    Arrêt global (`train.collect_sync`, défaut oui) : au lieu d'un quota fixe par worker (le lot attendait le plus lent,
    ~0,7-1 s perdues par lot de 400 k sur Kaggle), chaque worker publie son nombre de transitions en mémoire partagée
    et tous s'arrêtent dès que le TOTAL atteint le lot : les plus rapides en jouent plus. Observations par mémoire
    partagée (`train.collect_shared_obs`, défaut oui) : les ~490 Mo d'un lot ne passent plus par les pipes (ni pickle,
    ni décodage par le processus principal). Sans mémoire partagée utilisable : quota fixe et pipes, comme avant."""

    QUOTA_CAP = 2.5            # plafond par worker = 2,5 × la part égale (taille du tampon d'observations)

    def __init__(self, cfg: Dict[str, Any], n_workers: int, seed: int, pool=None,
                 build_actor_fn: Optional[Callable] = None, envs_per_worker: Optional[int] = None):
        self.cfg = cfg
        self.pool = pool
        self.hist_prob = float(cfg["selfplay"]["hist_prob"])
        self.envs_per_worker = int(envs_per_worker or cfg["train"].get("envs_per_worker", 1) or 1)
        self.conns, self.procs = _spawn_workers(cfg, n_workers, seed, self.envs_per_worker, local=True)
        n_params = 0
        for i in range(len(self.conns)):
            n_params = int(self._recv(i, timeout=300))
        self.weights: Optional[SharedWeights] = None
        try:
            self.weights = SharedWeights(n_params)
        except Exception as err:  # noqa: BLE001 — /dev/shm absent ou trop petit : les poids passent par les pipes
            print(f"[collect] mémoire partagée indisponible ({err}) : poids envoyés par pipe")
        tr = cfg["train"]
        self.shared_obs = bool(tr.get("collect_shared_obs", True))
        self.ctrl: Optional[SharedBlock] = None
        self.ctrl_arr: Optional[np.ndarray] = None
        if bool(tr.get("collect_sync", True)):
            self.ctrl = SharedBlock.make(8 * (len(self.conns) + 1))
            if self.ctrl is None:
                print("[collect] mémoire partagée indisponible : quota fixe par worker")
            else:
                self.ctrl_arr = np.ndarray((len(self.conns) + 1,), dtype=np.int64, buffer=self.ctrl.shm.buf)
                self.ctrl_arr[:] = 0
        self.gen = 0
        self.obs_block: Optional[SharedBlock] = None
        self.obs_rows = 0                # lignes par worker du tampon d'observations
        self.obs_dim = 0                 # connu après la 1re collecte (observations passées par les pipes)
        self.obs_failed = False
        self.last_stats: Dict[str, float] = {}

    @property
    def n_envs(self) -> int:
        return len(self.conns) * self.envs_per_worker

    def _hist_info(self) -> Dict[str, Any]:
        entries: List[Tuple[Dict[str, Any], float]] = []
        if self.pool is not None and len(self.pool) > 0:
            for e, p in self.pool.sampling_distribution():
                entries.append(({"id": e["id"], "path": str(e["path"]), "elo": float(e["elo"])}, float(p)))
        return {"hist_prob": self.hist_prob if entries else 0.0, "entries": entries}

    def _obs_buffer(self, rows: int) -> Optional[Tuple[str, int]]:
        """Tampon d'observations d'au moins `rows` lignes par worker (réalloué si le lot grandit)."""
        if not self.shared_obs or self.obs_failed or self.ctrl_arr is None or self.obs_dim <= 0:
            return None
        if self.obs_block is None or rows > self.obs_rows:
            if self.obs_block is not None:
                self.obs_block.close()
            self.obs_block = SharedBlock.make(4 * rows * self.obs_dim * len(self.conns))
            if self.obs_block is None:
                self.obs_failed = True
                print("[collect] pas assez de mémoire partagée pour les observations : envoi par pipe")
                return None
            self.obs_rows = rows
        return self.obs_block.name, self.obs_rows

    def _wait_total(self, n_steps: int, timeout: float) -> None:
        """Attend que le total des workers atteigne `n_steps`, puis donne le signal d'arrêt (ctrl[0] = génération).
        Panne d'un worker (processus mort) ou délai dépassé : on sort, _recv lèvera l'erreur."""
        deadline = time.perf_counter() + timeout
        next_check = time.perf_counter() + 1.0
        while int(self.ctrl_arr[1:].sum()) < n_steps:
            now = time.perf_counter()
            if now > deadline:
                break
            if now > next_check:
                next_check = now + 1.0
                if not all(p.is_alive() for p in self.procs):
                    break
            time.sleep(0.0005)
        self.ctrl_arr[0] = self.gen

    @torch.no_grad()
    def collect(self, actor: torch.nn.Module, n_steps: int, device: torch.device
                ) -> Tuple[List[Trajectory], List[Dict[str, Any]], int, float]:
        t0 = time.perf_counter()
        flat = flat_params(actor)
        if self.weights is not None:
            self.weights.write(flat)
            weights = ("shm", self.weights.name, len(flat))
        else:
            weights = ("raw", flat)
        hist = self._hist_info()
        W = len(self.conns)
        fair = int(math.ceil(max(0, n_steps) / W))
        timeout = 600.0 + 0.05 * fair             # ~ 20 steps/s minimum par worker avant de déclarer une panne
        sync = self.ctrl_arr is not None
        obs = None
        if sync:
            quota = int(math.ceil(self.QUOTA_CAP * fair)) + 1
            # une itération de la boucle d'un worker ajoute au plus (arènes × 6 voitures) transitions
            obs = self._obs_buffer(quota + 6 * self.envs_per_worker)
            self.gen += 1
            self.ctrl_arr[1:] = 0
            for w, conn in enumerate(self.conns):
                conn.send(("collect", quota, weights, hist,
                           {"ctrl": self.ctrl.name, "slots": W, "index": w, "gen": self.gen, "obs": obs}))
            self._wait_total(n_steps, timeout)
        else:
            for conn in self.conns:
                conn.send(("collect", fair, weights, hist))
        t_stop = time.perf_counter()
        trajs: List[Trajectory] = []
        episodes: List[Dict[str, Any]] = []
        per_worker: List[int] = []
        actives: List[float] = []
        collected = 0
        for w in range(W):
            payload = self._recv(w, timeout=timeout)
            t, e, n = payload[:3]
            if len(payload) > 3:
                actives.append(float(payload[3]["t_active"]))
            for traj in t:                         # index d'arène global, comme en collecte centrale
                traj.slot = w * self.envs_per_worker + traj.slot
            trajs.extend(t)
            episodes.extend(e)
            per_worker.append(int(n))
            collected += n
        t_recv = time.perf_counter()
        if obs is not None:
            self._attach_obs(trajs, per_worker)
        elif self.obs_dim <= 0 and trajs and trajs[0].obs is not None:
            self.obs_dim = int(np.asarray(trajs[0].obs).shape[-1])   # tampon partagé dès la collecte suivante
        t_end = time.perf_counter()
        pw = np.asarray(per_worker, dtype=np.float64)
        self.last_stats = {
            "collect/worker_steps_min": float(pw.min()), "collect/worker_steps_median": float(np.median(pw)),
            "collect/worker_steps_max": float(pw.max()), "collect/sync": float(sync),
            "collect/shared_obs": float(obs is not None),
            "time/collect_run_s": t_stop - t0,       # jusqu'au signal d'arrêt (quota fixe : jusqu'à l'envoi)
            "time/collect_recv_s": t_recv - t_stop,  # fin des workers + réception
            "time/collect_obs_s": t_end - t_recv,    # recopie des observations depuis la mémoire partagée
        }
        if actives:
            a = np.asarray(actives)
            self.last_stats.update({"collect/worker_active_min_s": float(a.min()),
                                    "collect/worker_active_max_s": float(a.max())})
        return trajs, episodes, collected, time.perf_counter() - t0

    def _attach_obs(self, trajs: List[Trajectory], per_worker: List[int]) -> None:
        """Observations du tampon partagé → UN tableau du lot (une seule copie, avant que les workers ne réécrivent le
        tampon à la collecte suivante) ; chaque trajectoire en reçoit une vue."""
        rows = self.obs_rows
        flat = np.ndarray((len(self.conns) * rows, self.obs_dim), dtype=np.float32, buffer=self.obs_block.shm.buf)
        obs_all = np.concatenate([flat[w * rows:w * rows + n] for w, n in enumerate(per_worker)])
        pos = 0
        for t in trajs:
            k = len(t)
            t.obs = obs_all[pos:pos + k]
            t.obs_base = (obs_all, pos)
            pos += k

    def close(self) -> None:
        self._close_workers()
        for blk in (self.weights, self.ctrl, self.obs_block):
            if blk is not None:
                blk.close()
        self.weights = self.ctrl = self.obs_block = None
        self.ctrl_arr = None


def make_collector(cfg: Dict[str, Any], n_workers: int, seed: int, pool=None,
                   build_actor_fn: Optional[Callable] = None):
    mode = str(cfg["train"].get("collect_mode", "central"))
    if mode == "local":
        return LocalCollector(cfg, n_workers, seed, pool=pool, build_actor_fn=build_actor_fn)
    if mode != "central":
        raise ValueError(f"train.collect_mode inconnu : {mode!r} (attendu 'central' ou 'local')")
    return Collector(cfg, n_workers, seed, pool=pool, build_actor_fn=build_actor_fn)
