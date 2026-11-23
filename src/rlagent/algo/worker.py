"""Processus d'environnement : fait tourner un ou plusieurs envs RocketSim et répond au processus principal.

Protocole à UN env (n_envs=None, tuples via multiprocessing.Pipe) :
  → ("step", actions: np.ndarray[n_agents])        ← ("ok", StepResult)
  → ("set_step", global_step: int)                  (pas de réponse)
  → ("set_info", {clé: valeur})                     (pas de réponse) : copié dans shared_info (ex. niveaux d'exercices)
  → ("close",)
Au démarrage le worker envoie ("ok", EpisodeStart) ; en cas d'erreur ("error", traceback).
Avec n_envs=k (plusieurs arènes par worker, `train.envs_per_worker`) : mêmes messages, mais les actions,
les EpisodeStart et les StepResult sont des listes (une entrée par arène). En plus :
  → ("step_part", début, [actions_env_début, ...])  ← ("ok", [StepResult, ...]) : seulement ces arènes
     (collecte en double tampon, `train.collect_pipeline`).
Mode local (`train.collect_mode: local`, local=True) : le worker a sa propre copie de l'acteur. Au démarrage il envoie
("ok", nb_de_paramètres_de_l_acteur), puis :
  → ("collect", quota, poids, adversaires[, sync])  ← ("ok", (trajectoires, épisodes, n[, infos])) : joue jusqu'à
     `quota` transitions apprenantes. poids = ("shm", nom, n) (mémoire partagée) ou ("raw", vecteur) ; adversaires =
     {"hist_prob", "entries": [(entrée du pool, probabilité), ...]}.
     sync (collecte à arrêt global, cf. collector.LocalCollector) = {"ctrl": nom, "slots": W, "index": w, "gen": g,
     "obs": (nom, lignes) ou None} : le worker publie son nombre de transitions dans ctrl[1 + w] et s'arrête dès que
     ctrl[0] == g (le principal y écrit g quand le TOTAL atteint le lot) ; `quota` n'est alors qu'un plafond. Avec
     "obs", les observations des trajectoires sont écrites dans la mémoire partagée (lignes [w·lignes, ...), dans
     l'ordre des trajectoires renvoyées) au lieu de passer par le pipe : Trajectory.obs = None au retour.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeStart:
    agents: List[str]
    teams: np.ndarray        # (n,) 0 = bleu, 1 = orange
    obs: np.ndarray          # (n, OBS_DIM)
    # (n,) action (indice de la table de 90, la même que Nexto) que Nexto choisirait dans cet état, −1 = non calculée
    # (`train.kickstart` : imitation de Nexto au début de la session 10) ; None = pas d'imitation en cours
    teacher: Optional[np.ndarray] = None


# Bots de sparring (selfplay.sparring) : l'ordre fixe le code d'action, identique dans tous les processus.
SPARRING_BOTS = ("nexto", "galactus", "kira", "rain", "necto")


# Action « jouée par un bot de sparring » envoyée par le collecteur : −(1 + 1000 · indice du bot + round(100 · beta)),
# beta dans [0, 1] ; le worker calcule les commandes du bot. Nexto (indice 0) garde le code de la session 10.
def bot_action_code(bot: str, beta: float) -> int:
    return -(1 + 1000 * SPARRING_BOTS.index(bot) + int(round(100.0 * float(beta))))


def bot_of_code(code: int):
    v = -int(code) - 1
    return SPARRING_BOTS[v // 1000], (v % 1000) / 100.0


def nexto_action_code(beta: float) -> int:
    return bot_action_code("nexto", beta)


def nexto_beta_of(code: int) -> float:
    return bot_of_code(code)[1]


def sparring_settings(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """`selfplay.sparring.bots` : {bot: {prob, beta_start}} -> {bot: {"prob", "beta"}} (bots à prob > 0 seulement).
    L'ancienne clé `selfplay.nexto` (session 10) vaut un seul bot. train.py fait monter `beta` (vers l'argmax) quand on
    prend assez de points au bot. Moteur factice des tests : aucun bot (pas d'arène)."""
    sp = cfg.get("selfplay") or {}
    if (cfg.get("env") or {}).get("engine", "rocketsim") == "fake":
        return {}
    bots = dict((sp.get("sparring") or {}).get("bots") or {})
    if not bots and sp.get("nexto"):
        bots = {"nexto": sp["nexto"]}
    out = {}
    for name, b in bots.items():
        if name not in SPARRING_BOTS:
            raise ValueError(f"selfplay.sparring : bot inconnu « {name} » ({', '.join(SPARRING_BOTS)})")
        if float((b or {}).get("prob", 0.0)) > 0.0:
            out[name] = {"prob": float(b["prob"]), "beta": float(b.get("beta_start", 1.0))}
    return out


def pick_sparring(sparring: Dict[str, Dict[str, float]], rng) -> Optional[str]:
    """Bot de sparring de la prochaine partie (chacun avec sa part `prob` des épisodes), ou None (self-play)."""
    u = rng.random()
    for name, b in sparring.items():
        u -= float(b["prob"])
        if u < 0.0:
            return name
    return None


def sparring_entry(name: str, sparring: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    return {"id": name, "kind": "bot", "bot": name, "beta": float(sparring[name].get("beta", 1.0)), "elo": 0.0}


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
        self.tracker = EpisodeTracker(rc.get("double_commit_radius", 900), rc.get("double_commit_min_speed", 400),
                                      rewards_cfg=rc)
        # Bots de sparring (selfplay.sparring : Nexto, GalactusV2, Kira, Rain) et Nexto professeur (train.kickstart).
        # Leurs commandes (8 lignes par step : kickoff codé en dur, délai d'action) passent par MixedRepeatAction,
        # identique à RepeatAction pour nos indices.
        self.seed = seed
        self.sparring = sparring_settings(cfg)
        self.kick_cfg = (cfg.get("train") or {}).get("kickstart") or {}
        if cfg["env"].get("engine", "rocketsim") == "fake":                # moteur factice des tests : pas d'arène
            self.kick_cfg = {}
        self.bots: Dict[str, Any] = {}         # adversaires de sparring de cette arène (créés au 1er besoin)
        self.teacher_obs = None                # NextoObs du professeur (sans état : un seul par arène)
        self.rng = np.random.default_rng(seed + 104729)
        if self.sparring:
            from ..opponents.necto import MixedRepeatAction
            self.env.action_parser = MixedRepeatAction(int(cfg["env"]["action_repeat"]))
        self.current = self.reset()

    def _pads(self):
        from ..opponents.necto import pad_locations_of
        return pad_locations_of(self.env)

    def _kick_frac(self) -> float:
        k = self.kick_cfg
        if not k.get("enabled"):
            return 0.0
        step = float(self.env.shared_info.get("global_step", 0))
        begin, steps = float(k.get("begin", 0)), float(k.get("steps", 0))
        return float(k.get("frac", 0.25)) if begin <= step < begin + steps else 0.0

    def _teacher(self, agents: List[str]) -> Optional[np.ndarray]:
        """Action de Nexto (argmax de sa politique, sans le kickoff codé en dur) pour une part `frac` des voitures."""
        frac = self._kick_frac()
        if frac <= 0.0:
            return None
        from ..opponents.nexto import NextoObs, load_model
        import torch
        if self.teacher_obs is None:
            self.teacher_obs = NextoObs(self._pads())
        state = self.env.state
        prev = self.env.shared_info.get("prev_actions", {})
        out = np.full(len(agents), -1, dtype=np.int16)
        model = load_model()
        for i, a in enumerate(agents):
            if self.rng.random() >= frac:
                continue
            q, kv, mask = self.teacher_obs.build(a, state, np.asarray(prev.get(a, np.zeros(8)), dtype=np.float64))
            with torch.no_grad():
                logits, _ = model(tuple(torch.as_tensor(x, dtype=torch.float32) for x in (q, kv, mask)))
            out[i] = int(torch.argmax(logits.reshape(-1)))
        return out

    def reset(self) -> EpisodeStart:
        env = self.env
        obs_dict = env.reset()
        agents = list(obs_dict.keys())
        self.tracker.reset(env.state, env.shared_info)
        for bot in self.bots.values():
            bot.on_reset(env.state)
        return EpisodeStart(
            agents=agents,
            teams=np.array([env.state.cars[a].team_num for a in agents], dtype=np.int8),
            obs=np.stack([obs_dict[a] for a in agents]).astype(np.float32),
            teacher=self._teacher(agents),
        )

    @property
    def nexto(self):
        return self.bots.get("nexto")

    def _bot_rows(self, name: str, agents: List[str], beta: float) -> Dict[str, np.ndarray]:
        bot = self.bots.get(name)
        if bot is None:
            from ..opponents.nexto import bot_from_spec
            bot = self.bots[name] = bot_from_spec(name, seed=self.seed + 31 * SPARRING_BOTS.index(name))
            bot.new_game(self._pads())
            bot.on_reset(self.env.state)
        bot.beta = float(beta)
        return bot.act(agents, self.env.state)

    def step(self, actions: np.ndarray) -> StepResult:
        env, current = self.env, self.current
        action_dict = {a: np.array([int(actions[i])], dtype=np.int64) for i, a in enumerate(current.agents)}
        by_bot: Dict[Any, List[str]] = {}
        for i, a in enumerate(current.agents):
            if int(actions[i]) < 0:
                by_bot.setdefault(bot_of_code(int(actions[i])), []).append(a)
        for (name, beta), agents in by_bot.items():
            action_dict.update(self._bot_rows(name, agents, beta))
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
            summary["reward_abs"] = {k: v / steps for k, v in env.shared_info.get("reward_log_abs", {}).items()}
            mut = getattr(env, "start_mutator", None)
            if mut is not None and "drill" in summary:
                mut.report("drill_" + summary["drill"], summary["drill_success"])
            self.current = self.reset()
            return StepResult(rewards, True, terminated, final_obs, summary, self.current)
        self.current = EpisodeStart(current.agents, current.teams,
                                    np.stack([obs_dict[a] for a in current.agents]).astype(np.float32),
                                    self._teacher(current.agents))
        return StepResult(rewards, False, False, None, None, self.current)

    def set_global_step(self, step: int) -> None:
        self.env.shared_info["global_step"] = int(step)

    def set_info(self, info: Dict[str, Any]) -> None:
        self.env.shared_info.update(info)


def env_seed(seed: int, env_index: int) -> int:
    """Graine de la k-ième arène d'un worker (la 1re garde la graine du worker : comportement inchangé à 1 arène)."""
    return seed + 7919 * env_index


class LocalPolicy:
    """Mode local : acteur courant + versions du pool, sur le CPU du worker (1 thread), et suivi des arènes."""

    def __init__(self, cfg: Dict[str, Any], seed: int):
        import random

        import torch

        from ..env.actions import N_ACTIONS
        from ..models.attention import actor_from_model_cfg

        torch.set_num_threads(1)
        self.n_actions = N_ACTIONS
        self.actor = actor_from_model_cfg(cfg["model"], N_ACTIONS).eval()
        self.n_params = int(sum(p.numel() for p in self.actor.parameters()))
        self.rng = random.Random(seed)
        self.stack_hist = bool(cfg["train"].get("stack_hist", True))
        self.hist: Dict[str, Any] = {"hist_prob": 0.0, "entries": []}
        self._hist_actors: "Dict[str, Any]" = {}
        self._shm = None
        self._sync_shm: Dict[str, Any] = {}       # nom → SharedMemory (contrôle de l'arrêt, tampon d'observations)
        self.rollout = None

    def load_weights(self, weights) -> None:
        import torch
        kind = weights[0]
        if kind == "shm":
            if self._shm is None or self._shm.name != weights[1]:
                from multiprocessing import shared_memory
                self._shm = shared_memory.SharedMemory(name=weights[1])
            flat = np.ndarray((int(weights[2]),), dtype=np.float32, buffer=self._shm.buf).copy()
        else:
            flat = np.asarray(weights[1], dtype=np.float32)
        torch.nn.utils.vector_to_parameters(torch.from_numpy(flat), self.actor.parameters())

    def new_slot(self, start: EpisodeStart):
        from .collector import EnvSlot
        slot = EnvSlot(episode=start)
        sparring = self.hist.get("sparring") or {}
        bot = pick_sparring(sparring, self.rng) if sparring else None
        if bot is not None:
            slot.hist_team = self.rng.choice([0, 1])
            slot.hist_entry = sparring_entry(bot, sparring)
            return slot
        entries = self.hist["entries"]
        # même tirage que le collecteur central (Collector._new_slot) avec la loi de tirage du pool
        if entries and self.rng.random() < float(self.hist["hist_prob"]):
            slot.hist_team = self.rng.choice([0, 1])
            slot.hist_entry = self.rng.choices([e for e, _ in entries], weights=[p for _, p in entries])[0]
        return slot

    def hist_actor(self, entry: Dict[str, Any]):
        import torch

        from ..models.attention import actor_from_model_cfg
        key = entry["id"]
        actor = self._hist_actors.get(key)
        if actor is None:
            from ..selfplay.pool import version_path
            data = torch.load(version_path(entry["path"]), map_location="cpu", weights_only=False)
            actor = actor_from_model_cfg(data["model_cfg"], self.n_actions)
            actor.load_state_dict(data["actor"])
            actor.eval()
            self._hist_actors[key] = actor
            if len(self._hist_actors) > len(self.hist["entries"]) + 4:   # versions sorties de la fenêtre du pool
                keep = {e["id"] for e, _ in self.hist["entries"]} | {key}
                if self.rollout is not None:      # (sauf si une partie en cours joue encore contre elles)
                    keep |= {s.hist_entry["id"] for s in self.rollout.slots if s.hist_entry is not None}
                for k in [k for k in self._hist_actors if k not in keep]:
                    del self._hist_actors[k]
        return actor

    def _attach(self, name: str):
        shm = self._sync_shm.get(name)
        if shm is None:
            from multiprocessing import shared_memory
            shm = self._sync_shm[name] = shared_memory.SharedMemory(name=name)
        return shm

    def _release_stale(self, keep) -> None:
        for name in [n for n in self._sync_shm if n not in keep]:  # tampon réalloué par le principal (lot plus gros)
            try:
                self._sync_shm.pop(name).close()
            except Exception:  # noqa: BLE001
                pass

    def collect(self, runners: List["EnvRunner"], quota: int, weights, hist: Dict[str, Any],
                sync: Optional[Dict[str, Any]] = None):
        import torch

        from .collector import Rollout
        if weights is not None:
            self.load_weights(weights)
        self.hist = hist
        if self.rollout is None:
            self.rollout = Rollout([r.current for r in runners], self.new_slot,
                                   stack_hist=self.stack_hist)
        ids = list(range(len(runners)))
        cpu = torch.device("cpu")
        ctrl, slot, gen, obs_buf = None, 0, 0, None
        if sync is not None:
            names = {sync["ctrl"]} | ({sync["obs"][0]} if sync.get("obs") else set())
            self._release_stale(names)
            ctrl = np.ndarray((int(sync["slots"]) + 1,), dtype=np.int64, buffer=self._attach(sync["ctrl"]).buf)
            slot, gen = 1 + int(sync["index"]), int(sync["gen"])
            if sync.get("obs"):
                name, rows = sync["obs"]
                dim = int(np.asarray(self.rollout.slots[0].episode.obs).shape[-1])
                flat = np.ndarray((int(sync["slots"]) * int(rows), dim), dtype=np.float32,
                                  buffer=self._attach(name).buf)
                obs_buf = flat[int(sync["index"]) * int(rows):(int(sync["index"]) + 1) * int(rows)]
        t0 = time.perf_counter()
        collected = 0
        while collected < quota and (ctrl is None or int(ctrl[0]) != gen):
            actions = self.rollout.act(ids, self.actor, cpu, self.hist_actor)
            collected += self.rollout.record(ids, [r.step(actions[i]) for i, r in enumerate(runners)])
            if ctrl is not None:
                ctrl[slot] = collected
        t_active = time.perf_counter() - t0
        trajs, episodes = self.rollout.cut()
        out = [t.compact() for t in trajs]
        if obs_buf is not None:
            pos = 0
            for t in out:
                k = len(t)
                obs_buf[pos:pos + k] = t.obs
                t.obs = None
                pos += k
        if sync is None:
            return out, episodes, collected
        return out, episodes, collected, {"t_active": t_active, "stopped": bool(int(ctrl[0]) == gen)}

    def close(self) -> None:
        self._release_stale(set())
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001
                pass


def worker_main(conn, cfg: Dict[str, Any], seed: int, worker_id: int, n_envs: Optional[int] = None,
                local: bool = False) -> None:
    """n_envs=None : protocole à UN env (messages simples). n_envs=k : k arènes, messages sous forme de listes
    (→ ("step", [actions_env0, ...]) ; ← ("ok", [StepResult_env0, ...])). local=True : mode local (voir en tête)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:  # Ctrl+C est géré par le processus principal (checkpoint propre), pas par les workers
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    policy = None
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)

        multi = n_envs is not None
        runners = [EnvRunner(cfg, env_seed(seed, k)) for k in range(n_envs if multi else 1)]
        if local:
            policy = LocalPolicy(cfg, seed)
            conn.send(("ok", policy.n_params))
        else:
            conn.send(("ok", [r.current for r in runners] if multi else runners[0].current))

        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "step":
                if multi:
                    conn.send(("ok", [r.step(a) for r, a in zip(runners, msg[1])]))
                else:
                    conn.send(("ok", runners[0].step(msg[1])))
            elif cmd == "step_part":
                lo = int(msg[1])
                conn.send(("ok", [runners[lo + i].step(a) for i, a in enumerate(msg[2])]))
            elif cmd == "collect":
                conn.send(("ok", policy.collect(runners, int(msg[1]), msg[2], msg[3],
                                                 msg[4] if len(msg) > 4 else None)))
            elif cmd == "set_step":
                for r in runners:
                    r.set_global_step(msg[1])
            elif cmd == "set_info":
                for r in runners:
                    r.set_info(msg[1])
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
        if policy is not None:
            policy.close()
        try:
            conn.close()
        except Exception:
            pass
