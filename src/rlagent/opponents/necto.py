"""Necto (bot RLBot v5 de NectoFamily) comme adversaire dans nos audits RocketSim.

Le clone `dist/opponents/NectoFamily` (ignoré par git) fournit le modèle TorchScript ; rien d'autre n'en est importé.
Ce module refait en numpy ce que font `necto/necto_obs.py` (observation) et `necto/agent.py` + `bot.py` (actions,
kickoff codé en dur), à partir de notre état RLGym 2 au lieu d'un `V1GameState` de `rlgym_compat`. Vérification :
`scripts/necto_check.py` compare cette observation à celle du code du clone sur des paquets RLBot équivalents.

Repris tels quels du bot, défauts compris, pour jouer le même Necto qu'en vrai match :
  - le minuteur de démo est une dent de scie (3 s qui décroît puis repart), sans lien avec les démos ;
  - les minuteurs de boost partent quand un pad est *disponible* et sont mis à zéro quand il est pris ;
  - `is_teammate` / `is_opponent` désignent bleu / orange, échangés ensuite pour l'orange.
Écarts connus avec le vrai jeu : Necto décide ici sur l'état au début du step et son action part tout de suite
(en RLBot elle est calculée 1 tick après le step et appliquée au step suivant, ~7 ticks de retard en plus) ;
`on_ground` vient des roues de RocketSim, pas de l'`air_state` du paquet (1 à 6 ticks d'écart après un saut) ;
le kickoff codé en dur est joué par blocs de 8 ticks et s'arrête au début du step où la balle a bougé.
Le kickoff codé en dur ne dure que 168 ticks (1,4 s : départ boosté puis flip) ; la politique finit l'approche.
Vérifié le 11/11 sur 3000 états (Necto bleu et orange) : obs identique au code du clone à 4e-5 près (angles d'Euler),
action argmax identique à 100 %.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.api import GameState

REPO_ROOT = Path(__file__).resolve().parents[3]
NECTO_DIR = REPO_ROOT / "dist" / "opponents" / "NectoFamily" / "necto"
TICK_SKIP = 8

_INVERT = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4, dtype=np.float64)
_NORM = np.array([1.0] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4, dtype=np.float64)

# bot.py : (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) par tick, 168 ticks.
_K = [(44, (1, 0, 0, 0, 0, 0, 1, 0)), (16, (1, -1, 0, 0, 0, 0, 1, 0)), (8, (1, 0, 0, 0, 0, 1, 1, 0)),
      (4, (1, 0, 0, 0, 0, 0, 1, 0)), (4, (1, 0, -0.7, 0.8, 0, 1, 1, 0)), (52, (1, 0, 1, 0, 0, 0, 1, 0)),
      (40, (1, 0, 0.5, 0, 1, 0, 0, 0))]
KICKOFF_TICKS = np.concatenate([np.tile(np.array(row, dtype=np.float32), (n, 1)) for n, row in _K])


def is_necto_spec(spec: str) -> bool:
    return spec == "necto" or spec.startswith("necto:")


class NectoObs:
    """`NectoObsBuilder` du clone, sur un `GameState` RLGym 2. Un objet par partie (minuteurs gardés entre les buts,
    comme le bot qui ne remet rien à zéro)."""

    def __init__(self, pad_locations: np.ndarray, tick_skip: int = TICK_SKIP):
        self.tick_skip = float(tick_skip)
        self.pad_locations = np.asarray(pad_locations, dtype=np.float64)
        self.pad_big = self.pad_locations[:, 2] > 72      # le bot lit is_full_boost ; z vaut 73 / 70 ici
        self.boost_timers = np.zeros(len(self.pad_locations))
        self.demo_timers: dict[str, float] = defaultdict(float)

    def build(self, me: str, state: GameState, prev_action: np.ndarray):
        """-> (q (1,1,32), kv (1,N,24), mask (1,N)) ; joueurs dans l'ordre du bot : soi, coéquipiers, adversaires."""
        my_team = state.cars[me].team_num
        order = [me] + [a for a, c in state.cars.items() if a != me and c.team_num == my_team] \
            + [a for a, c in state.cars.items() if c.team_num != my_team]
        n_p = len(order)
        qkv = np.zeros((1, 1 + n_p + len(self.pad_locations), 24))
        ball = state.ball
        qkv[0, 0, 3] = 1
        qkv[0, 0, 5:8] = ball.position
        qkv[0, 0, 8:11] = ball.linear_velocity
        qkv[0, 0, 17:20] = ball.angular_velocity
        for n, agent in enumerate(order, start=1):
            car = state.cars[agent]
            qkv[0, n, 1 if car.team_num == cv.BLUE_TEAM else 2] = 1
            ph = car.physics
            qkv[0, n, 5:8] = ph.position
            qkv[0, n, 8:11] = ph.linear_velocity
            qkv[0, n, 11:14] = ph.forward
            qkv[0, n, 14:17] = ph.up
            qkv[0, n, 17:20] = ph.angular_velocity
            qkv[0, n, 20] = car.boost_amount / 100.0
            qkv[0, n, 22] = car.on_ground
            qkv[0, n, 23] = car.has_flip
            t = self.demo_timers[agent]
            self.demo_timers[agent] = 3.0 if t <= 0 else max(t - self.tick_skip / 120, 0.0)
            qkv[0, n, 21] = self.demo_timers[agent] / 10
        p0 = 1 + n_p
        available = (np.asarray(state.boost_pad_timers) == 0).astype(np.float64)
        qkv[0, p0:, 4] = 1
        qkv[0, p0:, 5:8] = self.pad_locations
        qkv[0, p0:, 20] = 0.12 + 0.88 * self.pad_big
        grabs = (available == 1) & (self.boost_timers == 0)
        self.boost_timers[grabs] = 0.4 + 0.6 * (self.pad_locations[grabs, 2] > 72)
        self.boost_timers *= available
        qkv[0, p0:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200
        self.boost_timers[self.boost_timers < 0] = 0

        qkv = qkv / _NORM
        mask = np.zeros((1, qkv.shape[1]))
        qkv[0, 1, 0] = 1                                  # is_main
        if my_team == cv.ORANGE_TEAM:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]
            qkv *= _INVERT
        q = np.concatenate((qkv[0, 1, :], np.asarray(prev_action, dtype=np.float64)))[None, None]
        qkv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, qkv, mask


def parse_logits(out, beta: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """`Agent.act` du clone : 5 têtes (throttle/pitch, steer/yaw/roll, jump, boost, handbrake) -> 8 commandes."""
    import torch
    logits = [o.reshape(-1).double().numpy() for o in out]
    if beta == 1:
        a = np.array([int(np.argmax(l)) for l in logits])
    elif beta == -1:
        a = np.array([int(np.argmin(l)) for l in logits])
    else:
        rng = rng or np.random.default_rng()
        scale = 0.0 if beta == 0 else float(np.log((beta + 1) / (1 - beta)) / np.log(3))
        a = []
        for l in logits:
            p = torch.softmax(torch.as_tensor(l * scale), -1).numpy()
            a.append(int(rng.choice(len(p), p=p / p.sum())))
        a = np.array(a)
    thr, steer, jump, boost, hb = a[0] - 1, a[1] - 1, a[2], a[3], a[4]
    return np.array([thr, steer, thr, steer * (1 - hb), steer * hb, jump, boost, hb], dtype=np.float32)


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        import warnings

        import torch
        path = NECTO_DIR / "necto-model.pt"
        if not path.exists():
            raise FileNotFoundError(f"Modèle Necto absent : {path} (cloner NectoFamily dans dist/opponents)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _MODEL = torch.jit.load(str(path), map_location="cpu")
        _MODEL.eval()
    return _MODEL


class NectoOpponent:
    """Pilote une ou plusieurs voitures Necto. `new_game()` à chaque partie, `act()` à chaque step (8 ticks) :
    renvoie 8 lignes de commandes (une par tick), le kickoff codé en dur en a des différentes."""

    name = "necto"
    _load_model = staticmethod(lambda: load_model())

    def __init__(self, beta: float = 1.0, hardcoded_kickoffs: bool = True, seed: int | None = None):
        self.beta = float(beta)
        self.hardcoded_kickoffs = hardcoded_kickoffs
        self.rng = np.random.default_rng(seed)
        self.model = self._load_model()
        self.obs = None
        self.prev: dict[str, np.ndarray] = {}
        self.kickoff_index: dict[str, int] = {}

    @classmethod
    def from_spec(cls, spec: str, seed: int | None = None) -> NectoOpponent:
        """`necto` (argmax, comme le bot) ou `necto:<beta>` (1 argmax, 0,5 échantillonné, 0 aléatoire) ; idem nexto."""
        return cls(beta=float(spec.split(":", 1)[1]) if ":" in spec else 1.0, seed=seed)

    def new_game(self, pad_locations: np.ndarray) -> None:
        self.obs = NectoObs(pad_locations)
        self.prev = {}
        self.kickoff_index = {}

    def on_reset(self, state: GameState) -> None:
        """Remise au centre : le bot relance son kickoff codé en dur (1v1 : il est toujours le preneur)."""
        self.kickoff_index = {}

    def step_beta(self, state: GameState) -> float:
        """Beta de ce step (Necto : toujours celui du bot)."""
        return self.beta

    def _kickoff_rows(self, agent: str, state: GameState) -> np.ndarray | None:
        if not self.hardcoded_kickoffs:
            return None
        if agent not in self.kickoff_index:          # décidé une fois, au premier step de la remise au centre
            self.kickoff_index[agent] = 0 if self._is_taker(agent, state) else -2
        idx = self.kickoff_index[agent]
        if idx < 0 or idx >= len(KICKOFF_TICKS) or float(state.ball.position[1]) != 0.0:
            self.kickoff_index[agent] = -2
            return None
        rows = KICKOFF_TICKS[idx:idx + TICK_SKIP]
        if len(rows) < TICK_SKIP:
            rows = np.vstack([rows, np.repeat(rows[-1:], TICK_SKIP - len(rows), axis=0)])
        self.kickoff_index[agent] = idx + TICK_SKIP
        return rows

    @staticmethod
    def _is_taker(agent: str, state: GameState) -> bool:
        """bot.py : le plus proche de la balle (à 10 uu près) ; à égalité dans l'équipe, celui de gauche."""
        ball = state.ball.position[:2]
        d = {a: float(np.linalg.norm(c.physics.position[:2] - ball)) for a, c in state.cars.items()}
        me = state.cars[agent]
        if abs(min(d.values()) - d[agent]) > 10:
            return False
        for a, c in state.cars.items():
            if a != agent and c.team_num == me.team_num and abs(d[a] - d[agent]) <= 10:
                x_o, x_m = float(c.physics.position[0]), float(me.physics.position[0])
                if not (x_o < x_m if me.team_num == cv.BLUE_TEAM else x_o > x_m):
                    return False
        return True

    def build_obs(self, agent: str, state: GameState):
        return self.obs.build(agent, state, self.prev.get(agent, np.zeros(8)))

    def policy(self, q, kv, mask, beta: float | None = None) -> np.ndarray:
        import torch
        with torch.no_grad():
            out, _ = self.model(tuple(torch.as_tensor(x, dtype=torch.float32) for x in (q, kv, mask)))
        return parse_logits(out, self.beta if beta is None else beta, self.rng)

    def act(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]:
        """-> {agent: commandes (8 ticks, 8)}. L'observation est construite à chaque step même pendant le kickoff
        (le bot le fait aussi : les minuteurs avancent pareil)."""
        out = {}
        beta = self.step_beta(state)
        for agent in agents:
            q, kv, mask = self.build_obs(agent, state)
            rows = self._kickoff_rows(agent, state)
            if rows is None:
                rows = np.repeat(self.policy(q, kv, mask, beta)[None], TICK_SKIP, axis=0)
            self.prev[agent] = rows[-1].astype(np.float64)
            out[agent] = rows
        return out


class MixedRepeatAction:
    """Remplace `RepeatAction(TrackedLookupTableAction)` pour les audits : un indice de la table (shape (1,)) est
    traité exactement comme avant ; un tableau (8,) ou (repeats, 8) passe tel quel (commandes directes, ex. Necto).
    `shared_info["prev_actions"]` reste tenu à jour pour notre observation."""

    def __init__(self, repeats: int = TICK_SKIP):
        self.repeats = repeats
        self.lookup = LookupTableAction()

    def get_action_space(self, agent):
        return self.lookup.get_action_space(agent)

    def reset(self, agents, initial_state, shared_info) -> None:
        self.lookup.reset(agents, initial_state, shared_info)
        shared_info["prev_actions"] = {a: np.zeros(8, dtype=np.float32) for a in agents}

    def parse_actions(self, actions, state, shared_info):
        idx = {a: v for a, v in actions.items() if np.asarray(v).size == 1}
        parsed = self.lookup.parse_actions(idx, state, shared_info) if idx else {}
        out = {}
        prev = shared_info.setdefault("prev_actions", {})
        for agent, v in actions.items():
            a = np.asarray(parsed.get(agent, v), dtype=np.float32).reshape(-1, 8)
            if a.shape[0] == 1:
                a = a.repeat(self.repeats, axis=0)
            elif a.shape[0] != self.repeats:
                raise ValueError(f"{agent} : {a.shape[0]} lignes de commandes, {self.repeats} attendues")
            prev[agent] = a[-1].copy()
            out[agent] = a
        return out


def pad_locations_of(env) -> np.ndarray:
    """Positions des pads dans l'ordre de `state.boost_pad_timers` (celui de l'arène RocketSim)."""
    return np.array([p.get_pos().as_tuple() for p in env.transition_engine._arena.get_boost_pads()], dtype=np.float64)


CHECK_SUMMARY = REPO_ROOT / "dist" / "opponents" / "necto_check.json"
KNOWN_GAPS = ("écarts non mesurables sans le jeu : Necto agit ici sans le retard RLBot (~7 ticks de mieux, il est "
              "plutôt avantagé), physique RocketSim, kickoff codé en dur par blocs de 8 ticks ;\nla comparaison "
              "part de paquets refaits depuis RocketSim : air_state / dodge_timeout du vrai jeu et ordre des pads "
              "RLBot non vérifiés")


def check_summary_path(name: str = "necto") -> Path:
    return CHECK_SUMMARY.with_name(f"{name}_check.json")


def fidelity_report(name: str = "necto", known_gaps: str = KNOWN_GAPS) -> dict:
    """Résumé de la dernière vérification `scripts/necto_check.py` (adaptateur vs code du clone), pour les rapports."""
    path = check_summary_path(name)
    if not path.exists():
        return {"texte": f"adaptateur NON vérifié : lancer scripts/necto_check.py dump --bot {name} puis compare "
                         f"--bot {name}\n" + known_gaps}
    r = json.loads(path.read_text(encoding="utf-8"))
    worst = max(r["groupes"].items(), key=lambda kv: kv[1]["steps_ecart_%"])
    txt = (f"vs code du clone sur {r['steps']} états : obs identique {r['obs_identique_%']:.1f} % des steps "
           f"(pire groupe {worst[0]} : {worst[1]['steps_ecart_%']:.1f} %, écart max {worst[1]['max']:.3g}), "
           f"action argmax identique {r['actions_identiques_%']:.1f} %\n"
           f"on_ground du vrai jeu différent de RocketSim sur {r['on_ground_jeu_different_%']:.1f} % des steps "
           f"(estimation, air_state)\n" + known_gaps)
    return {**r, "texte": txt}
