"""Bots GigaLearnCPP du botpack RLBot v5 à observation `AdvancedObs` : Kira (phrog) et Rain (Rou_Ad, RichardsWorld).

Même méthode que `galactus.py` : seuls les poids `SHARED_HEAD.lt` / `POLICY.lt` des clones `dist/opponents/Kira` et
`dist/opponents/RainStream` (ignorés par git) sont chargés ; le reste est refait depuis leur C++ :
  - Kira : `AdvancedObs` (9 balle + 8 action précédente + 34 pads + 29 par joueur, coéquipiers puis adversaires,
    sans emplacement vide : 109 en 1v1, le seul format du bot), tick skip 8, action delay 7, réseau
    tête 1024x3 puis politique 1024, 1024, 512, 512 (Linear + LayerNorm + ReLU) ;
  - Rain : `AdvancedObsPadded(3)` (AdvancedObs + vecteurs balle -> buts, 2 coéquipiers et 3 adversaires, emplacements
    vides à 0 : 231), un modèle par format (`checkpoint_1v1/2v2/3v3`, choisi par la plus grande équipe), tick skip 6,
    action delay 5, réseau tête 1280 puis politique 1024, 768, 512, 256 (ReLU).
Table d'actions, masques et argmax : ceux de `DefaultAction` / `InferUnit`, identiques à GalactusV2.

Écarts connus :
  - Rain décide ici tous les 8 ticks (notre pas), pas tous les 6 : sur un pas, 5 ticks de l'action précédente puis 3
    de la nouvelle (son retard de 5 ticks est gardé, sa cadence est un peu plus lente qu'en vrai) ;
  - ordre des joueurs : celui de notre état (le bot suit l'ordre du paquet RLBot) ;
  - « retourné » estimé comme pour GalactusV2 ; physique RocketSim.
Défaut du bot repris tel quel : pour un joueur orange, Rain voit « but adverse » et « son but » échangés (but orange
pris dans le repère déjà retourné).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from rlagent.opponents.galactus import (_INV, ANG_VEL_COEF, POS_COEF, VEL_COEF, GalactusObs, GalactusOpponent,
                                        action_mask, choose)
from rlagent.opponents.necto import REPO_ROOT
from rlagent.opponents.nexto import LOOKUP_TABLE, NEXTO_PADS

OPP_DIR = REPO_ROOT / "dist" / "opponents"
BLUE_GOAL_BACK = np.array([0.0, -6000.0, 642.775 / 2])
ORANGE_GOAL_BACK = np.array([0.0, 6000.0, 642.775 / 2])


@dataclass(frozen=True)
class GigaBot:
    name: str
    folders: dict          # joueurs par équipe -> dossier des .lt (relatif à dist/opponents)
    n_obs: int
    head: tuple
    policy: tuple
    activation: str        # "relu" | "leaky_relu"
    tick_skip: int
    action_delay: int
    padded: bool           # AdvancedObsPadded(3) (Rain) ou AdvancedObs (Kira)
    env_var: str


BOTS = {
    "kira": GigaBot("kira", {1: "Kira/rlbot"}, 109, (1024, 1024, 1024), (1024, 1024, 512, 512), "relu", 8, 7,
                    False, "RLAGENT_KIRA_DIR"),
    "rain": GigaBot("rain", {1: "RainStream/rlbot/checkpoint_1v1", 2: "RainStream/rlbot/checkpoint_2v2",
                             3: "RainStream/rlbot/checkpoint_3v3"}, 231, (1280,), (1024, 768, 512, 256), "relu", 6, 5,
                    True, "RLAGENT_RAIN_DIR"),
}


def bot_name_of(spec: str) -> str | None:
    name = spec.split(":", 1)[0]
    return name if name in BOTS else None


def is_gigalearn_spec(spec: str) -> bool:
    return bot_name_of(spec) is not None


def model_dir(bot: GigaBot, per_team: int) -> Path:
    """Dossier des poids ; `RLAGENT_<BOT>_DIR` (Kaggle) remplace `dist/opponents` (même arborescence que le clone)."""
    base = Path(os.environ.get(bot.env_var) or OPP_DIR)
    rel = bot.folders[min(per_team, max(bot.folders))] if bot.padded else bot.folders[1]
    return base / rel


def build_model(folder: Path, n_obs: int, head: tuple, policy: tuple, activation: str):
    """Réseau `GGL::Model` : chaque couche cachée Linear + LayerNorm + activation, sortie 90 ; poids des .lt."""
    import torch
    from torch import nn

    act = nn.LeakyReLU if activation == "leaky_relu" else nn.ReLU

    def mlp(n_in, sizes, n_out):
        layers, last = [], n_in
        for sz in sizes:
            layers += [nn.Linear(last, sz), nn.LayerNorm(sz), act()]
            last = sz
        if n_out:
            layers.append(nn.Linear(last, n_out))
        return nn.Sequential(*layers)

    nets = {"SHARED_HEAD": mlp(n_obs, head, 0), "POLICY": mlp(head[-1], policy, len(LOOKUP_TABLE))}
    for name, net in nets.items():
        path = folder / f"{name}.lt"
        if not path.exists():
            raise FileNotFoundError(f"Poids absents : {path} (cloner le bot dans dist/opponents)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            src = torch.jit.load(str(path), map_location="cpu")
        net.load_state_dict({k: v.detach().clone() for k, v in src.named_parameters()}, strict=True)
    model = nn.Sequential(nets["SHARED_HEAD"], nets["POLICY"]).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


_MODELS: dict = {}


def load_model(bot: GigaBot, per_team: int):
    folder = model_dir(bot, per_team)
    if folder not in _MODELS:
        _MODELS[folder] = build_model(folder, bot.n_obs, bot.head, bot.policy, bot.activation)
    return _MODELS[folder]


class AdvancedObs:
    """`AdvancedObs::BuildObs` (Kira) ou `AdvancedObsPadded::BuildObs` (Rain) sur un `GameState` RLGym 2."""

    def __init__(self, pad_locations: np.ndarray, padded: bool):
        pads = np.asarray(pad_locations, dtype=np.float64)
        self.order = np.array([int(np.linalg.norm(pads[:, :2] - p[:2], axis=1).argmin()) for p in NEXTO_PADS])
        if len(set(self.order.tolist())) != len(NEXTO_PADS):
            raise ValueError("pads de l'arène sans correspondance un à un avec BOOST_LOCATIONS")
        self.padded = padded

    def build(self, me: str, state: GameState, prev_action: np.ndarray) -> np.ndarray:
        car = state.cars[me]
        inv = car.team_num == cv.ORANGE_TEAM
        s = _INV if inv else 1.0
        ball = state.ball
        b_pos, b_vel, b_ang = ball.position * s, ball.linear_velocity * s, ball.angular_velocity * s
        timers = np.asarray(state.boost_pad_timers, dtype=np.float64)[self.order]
        if inv:
            timers = timers[::-1]
        pads = np.where(timers <= 0, 1.0, 1.0 / (1.0 + timers))
        parts = [b_pos * POS_COEF, b_vel * VEL_COEF, b_ang * ANG_VEL_COEF]
        if self.padded:
            own, enemy = (ORANGE_GOAL_BACK, BLUE_GOAL_BACK) if inv else (BLUE_GOAL_BACK, ORANGE_GOAL_BACK)
            parts += [(own - b_pos) * POS_COEF, (enemy - b_pos) * POS_COEF]
        parts += [np.asarray(prev_action, dtype=np.float64), pads, GalactusObs._player(car, inv, b_pos, b_vel)]
        mates = [GalactusObs._player(c, inv, b_pos, b_vel) for a, c in state.cars.items()
                 if a != me and c.team_num == car.team_num]
        opps = [GalactusObs._player(c, inv, b_pos, b_vel) for c in state.cars.values() if c.team_num != car.team_num]
        if self.padded:
            mates = (mates + [np.zeros(29)] * 2)[:2]
            opps = (opps + [np.zeros(29)] * 3)[:3]
        return np.concatenate(parts + mates + opps)


class GigaLearnOpponent(GalactusOpponent):
    """Pilote des voitures Kira ou Rain ; interface de `NectoOpponent`. Par pas de 8 ticks : `action_delay` ticks de
    l'action précédente puis la nouvelle (Rain : 5 + 3 ; Kira : 7 + 1)."""

    _load_model = staticmethod(lambda: None)            # un modèle par format, chargé à la demande

    def __init__(self, bot: str, beta: float = 1.0, seed: int | None = None):
        super().__init__(beta=beta, seed=seed)
        self.bot = BOTS[bot]
        self.name = bot
        load_model(self.bot, 1)                         # échoue tout de suite si les poids manquent

    @classmethod
    def from_spec(cls, spec: str, seed: int | None = None) -> GigaLearnOpponent:
        """`kira`, `rain` (argmax, comme les bots) ou `<bot>:<beta>`."""
        name, _, beta = spec.partition(":")
        return cls(name, beta=float(beta) if beta else 1.0, seed=seed)

    def new_game(self, pad_locations: np.ndarray) -> None:
        self.obs = AdvancedObs(pad_locations, self.bot.padded)
        self.prev = {}
        self.kickoff_index = {}

    def act(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]:
        import torch
        if not agents:
            return {}
        per_team = max(sum(c.team_num == t for c in state.cars.values()) for t in (0, 1))
        if not self.bot.padded and len(state.cars) != 2:
            raise ValueError(f"{self.name} ne joue qu'en 1v1")
        model = load_model(self.bot, per_team)
        beta = self.step_beta(state)
        obs = np.stack([self.obs.build(a, state, self.prev.get(a, np.zeros(8))) for a in agents])
        assert obs.shape[1] == self.bot.n_obs, obs.shape
        with torch.no_grad():
            logits = model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        delay = min(self.bot.action_delay, 7)
        out = {}
        for i, agent in enumerate(agents):
            new = LOOKUP_TABLE[choose(logits[i], action_mask(state.cars[agent]), beta, self.rng)]
            old = self.prev.get(agent, np.zeros(8)).astype(np.float32)
            out[agent] = np.vstack([np.repeat(old[None], delay, axis=0), np.repeat(new[None], 8 - delay, axis=0)])
            self.prev[agent] = new.astype(np.float64)
        return out


KNOWN_GAPS = {
    "kira": "adaptateur refait depuis le C++ (pas de binaire de référence) ; 1v1 seulement ; retourné estimé, "
            "physique RocketSim",
    "rain": "adaptateur refait depuis le C++ (pas de binaire de référence) ; décide tous les 8 ticks au lieu de 6 "
            "(5 ticks d'action précédente + 3 de la nouvelle) ; retourné estimé, physique RocketSim",
}


def fidelity_report(spec: str) -> dict:
    return {"texte": KNOWN_GAPS[bot_name_of(spec)]}
