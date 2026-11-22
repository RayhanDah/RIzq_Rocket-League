"""GalactusV2 (SoldierDoge, botpack RLBot v5, GigaLearnCPP) comme adversaire dans nos audits RocketSim.

Le clone `dist/opponents/GalactusV2` (ignoré par git) fournit les poids `rlbot/SHARED_HEAD.lt` et `rlbot/POLICY.lt`
(modules `torch::nn::Sequential` sérialisés, chargés avec `torch.jit.load`) ; rien d'autre n'en est exécuté.
Ce module refait en numpy ce que font, côté C++ :
  - `inc/RLGymCPP/ObsBuilders/GalactusOBS.cpp` : observation de 258 valeurs (maxPlayers 3, emplacements vides à 0) ;
  - `inc/RLGymCPP/ActionParsers/DefaultAction.cpp` : table de 90 actions (identique à celle de Nexto) et masques ;
  - `inc/GigaLearnCPP/InferenceModels.cpp` : logits masqués (-1e10), argmax en match (`deterministic = true`) ;
  - `src/RLBotMain.cpp` / `RLBotClient.cpp` : tick skip 8, action delay 7 (la nouvelle action ne part qu'au 8e tick ;
    les 7 premiers rejouent l'action précédente), `prevAction` = dernière action décidée.
La prédiction de balle (5 échantillons à +30, +60, +120, +180, +240 ticks, comme les +0,25 … +2 s du paquet RLBot)
vient d'une arène RocketSim privée qui ne contient que la balle (la prédiction RLBot ignore aussi les voitures).

Écarts connus avec le vrai bot :
  - « retourné » (masque de saut) estimé : toit vers le bas (up.z < -0,9) et voiture basse (z < 50), RLGym ne donnant
    pas le contact avec le sol ;
  - `HasFlipOrJump` = au sol ou `has_flip` de RLGym (même formule que RocketSim) ;
  - pas de kickoff codé en dur (le bot n'en a pas) ; les emplacements des coéquipiers / adversaires sont mélangés
    au hasard à chaque décision, comme dans le bot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from rlagent.opponents.necto import REPO_ROOT, NectoOpponent
from rlagent.opponents.nexto import LOOKUP_TABLE, NEXTO_PADS

GALACTUS_DIR = REPO_ROOT / "dist" / "opponents" / "GalactusV2" / "rlbot"
TICK_SKIP, ACTION_DELAY = 8, 7
MAX_PLAYERS = 3
OBS_SIZE = 258
POS_COEF, VEL_COEF, ANG_VEL_COEF = 1 / 5000, 1 / 2300, 1 / 3
BIG_PADS = (3, 4, 15, 18, 29, 30)
BIG_PAD_GATE = 1500.0
PRED_TICKS = (30, 60, 120, 180, 240)
_INV = np.array([-1.0, -1.0, 1.0])

# Masques de DefaultAction : 24 actions au sol puis 66 en l'air ; sol = indices < 24.
N_GROUND = 24
JUMP_MASK = LOOKUP_TABLE[:, 5] == 1
BOOST_MASK = LOOKUP_TABLE[:, 6] == 1
GROUND_MASK = np.arange(len(LOOKUP_TABLE)) < N_GROUND
AIR_MASK = (np.arange(len(LOOKUP_TABLE)) > N_GROUND) & ~JUMP_MASK          # « i > numGroundActions » : 24 exclu
AIR_MASK |= GROUND_MASK & (LOOKUP_TABLE[:, 0] == LOOKUP_TABLE[:, 6]) & ((LOOKUP_TABLE[:, 3] != 0) == (LOOKUP_TABLE[:, 7] == 1))


def is_galactus_spec(spec: str) -> bool:
    return spec == "galactus" or spec.startswith("galactus:")


def action_mask(car) -> np.ndarray:
    """`DefaultAction::GetActionMask`."""
    ph = car.physics
    on_ground = bool(car.on_ground)
    m = GROUND_MASK.copy() if on_ground else AIR_MASK.copy()
    if car.boost_amount <= 0:
        m &= ~BOOST_MASK
    turtled = (not on_ground) and float(ph.up[2]) < -0.9 and float(ph.position[2]) < 50.0
    if on_ground or car.has_flip or turtled:
        m |= JUMP_MASK
    return m


class BallPredictor:
    """Arène RocketSim avec la balle seule : `samples(state)` -> (5, 6) positions / vitesses brutes, repère du jeu."""

    def __init__(self):
        import RocketSim as rs
        self.rs = rs
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self._key = None
        self._cache = None

    def samples(self, state: GameState) -> np.ndarray:
        b = state.ball
        key = (tuple(np.round(b.position, 4)), tuple(np.round(b.linear_velocity, 4)),
               tuple(np.round(b.angular_velocity, 4)))
        if key == self._key:
            return self._cache
        rs = self.rs
        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(*map(float, b.position))
        bs.vel = rs.Vec(*map(float, b.linear_velocity))
        bs.ang_vel = rs.Vec(*map(float, b.angular_velocity))
        self.arena.ball.set_state(bs)
        pred = self.arena.get_ball_prediction(PRED_TICKS[-1] + 1, 1)      # indice k = après k ticks
        out = np.array([[p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z]
                        for p in (pred[t] for t in PRED_TICKS)], dtype=np.float64)
        self._key, self._cache = key, out
        return out


class GalactusObs:
    """`GalactusOBS::BuildObsInto` sur un `GameState` RLGym 2."""

    def __init__(self, pad_locations: np.ndarray, rng: np.random.Generator, predictor: BallPredictor | None = None):
        pads = np.asarray(pad_locations, dtype=np.float64)
        self.order = np.array([int(np.linalg.norm(pads[:, :2] - p[:2], axis=1).argmin()) for p in NEXTO_PADS])
        if len(set(self.order.tolist())) != len(NEXTO_PADS):
            raise ValueError("pads de l'arène sans correspondance un à un avec BOOST_LOCATIONS")
        self.rng = rng
        self.predictor = predictor or BallPredictor()

    @staticmethod
    def _player(car, inv: bool, ball_pos: np.ndarray, ball_vel: np.ndarray) -> np.ndarray:
        ph = car.physics
        s = _INV if inv else 1.0
        pos, vel, ang = ph.position * s, ph.linear_velocity * s, ph.angular_velocity * s
        fw, right, up = ph.forward * s, ph.right * s, ph.up * s
        rot = np.stack([fw, right, up])                                     # RotMat::Dot : (fw·v, right·v, up·v)
        return np.concatenate([
            pos * POS_COEF, fw, up, vel * VEL_COEF, ang * ANG_VEL_COEF, rot @ ang * ANG_VEL_COEF,
            rot @ (ball_pos - pos) * POS_COEF, rot @ (ball_vel - vel) * VEL_COEF,
            [car.boost_amount / 100.0, float(car.on_ground), float(car.on_ground or car.has_flip),
             float(car.is_demoed), float(car.has_jumped)]])

    def build(self, me: str, state: GameState, prev_action: np.ndarray) -> np.ndarray:
        car = state.cars[me]
        inv = car.team_num == cv.ORANGE_TEAM
        s = _INV if inv else 1.0
        ball = state.ball
        b_pos, b_vel, b_ang = ball.position * s, ball.linear_velocity * s, ball.angular_velocity * s
        timers = np.asarray(state.boost_pad_timers, dtype=np.float64)[self.order]
        if inv:
            timers = timers[::-1]                                           # boostPadsInv[i] = pad 33 - i
        active = timers <= 0
        pads = np.where(active, 1.0, 1.0 / (1.0 + timers))
        me_pos = car.physics.position * s
        near, best = np.zeros(3), BIG_PAD_GATE ** 2
        for i in BIG_PADS:
            if not active[i]:
                continue
            d2 = float(np.sum((NEXTO_PADS[i, :2] - me_pos[:2]) ** 2))
            if d2 <= best:
                best, near = d2, NEXTO_PADS[i] - me_pos
        parts = [b_pos * POS_COEF, b_vel * VEL_COEF, b_ang * ANG_VEL_COEF, np.asarray(prev_action, dtype=np.float64),
                 pads, near * POS_COEF, self._player(car, inv, b_pos, b_vel)]
        mates = [c for a, c in state.cars.items() if a != me and c.team_num == car.team_num]
        opps = [c for c in state.cars.values() if c.team_num != car.team_num]
        if len(mates) > MAX_PLAYERS - 1 or len(opps) > MAX_PLAYERS:
            raise ValueError("GalactusV2 : 3 joueurs par équipe au plus")
        for group, n in ((mates, MAX_PLAYERS - 1), (opps, MAX_PLAYERS)):
            slots = group + [None] * (n - len(group))
            self.rng.shuffle(slots)                                         # std::shuffle, vides compris
            parts += [self._player(c, inv, b_pos, b_vel) if c is not None else np.zeros(29) for c in slots]
        pred = self.predictor.samples(state)
        if inv:
            pred = pred * np.concatenate([_INV, _INV])
        parts.append((pred * np.array([POS_COEF] * 3 + [VEL_COEF] * 3)).reshape(-1))
        obs = np.concatenate(parts)
        assert obs.shape == (OBS_SIZE,), obs.shape
        return obs


def _load(name: str):
    import os
    import warnings

    import torch
    folder = Path(os.environ.get("RLAGENT_GALACTUS_DIR") or GALACTUS_DIR)
    path = folder / f"{name}.lt"
    if not path.exists():
        raise FileNotFoundError(f"Modèle GalactusV2 absent : {path} (cloner SoldierDoge/GalactusV2 dans dist/opponents)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.jit.load(str(path), map_location="cpu")


def build_model():
    """Même réseau que `RLBotMain.cpp` : tête partagée 258->1024->1024 puis politique 1024x4 -> 90, chaque couche
    cachée Linear + LayerNorm + LeakyReLU(0,01) ; poids copiés depuis les .lt."""
    import torch
    from torch import nn

    def mlp(n_in, sizes, n_out):
        layers, last = [], n_in
        for sz in sizes:
            layers += [nn.Linear(last, sz), nn.LayerNorm(sz), nn.LeakyReLU()]
            last = sz
        if n_out:
            layers.append(nn.Linear(last, n_out))
        return nn.Sequential(*layers)

    head, pol = mlp(OBS_SIZE, (1024, 1024), 0), mlp(1024, (1024,) * 4, len(LOOKUP_TABLE))
    for net, name in ((head, "SHARED_HEAD"), (pol, "POLICY")):
        sd = {k: v.detach().clone() for k, v in _load(name).named_parameters()}
        net.load_state_dict(sd, strict=True)
    model = nn.Sequential(head, pol).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(torch.get_num_threads())
    return model


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


def choose(logits: np.ndarray, mask: np.ndarray, beta: float, rng: np.random.Generator) -> int:
    """Logits masqués comme `InferPolicyProbsFromModels` ; beta 1 = argmax (le bot en match), sinon échantillonné
    (même échelle de température que Necto / Nexto)."""
    l = np.where(mask, logits.astype(np.float64), -1e10)
    if beta >= 1:
        return int(np.argmax(l))
    scale = 0.0 if beta <= 0 else float(np.log((beta + 1) / (1 - beta)) / np.log(3))
    z = np.where(mask, l * scale, -np.inf)
    p = np.exp(z - z.max())
    return int(rng.choice(len(p), p=p / p.sum()))


class GalactusOpponent(NectoOpponent):
    """Pilote une ou plusieurs voitures GalactusV2 ; même interface que `NectoOpponent` (`new_game`, `on_reset`,
    `act` -> 8 lignes de commandes par voiture : 7 de l'action précédente, 1 de la nouvelle)."""

    name = "galactus"
    _load_model = staticmethod(lambda: load_model())

    def __init__(self, beta: float = 1.0, seed: int | None = None):
        super().__init__(beta=beta, hardcoded_kickoffs=False, seed=seed)
        self.predictor = None

    def new_game(self, pad_locations: np.ndarray) -> None:
        if self.predictor is None:
            self.predictor = BallPredictor()
        self.obs = GalactusObs(pad_locations, self.rng, self.predictor)
        self.prev = {}
        self.kickoff_index = {}

    def act(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]:
        import torch
        if not agents:
            return {}
        beta = self.step_beta(state)
        obs = np.stack([self.obs.build(a, state, self.prev.get(a, np.zeros(8))) for a in agents])
        with torch.no_grad():
            logits = self.model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        out = {}
        for i, agent in enumerate(agents):
            new = LOOKUP_TABLE[choose(logits[i], action_mask(state.cars[agent]), beta, self.rng)]
            old = self.prev.get(agent, np.zeros(8)).astype(np.float32)
            rows = np.vstack([np.repeat(old[None], ACTION_DELAY, axis=0),
                              np.repeat(new[None], TICK_SKIP - ACTION_DELAY, axis=0)])
            self.prev[agent] = new.astype(np.float64)
            out[agent] = rows
        return out


KNOWN_GAPS = ("adaptateur refait depuis le C++ (pas de binaire de référence) : vérifié par revue du code et par le "
              "jeu en simulation ;\nretourné estimé (up.z < -0,9, z < 50), physique RocketSim, prédiction de balle "
              "RocketSim au lieu de celle de RLBot")


def fidelity_report() -> dict:
    return {"texte": KNOWN_GAPS}
