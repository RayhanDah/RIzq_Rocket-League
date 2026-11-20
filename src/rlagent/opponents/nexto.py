"""Nexto (version 2 de Necto, NectoFamily) comme adversaire dans nos audits RocketSim.

Même méthode que `necto.py` : seul le modèle TorchScript `dist/opponents/NectoFamily/nexto/nexto-model.pt` est
chargé ; l'observation de `nexto/nexto_obs.py` et la table d'actions de `nexto/agent.py` sont refaites en numpy à
partir de notre état RLGym 2. Le kickoff codé en dur, le choix du preneur et le rythme des actions sont ceux de Necto
(code identique dans `nexto/bot.py`), d'où la sous-classe. Vérification : `scripts/necto_check.py --bot nexto`.

Différences avec Necto, reprises du bot :
  - entités dans l'ordre soi, coéquipiers, adversaires, balle, 34 pads (ordre `BOOST_LOCATIONS`, hauteurs 70 / 73) ;
  - coordonnées relatives à soi puis tournées autour de Z selon le cap de la voiture (`convert_to_relative`) ;
    seules les positions sont décalées, les vitesses sont absolues mais tournées ;
  - « minuteur » des pads = pad disponible (0/1), « minuteur » de démo = voiture démolie (0/1) (FIXME du bot) ;
  - une seule tête de 90 logits sur la table `make_lookup_table` ;
  - kickoffs stochastiques : beta 0,5 pendant la phase Kickoff de RLBot (fin du compte à rebours -> première
    touche), ici tant que la balle n'a pas bougé depuis la remise au centre.
"""
from __future__ import annotations

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from rlagent.opponents.necto import (KNOWN_GAPS as NECTO_GAPS, REPO_ROOT, NectoOpponent, fidelity_report as _report,
                                     is_necto_spec)

NEXTO_DIR = REPO_ROOT / "dist" / "opponents" / "NectoFamily" / "nexto"
KICKOFF_BETA = 0.5

# nexto_obs.py : ordre et hauteurs des pads vus par le modèle (V1GameState les donne dans cet ordre).
NEXTO_PADS = np.array([
    (0.0, -4240.0, 70.0), (-1792.0, -4184.0, 70.0), (1792.0, -4184.0, 70.0), (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0), (-940.0, -3308.0, 70.0), (940.0, -3308.0, 70.0), (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0), (3584.0, -2484.0, 70.0), (-1788.0, -2300.0, 70.0), (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0), (0.0, -1024.0, 70.0), (2048.0, -1036.0, 70.0), (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0), (1024.0, 0.0, 70.0), (3584.0, 0.0, 73.0), (-2048.0, 1036.0, 70.0), (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0), (-1788.0, 2300.0, 70.0), (1788.0, 2300.0, 70.0), (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0), (0.0, 2816.0, 70.0), (-940.0, 3310.0, 70.0), (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0), (3072.0, 4096.0, 73.0), (-1792.0, 4184.0, 70.0), (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0)], dtype=np.float64)

IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS, LIN_VEL, FW, UP, ANG_VEL = slice(5, 8), slice(8, 11), slice(11, 14), slice(14, 17), slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
_INVERT = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4, dtype=np.float64)
_NORM = np.array([1.0] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4, dtype=np.float64)


def is_nexto_spec(spec: str) -> bool:
    return spec == "nexto" or spec.startswith("nexto:")


def make_lookup_table() -> np.ndarray:
    """`nexto/agent.py` : 24 actions au sol puis 66 en l'air (90)."""
    actions = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.array(actions, dtype=np.float32)


LOOKUP_TABLE = make_lookup_table()


class NextoObs:
    """`NextoObsBuilder` du clone (sans état : le bot ne garde aucun minuteur), sur un `GameState` RLGym 2."""

    def __init__(self, pad_locations: np.ndarray):
        """`pad_locations` : ordre de `state.boost_pad_timers` ; remis dans l'ordre `NEXTO_PADS` (pad le plus proche)."""
        pads = np.asarray(pad_locations, dtype=np.float64)
        self.order = np.array([int(np.linalg.norm(pads[:, :2] - p[:2], axis=1).argmin()) for p in NEXTO_PADS])
        if len(set(self.order.tolist())) != len(NEXTO_PADS):
            raise ValueError("pads de l'arène sans correspondance un à un avec BOOST_LOCATIONS")
        # bot.py avec field_info : x, y du jeu (un pad diffère de 2 uu de BOOST_LOCATIONS), z de BOOST_LOCATIONS
        self.locations = np.column_stack([pads[self.order, :2], NEXTO_PADS[:, 2]])

    def build(self, me: str, state: GameState, prev_action: np.ndarray):
        """-> (q (1,1,32), kv (1,N,24), mask (1,N)) ; entités : soi, coéquipiers, adversaires, balle, pads."""
        my_team = state.cars[me].team_num
        order = [me] + [a for a, c in state.cars.items() if a != me and c.team_num == my_team] \
            + [a for a, c in state.cars.items() if c.team_num != my_team]
        n_p = len(order)
        kv = np.zeros((1, n_p + 1 + len(NEXTO_PADS), 24))
        for n, agent in enumerate(order):
            car = state.cars[agent]
            ph = car.physics
            kv[0, n, IS_MATE] = 1 - car.team_num               # bleu = coéquipier par défaut, échangé pour l'orange
            kv[0, n, IS_OPP] = car.team_num
            kv[0, n, POS] = ph.position
            kv[0, n, LIN_VEL] = ph.linear_velocity
            kv[0, n, FW] = ph.forward
            kv[0, n, UP] = ph.up
            kv[0, n, ANG_VEL] = ph.angular_velocity
            kv[0, n, BOOST] = car.boost_amount / 100.0
            kv[0, n, DEMO] = float(car.is_demoed)
            kv[0, n, ON_GROUND] = car.on_ground
            kv[0, n, HAS_FLIP] = car.has_flip
        kv[0, 0, IS_SELF] = 1
        b = n_p
        kv[0, b, IS_BALL] = 1
        kv[0, b, POS] = state.ball.position
        kv[0, b, LIN_VEL] = state.ball.linear_velocity
        kv[0, b, ANG_VEL] = state.ball.angular_velocity
        kv[0, b + 1:, IS_BOOST] = 1
        kv[0, b + 1:, POS] = self.locations
        kv[0, b + 1:, BOOST] = 0.12 + 0.88 * (NEXTO_PADS[:, 2] > 72)
        kv[0, b + 1:, DEMO] = np.asarray(state.boost_pad_timers)[self.order] == 0

        if my_team == cv.ORANGE_TEAM:
            kv *= _INVERT
            kv[0, :, (IS_MATE, IS_OPP)] = kv[0, :, (IS_OPP, IS_MATE)]
        kv /= _NORM
        q = np.zeros((1, 1, 32))
        q[0, 0, :24] = kv[0, 0]
        q[0, 0, 24:] = prev_action
        kv[..., POS] -= q[..., POS]                          # convert_to_relative
        theta = np.arctan2(q[0, 0, FW.start], q[0, 0, FW.start + 1])
        ct, st = np.cos(theta), np.sin(theta)
        xs, ys = kv[..., POS.start:ANG_VEL.stop:3].copy(), kv[..., POS.start + 1:ANG_VEL.stop:3].copy()
        kv[..., POS.start:ANG_VEL.stop:3] = ct * xs - st * ys
        kv[..., POS.start + 1:ANG_VEL.stop:3] = st * xs + ct * ys
        return q, kv, np.zeros((1, kv.shape[1]))


def parse_logits(logits, beta: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """`Agent.act` du clone : une tête de 90 logits -> ligne de la table (8 commandes)."""
    import torch
    l = torch.as_tensor(logits).reshape(-1).double()
    if beta == 1:
        i = int(torch.argmax(l))
    elif beta == -1:
        i = int(torch.argmin(l))
    else:
        rng = rng or np.random.default_rng()
        scale = 0.0 if beta == 0 else float(np.log((beta + 1) / (1 - beta)) / np.log(3))
        p = torch.softmax(l * scale, -1).numpy()
        i = int(rng.choice(len(p), p=p / p.sum()))
    return LOOKUP_TABLE[i].copy()


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        import torch
        path = NEXTO_DIR / "nexto-model.pt"
        if not path.exists():
            raise FileNotFoundError(f"Modèle Nexto absent : {path} (cloner NectoFamily dans dist/opponents)")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _MODEL = torch.jit.load(str(path), map_location="cpu")
        _MODEL.eval()
    return _MODEL


class NextoOpponent(NectoOpponent):
    """Pilote une ou plusieurs voitures Nexto ; même interface que `NectoOpponent` (kickoff codé en dur compris)."""

    name = "nexto"
    _load_model = staticmethod(lambda: load_model())

    def __init__(self, beta: float = 1.0, hardcoded_kickoffs: bool = True, stochastic_kickoffs: bool = True,
                 seed: int | None = None):
        super().__init__(beta=beta, hardcoded_kickoffs=hardcoded_kickoffs, seed=seed)
        self.stochastic_kickoffs = stochastic_kickoffs
        self.in_kickoff = False

    def new_game(self, pad_locations: np.ndarray) -> None:
        self.obs = NextoObs(pad_locations)
        self.prev = {}
        self.kickoff_index = {}
        self.in_kickoff = False

    def on_reset(self, state: GameState) -> None:
        super().on_reset(state)
        self.in_kickoff = True

    def step_beta(self, state: GameState) -> float:
        """Phase Kickoff de RLBot : jusqu'à ce que la balle bouge (première touche)."""
        if self.in_kickoff and (float(state.ball.position[1]) != 0.0 or float(state.ball.position[0]) != 0.0
                                or np.any(np.asarray(state.ball.linear_velocity) != 0.0)):
            self.in_kickoff = False
        return KICKOFF_BETA if (self.in_kickoff and self.stochastic_kickoffs) else self.beta

    def policy(self, q, kv, mask, beta: float | None = None) -> np.ndarray:
        import torch
        with torch.no_grad():
            out, _ = self.model(tuple(torch.as_tensor(x, dtype=torch.float32) for x in (q, kv, mask)))
        return parse_logits(out, self.beta if beta is None else beta, self.rng)


KNOWN_GAPS = (NECTO_GAPS.replace("Necto agit", "Nexto agit") + " ;\nphase Kickoff (beta 0,5) supposée finir quand "
              "la balle bouge, sans compte à rebours ici")


def fidelity_report() -> dict:
    return _report("nexto", KNOWN_GAPS)


# Point d'entrée commun aux scripts d'audit : spécifications `necto[:beta]` et `nexto[:beta]`.
def is_bot_spec(spec: str) -> bool:
    return is_necto_spec(spec) or is_nexto_spec(spec)


def bot_from_spec(spec: str, seed: int | None = None) -> NectoOpponent:
    return (NextoOpponent if is_nexto_spec(spec) else NectoOpponent).from_spec(spec, seed=seed)


def bot_fidelity(spec: str) -> dict:
    from rlagent.opponents.necto import fidelity_report as necto_report
    return fidelity_report() if is_nexto_spec(spec) else necto_report()

