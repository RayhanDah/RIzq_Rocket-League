"""Rewards (ARCHITECTURE §7).

Chaque terme renvoie une valeur BRUTE par agent. `TeamReward` les pondère (avec décroissance programmée),
puis applique l'esprit d'équipe τ et la soustraction adverse :

    r_final(i) = (1 - τ) * r_i + τ * moyenne(r de l'équipe de i) - zero_sum * moyenne(r de l'équipe adverse)

Option `rewards.zero_sum_terms` : liste des termes concernés par la soustraction adverse (null = tous, comportement
d'origine). Exemple [goal, vel_ball_to_goal] : les rewards « d'apprentissage des bases » (toucher, aller vers la
balle, s'orienter) ne sont plus annulées quand l'adversaire fait la même chose.

La progression (nombre global de pas) est lue dans shared_info["global_step"], mise à jour par le trainer.
Les sommes par terme de l'épisode sont accumulées dans shared_info["reward_log"] pour le suivi (moyenne sur les agents :
≈ 0 pour un terme antisymétrique entre équipes), et leur valeur absolue dans shared_info["reward_log_abs"] (poids réel
de chaque terme dans le signal d'apprentissage).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from .common import OPP_GOAL_BACK, OWN_GOAL_BACK, is_flip_reset, norm3
from .predict import ETA_CAP, goal_threat, race_info, team_etas


def _dot_dir(v, a, b) -> float:
    """v · unit(b - a) pour des vecteurs 3D (listes ou tableaux), sans numpy (0 si a ≈ b)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    return 0.0 if n <= 1e-6 else (v[0] * dx + v[1] * dy + v[2] * dz) / n


_OPP_GOAL = OPP_GOAL_BACK.tolist()
_OPP_GOAL_ORANGE = [_OPP_GOAL[0], -_OPP_GOAL[1], _OPP_GOAL[2]]     # but adverse de l'orange, repère du monde
_OWN_GOAL = OWN_GOAL_BACK.tolist()
_OWN_GOAL_ORANGE = [_OWN_GOAL[0], -_OWN_GOAL[1], _OWN_GOAL[2]]


class RewardTerm:
    name = "term"

    def reset(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class GoalTerm(RewardTerm):
    """+1 pour chaque membre de l'équipe qui marque (la soustraction adverse donne -1 à l'équipe qui encaisse)."""
    name = "goal"

    def get(self, agents, state, shared_info):
        scorer = state.scoring_team if state.goal_scored else None
        return {a: (1.0 if scorer is not None and state.cars[a].team_num == scorer else 0.0) for a in agents}


class TouchTerm(RewardTerm):
    """1 par touche, majorée quand la balle est haute (encourage les aériennes)."""
    name = "touch"

    def __init__(self, aerial_bonus: float = 1.0):
        self.aerial_bonus = aerial_bonus

    def get(self, agents, state, shared_info):
        height = float(np.clip(state.ball.position[2] / cv.CEILING_Z, 0.0, 1.0))
        mult = 1.0 + self.aerial_bonus * height
        return {a: (mult if state.cars[a].ball_touches > 0 else 0.0) for a in agents}


class VelPlayerToBallTerm(RewardTerm):
    """Vitesse vers la balle / 2300. `cap` (uu/s) : vitesse au-delà de laquelle on ne paie plus (s7d : 1410, vitesse
    max sans boost) : l'aide paie le fait d'aller vers la balle, pas celui de brûler son boost pour y aller.
    `signed` (s10, comme GigaLearn) : s'éloigner de la balle coûte autant que s'en approcher rapporte."""
    name = "vel_player_to_ball"

    def __init__(self, cap: Optional[float] = None, signed: bool = False):
        self.cap = None if cap is None else float(cap)
        self.signed = bool(signed)

    def get(self, agents, state, shared_info):
        ball = state.ball.position.tolist()
        out = {}
        for a in agents:
            p = state.cars[a].physics
            v = _dot_dir(p.linear_velocity.tolist(), p.position.tolist(), ball)
            if not self.signed:
                v = max(0.0, v)
            out[a] = (v if self.cap is None else min(v, self.cap)) / cv.CAR_MAX_SPEED
        return out


class VelBallToGoalTerm(RewardTerm):
    """Vitesse de la balle vers le but adverse (signée : négatif si elle va vers notre but)."""
    name = "vel_ball_to_goal"

    def get(self, agents, state, shared_info):
        pos, vel = state.ball.position.tolist(), state.ball.linear_velocity.tolist()
        per_team = {cv.BLUE_TEAM: _dot_dir(vel, pos, _OPP_GOAL) / cv.BALL_MAX_SPEED,
                    cv.ORANGE_TEAM: _dot_dir(vel, pos, _OPP_GOAL_ORANGE) / cv.BALL_MAX_SPEED}
        return {a: per_team[state.cars[a].team_num] for a in agents}


class FaceBallTerm(RewardTerm):
    name = "face_ball"

    def get(self, agents, state, shared_info):
        ball = state.ball.position.tolist()
        out = {}
        for a in agents:
            p = state.cars[a].physics
            out[a] = _dot_dir(p.forward.tolist(), p.position.tolist(), ball)
        return out


class BoostGainTerm(RewardTerm):
    """Gain de boost en racine carrée (le boost bas vaut plus cher). Dépense pénalisée par loss_coef.
    `below` : ramassage payé seulement si la voiture avait moins de `below` de boost (recharger quand on est à sec,
    pas faire le plein à 90 ; défaut 100 = toujours payé, comportement d'origine)."""
    name = "boost_gain"

    def __init__(self, loss_coef: float = 0.0, below: float = 100.0):
        self.loss_coef = loss_coef
        self.below = float(below)
        self.prev: Dict[str, float] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: state.cars[a].boost_amount for a in agents}

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            b = state.cars[a].boost_amount
            prev = self.prev.get(a, b)
            delta = math.sqrt(max(b, 0.0) / 100.0) - math.sqrt(max(prev, 0.0) / 100.0)
            out[a] = (delta if prev < self.below else 0.0) if delta >= 0 else self.loss_coef * delta
            self.prev[a] = b
        return out


class SmallPadTerm(RewardTerm):
    """Petit pad ramassé (17/11, palier 14, demande du PO : il ne va que sur les gros pads). +Δboost / 12 pour un gain de
    boost d'au plus `max_gain` (un petit pad donne 12 ; un gros pad, une réapparition ou un kickoff donnent plus et ne
    comptent pas). Compte en plus de `boost_gain` (Δ√boost), qui paie un gros pad à sec ~4 fois un petit : ici, le petit
    pad pris sur le chemin vaut enfin un peu. Borné par les pads (4 s de réapparition) et par `boost_spend`, qui fait
    payer le boost brûlé au sol pour les reprendre (tests/test_exploits.py)."""
    name = "small_pad"

    def __init__(self, max_gain: float = 13.0):
        self.max_gain = float(max_gain)
        self.prev: Dict[str, float] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: state.cars[a].boost_amount for a in agents}

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            car = state.cars[a]
            b = float(car.boost_amount)
            gain = b - self.prev.get(a, b)
            self.prev[a] = b
            out[a] = min(gain, 12.0) / 12.0 if 0.0 < gain <= self.max_gain and not car.is_demoed else 0.0
        return out


class BoostSpendTerm(RewardTerm):
    """Dépense de boost près du sol (palier 6, recette de Necto) : Δ√boost (négatif), atténué avec la hauteur, nul à partir
    de la barre : le vol reste gratuit. `boost_gain` ne paie que les gains et paie plus à sec : sans ce terme, brûler son
    boost pour le reprendre rapporte, et l'agent roule sous 10 de boost 65 % du temps (audit du 06/11)."""
    name = "boost_spend"

    def __init__(self):
        self.prev: Dict[str, float] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: state.cars[a].boost_amount for a in agents}

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            car = state.cars[a]
            b = car.boost_amount
            delta = math.sqrt(max(b, 0.0) / 100.0) - math.sqrt(max(self.prev.get(a, b), 0.0) / 100.0)
            self.prev[a] = b
            low = max(0.0, 1.0 - float(car.physics.position[2]) / cv.GOAL_HEIGHT)
            out[a] = delta * low if delta < 0.0 and not car.is_demoed else 0.0
        return out


class DoubleCommitTerm(RewardTerm):
    """-1 pour chaque agent "engagé" sur la balle quand au moins un coéquipier l'est aussi (2v2/3v3)."""
    name = "double_commit"

    def __init__(self, radius: float = 900.0, min_speed: float = 400.0):
        self.radius = radius
        self.min_speed = min_speed

    def committed(self, state: GameState, agent: str) -> bool:
        p = state.cars[agent].physics
        pos, ball = p.position.tolist(), state.ball.position.tolist()
        dx, dy, dz = ball[0] - pos[0], ball[1] - pos[1], ball[2] - pos[2]
        if dx * dx + dy * dy + dz * dz > self.radius * self.radius:
            return False
        return _dot_dir(p.linear_velocity.tolist(), pos, ball) > self.min_speed

    def get(self, agents, state, shared_info):
        committed = {a: self.committed(state, a) for a in state.cars}
        per_team: Dict[int, int] = {}
        for a, c in committed.items():
            if c:
                t = state.cars[a].team_num
                per_team[t] = per_team.get(t, 0) + 1
        return {a: (-1.0 if committed[a] and per_team.get(state.cars[a].team_num, 0) >= 2 else 0.0)
                for a in agents}


class TouchStrengthTerm(RewardTerm):
    """Puissance de frappe : à chaque touche, gain de vitesse de la balle (normalisé par sa vitesse max).
    Récompense les vraies frappes plutôt que les simples contacts. Poids absent du YAML = terme désactivé."""
    name = "touch_strength"

    def __init__(self):
        self.prev_ball_vel = np.zeros(3, dtype=np.float32)

    def reset(self, agents, state, shared_info):
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()

    def get(self, agents, state, shared_info):
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        gain = norm3(vel - self.prev_ball_vel) / cv.BALL_MAX_SPEED
        self.prev_ball_vel = vel.copy()
        return {a: (gain if state.cars[a].ball_touches > 0 else 0.0) for a in agents}


class ShotTerm(RewardTerm):
    """Frappe orientée vers le but adverse : à chaque touche, variation de la vitesse de la balle DANS L'AXE du but
    adverse (normalisée par sa vitesse max). Négative si la touche renvoie la balle vers notre but.
    Contrairement à `touch_strength`, une frappe forte vers les côtés ou vers notre camp ne rapporte rien.
    `off_target_factor` (palier 5) : dans le camp adverse, une frappe qui ne cadre pas ne touche que cette part du gain.
    Un dégagement depuis notre camp reste payé en entier : ce n'est pas un tir."""
    name = "shot"

    def __init__(self, off_target_factor: float = 1.0):
        self.off_target_factor = float(off_target_factor)
        self.prev_ball_vel = np.zeros(3, dtype=np.float32)

    def reset(self, agents, state, shared_info):
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()

    def get(self, agents, state, shared_info):
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        dv = (vel - self.prev_ball_vel).tolist()
        self.prev_ball_vel = vel.copy()
        ball, ball_vel = state.ball.position.tolist(), vel.tolist()
        out = {}
        for a in agents:
            car = state.cars[a]
            if car.ball_touches <= 0:
                out[a] = 0.0
                continue
            # axe balle → but adverse, dans le repère du monde (inversé pour l'orange)
            goal = _OPP_GOAL if car.team_num == cv.BLUE_TEAM else _OPP_GOAL_ORANGE
            r = _dot_dir(dv, ball, goal) / cv.BALL_MAX_SPEED
            sign = _team_sign(car.team_num)
            if (r > 0.0 and self.off_target_factor != 1.0 and ball[1] * sign > 0.0
                    and goal_threat(ball, ball_vel, sign) == 0.0):
                r *= self.off_target_factor
            out[a] = r
        return out


class AlignBallGoalTerm(RewardTerm):
    """Placement : être derrière la balle par rapport au but adverse (cos de l'angle voiture→balle / balle→but).
    +1 = voiture, balle et but alignés dans le bon ordre ; -1 = la voiture est entre la balle et le but adverse."""
    name = "align_ball_goal"

    def get(self, agents, state, shared_info):
        ball = state.ball.position.tolist()
        out = {}
        for a in agents:
            car = state.cars[a]
            pos = car.physics.position.tolist()
            goal = _OPP_GOAL if car.team_num == cv.BLUE_TEAM else _OPP_GOAL_ORANGE   # repère du monde
            dx, dy, dz = ball[0] - pos[0], ball[1] - pos[1], ball[2] - pos[2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            out[a] = 0.0 if n <= 1e-6 else _dot_dir([dx / n, dy / n, dz / n], ball, goal)
        return out


class InAirTerm(RewardTerm):
    """1 quand la voiture ne touche pas le sol : petit encouragement à sauter (sinon le bot oublie les sauts et passe
    sous les balles hautes). Poids très faible."""
    name = "in_air"

    def get(self, agents, state, shared_info):
        return {a: (0.0 if state.cars[a].on_ground else 1.0) for a in agents}


class BoostKeepTerm(RewardTerm):
    """Réserve de boost en racine carrée (le boost bas vaut plus cher) : apprendre à garder du boost au lieu de
    rouler à vide. Poids très faible, sinon il le stockerait sans s'en servir."""
    name = "boost_keep"

    def get(self, agents, state, shared_info):
        return {a: math.sqrt(max(state.cars[a].boost_amount, 0.0) / 100.0) for a in agents}


class SaveTerm(RewardTerm):
    """Arrêt : à chaque touche, réduction de la vitesse de la balle VERS NOTRE BUT quand elle y allait (> 500 uu/s,
    dans notre moitié). Normalisé par la vitesse max de la balle. Complète `shot` (qui paie le renvoi vers l'avant).

    `opp_only` (17/11, palier 12) : seulement si la dernière touche avant celle-ci venait de l'adversaire (ou d'aucune
    voiture : début d'exercice). Sans lui, pousser la balle vers son propre but puis l'arrêter se paie en boucle
    (tests/test_exploits.py)."""
    name = "save"

    def __init__(self, min_speed: float = 500.0, opp_only: bool = False):
        self.min_speed = min_speed
        self.opp_only = bool(opp_only)
        self.last_team: Optional[int] = None
        self.prev_ball_vel = np.zeros(3, dtype=np.float32)
        self.prev_ball_pos = np.zeros(3, dtype=np.float32)

    def reset(self, agents, state, shared_info):
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()
        self.prev_ball_pos = np.asarray(state.ball.position, dtype=np.float32).copy()
        self.last_team = None

    def get(self, agents, state, shared_info):
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        out = {}
        touched = set()
        for a in agents:
            car = state.cars[a]
            out[a] = 0.0
            if car.ball_touches <= 0:
                continue
            touched.add(int(car.team_num))
            if self.opp_only and self.last_team is not None and self.last_team == car.team_num:
                continue
            own_goal = _OWN_GOAL if car.team_num == cv.BLUE_TEAM else _OWN_GOAL_ORANGE
            prev_pos = self.prev_ball_pos.tolist()
            in_own_half = prev_pos[1] * own_goal[1] > 0
            before = _dot_dir(self.prev_ball_vel.tolist(), prev_pos, own_goal)
            if in_own_half and before > self.min_speed:
                after = _dot_dir(vel.tolist(), prev_pos, own_goal)
                out[a] = max(0.0, before - max(after, 0.0)) / cv.BALL_MAX_SPEED
        if len(touched) == 1:
            self.last_team = touched.pop()
        elif touched:
            self.last_team = None                            # contact simultané des deux camps
        self.prev_ball_vel = vel.copy()
        self.prev_ball_pos = np.asarray(state.ball.position, dtype=np.float32).copy()
        return out


class BoostWasteTerm(RewardTerm):
    """-1 quand la voiture booste au sol alors qu'elle est déjà supersonique (le boost n'y sert presque plus à rien)."""
    name = "boost_waste"

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            c = state.cars[a]
            out[a] = -1.0 if (c.is_boosting and c.is_supersonic and c.on_ground) else 0.0
        return out


class GoalsideTerm(RewardTerm):
    """Placement défensif : quand la balle est dans notre moitié, +1 si la voiture est entre la balle et son but
    (plus proche de son but que la balle, en profondeur), -1 sinon. 0 quand la balle est chez l'adversaire."""
    name = "goalside"

    def get(self, agents, state, shared_info):
        ball_y = float(state.ball.position[1])
        out = {}
        for a in agents:
            car = state.cars[a]
            sign = 1.0 if car.team_num == cv.BLUE_TEAM else -1.0     # perspective d'équipe : on attaque vers +Y
            by, cy = sign * ball_y, sign * float(car.physics.position[1])
            out[a] = 0.0 if by > 0 else (1.0 if cy < by else -1.0)
        return out


# --------------------------------------------------------------------------- rewards « s7 »
# Principe : les rewards denses de PLACEMENT et de VALEUR (balle → but, menace de tir, boost, course à la balle) sont
# des potentiels : r = γ·Φ(s') − Φ(s) (Ng, Harada & Russell 1999). Ils accélèrent l'apprentissage sans pouvoir être
# « farmés » : leur somme sur un épisode se télescope, donc la stratégie optimale reste celle du reward de match.
# Les seuls rewards non potentiels sont des ÉVÉNEMENTS de compétence (aérienne, kickoff, but encaissé) ou des aides
# de démarrage qui décroissent. Valeur terminale : Φ(s_T) est gardée (ni remise à 0, ni bootstrap) : marquer vaut
# donc un peu plus que `goal` (balle au fond = potentiel maximal), encaisser un peu plus cher.


class PotentialTerm(RewardTerm):
    """Base des rewards potentiels : r = γ·Φ(s') − Φ(s). Sous-classes : `potentials` → {agent: Φ}."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = float(gamma)
        self.prev: Dict[str, float] = {}

    def potentials(self, agents, state, shared_info) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self, agents, state, shared_info):
        self.prev = self.potentials(agents, state, shared_info)

    def get(self, agents, state, shared_info):
        phi = self.potentials(agents, state, shared_info)
        out = {a: self.gamma * phi[a] - self.prev.get(a, phi[a]) for a in agents}
        self.prev = phi
        return out


def _team_sign(team: int) -> float:
    return 1.0 if team == cv.BLUE_TEAM else -1.0


class BallGoalPotential(PotentialTerm):
    """Φ = (d_notre_but − d_but_adverse) / (somme) ∈ [-1, 1] : la balle proche du but adverse vaut +1, du nôtre -1.
    Antisymétrique entre équipes (pas de soustraction adverse nécessaire). Remplace `vel_ball_to_goal`."""
    name = "ball_goal"

    def potentials(self, agents, state, shared_info):
        bx, by, bz = (float(c) for c in state.ball.position)
        gy, gz = float(cv.BACK_NET_Y), float(cv.GOAL_HEIGHT) / 2
        d_pos = math.sqrt(bx * bx + (by - gy) ** 2 + (bz - gz) ** 2)     # but en +Y (orange)
        d_neg = math.sqrt(bx * bx + (by + gy) ** 2 + (bz - gz) ** 2)     # but en -Y (bleu)
        blue = (d_neg - d_pos) / max(d_neg + d_pos, 1.0)
        return {a: blue * _team_sign(state.cars[a].team_num) for a in agents}


class ThreatPotential(PotentialTerm):
    """Φ = urgence d'un tir cadré vers le but adverse − urgence d'un tir cadré vers NOTRE but (predict.goal_threat :
    1 − t/3 s si la trajectoire entre dans le cadre, 0 sinon). Effets :
      - tir cadré : récompense immédiate (avant le but), qui grandit à l'approche du but ;
      - arrêt / dégagement d'un tir cadré : récompense immédiate pour le défenseur, reprise au tireur ;
      - balle qui file vers notre but : pénalité qui grandit chaque step tant que rien n'est fait (« urgence »).
    C'est le cœur du « on ne marque pas contre nous » : la menace est punie dès qu'elle existe, pas au but."""
    name = "threat"

    def potentials(self, agents, state, shared_info):
        g = race_info(state, shared_info).goal_in
        out = {}
        for a in agents:
            team = state.cars[a].team_num
            out[a] = g.get(1 - team, 0.0) - g.get(team, 0.0)
        return out


class ShotThreatPotential(PotentialTerm):
    """Φ = valeur d'une balle cadrée vers le but adverse − valeur d'une balle cadrée vers NOTRE but (palier 5).
    Valeur = 0 hors cadre, sinon `floor` + (1 − floor) × urgence (predict.goal_threat, balle seule : pas d'ETA, peu
    coûteux). Le saut à `floor` donne un signal net au moment où la balle devient cadrée :
      - tir cadré : payé tout de suite, à qui l'a frappé ;
      - balle cadrée déviée par son propre tireur : la même somme, retirée tout de suite ;
      - arrêt : payé au défenseur, repris au tireur.
    Potentiel : la somme sur un épisode se télescope, il ne se farme pas. Antisymétrique : pas de soustraction adverse."""
    name = "shot_threat"

    def __init__(self, gamma: float = 0.99, floor: float = 0.5):
        super().__init__(gamma)
        self.floor = float(floor)

    def _value(self, pos, vel, goal_sign: float) -> float:
        g = goal_threat(pos, vel, goal_sign)
        return 0.0 if g <= 0.0 else self.floor + (1.0 - self.floor) * g

    def potentials(self, agents, state, shared_info):
        pos, vel = state.ball.position.tolist(), state.ball.linear_velocity.tolist()
        blue = self._value(pos, vel, 1.0) - self._value(pos, vel, -1.0)
        return {a: blue * _team_sign(state.cars[a].team_num) for a in agents}


class BoostPotential(PotentialTerm):
    """Φ = √(boost/100) par voiture : ramasser rapporte tout de suite (surtout à sec), dépenser coûte tout de suite ;
    le boost n'est rentable que s'il sert (touche, tir, arrêt, course gagnée). Remplace `boost_keep` (qui payait le
    stockage à chaque step). Avec γ < 1, garder du boost coûte (1 − γ)·Φ à chaque step : le bot vide son réservoir
    (s7c : à 0 de boost 81 % du temps). `rewards.boost_potential_gamma: 1.0` (s7d) supprime cette fuite : r = Φ(s') − Φ(s)."""
    name = "boost_potential"

    def potentials(self, agents, state, shared_info):
        return {a: math.sqrt(min(max(state.cars[a].boost_amount, 0.0), 100.0) / 100.0) for a in agents}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class PositioningPotential(PotentialTerm):
    """Placement selon la course à la balle (« qui va toucher en premier ? », predict.car_eta) :

        Φ = (1 − p)·A + p·D,   p = P(l'adversaire touche avant moi) = σ((mon ETA − ETA adverse) / 0,3 s)

    A = 1 − ETA/6 s (aller vite à la balle : vitesse, boost, flips, virages courts) ; D = être du bon côté de la balle
    (cos de l'angle balle→notre but / balle→voiture, ∈ [-1, 1]). Course gagnée → on attaque la balle ; course perdue
    → on se replace entre la balle et notre but au lieu de foncer dans le vide (et de se faire dribbler).
    En 2v2/3v3, seul le 1er homme (plus petite ETA de l'équipe) attaque ; les autres ont p ≥ `support_p`.
    Remplace `vel_player_to_ball`, `align_ball_goal` et `goalside`."""
    name = "positioning"

    def __init__(self, gamma: float = 0.99, temperature: float = 0.3, support_p: float = 0.8):
        super().__init__(gamma)
        self.temperature = float(temperature)
        self.support_p = float(support_p)

    def potentials(self, agents, state, shared_info):
        info = race_info(state, shared_info)
        bx, by = float(state.ball.position[0]), float(state.ball.position[1])
        out = {}
        for a in agents:
            car = state.cars[a]
            mine, mate, opp = team_etas(info, state, a)
            p = _sigmoid((mine - opp) / self.temperature)
            if mate < mine:                                   # un coéquipier arrive avant : soutien
                p = max(p, self.support_p)
            approach = 1.0 - mine / ETA_CAP
            gx, gy = 0.0, -_team_sign(car.team_num) * float(cv.BACK_WALL_Y)   # notre but
            px, py = float(car.physics.position[0]), float(car.physics.position[1])
            ux, uy, cx, cy = gx - bx, gy - by, px - bx, py - by
            nu, nc = math.sqrt(ux * ux + uy * uy), math.sqrt(cx * cx + cy * cy)
            defend = (ux * cx + uy * cy) / (nu * nc) if nu > 1.0 and nc > 1.0 else 1.0
            out[a] = (1.0 - p) * approach + p * defend
        return out


class AerialTouchTerm(RewardTerm):
    """Touche AÉRIENNE (voiture en l'air, balle au-dessus de `min_z`) : 0,5 au ras du sol → 2 vers le plafond, ×1,5 puis
    ×2 pour les touches aériennes suivantes sans retomber (double touche, air dribble). Les touches au mur ne comptent
    pas (roues au contact). Au plus une touche payée par `cooldown_s` et par voiture : un contact prolongé (balle posée
    sur le capot) ne se farme pas step après step. Événement de compétence : ne décroît pas (identité de l'agent).
    `min_air_boost_s` (palier 5) : boost exigé en l'air depuis le décollage. Sans lui, 86 % des touches payées venaient
    d'un saut depuis un mur, sans boost (audit du 04/11).

    `useful_speed` (08/11, après la session 4) : la touche est payée selon ce qu'elle apporte, g = gain de vitesse de la
    balle vers le but adverse / `useful_speed` (borné à [0, 1]). 1re touche du vol : `first_floor` + le reste × g ;
    touches suivantes sans retomber : g × `follow`^k (k = rang dans le vol). Le vol ne se termine qu'au vrai retour au sol
    (roues au sol ou au mur, pas sur la balle). Sans cette option, le ×1,5 puis ×2 des touches enchaînées payait le
    jonglage sur place ~30 / s contre 10 pour un but : l'agent l'a trouvé à 8,7 G et a cessé de marquer (Elo 916 → 270).
    Avec elle, jongler sur place un vol entier rapporte moins d'une 1re touche (tests/test_exploits.py). Une parade en
    l'air est payée comme une frappe : arrêter une balle qui file vers notre but, c'est lui ajouter de la vitesse vers le
    but adverse (test_aerial_save_is_already_paid_like_a_shot)."""
    name = "aerial_touch"

    def __init__(self, min_z: float = 220.0, full_z: float = 1720.0, cooldown_s: float = 0.5,
                 min_air_boost_s: float = 0.0, useful_speed: Optional[float] = None, first_floor: float = 0.25,
                 follow: float = 0.5):
        self.min_z, self.full_z = float(min_z), float(full_z)
        self.cooldown = int(round(cooldown_s * cv.TICKS_PER_SECOND))
        self.min_air_boost = float(min_air_boost_s) * cv.TICKS_PER_SECOND
        self.useful_speed = None if useful_speed is None else float(useful_speed)
        self.first_floor, self.follow = float(first_floor), float(follow)
        self.chain: Dict[str, int] = {}
        self.last_tick: Dict[str, int] = {}
        self.air_boost: Dict[str, int] = {}               # ticks de boost depuis le décollage
        self.prev_tick = 0
        self.prev_ball_vel = np.zeros(3, dtype=np.float32)

    def reset(self, agents, state, shared_info):
        self.chain = {a: 0 for a in agents}
        self.last_tick = {}
        self.air_boost = {a: 0 for a in agents}
        self.prev_tick = int(getattr(state, "tick_count", 0) or 0)
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()

    def get(self, agents, state, shared_info):
        z = float(state.ball.position[2])
        tick = int(getattr(state, "tick_count", 0) or 0)
        dt_ticks = max(0, tick - self.prev_tick)
        self.prev_tick = tick
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        dv = (vel - self.prev_ball_vel).tolist()
        self.prev_ball_vel = vel.copy()
        useful = self.useful_speed is not None
        ball = state.ball.position.tolist()
        height = min(1.0, max(0.0, (z - self.min_z) / (self.full_z - self.min_z)))
        out = {}
        for a in agents:
            car = state.cars[a]
            out[a] = 0.0
            if car.on_ground:
                # roues contre la balle (flip reset) : RocketSim dit « au sol », mais le vol continue (mode useful)
                if not useful or car.ball_touches <= 0:
                    self.chain[a] = 0
                    self.air_boost[a] = 0
                continue
            if car.is_boosting:
                self.air_boost[a] = self.air_boost.get(a, 0) + dt_ticks
            if car.ball_touches <= 0 or z <= self.min_z or self.air_boost.get(a, 0) < self.min_air_boost:
                continue
            last = self.last_tick.get(a)
            if last is not None and tick - last < self.cooldown:
                continue
            k = self.chain.get(a, 0)
            base = 0.5 + 1.5 * height
            if not useful:
                out[a] = base * min(1.0 + 0.5 * k, 2.0)
            else:
                goal = _OPP_GOAL if car.team_num == cv.BLUE_TEAM else _OPP_GOAL_ORANGE
                g = min(1.0, max(0.0, _dot_dir(dv, ball, goal) / self.useful_speed))
                share = self.first_floor + (1.0 - self.first_floor) * g if k == 0 else g
                out[a] = base * share * self.follow ** k
            self.chain[a] = k + 1
            self.last_tick[a] = tick
        return out


class FlipResetTerm(RewardTerm):
    """Flip reset (palier 6, `flip_reset_w` de Necto) : +1 quand les roues touchent la balle en l'air (common.is_flip_reset).
    Les bots ne le découvrent pas seuls (guide RLGym-PPO). `cooldown_s` par voiture : rester posé sur une balle en l'air
    ne se farme pas contact après contact, enchaîner de vrais resets reste payé.

    Options du 08/11 (revue des boucles après la session 4) : `require_spent` ne paie que si le saut avait été utilisé
    avant le contact (sinon rebondir sur la balle sans sauter rapportait 5 par seconde) ; `max_per_flight` resets payés
    par vol, le k-ième valant `follow`^k (un triple reset : 1 + 0,5 + 0,25). Le vol se termine au vrai retour au sol."""
    name = "flip_reset"

    def __init__(self, cooldown_s: float = 1.0, require_spent: bool = False, max_per_flight: Optional[int] = None,
                 follow: float = 0.5):
        self.cooldown = int(round(cooldown_s * cv.TICKS_PER_SECOND))
        self.require_spent = bool(require_spent)
        self.max_per_flight = None if max_per_flight is None else int(max_per_flight)
        self.follow = float(follow)
        self.prev: Dict[str, Tuple[bool, bool]] = {}
        self.last_tick: Dict[str, int] = {}
        self.flight: Dict[str, int] = {}                  # resets depuis le dernier vrai retour au sol

    def reset(self, agents, state, shared_info):
        self.prev = {a: (bool(state.cars[a].on_ground), bool(state.cars[a].has_jumped)) for a in agents}
        self.last_tick = {}
        self.flight = {a: 0 for a in agents}

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        out = {}
        for a in agents:
            car = state.cars[a]
            ground, jumped = self.prev.get(a, (True, False))
            self.prev[a] = (bool(car.on_ground), bool(car.has_jumped))
            out[a] = 0.0
            if car.on_ground and car.ball_touches <= 0:
                self.flight[a] = 0
            last = self.last_tick.get(a)
            if ((last is None or tick - last >= self.cooldown) and (jumped or not self.require_spent)
                    and is_flip_reset(car, state.ball.position, ground, jumped)):
                k = self.flight.get(a, 0)
                if self.max_per_flight is None:
                    out[a] = 1.0
                elif k < self.max_per_flight:
                    out[a] = self.follow ** k
                self.flight[a] = k + 1
                self.last_tick[a] = tick
        return out


class AirApproachTerm(RewardTerm):
    """En l'air, balle haute (> `min_z`) : vitesse vers la balle / 2300, pour le 1er homme seulement (plus petite ETA
    de l'équipe). Apprend le contrôle aérien (boost + orientation vers la balle) ; sauter quand la balle est au sol ou
    s'envoler à côté ne rapporte rien. Aide d'apprentissage : à faire décroître.
    `boost_only` (palier 5) : seulement en boostant, pour payer le vol et plus les sauts et flips vers une balle haute.
    Options du 08/11 (revue des boucles après la session 4) : `until_touch` arrête de payer après la 1re touche du vol
    (jusqu'au vrai retour au sol) : planer sous la balle en la retouchant ne rapporte plus ; `signed` compte aussi
    l'éloignement, pour qu'aller-retour vers la balle sans la toucher ne rapporte rien."""
    name = "air_approach"

    def __init__(self, min_z: float = 300.0, boost_only: bool = False, until_touch: bool = False,
                 signed: bool = False):
        self.min_z = float(min_z)
        self.boost_only = bool(boost_only)
        self.until_touch, self.signed = bool(until_touch), bool(signed)
        self.touched: Dict[str, bool] = {}

    def reset(self, agents, state, shared_info):
        self.touched = {}

    def get(self, agents, state, shared_info):
        if self.until_touch:
            for a in agents:
                car = state.cars[a]
                if car.on_ground and car.ball_touches <= 0:
                    self.touched[a] = False
                elif car.ball_touches > 0:
                    self.touched[a] = True
        ball = state.ball.position.tolist()
        if ball[2] <= self.min_z:
            return {a: 0.0 for a in agents}
        info = race_info(state, shared_info)
        out = {}
        for a in agents:
            car = state.cars[a]
            if car.on_ground or (self.boost_only and not car.is_boosting) or self.touched.get(a, False):
                out[a] = 0.0
                continue
            mine, mate, _ = team_etas(info, state, a)
            if mate < mine:
                out[a] = 0.0
                continue
            p = car.physics
            v = _dot_dir(p.linear_velocity.tolist(), p.position.tolist(), ball) / cv.CAR_MAX_SPEED
            out[a] = v if self.signed else max(0.0, v)
        return out


class FlipSpeedTerm(RewardTerm):
    """Flip (dodge) qui fait aller plus vite vers la balle : flip avant, speedflip, wavedash, half-flip pour faire
    demi-tour. Le flip ajoute ~500 uu/s d'un coup : sans boost, c'est le moyen le plus rapide de se déplacer.

    Au début d'un flip (`has_flipped` passe à vrai), on retient la MEILLEURE vitesse horizontale vers la balle de la
    dernière seconde ; `delay_s` plus tard : r = clip((vitesse vers la balle − ce record − part du boost) / 500, 0, 1),
    seulement si l'on va alors à plus de `min_speed`. Freiner pour reflipper ne rapporte rien (le record reste celui
    d'avant le freinage), un flip déjà à pleine vitesse non plus, et le boost brûlé pendant le flip est retiré du gain
    (~66 uu/s par step de boost) : seul le flip est payé."""
    name = "flip_speed"
    BOOST_ACCEL = 991.0 + 2.0 / 3.0      # uu/s² (accélération du boost)

    def __init__(self, delay_s: float = 0.3, memory_s: float = 1.0, full_gain: float = 500.0,
                 min_speed: float = 1000.0):
        self.delay = int(round(delay_s * cv.TICKS_PER_SECOND))
        self.memory = int(round(memory_s * cv.TICKS_PER_SECOND))
        self.full_gain, self.min_speed = float(full_gain), float(min_speed)
        self.hist: Dict[str, List[Tuple[int, float]]] = {}
        self.pending: Dict[str, List[float]] = {}         # agent → [tick de début, record, ticks de boost]
        self.flipped: Dict[str, bool] = {}
        self.last_tick = 0

    def reset(self, agents, state, shared_info):
        self.hist = {a: [] for a in agents}
        self.pending = {}
        self.flipped = {a: bool(state.cars[a].has_flipped) for a in agents}
        self.last_tick = int(getattr(state, "tick_count", 0) or 0)

    @staticmethod
    def speed_to_ball(car, ball) -> float:
        p = car.physics
        pos, vel = p.position.tolist(), p.linear_velocity.tolist()
        dx, dy = ball[0] - pos[0], ball[1] - pos[1]
        n = math.sqrt(dx * dx + dy * dy)
        return 0.0 if n <= 1e-6 else (vel[0] * dx + vel[1] * dy) / n

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        dt_ticks = max(0, tick - self.last_tick)
        self.last_tick = tick
        ball = state.ball.position.tolist()
        out = {}
        for a in agents:
            car = state.cars[a]
            v = self.speed_to_ball(car, ball)
            out[a] = 0.0
            pend = self.pending.get(a)
            if pend is not None:
                if car.is_boosting:
                    pend[2] += dt_ticks
                if tick - pend[0] >= self.delay:
                    gain = v - pend[1] - self.BOOST_ACCEL * pend[2] / cv.TICKS_PER_SECOND
                    if v > self.min_speed and gain > 0.0:
                        out[a] = min(1.0, gain / self.full_gain)
                    del self.pending[a]
            hist = self.hist.setdefault(a, [])
            flipped = bool(car.has_flipped)
            if flipped and not self.flipped.get(a, flipped) and a not in self.pending:
                best = max((x for t, x in hist if tick - t <= self.memory), default=v)
                self.pending[a] = [float(tick), best, 0.0]
            self.flipped[a] = flipped
            hist.append((tick, v))
            while hist and tick - hist[0][0] > self.memory:
                hist.pop(0)
        return out


class _KickoffPhase:
    """Suivi de la phase de kickoff : active si l'épisode commence par un kickoff, jusqu'à la 1re touche."""

    def reset_phase(self, state, shared_info):
        self.kickoff = shared_info.get("start_kind") == "kickoff"
        self.start_tick = int(getattr(state, "tick_count", 0) or 0)
        self.touched = False

    def elapsed(self, state) -> float:
        return (int(getattr(state, "tick_count", 0) or 0) - self.start_tick) / cv.TICKS_PER_SECOND


class KickoffRushTerm(RewardTerm, _KickoffPhase):
    """Kickoff : vitesse vers la balle / 2300 pour le 1er homme de chaque équipe (le plus proche), jusqu'à la 1re
    touche. Apprend le départ boost + flip (speedflip) ; les autres coéquipiers ne sont pas poussés à y aller."""
    name = "kickoff_rush"

    def reset(self, agents, state, shared_info):
        self.reset_phase(state, shared_info)

    def get(self, agents, state, shared_info):
        if not self.kickoff or self.touched:
            return {a: 0.0 for a in agents}
        if any(c.ball_touches > 0 for c in state.cars.values()) or self.elapsed(state) > 4.0:
            self.touched = True
            return {a: 0.0 for a in agents}
        ball = state.ball.position.tolist()
        first: Dict[int, Any] = {}
        for a, car in state.cars.items():
            d = norm3(car.physics.position - state.ball.position)
            if car.team_num not in first or d < first[car.team_num][0]:
                first[car.team_num] = (d, a)
        out = {}
        for a in agents:
            car = state.cars[a]
            if first[car.team_num][1] != a:
                out[a] = 0.0
                continue
            p = car.physics
            out[a] = max(0.0, _dot_dir(p.linear_velocity.tolist(), p.position.tolist(), ball)) / cv.CAR_MAX_SPEED
        return out


class KickoffWinTerm(RewardTerm, _KickoffPhase):
    """Résultat du kickoff, versé UNE fois 3 s après le coup d'envoi : position de la balle dans l'axe du terrain
    (perspective d'équipe), clip(y / 2000, -1, 1) ; ±1 si un but est marqué avant. Antisymétrique (pas de
    soustraction adverse). Même critère que la métrique `vs_hist/kickoff_win_rate`."""
    name = "kickoff_win"

    def __init__(self, eval_s: float = 3.0):
        self.eval_s = float(eval_s)

    def reset(self, agents, state, shared_info):
        self.reset_phase(state, shared_info)
        self.done = not self.kickoff

    def get(self, agents, state, shared_info):
        if self.done:
            return {a: 0.0 for a in agents}
        if state.goal_scored:
            self.done = True
            blue = 1.0 if state.scoring_team == cv.BLUE_TEAM else -1.0
        elif self.elapsed(state) >= self.eval_s:
            self.done = True
            blue = min(1.0, max(-1.0, float(state.ball.position[1]) / 2000.0))
        else:
            return {a: 0.0 for a in agents}
        return {a: blue * _team_sign(state.cars[a].team_num) for a in agents}


class ConcedeTerm(RewardTerm):
    """-1 pour chaque membre de l'équipe qui encaisse, SANS contrepartie pour l'équipe qui marque (ne pas le mettre
    dans `zero_sum_terms`). Rend l'encaissement plus cher que le but ne rapporte : priorité à la défense."""
    name = "concede"

    def get(self, agents, state, shared_info):
        if not state.goal_scored:
            return {a: 0.0 for a in agents}
        return {a: (-1.0 if state.cars[a].team_num != state.scoring_team else 0.0) for a in agents}


class StrongTouchTerm(RewardTerm):
    """Frappe forte (s10, `StrongTouchReward` de GigaLearn) : à chaque touche, gain de vitesse de la balle |Δv|,
    0 sous `min_speed` (simples contacts, conduites de balle), puis |Δv| / `max_speed` plafonné à 1.
    Défauts = 20 et 100 km/h. Pousse à frapper fort (flip dans la balle) plutôt qu'à pousser la balle."""
    name = "strong_touch"

    def __init__(self, min_speed: float = 556.0, max_speed: float = 2778.0):
        self.min_speed, self.max_speed = float(min_speed), float(max_speed)
        self.prev_ball_vel = np.zeros(3, dtype=np.float32)

    def reset(self, agents, state, shared_info):
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()

    def get(self, agents, state, shared_info):
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        dv = norm3(vel - self.prev_ball_vel)
        self.prev_ball_vel = vel.copy()
        r = 0.0 if dv < self.min_speed else min(1.0, dv / self.max_speed)
        return {a: (r if state.cars[a].ball_touches > 0 else 0.0) for a in agents}


def _contact_victim(car) -> Optional[str]:
    return getattr(car, "bump_victim_id", None)


class DemoTerm(RewardTerm):
    """+1 quand on démolit un adversaire (s10). L'auteur est la voiture dont le dernier contact est la victime
    (`bump_victim_id`) ; à défaut, l'adversaire le plus proche de la victime. Dans `zero_sum_terms` : la victime
    reçoit l'opposé."""
    name = "demo"

    def __init__(self):
        self.demoed: Dict[str, bool] = {}

    def reset(self, agents, state, shared_info):
        self.demoed = {a: bool(state.cars[a].is_demoed) for a in agents}

    def get(self, agents, state, shared_info):
        out = {a: 0.0 for a in agents}
        for v in agents:
            victim = state.cars[v]
            now = bool(victim.is_demoed)
            if now and not self.demoed.get(v, now):
                opps = [a for a in agents if state.cars[a].team_num != victim.team_num]
                authors = [a for a in opps if _contact_victim(state.cars[a]) == v]
                if not authors and opps:
                    authors = [min(opps, key=lambda a: norm3(state.cars[a].physics.position - victim.physics.position))]
                for a in authors[:1]:
                    out[a] += 1.0
            self.demoed[v] = now
        return out


class BumpTerm(RewardTerm):
    """+1 au début de chaque contact avec un adversaire (s10, `BumpReward` de GigaLearn) : le contact dure le temps
    du délai de RocketSim (`bump_victim_id` non vide), il n'est payé qu'une fois. Dans `zero_sum_terms` : la victime
    reçoit l'opposé."""
    name = "bump"

    def __init__(self):
        self.prev: Dict[str, Optional[str]] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: _contact_victim(state.cars[a]) for a in agents}

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            car = state.cars[a]
            victim = _contact_victim(car)
            new = victim is not None and victim != self.prev.get(a) and victim in state.cars
            out[a] = 1.0 if new and state.cars[victim].team_num != car.team_num else 0.0
            self.prev[a] = victim
        return out


# --------------------------------------------------------------------------- paliers 15-17 (20/11/2026)
# Temps de la 1re touche d'un kickoff foncé (Necto, qui fonce toujours, contre v23037 ; RocketSim, 200 kickoffs, 20/11)
# selon la distance de la voiture la plus proche à la balle : diagonale, décalée, centre.
KICKOFF_RUSH_DIST = (3279.0, 3849.0, 4609.0)
KICKOFF_RUSH_S = (2.067, 2.333, 2.600)


def kickoff_rush_time(dist: float) -> float:
    """Temps attendu de la 1re touche quand on fonce, interpolé sur la distance (borné aux positions mesurées)."""
    return float(np.interp(dist, KICKOFF_RUSH_DIST, KICKOFF_RUSH_S))


class KickoffStallTerm(RewardTerm, _KickoffPhase):
    """Kickoff bloqué ou lent (palier 15, version du PO du 20/11). Payé une fois par kickoff, jamais positif, hors
    zéro-somme (attendre coûte aux deux camps) :
    - toutes les voitures : 0 jusqu'à t0 = temps d'un kickoff foncé (`kickoff_rush_time`, selon la distance de la voiture
      la plus proche) + `margin_s`, puis de 0 à −1 en `ramp_s` secondes, selon l'heure de la 1re touche (−1 si
      personne n'a touché à t0 + ramp_s) ;
    - en plus, `loser` pour la voiture adverse la plus proche de la balle à la 1re touche : elle a perdu la course, qu'elle
      touche ensuite ou non (choix du PO : contester la balle juste après ne coûte rien de plus). Rien si les deux
      équipes touchent au même step.
    Mesure du 20/11 (RocketSim, 300 kickoffs) : en auto-jeu, v18160 touchait avant 3 s dans 99 % des kickoffs, v23037
    dans 8 % (1re touche vers 4 s, les deux ralentissent pour arriver second)."""
    name = "kickoff_stall"

    def __init__(self, margin_s: float = 0.1, ramp_s: float = 1.0, loser: float = 0.125):
        self.margin_s, self.ramp_s, self.loser = float(margin_s), max(1e-6, float(ramp_s)), float(loser)
        self.t0 = self.t1 = 0.0
        self.paid = 0.0                                   # part « heure de la 1re touche » versée, ∈ [0, 1] (métrique)

    def reset(self, agents, state, shared_info):
        self.reset_phase(state, shared_info)
        self.done = not self.kickoff
        self.paid = 0.0
        if self.kickoff:
            ball = state.ball.position
            d = min(norm3(c.physics.position - ball) for c in state.cars.values())
            self.t0 = kickoff_rush_time(d) + self.margin_s
            self.t1 = self.t0 + self.ramp_s

    def get(self, agents, state, shared_info):
        out = {a: 0.0 for a in agents}
        if self.done:
            return out
        t = self.elapsed(state)
        touchers = [a for a, c in state.cars.items() if c.ball_touches > 0]
        if touchers or state.goal_scored:
            self.done = True
            self.paid = min(max((t - self.t0) / self.ramp_s, 0.0), 1.0)
            for a in agents:
                out[a] = -self.paid
            teams = {state.cars[a].team_num for a in touchers}
            if len(teams) == 1:
                first = teams.pop()
                opps = [a for a in agents if state.cars[a].team_num != first]
                if opps:
                    loser = min(opps, key=lambda a: norm3(state.cars[a].physics.position - state.ball.position))
                    out[loser] -= self.loser
            return out
        if t >= self.t1:
            self.done = True
            self.paid = 1.0
            return {a: -1.0 for a in agents}
        return out


def _own_goal_dist(car) -> float:
    """Distance de la voiture au fond de son propre but."""
    own = _OWN_GOAL if car.team_num == cv.BLUE_TEAM else _OWN_GOAL_ORANGE
    p = car.physics.position
    return math.sqrt((p[0] - own[0]) ** 2 + (p[1] - own[1]) ** 2 + (p[2] - own[2]) ** 2)


class DemoGoalTerm(RewardTerm):
    """Bonus à l'auteur d'une démo si son équipe marque juste après (palier 16) : 1 si le but arrive dans les `full_s`
    secondes, puis décroissance linéaire jusqu'à 0 à `window_s` secondes (version du PO du 20/11, comme kickoff_stall).
    Sur 4 998 buts d'auto-jeu de v23037 : la démo du gardien garde ~97 % du bonus d'une fenêtre fixe, une démo loin du
    but ~38 %. Mécanique du
    haut niveau : garder la balle sur le capot, démolir le gardien, la balle entre. `last_defender_only` (choix du PO) :
    seulement si la victime était, juste avant la démo, la voiture de son équipe la plus proche de son propre but (en
    1v1, c'est toujours le cas). `own_half_only` (PO, 20/11) : la victime doit aussi être dans son propre camp, sinon
    c'est un attaquant démoli suivi d'une contre-attaque. Mesure du 20/11 (auto-jeu de v23037, 4 998 buts) : à 5 s sans
    cette condition, 21 % des bonus payaient la démo d'un attaquant ; à 2,8 s avec, aucun, et 42 des 53 démos du gardien
    (but en 0,6 s en médiane) restent payées. Auteur désigné comme pour `demo`. Dans `zero_sum_terms` : l'équipe qui
    encaisse perd autant. Au plus un paiement par but (l'épisode finit au but), et rien sans but : pas de boucle."""
    name = "demo_goal"

    def __init__(self, window_s: float = 3.0, last_defender_only: bool = True, own_half_only: bool = True,
                 full_s: float = 0.75):
        self.window = int(round(float(window_s) * cv.TICKS_PER_SECOND))
        self.full = int(round(min(float(full_s), float(window_s)) * cv.TICKS_PER_SECOND))
        self.last_defender_only = bool(last_defender_only)
        self.own_half_only = bool(own_half_only)

    def reset(self, agents, state, shared_info):
        self.demoed = {a: bool(state.cars[a].is_demoed) for a in agents}
        self.prev_dist = {a: _own_goal_dist(state.cars[a]) for a in agents}
        self.prev_in_own_half = {a: self._in_own_half(state.cars[a]) for a in agents}
        self.demos: List[Tuple[int, str]] = []          # (tick, auteur) des démos qui comptent

    @staticmethod
    def _in_own_half(car) -> bool:
        y = float(car.physics.position[1])
        return y < 0.0 if car.team_num == cv.BLUE_TEAM else y > 0.0

    def _is_last_defender(self, victim: str, agents, state) -> bool:
        team = state.cars[victim].team_num
        mates = [a for a in agents if a != victim and state.cars[a].team_num == team and not self.demoed.get(a, False)]
        return all(self.prev_dist[victim] <= self.prev_dist[a] for a in mates)

    def get(self, agents, state, shared_info):
        out = {a: 0.0 for a in agents}
        tick = int(getattr(state, "tick_count", 0) or 0)
        for v in agents:
            victim = state.cars[v]
            now = bool(victim.is_demoed)
            if now and not self.demoed.get(v, now):
                opps = [a for a in agents if state.cars[a].team_num != victim.team_num]
                authors = [a for a in opps if _contact_victim(state.cars[a]) == v]
                if not authors and opps:
                    authors = [min(opps, key=lambda a: norm3(state.cars[a].physics.position - victim.physics.position))]
                if (authors and (not self.last_defender_only or self._is_last_defender(v, agents, state))
                        and (not self.own_half_only or self.prev_in_own_half.get(v, False))):
                    self.demos.append((tick, authors[0]))
        for v in agents:
            self.demoed[v] = bool(state.cars[v].is_demoed)
            if not self.demoed[v]:
                self.prev_dist[v] = _own_goal_dist(state.cars[v])
                self.prev_in_own_half[v] = self._in_own_half(state.cars[v])
        if state.goal_scored:
            recent = [(tick - t, a) for t, a in self.demos
                      if tick - t < self.window and state.cars[a].team_num == state.scoring_team]
            if recent:
                delay, author = recent[-1]                     # la démo la plus récente : le plus gros bonus
                span = max(1, self.window - self.full)
                out[author] = 1.0 if delay <= self.full else max(0.0, (self.window - delay) / span)
            self.demos = []
        return out


class HighGoalTerm(RewardTerm):
    """+1 au buteur si la dernière touche avant le but était la sienne avec la balle à `min_z` uu ou plus (palier 17) :
    aérienne haute, tir depuis le haut d'un mur ou du plafond. 600 uu : au-dessus de ce qu'atteint un double saut (balle
    vers 500 uu), sous la barre (642 uu). Rien si la dernière touche est adverse (but contre son camp, déviation) ou si
    les deux équipes ont touché au même step. Dans `zero_sum_terms` : encaisser un but haut coûte autant. Au plus un
    paiement par but (l'épisode finit au but), et rien sans but : pas de boucle possible."""
    name = "high_goal"

    def __init__(self, min_z: float = 600.0):
        self.min_z = float(min_z)

    def reset(self, agents, state, shared_info):
        self.last: Optional[Tuple[str, int, float]] = None   # (voiture, équipe, hauteur de la balle)

    def get(self, agents, state, shared_info):
        out = {a: 0.0 for a in agents}
        touchers = [a for a in agents if state.cars[a].ball_touches > 0]
        if touchers:
            teams = {state.cars[a].team_num for a in touchers}
            if len(teams) > 1:
                self.last = None
            else:
                a = min(touchers, key=lambda x: norm3(state.cars[x].physics.position - state.ball.position))
                self.last = (a, state.cars[a].team_num, float(state.ball.position[2]))
        if state.goal_scored:
            if self.last is not None and self.last[1] == state.scoring_team and self.last[2] >= self.min_z:
                out[self.last[0]] = 1.0
            self.last = None
        return out


# --------------------------------------------------------------------------- paliers 18-20 (22/11/2026)
class BumpGoalTerm(DemoGoalTerm):
    """Comme `demo_goal`, pour un BUMP sans démolition (palier 18, PO du 22/11 : le bot ne bumpe jamais le gardien juste
    avant de marquer). Événement : début d'un contact (`bump_victim_id`, comme `BumpTerm`) avec un adversaire qui n'est
    pas démoli à ce step (la démo relève de `demo_goal`), dernier défenseur de son équipe et dans son camp. Bonus plein si
    l'équipe de l'auteur marque dans les `full_s` secondes, puis linéaire jusqu'à 0 à `window_s`. Zéro-somme. Au plus un
    paiement par but, rien sans but : pas de boucle."""
    name = "bump_goal"

    def reset(self, agents, state, shared_info):
        super().reset(agents, state, shared_info)
        self.prev_victim = {a: _contact_victim(state.cars[a]) for a in agents}

    def get(self, agents, state, shared_info):
        out = {a: 0.0 for a in agents}
        tick = int(getattr(state, "tick_count", 0) or 0)
        for a in agents:
            car = state.cars[a]
            v = _contact_victim(car)
            new = v is not None and v != self.prev_victim.get(a) and v in state.cars
            self.prev_victim[a] = v
            if not new or state.cars[v].team_num == car.team_num or state.cars[v].is_demoed:
                continue
            if ((not self.last_defender_only or self._is_last_defender(v, agents, state))
                    and (not self.own_half_only or self.prev_in_own_half.get(v, False))):
                self.demos.append((tick, a))
        for v in agents:
            self.demoed[v] = bool(state.cars[v].is_demoed)
            if not self.demoed[v]:
                self.prev_dist[v] = _own_goal_dist(state.cars[v])
                self.prev_in_own_half[v] = self._in_own_half(state.cars[v])
        if state.goal_scored:
            recent = [(tick - t, a) for t, a in self.demos
                      if tick - t < self.window and state.cars[a].team_num == state.scoring_team]
            if recent:
                delay, author = recent[-1]
                span = max(1, self.window - self.full)
                out[author] = 1.0 if delay <= self.full else max(0.0, (self.window - delay) / span)
            self.demos = []
        return out


class _FlightTerm(RewardTerm):
    """Base des paliers 19-20 : suit chaque vol (décollage → retour au sol) et paie AU PLUS la 1re touche du vol, balle à
    `min_z` ou plus, selon ce qu'elle apporte : `floor` + (1 − `floor`) × g, g = gain de vitesse de la balle vers le but
    adverse / `useful_speed` (borné à [0, 1]), comme `aerial_touch` en mode utile. Une parade compte (arrêter une balle
    qui file vers notre but lui ajoute de la vitesse vers le but adverse). Le jonglage ne paie rien : une seule touche
    par vol, et il faut retoucher le sol (ou un mur) pour un nouveau vol."""

    def __init__(self, min_z: float, useful_speed: float = 1000.0, floor: float = 0.25):
        self.min_z, self.useful_speed, self.floor = float(min_z), float(useful_speed), float(floor)

    def reset(self, agents, state, shared_info):
        self.flight: Dict[str, Optional[Dict[str, Any]]] = {a: None for a in agents}
        self.ground_up_z = {a: float(state.cars[a].physics.up[2]) for a in agents}
        self.prev_ground = {a: bool(state.cars[a].on_ground) for a in agents}
        self.prev_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32).copy()

    def eligible(self, car, flight: Dict[str, Any], tick: int) -> float:
        """Facteur du vol (0 = rien) ; à définir dans la classe fille."""
        raise NotImplementedError

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        dv = (vel - self.prev_ball_vel).tolist()
        self.prev_ball_vel = vel.copy()
        ball = state.ball.position.tolist()
        out = {}
        for a in agents:
            car = state.cars[a]
            out[a] = 0.0
            ground = bool(car.on_ground) and car.ball_touches <= 0     # roues sur la balle : le vol continue
            if ground:
                self.ground_up_z[a] = float(car.physics.up[2])
                self.flight[a] = None
            elif self.flight.get(a) is None:                           # décollage (ou départ en l'air d'un exercice)
                self.flight[a] = {"t0": tick, "wall": self.prev_ground.get(a, False)
                                  and self.ground_up_z.get(a, 1.0) < 0.6, "paid": False}
            self.prev_ground[a] = ground
            f = self.flight.get(a)
            if f is None or f["paid"] or car.ball_touches <= 0 or float(ball[2]) < self.min_z:
                continue
            f["paid"] = True                                           # 1re touche haute du vol, payée ou non
            factor = self.eligible(car, f, tick)
            if factor <= 0.0:
                continue
            goal = _OPP_GOAL if car.team_num == cv.BLUE_TEAM else _OPP_GOAL_ORANGE
            g = min(1.0, max(0.0, _dot_dir(dv, ball, goal) / self.useful_speed))
            out[a] = factor * (self.floor + (1.0 - self.floor) * g)
        return out


class WallTouchTerm(_FlightTerm):
    """Palier 19 (PO du 22/11 : plus à l'aise avec les murs et les sauts depuis le mur pour atteindre la balle plus vite).
    1re touche d'un vol parti d'un mur (roues au contact, vecteur « haut » de la voiture à moins de 0,6 en z au dernier
    instant au sol), balle à `min_z` ou plus. Zéro-somme, comme aerial_touch."""
    name = "wall_touch"

    def eligible(self, car, flight, tick):
        return 1.0 if flight["wall"] else 0.0


class FastAerialTerm(_FlightTerm):
    """Palier 20 (PO du 22/11 : Necto et Nexto vont plus vite en l'air grâce aux doubles sauts). 1re touche d'un vol
    avec double saut utilisé, balle à `min_z` (500) ou plus : plein si la touche arrive dans les `fast_s` secondes après
    le décollage, puis linéaire jusqu'à 0 à `slow_s`. Zéro-somme."""
    name = "fast_aerial"

    def __init__(self, min_z: float = 500.0, useful_speed: float = 1000.0, floor: float = 0.25,
                 fast_s: float = 0.8, slow_s: float = 2.0):
        super().__init__(min_z, useful_speed, floor)
        self.fast = float(fast_s) * cv.TICKS_PER_SECOND
        self.slow = max(float(slow_s) * cv.TICKS_PER_SECOND, self.fast + 1.0)

    def eligible(self, car, flight, tick):
        if not car.has_double_jumped:
            return 0.0
        t = tick - flight["t0"]
        return 1.0 if t <= self.fast else max(0.0, (self.slow - t) / (self.slow - self.fast))


def linear_schedule(start: float, end: float, steps: float, t: float) -> float:
    if steps <= 0:
        return end
    frac = min(max(t / steps, 0.0), 1.0)
    return start + (end - start) * frac


class KickoffFirstTouchTerm(RewardTerm, _KickoffPhase):
    """Palier 21 (25/11, PO) : 1re touche du kickoff. +1, une fois par kickoff, à la voiture qui touche la balle la
    première dans les `max_s` secondes, si sa touche ne renvoie pas la balle vers son propre but (vitesse de la balle
    dans l'axe du terrain, perspective d'équipe, au moins −`max_back` uu/s). Deux camps au contact au même step (50/50) :
    rien. Zéro-somme : l'adversaire perd ce que l'autre gagne.
    Mesure du 24/11 : v30160 ne touche le premier que dans 0-4 % des kickoffs contre Necto et Nexto, en simulation comme
    dans le vrai jeu ; kickoff_win (balle chez l'adversaire à 3 s) paie autant « attendre puis contrer » qu'un kickoff
    gagné, et dans le vrai jeu les bots externes marquent sur kickoff."""
    name = "kickoff_first"

    def __init__(self, max_s: float = 3.0, max_back: float = 200.0):
        self.max_s, self.max_back = float(max_s), float(max_back)

    def reset(self, agents, state, shared_info):
        self.reset_phase(state, shared_info)
        self.done = not self.kickoff

    def get(self, agents, state, shared_info):
        if self.done:
            return {a: 0.0 for a in agents}
        if self.elapsed(state) > self.max_s or state.goal_scored:
            self.done = True
            return {a: 0.0 for a in agents}
        touchers = [a for a, c in state.cars.items() if c.ball_touches > 0]
        if not touchers:
            return {a: 0.0 for a in agents}
        self.done = True
        if len({state.cars[a].team_num for a in touchers}) > 1:
            return {a: 0.0 for a in agents}
        vy = float(state.ball.linear_velocity[1])
        return {a: 1.0 if a in touchers and vy * _team_sign(state.cars[a].team_num) >= -self.max_back else 0.0
                for a in agents}


class BigPadLowTerm(RewardTerm):
    """Palier 23 (25/11, PO : il roule souvent à sec ; récompenser le boost sans freiner sa consommation). Gros pad pris
    avec moins de `below` de boost : (below − boost avant) / below, donc 1 à sec, 0 à `below`. Ne pénalise jamais la
    dépense (rien ne change tant qu'on ne ramasse pas). Une fois toutes les `cooldown_s` secondes au plus par voiture :
    la boucle « brûler son boost puis reprendre un gros pad » rapporte au plus 6 / min × poids
    (tests/test_paliers_21_25.py). Pas après une démo (la réapparition redonne 33 de boost)."""
    name = "big_pad_low"

    def __init__(self, below: float = 25.0, cooldown_s: float = 10.0, min_gain: float = 13.0):
        self.below, self.cooldown_s, self.min_gain = float(below), float(cooldown_s), float(min_gain)
        self.prev: Dict[str, float] = {}
        self.demoed: Dict[str, bool] = {}
        self.last: Dict[str, int] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: float(state.cars[a].boost_amount) for a in agents}
        self.demoed = {a: bool(state.cars[a].is_demoed) for a in agents}
        self.last = {}

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        out = {}
        for a in agents:
            car = state.cars[a]
            b = float(car.boost_amount)
            prev = self.prev.get(a, b)
            was_demoed = self.demoed.get(a, False)
            self.prev[a], self.demoed[a] = b, bool(car.is_demoed)
            ready = tick - self.last.get(a, -10**9) >= self.cooldown_s * cv.TICKS_PER_SECOND
            if b - prev > self.min_gain and prev < self.below and not car.is_demoed and not was_demoed and ready:
                self.last[a] = tick
                out[a] = (self.below - prev) / self.below
            else:
                out[a] = 0.0
        return out


class SlowTurnTerm(RewardTerm):
    """Palier 25 (25/11, PO : il fait de longs demi-tours inutiles ; plus mobile, plus agile). −1 par seconde passée au
    sol, à plus de `min_dist` de la balle, la balle dans le dos (angle entre l'avant de la voiture et la balle au-delà de
    `min_angle_deg`) et à moins de `max_speed` uu/s vers l'avant : le demi-tour lent coûte, le demi-flip, le powerslide
    ou une remise en route rapide coûtent peu. Pénalité seule : rien à accumuler. Hors zéro-somme (comme kickoff_stall)."""
    name = "slow_turn"

    def __init__(self, min_angle_deg: float = 110.0, max_speed: float = 800.0, min_dist: float = 700.0):
        self.cos_max = math.cos(math.radians(float(min_angle_deg)))
        self.max_speed, self.min_dist = float(max_speed), float(min_dist)
        self.tick: Optional[int] = None

    def reset(self, agents, state, shared_info):
        self.tick = int(getattr(state, "tick_count", 0) or 0)

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        dt = max(0, tick - (self.tick if self.tick is not None else tick)) / cv.TICKS_PER_SECOND
        self.tick = tick
        out = {}
        for a in agents:
            car = state.cars[a]
            p = car.physics
            rel = state.ball.position - p.position
            dist = norm3(rel)
            if not car.on_ground or car.is_demoed or dist < self.min_dist or dt <= 0.0:
                out[a] = 0.0
                continue
            fwd = p.forward
            behind = float(np.dot(fwd, rel)) / max(dist, 1e-6) < self.cos_max
            slow = float(np.dot(fwd, p.linear_velocity)) < self.max_speed
            out[a] = -dt if behind and slow else 0.0
        return out


class PowerslideTurnTerm(RewardTerm):
    """Palier 25 (25/11, PO : le bot a extrêmement besoin des dérapages). Un dérapage (frein à main au sol à plus de
    `min_speed` uu/s, au moins `min_s` secondes) est jugé quand il se termine : (cos final − cos initial) / 2 de l'angle
    entre l'avant de la voiture et la balle, × min(1, vitesse finale / 1 000). Signé : déraper pour se détourner de la
    balle coûte autant que déraper vers elle rapporte, tourner en rond ne rapporte rien. Seulement si la balle était à
    90° ou plus au départ et à plus de `min_dist`. Au plus un dérapage payé toutes les `cooldown_s` secondes : la boucle
    « se détourner sans frein, revenir en dérapant » rapporte au plus 60 / cooldown_s par minute × poids
    (tests/test_paliers_21_25.py)."""
    name = "powerslide"

    def __init__(self, min_speed: float = 500.0, min_s: float = 0.1, min_dist: float = 1000.0, cooldown_s: float = 2.0):
        self.min_speed, self.min_s, self.min_dist = float(min_speed), float(min_s), float(min_dist)
        self.cooldown_s = float(cooldown_s)
        self.slide: Dict[str, Tuple[int, float]] = {}
        self.last: Dict[str, int] = {}

    def reset(self, agents, state, shared_info):
        self.slide, self.last = {}, {}

    @staticmethod
    def _facing(car, state) -> Tuple[float, float]:
        p = car.physics
        rel = state.ball.position - p.position
        dist = norm3(rel)
        return float(np.dot(p.forward, rel)) / max(dist, 1e-6), dist

    def get(self, agents, state, shared_info):
        tick = int(getattr(state, "tick_count", 0) or 0)
        out = {}
        for a in agents:
            car = state.cars[a]
            out[a] = 0.0
            speed = norm3(car.physics.linear_velocity)
            sliding = bool(car.handbrake > 0.5) and car.on_ground and not car.is_demoed and speed > self.min_speed
            cos, dist = self._facing(car, state)
            if sliding and a not in self.slide:
                self.slide[a] = (tick, cos)
            elif not sliding and a in self.slide:
                start, cos0 = self.slide.pop(a)
                long_enough = (tick - start) >= self.min_s * cv.TICKS_PER_SECOND
                ready = tick - self.last.get(a, -10**9) >= self.cooldown_s * cv.TICKS_PER_SECOND
                if long_enough and ready and cos0 <= 0.0 and dist > self.min_dist and not car.is_demoed:
                    self.last[a] = tick
                    out[a] = (cos - cos0) / 2.0 * min(1.0, speed / 1000.0)
        return out


class TeamReward(RewardFunction[str, GameState, float]):
    """`stage_weights` : poids possibles des paliers du curriculum (`curriculum.stages`, curriculum.py). Les termes
    utilisés par au moins un palier sont construits dès le départ ; les poids du moment arrivent par
    shared_info["reward_weights"] (envoyés par le processus principal) et remplacent alors `weights`."""

    def __init__(self, terms: List[RewardTerm], weights: Dict[str, float],
                 anneal: Optional[Dict[str, Dict[str, float]]] = None,
                 team_spirit: Optional[Dict[str, float]] = None, zero_sum: float = 1.0,
                 zero_sum_terms: Optional[List[str]] = None,
                 stage_weights: Optional[List[Dict[str, float]]] = None):
        used = [weights] + list(stage_weights or [])
        self.terms = [t for t in terms if any(w.get(t.name, 0.0) != 0.0 for w in used)]
        self.weights = dict(weights)
        self.anneal = anneal or {}
        self.team_spirit = team_spirit or {"start": 0.0, "end": 0.0, "steps": 1}
        self.zero_sum = zero_sum
        self.zero_sum_terms = None if zero_sum_terms is None else set(zero_sum_terms)
        self._override: Optional[Dict[str, float]] = None

    def _weights_at(self, step: float, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if getattr(self, "_cached_step", None) != step or override is not self._override:
            self._cached_step = step
            self._override = override        # référence gardée : un nouveau dict envoyé = nouveau calcul
            self._cached_weights = {t.name: self.current_weight(t.name, step, override) for t in self.terms}
            self._cached_tau = self.current_tau(step)
        return self._cached_weights

    def current_weight(self, name: str, step: float, override: Optional[Dict[str, float]] = None) -> float:
        """Poids × multiplicateur de décroissance : 1 jusqu'au step `begin` (0 par défaut), puis linéaire jusqu'à
        `end` en `steps` pas. `begin` sert aux démarrages à chaud (compteur de steps hérité d'un autre run).
        `override` : poids du palier courant du curriculum (à la place de `weights`)."""
        w = (self.weights if override is None else override).get(name, 0.0)
        sched = self.anneal.get(name)
        if sched:
            begin = float(sched.get("begin", 0.0))
            w *= linear_schedule(1.0, float(sched["end"]), float(sched["steps"]), step - begin)
        return w

    def current_tau(self, step: float) -> float:
        ts = self.team_spirit
        return linear_schedule(float(ts["start"]), float(ts["end"]), float(ts["steps"]), step)

    def reset(self, agents, initial_state, shared_info):
        for t in self.terms:
            t.reset(agents, initial_state, shared_info)
        shared_info["reward_log"] = {t.name: 0.0 for t in self.terms}
        shared_info["reward_log_abs"] = {t.name: 0.0 for t in self.terms}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> Dict[str, float]:
        step = float(shared_info.get("global_step", 0))
        raw = {a: 0.0 for a in agents}
        raw_zs = raw if self.zero_sum_terms is None else {a: 0.0 for a in agents}
        log = shared_info.setdefault("reward_log", {})
        log_abs = shared_info.setdefault("reward_log_abs", {})
        n = max(1, len(agents))
        weights = self._weights_at(step, shared_info.get("reward_weights"))
        for term in self.terms:
            w = weights[term.name]
            vals = term.get(agents, state, shared_info)
            in_zs = self.zero_sum_terms is not None and term.name in self.zero_sum_terms
            total = total_abs = 0.0
            for a in agents:
                v = w * vals[a]
                raw[a] += v
                if in_zs:
                    raw_zs[a] += v
                total += v
                total_abs += abs(v)
            log[term.name] = log.get(term.name, 0.0) + total / n
            log_abs[term.name] = log_abs.get(term.name, 0.0) + total_abs / n
        return combine_team_rewards(raw, {a: state.cars[a].team_num for a in agents},
                                    self._cached_tau, self.zero_sum, raw_opp=raw_zs)


def _team_means(vals: Dict[str, float], teams: Dict[str, int]) -> Dict[int, float]:
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for a, r in vals.items():
        t = teams[a]
        sums[t] = sums.get(t, 0.0) + r
        counts[t] = counts.get(t, 0) + 1
    return {t: sums[t] / counts[t] for t in sums}


def combine_team_rewards(raw: Dict[str, float], teams: Dict[str, int], tau: float, zero_sum: float,
                         raw_opp: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """raw_opp : part des rewards soumise à la soustraction adverse (par défaut : toutes)."""
    team_mean = _team_means(raw, teams)
    opp_team_mean = team_mean if raw_opp is None or raw_opp is raw else _team_means(raw_opp, teams)
    out = {}
    for a, r in raw.items():
        t = teams[a]
        opp = [m for tt, m in opp_team_mean.items() if tt != t]
        opp_mean = sum(opp) / len(opp) if opp else 0.0
        out[a] = (1.0 - tau) * r + tau * team_mean[t] - zero_sum * opp_mean
    return out


def make_reward(cfg: Dict[str, Any]) -> TeamReward:
    rc = cfg["rewards"]
    gamma = float(cfg.get("ppo", {}).get("gamma", 0.99))
    terms = [
        GoalTerm(),
        TouchTerm(aerial_bonus=rc.get("touch_aerial_bonus", 1.0)),
        VelPlayerToBallTerm(cap=rc.get("vel_player_to_ball_cap"), signed=rc.get("vel_player_to_ball_signed", False)),
        VelBallToGoalTerm(),
        FaceBallTerm(),
        BoostGainTerm(loss_coef=rc.get("boost_loss_coef", 0.0), below=rc.get("boost_gain_below", 100.0)),
        DoubleCommitTerm(radius=rc.get("double_commit_radius", 900), min_speed=rc.get("double_commit_min_speed", 400)),
        TouchStrengthTerm(),
        ShotTerm(off_target_factor=rc.get("shot_off_target_factor", 1.0)),
        AlignBallGoalTerm(),
        InAirTerm(),
        BoostKeepTerm(),
        SaveTerm(opp_only=rc.get("save_opp_only", False)),
        BoostWasteTerm(),
        GoalsideTerm(),
        # s7 (cf. section « rewards s7 ») : potentiels au γ du PPO, pour que r = γΦ(s') − Φ(s) soit exact
        BallGoalPotential(gamma),
        ThreatPotential(gamma),
        BoostPotential(float(rc.get("boost_potential_gamma", gamma))),
        PositioningPotential(gamma, temperature=rc.get("race_temperature", 0.3),
                             support_p=rc.get("support_defend_p", 0.8)),
        AerialTouchTerm(min_z=rc.get("aerial_min_z", 220.0), min_air_boost_s=rc.get("aerial_min_air_boost_s", 0.0),
                        useful_speed=rc.get("aerial_useful_speed"), first_floor=rc.get("aerial_first_floor", 0.25),
                        follow=rc.get("aerial_follow", 0.5)),
        AirApproachTerm(boost_only=rc.get("air_approach_boost_only", False),
                        until_touch=rc.get("air_approach_until_touch", False),
                        signed=rc.get("air_approach_signed", False)),
        FlipSpeedTerm(min_speed=rc.get("flip_speed_min_speed", 1000.0)),
        KickoffRushTerm(),
        KickoffWinTerm(),
        ConcedeTerm(),
        # s10 (refonte du 23/10/2026)
        StrongTouchTerm(min_speed=rc.get("strong_touch_min_speed", 556.0),
                        max_speed=rc.get("strong_touch_max_speed", 2778.0)),
        DemoTerm(),
        BumpTerm(),
        # palier 5 « finition » (04/11/2026)
        ShotThreatPotential(gamma, floor=rc.get("shot_threat_floor", 0.5)),
        # palier 6 « maitrise » (06/11/2026)
        BoostSpendTerm(),
        # palier 14 « petits_pads » (17/11/2026)
        SmallPadTerm(),
        FlipResetTerm(cooldown_s=rc.get("flip_reset_cooldown_s", 1.0),
                      require_spent=rc.get("flip_reset_require_spent", False),
                      max_per_flight=rc.get("flip_reset_max_per_flight"), follow=rc.get("flip_reset_follow", 0.5)),
        # paliers 15-17 (20/11/2026)
        KickoffStallTerm(margin_s=rc.get("kickoff_stall_margin_s", 0.1), ramp_s=rc.get("kickoff_stall_ramp_s", 1.0),
                         loser=rc.get("kickoff_stall_loser", 0.125)),
        DemoGoalTerm(window_s=rc.get("demo_goal_window_s", 3.0),
                     last_defender_only=rc.get("demo_goal_last_defender", True),
                     own_half_only=rc.get("demo_goal_own_half", True), full_s=rc.get("demo_goal_full_s", 0.75)),
        HighGoalTerm(min_z=rc.get("high_goal_min_z", 600.0)),
        # paliers 18-20 (22/11/2026)
        BumpGoalTerm(window_s=rc.get("demo_goal_window_s", 3.0),
                     last_defender_only=rc.get("demo_goal_last_defender", True),
                     own_half_only=rc.get("demo_goal_own_half", True), full_s=rc.get("demo_goal_full_s", 0.75)),
        WallTouchTerm(min_z=rc.get("wall_touch_min_z", 300.0), useful_speed=rc.get("aerial_useful_speed") or 1000.0,
                      floor=rc.get("flight_touch_floor", 0.25)),
        FastAerialTerm(min_z=rc.get("fast_aerial_min_z", 500.0), useful_speed=rc.get("aerial_useful_speed") or 1000.0,
                       floor=rc.get("flight_touch_floor", 0.25), fast_s=rc.get("fast_aerial_fast_s", 0.8),
                       slow_s=rc.get("fast_aerial_slow_s", 2.0)),
        # paliers 21-25 (25/11/2026)
        KickoffFirstTouchTerm(max_s=rc.get("kickoff_first_max_s", 3.0), max_back=rc.get("kickoff_first_max_back", 200.0)),
        BigPadLowTerm(below=rc.get("big_pad_low_below", 25.0), cooldown_s=rc.get("big_pad_low_cooldown_s", 10.0)),
        SlowTurnTerm(min_angle_deg=rc.get("slow_turn_min_angle_deg", 110.0),
                     max_speed=rc.get("slow_turn_max_speed", 800.0), min_dist=rc.get("slow_turn_min_dist", 700.0)),
        PowerslideTurnTerm(cooldown_s=rc.get("powerslide_cooldown_s", 2.0)),
    ]
    stages = (cfg.get("curriculum") or {}).get("stages") or []
    return TeamReward(terms, rc["weights"], rc.get("anneal"), rc.get("team_spirit"), rc.get("zero_sum", 1.0),
                      rc.get("zero_sum_terms"), stage_weights=[s.get("rewards") or {} for s in stages])
