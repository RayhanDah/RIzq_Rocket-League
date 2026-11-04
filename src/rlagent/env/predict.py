"""Prédictions physiques rapides (sans simulation), partagées par les observations, les rewards et les métriques.

- Trajectoire analytique de la balle : gravité, rebonds successifs au sol (restitution 0,6), murs en miroir,
  frottement de l'air. Fiable ~1-2 s (ni effets, ni contacts avec les voitures).
- Tir cadré : la trajectoire franchit-elle la ligne de but DANS le cadre, et dans combien de temps.
- ETA : temps approximatif pour qu'une voiture atteigne la balle (vitesse actuelle, accélération avec / sans boost,
  virage, hauteur de la balle), avec point d'interception. Sert à savoir qui gagnera la course à la balle.

Tout est en repère du MONDE et en arithmétique Python scalaire (numpy est lent sur des vecteurs de 3 valeurs).
`race_info` met le résultat en cache dans shared_info : l'ObsBuilder (appelé en premier par RLGym) et les rewards
du même step partagent le calcul.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

GRAVITY = 650.0
BOUNCE = 0.6                  # restitution verticale au rebond au sol
BALL_DRAG = 0.03              # frottement de l'air de la balle (1/s, RocketSim : BALL_DRAG)
R = float(cv.BALL_RADIUS)
X_WALL = float(cv.SIDE_WALL_X) - R
Y_WALL = float(cv.BACK_WALL_Y) - R
Z_CEIL = float(cv.CEILING_Z) - R
GOAL_Y = float(cv.BACK_WALL_Y)
POST_X = float(cv.GOAL_CENTER_TO_POST) - 0.25 * R     # centre de la balle dans le cadre (un peu de tolérance)
BAR_Z = float(cv.GOAL_HEIGHT) - 0.25 * R

THREAT_HORIZON = 3.0          # s : un tir qui arrive plus tard ne compte pas (trajectoire trop incertaine)
PRED_HORIZON = 3.0            # s : au-delà, on garde la position prédite à 3 s
ETA_CAP = 6.0                 # s : ETA plafonnée
CONTACT = 130.0               # uu : distance centre voiture → centre balle à laquelle on touche
TURN_TIME = 0.9               # s pour un demi-tour complet
BOOST_PER_S = 100.0 / 3.0     # consommation de boost (unités/s)
A_BOOST, V_BOOST = 1800.0, 2300.0   # accélération moyenne gaz + boost, vitesse max
A_THROTTLE, V_THROTTLE = 800.0, 1410.0
A_AIR_BOOST = 991.7


def fold(x: float, limit: float) -> float:
    """Réflexion sur des murs en ±limit (onde triangulaire) : trajectoire avec rebonds parfaits sur les murs."""
    period = 4.0 * limit
    return limit - abs(math.fmod(math.fmod(x + limit, period) + period, period) - 2.0 * limit)


def drag_dist(t: float) -> float:
    """Distance parcourue à vitesse unité en t secondes avec frottement linéaire : (1 - e^{-kt}) / k."""
    return (1.0 - math.exp(-BALL_DRAG * t)) / BALL_DRAG


def ball_z_at(z0: float, vz: float, t: float) -> float:
    """Hauteur du centre de la balle à t, avec gravité et rebonds successifs au sol."""
    h, v, rem = max(z0 - R, 0.0), vz, t
    for _ in range(8):
        if h <= 1e-3 and v < 60.0:               # balle qui roule (ou rebonds négligeables)
            return R
        t_hit = (v + math.sqrt(v * v + 2.0 * GRAVITY * h)) / GRAVITY
        if rem <= t_hit:
            return min(R + h + v * rem - 0.5 * GRAVITY * rem * rem, Z_CEIL)
        rem -= t_hit
        v = BOUNCE * (GRAVITY * t_hit - v)       # vitesse verticale (vers le haut) après le rebond
        h = 0.0
    return R


def ball_at(pos, vel, t: float) -> Tuple[float, float, float]:
    """Position prédite de la balle à t (repère du monde)."""
    t = min(max(t, 0.0), PRED_HORIZON)
    d = drag_dist(t)
    return (fold(pos[0] + vel[0] * d, X_WALL), fold(pos[1] + vel[1] * d, Y_WALL), ball_z_at(pos[2], vel[2], t))


def _line_crossing(y0: float, vy_world: float, goal_sign: float, horizon: float) -> Optional[Tuple[float, float]]:
    """(distance à vitesse unité, temps) pour atteindre la ligne de but, None si la balle n'y arrive pas dans `horizon`."""
    vy = vy_world * goal_sign
    if vy <= 1.0:
        return None
    need = (GOAL_Y - goal_sign * y0) / vy        # distance « à vitesse unité » nécessaire
    if need <= 0.0:
        return None
    k_need = BALL_DRAG * need
    if k_need >= 1.0:                            # le frottement l'arrête avant
        return None
    t = -math.log(1.0 - k_need) / BALL_DRAG
    return None if t > horizon else (need, t)


def goal_line_time(pos, vel, goal_sign: float, horizon: float = THREAT_HORIZON) -> Optional[float]:
    """Temps pour que la balle atteigne la ligne de but en y = goal_sign × 5120, dans le cadre ou non."""
    cross = _line_crossing(pos[1], vel[1], goal_sign, horizon)
    return None if cross is None else cross[1]


def goal_entry_time(pos, vel, goal_sign: float, horizon: float = THREAT_HORIZON) -> Optional[float]:
    """Temps avant que la balle entre dans le but situé en y = goal_sign × 5120 (None si elle n'y va pas, sort du
    cadre ou arrive après `horizon`). Balle déjà sur la ligne dans le cadre → 0."""
    x0, y0, z0 = pos[0], pos[1], pos[2]
    if goal_sign * y0 >= GOAL_Y - R and abs(x0) < POST_X and z0 < BAR_Z:
        return 0.0
    cross = _line_crossing(y0, vel[1], goal_sign, horizon)
    if cross is None:
        return None
    need, t = cross
    x = fold(x0 + vel[0] * need, X_WALL)
    z = ball_z_at(z0, vel[2], t)
    if abs(x) < POST_X and z < BAR_Z:
        return t
    return None


def goal_threat(pos, vel, goal_sign: float, horizon: float = THREAT_HORIZON) -> float:
    """Urgence d'un tir cadré vers le but en y = goal_sign × 5120 : 1 - t/horizon (1 = but imminent), 0 sinon."""
    t = goal_entry_time(pos, vel, goal_sign, horizon)
    return 0.0 if t is None else 1.0 - t / horizon


def _time_to_cover(dist: float, v0: float, boost: float, on_ground: bool) -> float:
    """Temps pour parcourir `dist` en ligne droite depuis la vitesse v0 (≥ 0), boost puis gaz seul."""
    if dist <= 0.0:
        return 0.0
    t, v, d = 0.0, v0, dist
    # phase 1 : boost (au sol : gaz + boost ; en l'air : boost seul)
    tb = boost / BOOST_PER_S
    a = A_BOOST if on_ground else A_AIR_BOOST
    if tb > 0.0 and v < V_BOOST:
        dt_v = (V_BOOST - v) / a
        dt = min(tb, dt_v)
        reach = v * dt + 0.5 * a * dt * dt
        if reach >= d:
            return t + (-v + math.sqrt(v * v + 2.0 * a * d)) / a
        t, d, v = t + dt, d - reach, v + a * dt
    # phase 2 : gaz seul (au sol)
    if on_ground and v < V_THROTTLE:
        a = A_THROTTLE
        dt = (V_THROTTLE - v) / a
        reach = v * dt + 0.5 * a * dt * dt
        if reach >= d:
            return t + (-v + math.sqrt(v * v + 2.0 * a * d)) / a
        t, d, v = t + dt, d - reach, V_THROTTLE
    # phase 3 : vitesse constante
    return t + d / max(v, 300.0)


def _time_to_land(height: float, vz: float) -> float:
    """Temps avant que la voiture retombe au sol (chute libre), plafonné à 1,5 s."""
    h = max(height - 17.0, 0.0)
    return min((vz + math.sqrt(vz * vz + 2.0 * GRAVITY * h)) / GRAVITY, 1.5) if (h > 0.0 or vz > 0.0) else 0.0


def _time_to_cover_airborne(dist: float, v0: float, boost: float, t_land: float) -> float:
    """Voiture en l'air qui va retomber : elle garde sa vitesse jusqu'à l'atterrissage, puis accélère au sol.
    Continu avec le cas au sol au décollage (seule l'accélération perdue pendant le vol compte), contrairement à un
    modèle « sans gaz en l'air » qui faisait bondir l'ETA au plafond à chaque saut (bruit sur `positioning`)."""
    v, d, t = v0, dist, 0.0
    tb = min(boost / BOOST_PER_S, t_land) if v < V_BOOST else 0.0   # boost en l'air pendant le vol
    if tb > 0.0:
        a = A_AIR_BOOST
        dt = min(tb, (V_BOOST - v) / a)
        reach = v * dt + 0.5 * a * dt * dt
        if reach >= d:
            return (-v + math.sqrt(v * v + 2.0 * a * d)) / a
        t, d, v, boost = dt, d - reach, v + a * dt, boost - dt * BOOST_PER_S
    glide = t_land - t
    if v * glide >= d:
        return t + d / max(v, 1e-3)
    return t_land + _time_to_cover(d - v * glide, v, boost, True)


def car_eta(car, pos, vel, iters: int = 2) -> Tuple[float, Tuple[float, float, float]]:
    """(ETA en s, point d'interception) d'une voiture vers la balle (pos, vel : balle, repère du monde).
    Point fixe : temps pour atteindre la balle → où sera la balle à ce temps → nouveau temps…"""
    if car.demo_respawn_timer > 0.0:
        return ETA_CAP, (pos[0], pos[1], pos[2])
    p = car.physics
    px, py, pz = p.position.tolist()
    vx, vy, vz = p.linear_velocity.tolist()
    fx, fy, _ = p.rotation_mtx[:, 0].tolist()              # colonne 0 = avant de la voiture
    fn = math.sqrt(fx * fx + fy * fy)
    boost, ground = float(car.boost_amount), bool(car.on_ground)
    t_land = 0.0 if ground else _time_to_land(pz, vz)
    target = (float(pos[0]), float(pos[1]), float(pos[2]))
    t = 0.0
    for i in range(iters + 1):
        dx, dy = target[0] - px, target[1] - py
        dxy = math.sqrt(dx * dx + dy * dy)
        dz = max(0.0, target[2] - max(pz, 0.0) - 110.0)       # hauteur à gagner (saut / aérienne)
        dist = math.sqrt(dxy * dxy + (1.5 * dz) ** 2) - CONTACT
        if dist <= 0.0:
            t_new = 0.0
        else:
            if dxy > 1e-3:
                ux, uy = dx / dxy, dy / dxy
                v0 = max(0.0, vx * ux + vy * uy)
                cos_a = (fx * ux + fy * uy) / fn if fn > 1e-3 else 1.0
            else:
                v0, cos_a = 0.0, 1.0
            turn = TURN_TIME * 0.5 * (1.0 - cos_a) * (1.0 - min(v0, 1400.0) / 2800.0)
            if ground or (dz > 0.0 and boost > 0.0):          # au sol, ou en vol vers une balle plus haute
                cover = _time_to_cover(dist, v0, boost, ground)
            else:                                             # en l'air, va retomber puis rouler
                cover = _time_to_cover_airborne(dist, v0, boost, t_land)
            t_new = turn + cover + (0.2 if dz > 0.0 else 0.0)
        t_new = min(t_new, ETA_CAP)
        if abs(t_new - t) < 0.02 and i > 0:
            t = t_new
            break
        t = t_new
        target = ball_at(pos, vel, t)
    return t, target


@dataclass
class RaceInfo:
    """Prédictions d'un step (repère du monde)."""
    eta: Dict[Any, float]
    intercept: Dict[Any, Tuple[float, float, float]]
    goal_in: Dict[int, float]        # équipe → urgence d'un tir cadré vers SON but (0 = pas de menace)


def compute_race(state: GameState) -> RaceInfo:
    ball = state.ball
    pos = [float(c) for c in ball.position]
    vel = [float(c) for c in ball.linear_velocity]
    eta, inter = {}, {}
    for a, car in state.cars.items():
        eta[a], inter[a] = car_eta(car, pos, vel)
    # le but du bleu est en -Y, celui de l'orange en +Y
    goal_in = {cv.BLUE_TEAM: goal_threat(pos, vel, -1.0), cv.ORANGE_TEAM: goal_threat(pos, vel, 1.0)}
    return RaceInfo(eta, inter, goal_in)


def race_info(state: GameState, shared_info: Optional[Dict[str, Any]]) -> RaceInfo:
    """RaceInfo du state, calculée une seule fois par step (cache dans shared_info)."""
    b = state.ball.position
    key = (int(getattr(state, "tick_count", 0) or 0), float(b[0]), float(b[1]), float(b[2]), len(state.cars))
    if shared_info is not None:
        cached = shared_info.get("_race")
        if cached is not None and cached[0] == key:
            return cached[1]
    info = compute_race(state)
    if shared_info is not None:
        shared_info["_race"] = (key, info)
    return info


def team_etas(info: RaceInfo, state: GameState, agent) -> Tuple[float, float, float]:
    """(mon ETA, meilleure ETA d'un coéquipier, meilleure ETA adverse) ; ETA_CAP si personne."""
    team = state.cars[agent].team_num
    mine = info.eta[agent]
    mate, opp = ETA_CAP, ETA_CAP
    for o, car in state.cars.items():
        if o == agent:
            continue
        if car.team_num == team:
            mate = min(mate, info.eta[o])
        else:
            opp = min(opp, info.eta[o])
    return mine, mate, opp
