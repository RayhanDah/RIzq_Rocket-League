"""Observations pour l'acteur à attention (ARCHITECTURE §3).

Vecteur plat par agent, découpé ensuite par le modèle :
    [ soi + global (SELF_DIM) | autres voitures (MAX_OTHERS x OTHER_DIM) | masque (MAX_OTHERS) ]

Tout est exprimé du point de vue de l'équipe : pour l'orange, le terrain est tourné de 180° autour de Z,
donc chaque agent attaque toujours vers +Y. Un même modèle sert en 1v1, 2v2 et 3v3 : les voitures absentes
sont masquées (masque = 0), jamais simplement mises à zéro.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import ObsBuilder
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import Car, GameState

from .common import (OPP_GOAL_BACK, OWN_GOAL_BACK, ball_phys, car_phys, is_kickoff_state, norm3,
                     team_proximity_ranks)
from .predict import ETA_CAP, RaceInfo, race_info, team_etas

MAX_PLAYERS_PER_TEAM = 3
MAX_OTHERS = 2 * MAX_PLAYERS_PER_TEAM - 1

POS_COEF = np.array([1 / cv.SIDE_WALL_X, 1 / cv.BACK_NET_Y, 1 / cv.CEILING_Z], dtype=np.float32)
REL_COEF = 1 / 2300.0
CAR_VEL_COEF = 1 / cv.CAR_MAX_SPEED
BALL_VEL_COEF = 1 / cv.BALL_MAX_SPEED
ANG_VEL_COEF = 1 / cv.CAR_MAX_ANG_VEL
BALL_ANG_VEL_COEF = 1 / 6.0
PAD_COEF = 1 / 10.0
N_PADS = len(cv.BOOST_LOCATIONS)

CAR_STATE_DIM = 3 + 3 + 3 + 3 + 3 + 6       # pos, fwd, up, vel, angvel, [boost, ground, demo, boosting, ssonic, flip]
SELF_EXTRA_DIM = 3 + 10 + 3 + 3 + 3 + 1 + 8 + 4   # vel locale, états saut/flip, relatif balle (monde, local, vitesse), dist, action préc., contexte
GLOBAL_DIM = 3 + 3 + 3 + N_PADS + 3 + 3          # balle pos/vel/angvel, pads, balle→buts
SELF_DIM = CAR_STATE_DIM + SELF_EXTRA_DIM + GLOBAL_DIM
OTHER_DIM = 2 + CAR_STATE_DIM + 3 + 3 + 3 + 1 + 1  # rôle, état, rel pos (monde), rel vel, rel pos (local), dist balle, rang
OBS_DIM = SELF_DIM + MAX_OTHERS * OTHER_DIM + MAX_OTHERS

# Prédiction de trajectoire de la balle (option `env.ball_prediction`) : BALL_PRED_DIM features ajoutées à la FIN du
# bloc « soi + global ». Sans l'option, l'observation est exactement celle d'origine (OBS_DIM).
BALL_PRED_DIM = 14
BALL_PRED_TIMES = (0.5, 1.0, 1.5)
BALL_PRED_HORIZON = 3.0
GRAVITY = 650.0
BOUNCE = 0.6           # restitution verticale approchée au rebond au sol / aux murs


# « Course à la balle » (option `env.race_features`, exige `env.ball_prediction`) : RACE_DIM features ajoutées APRÈS
# la prédiction de balle (predict.py) : mon ETA, meilleure ETA d'un coéquipier / adverse, avance sur l'adversaire et
# sur le coéquipier, point d'interception, urgence d'un tir cadré vers notre but / le but adverse.
RACE_DIM = 10
RACE_DIFF_CAP = 2.0

# Pads de boost « utiles » (option `env.boost_features`, s7e, exige `env.race_features`) : BOOST_DIM features ajoutées
# APRÈS la course à la balle. Les 34 minuteurs du bloc global sont rangés par numéro de pad : le réseau devait apprendre
# seul où est chaque pad par rapport à lui (s7c : à 0 de boost 75 % du temps, presque jamais sur les gros pads). Ici,
# les 2 gros pads et le petit pad qu'on peut prendre le plus tôt, dans le repère de la voiture.
BOOST_DIM = 12
BOOST_SLOTS = (True, True, False)              # gros, gros, petit
BOOST_REACH_SPEED = 1400.0                     # uu/s : vitesse supposée pour estimer « quand je peux le prendre »
_PADS = np.asarray(cv.BOOST_LOCATIONS, dtype=np.float64)
_BIG = _PADS[:, 2] > 71.0                      # gros pads : z = 73 (petits : 70)


def self_dim(ball_prediction: bool = False, race_features: bool = False, boost_features: bool = False) -> int:
    if race_features and not ball_prediction:
        raise ValueError("env.race_features exige env.ball_prediction (les features sont rangées après)")
    if boost_features and not race_features:
        raise ValueError("env.boost_features exige env.race_features (les features sont rangées après)")
    return (SELF_DIM + (BALL_PRED_DIM if ball_prediction else 0) + (RACE_DIM if race_features else 0)
            + (BOOST_DIM if boost_features else 0))


def obs_dim(ball_prediction: bool = False, race_features: bool = False, boost_features: bool = False) -> int:
    return self_dim(ball_prediction, race_features, boost_features) + MAX_OTHERS * OTHER_DIM + MAX_OTHERS


def env_obs_flags(env_cfg: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """(ball_prediction, race_features, boost_features) d'une section `env` de config."""
    return (bool(env_cfg.get("ball_prediction", False)), bool(env_cfg.get("race_features", False)),
            bool(env_cfg.get("boost_features", False)))


def boost_pad_features(pos: np.ndarray, fwd: np.ndarray, right: np.ndarray, timers: np.ndarray) -> List[float]:
    """Pads de boost à prendre, dans une perspective d'équipe (positions et minuteurs déjà tournés pour l'orange).

    Coût d'un pad = max(distance / 1400 uu/s, temps avant réapparition) : un pad qui réapparaît avant qu'on y soit
    compte comme disponible. Par emplacement (2 gros, 1 petit, les moins coûteux) :
    [position locale avant, à droite (/2300), coût / 6 s (borné à 1), minuteur / 10]."""
    rel = _PADS - pos
    dist = np.sqrt((rel * rel).sum(axis=1))
    cost = np.maximum(dist / BOOST_REACH_SPEED, np.asarray(timers, dtype=np.float64))
    out: List[float] = []
    taken = set()
    for big in BOOST_SLOTS:
        cand = np.where(_BIG == big)[0]
        order = cand[np.argsort(cost[cand], kind="stable")]
        i = next(int(j) for j in order if int(j) not in taken)
        taken.add(i)
        out += [float(rel[i] @ fwd) * REL_COEF, float(rel[i] @ right) * REL_COEF, min(float(cost[i]) / ETA_CAP, 1.0),
                float(timers[i]) * PAD_COEF]
    return out


def race_features(state: GameState, agent: str, info: RaceInfo) -> List[float]:
    """Features « qui touche en premier ? » d'un agent, dans la perspective de son équipe (RACE_DIM valeurs)."""
    team = state.cars[agent].team_num
    mine, mate, opp = team_etas(info, state, agent)
    ix, iy, iz = info.intercept[agent]
    if team == cv.ORANGE_TEAM:
        ix, iy = -ix, -iy
    c = RACE_DIFF_CAP
    return [mine / ETA_CAP, mate / ETA_CAP, opp / ETA_CAP,
            min(max(opp - mine, -c), c) / c, min(max(mate - mine, -c), c) / c,
            ix * _PX, iy * _PY, iz * _PZ,
            info.goal_in.get(team, 0.0), info.goal_in.get(1 - team, 0.0)]


def split_dims() -> Tuple[int, int, int]:
    return SELF_DIM, OTHER_DIM, MAX_OTHERS


def _fold(x: float, limit: float) -> float:
    """Réflexion sur des murs en ±limit (onde triangulaire) : trajectoire horizontale avec rebonds, sans amortissement."""
    period = 4.0 * limit
    return limit - abs(math.fmod(math.fmod(x + limit, period) + period, period) - 2.0 * limit)


_R = float(cv.BALL_RADIUS)
_XW, _YW, _ZC = float(cv.SIDE_WALL_X) - _R, float(cv.BACK_WALL_Y) - _R, float(cv.CEILING_Z) - _R
_PX, _PY, _PZ = (float(c) for c in POS_COEF)


def ball_prediction_features(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Trajectoire ANALYTIQUE de la balle (gravité, un rebond au sol, murs en miroir ; ni frottement ni effets) dans la
    perspective donnée. Peu coûteux (pas de simulation, calcul scalaire). Fiable ~1-1,5 s ; au-delà, indicatif.

    [temps avant le sol, x/y au sol, hauteur max, temps de la hauteur max, puis x/y/z à 0,5 / 1 / 1,5 s] (14 valeurs)."""
    x0, y0, z0 = float(pos[0]), float(pos[1]), max(float(pos[2]), _R)
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    g = GRAVITY
    h = z0 - _R
    if h < 1.0 and abs(vz) < 1.0:                      # balle qui roule
        t_land, v_impact = 0.0, 0.0
    else:
        t_land = (vz + math.sqrt(vz * vz + 2.0 * g * h)) / g
        v_impact = g * t_land - vz                     # vitesse verticale (vers le bas) à l'impact
    if vz > 0:
        t_apex, z_apex = vz / g, z0 + vz * vz / (2.0 * g)
    else:
        t_apex, z_apex = 0.0, z0
    z_apex = min(z_apex, _ZC)
    feats = [min(t_land, BALL_PRED_HORIZON) / BALL_PRED_HORIZON,
             _fold(x0 + vx * t_land, _XW) * _PX, _fold(y0 + vy * t_land, _YW) * _PY,
             z_apex * _PZ, min(t_apex, BALL_PRED_HORIZON) / BALL_PRED_HORIZON]
    vb = BOUNCE * v_impact
    for t in BALL_PRED_TIMES:
        if t <= t_land:
            z = z0 + vz * t - 0.5 * g * t * t
        else:
            dt = t - t_land
            z = _R + max(0.0, vb * dt - 0.5 * g * dt * dt)
        z = min(max(z, _R), _ZC)
        feats += [_fold(x0 + vx * t, _XW) * _PX, _fold(y0 + vy * t, _YW) * _PY, z * _PZ]
    return np.asarray(feats, dtype=np.float32)


def _car_state(car: Car, inverted: bool) -> np.ndarray:
    p = car_phys(car, inverted)
    return np.concatenate([
        p.position * POS_COEF,
        p.forward,
        p.up,
        p.linear_velocity * CAR_VEL_COEF,
        p.angular_velocity * ANG_VEL_COEF,
        np.array([
            car.boost_amount / 100.0,
            float(car.on_ground),
            min(car.demo_respawn_timer, 3.0) / 3.0,
            float(car.is_boosting),
            float(car.is_supersonic),
            float(car.has_flip),
        ], dtype=np.float32),
    ]).astype(np.float32)


class AttentionObs(ObsBuilder[str, np.ndarray, GameState, Tuple[str, int]]):
    def __init__(self, ball_prediction: bool = False, race_features: bool = False, boost_features: bool = False):
        self.ball_prediction = bool(ball_prediction)
        self.race_features = bool(race_features)
        self.boost_features = bool(boost_features)
        self_dim(self.ball_prediction, self.race_features, self.boost_features)       # valide la combinaison

    def get_obs_space(self, agent: str) -> Tuple[str, int]:
        return "real", obs_dim(self.ball_prediction, self.race_features, self.boost_features)

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = self.build_obs_base(agents, state, shared_info)
        if not self.ball_prediction:
            return out
        info = race_info(state, shared_info) if self.race_features else None
        pred = {}
        for agent in agents:
            inv = state.cars[agent].team_num == cv.ORANGE_TEAM
            if inv not in pred:
                b = ball_phys(state, inv)
                pred[inv] = ball_prediction_features(b.position, b.linear_velocity)
            v = out[agent]
            extra = [pred[inv]]
            if info is not None:
                extra.append(np.asarray(race_features(state, agent, info), dtype=np.float32))
            if self.boost_features:
                p = car_phys(state.cars[agent], inv)
                rot = p.rotation_mtx                   # colonnes = forward, right, up
                pads = state.inverted_boost_pad_timers if inv else state.boost_pad_timers
                extra.append(np.asarray(boost_pad_features(np.asarray(p.position, dtype=np.float64), rot[:, 0],
                                                           rot[:, 1], pads), dtype=np.float32))
            out[agent] = np.concatenate([v[:SELF_DIM], *extra, v[SELF_DIM:]])
        return out

    def build_obs_base(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Version rapide : mêmes valeurs que `build_obs_reference` (tests d'équivalence), calculées en arithmétique
        Python sur des listes (numpy est lent sur des vecteurs de 3 valeurs). Chaque voiture est calculée UNE fois par
        perspective (bleu / orange) puis réutilisée comme « soi » et comme « autre voiture »."""
        ids = sorted(state.cars.keys(), key=str)
        ranks = team_proximity_ranks(state)
        kickoff = float(is_kickoff_state(state))
        prev_actions = shared_info.get("prev_actions", {})
        team_sizes: Dict[int, int] = {}
        for c in state.cars.values():
            team_sizes[c.team_num] = team_sizes.get(c.team_num, 0) + 1
        persp: Dict[bool, tuple] = {}
        out: Dict[str, np.ndarray] = {}
        for agent in agents:
            car = state.cars[agent]
            inv = car.team_num == cv.ORANGE_TEAM
            if inv not in persp:
                persp[inv] = self._perspective_lists(state, ids, inv)
            cars, ball_pos, ball_vel, glob = persp[inv]
            me = cars[agent]
            px, py, pz = me["pos"]
            vx, vy, vz = me["vel"]
            f, r, u = me["fwd"], me["right"], me["up"]
            bx, by, bz = ball_pos
            rbx, rby, rbz = bx - px, by - py, bz - pz
            n_mates = team_sizes.get(car.team_num, 1)
            n_opps = sum(v for k, v in team_sizes.items() if k != car.team_num)
            prev = prev_actions.get(agent)
            prev = [0.0] * 8 if prev is None else np.asarray(prev, dtype=np.float32).reshape(-1)[:8].tolist()
            vals = list(me["state"])
            vals += [(f[0] * vx + f[1] * vy + f[2] * vz) * CAR_VEL_COEF, (r[0] * vx + r[1] * vy + r[2] * vz) * CAR_VEL_COEF,
                     (u[0] * vx + u[1] * vy + u[2] * vz) * CAR_VEL_COEF]
            vals += [float(car.has_jumped), float(car.is_jumping), float(car.has_flipped), float(car.is_flipping),
                     float(car.has_double_jumped), float(car.can_flip), min(car.air_time_since_jump, 1.25) / 1.25,
                     min(car.jump_time, 0.2) / 0.2, float(car.handbrake), float(car.is_holding_jump)]
            vals += [rbx * REL_COEF, rby * REL_COEF, rbz * REL_COEF,
                     (f[0] * rbx + f[1] * rby + f[2] * rbz) * REL_COEF, (r[0] * rbx + r[1] * rby + r[2] * rbz) * REL_COEF,
                     (u[0] * rbx + u[1] * rby + u[2] * rbz) * REL_COEF,
                     (ball_vel[0] - vx) * BALL_VEL_COEF, (ball_vel[1] - vy) * BALL_VEL_COEF,
                     (ball_vel[2] - vz) * BALL_VEL_COEF,
                     math.sqrt(rbx * rbx + rby * rby + rbz * rbz) * REL_COEF]
            vals += prev
            vals += [ranks.get(agent, 0) / 2.0, n_mates / 3.0, n_opps / 3.0, kickoff]
            vals += glob
            mates = [o for o in ids if o != agent and state.cars[o].team_num == car.team_num]
            opps = [o for o in ids if o != agent and state.cars[o].team_num != car.team_num]
            others = (mates + opps)[:MAX_OTHERS]
            for o in others:
                oc = cars[o]
                is_mate = state.cars[o].team_num == car.team_num
                ox, oy, oz = oc["pos"]
                rx, ry, rz = ox - px, oy - py, oz - pz
                ovx, ovy, ovz = oc["vel"]
                dbx, dby, dbz = bx - ox, by - oy, bz - oz
                vals += [float(is_mate), float(not is_mate)]
                vals += oc["state"]
                vals += [rx * REL_COEF, ry * REL_COEF, rz * REL_COEF,
                         (ovx - vx) * CAR_VEL_COEF, (ovy - vy) * CAR_VEL_COEF, (ovz - vz) * CAR_VEL_COEF,
                         (f[0] * rx + f[1] * ry + f[2] * rz) * REL_COEF, (r[0] * rx + r[1] * ry + r[2] * rz) * REL_COEF,
                         (u[0] * rx + u[1] * ry + u[2] * rz) * REL_COEF,
                         math.sqrt(dbx * dbx + dby * dby + dbz * dbz) * REL_COEF, ranks.get(o, 0) / 2.0]
            k = len(others)
            vals += [0.0] * ((MAX_OTHERS - k) * OTHER_DIM)
            vals += [1.0] * k + [0.0] * (MAX_OTHERS - k)
            out[agent] = np.array(vals, dtype=np.float32)
        return out

    @staticmethod
    def _perspective_lists(state: GameState, ids: List[str], inverted: bool):
        """Voitures et balle dans une perspective d'équipe, en listes Python (calculées une fois par step)."""
        cars = {}
        for a in ids:
            car = state.cars[a]
            p = car_phys(car, inverted)
            pos, vel, ang = p.position.tolist(), p.linear_velocity.tolist(), p.angular_velocity.tolist()
            rot = p.rotation_mtx.tolist()             # colonnes = forward, right, up
            fwd = [rot[0][0], rot[1][0], rot[2][0]]
            right = [rot[0][1], rot[1][1], rot[2][1]]
            up = [rot[0][2], rot[1][2], rot[2][2]]
            st = [pos[0] * _PX, pos[1] * _PY, pos[2] * _PZ, *fwd, *up,
                  vel[0] * CAR_VEL_COEF, vel[1] * CAR_VEL_COEF, vel[2] * CAR_VEL_COEF,
                  ang[0] * ANG_VEL_COEF, ang[1] * ANG_VEL_COEF, ang[2] * ANG_VEL_COEF,
                  car.boost_amount / 100.0, float(car.on_ground), min(car.demo_respawn_timer, 3.0) / 3.0,
                  float(car.is_boosting), float(car.is_supersonic), float(car.has_flip)]
            cars[a] = {"pos": pos, "vel": vel, "fwd": fwd, "right": right, "up": up, "state": st}
        ball = ball_phys(state, inverted)
        bp, bv, ba = ball.position.tolist(), ball.linear_velocity.tolist(), ball.angular_velocity.tolist()
        pads = (state.inverted_boost_pad_timers if inverted else state.boost_pad_timers)
        og, ow = OPP_GOAL_BACK.tolist(), OWN_GOAL_BACK.tolist()
        glob = [bp[0] * _PX, bp[1] * _PY, bp[2] * _PZ,
                bv[0] * BALL_VEL_COEF, bv[1] * BALL_VEL_COEF, bv[2] * BALL_VEL_COEF,
                ba[0] * BALL_ANG_VEL_COEF, ba[1] * BALL_ANG_VEL_COEF, ba[2] * BALL_ANG_VEL_COEF]
        glob += (np.asarray(pads, dtype=np.float32) * PAD_COEF).tolist()
        glob += [(og[i] - bp[i]) * REL_COEF for i in range(3)] + [(ow[i] - bp[i]) * REL_COEF for i in range(3)]
        return cars, bp, bv, glob

    def build_obs_reference(self, agents: List[str], state: GameState,
                            shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Implémentation d'origine (lisible, lente), conservée comme référence pour les tests d'équivalence."""
        ranks = team_proximity_ranks(state)
        kickoff = float(is_kickoff_state(state))
        prev_actions = shared_info.get("prev_actions", {})
        team_sizes: Dict[int, int] = {}
        for car in state.cars.values():
            team_sizes[car.team_num] = team_sizes.get(car.team_num, 0) + 1
        return {
            agent: self._build(agent, state, ranks, kickoff, prev_actions.get(agent), team_sizes)
            for agent in agents
        }

    def _build(self, agent: str, state: GameState, ranks: Dict[str, int], kickoff: float,
               prev_action, team_sizes: Dict[int, int]) -> np.ndarray:
        car = state.cars[agent]
        inv = car.team_num == cv.ORANGE_TEAM
        me = car_phys(car, inv)
        ball = ball_phys(state, inv)
        pads = state.inverted_boost_pad_timers if inv else state.boost_pad_timers
        rot = me.rotation_mtx  # colonnes = forward, right, up

        rel_ball = ball.position - me.position
        n_mates = team_sizes.get(car.team_num, 1)
        n_opps = sum(v for k, v in team_sizes.items() if k != car.team_num)
        if prev_action is None:
            prev_action = np.zeros(8, dtype=np.float32)

        self_vec = np.concatenate([
            _car_state(car, inv),
            (rot.T @ me.linear_velocity) * CAR_VEL_COEF,
            np.array([
                float(car.has_jumped), float(car.is_jumping), float(car.has_flipped), float(car.is_flipping),
                float(car.has_double_jumped), float(car.can_flip),
                min(car.air_time_since_jump, 1.25) / 1.25, min(car.jump_time, 0.2) / 0.2,
                float(car.handbrake), float(car.is_holding_jump),
            ], dtype=np.float32),
            rel_ball * REL_COEF,
            (rot.T @ rel_ball) * REL_COEF,
            (ball.linear_velocity - me.linear_velocity) * BALL_VEL_COEF,
            np.array([norm3(rel_ball) * REL_COEF], dtype=np.float32),
            np.asarray(prev_action, dtype=np.float32).reshape(-1)[:8],
            np.array([ranks.get(agent, 0) / 2.0, n_mates / 3.0, n_opps / 3.0, kickoff], dtype=np.float32),
            # global
            ball.position * POS_COEF,
            ball.linear_velocity * BALL_VEL_COEF,
            ball.angular_velocity * BALL_ANG_VEL_COEF,
            np.asarray(pads, dtype=np.float32) * PAD_COEF,
            (OPP_GOAL_BACK - ball.position) * REL_COEF,
            (OWN_GOAL_BACK - ball.position) * REL_COEF,
        ]).astype(np.float32)

        mates, opps = [], []
        for other_id in sorted(state.cars.keys(), key=str):
            if other_id == agent:
                continue
            other = state.cars[other_id]
            op = car_phys(other, inv)
            is_mate = other.team_num == car.team_num
            rel = op.position - me.position
            token = np.concatenate([
                np.array([float(is_mate), float(not is_mate)], dtype=np.float32),
                _car_state(other, inv),
                rel * REL_COEF,
                (op.linear_velocity - me.linear_velocity) * CAR_VEL_COEF,
                (rot.T @ rel) * REL_COEF,
                np.array([norm3(ball.position - op.position) * REL_COEF,
                          ranks.get(other_id, 0) / 2.0], dtype=np.float32),
            ]).astype(np.float32)
            (mates if is_mate else opps).append(token)

        others = (mates + opps)[:MAX_OTHERS]
        other_block = np.zeros((MAX_OTHERS, OTHER_DIM), dtype=np.float32)
        mask = np.zeros(MAX_OTHERS, dtype=np.float32)
        for i, tok in enumerate(others):
            other_block[i] = tok
            mask[i] = 1.0

        out = np.concatenate([self_vec, other_block.reshape(-1), mask])
        assert out.shape[0] == OBS_DIM, (out.shape, OBS_DIM)
        return out
