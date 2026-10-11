"""Observations pour l'acteur à attention (ARCHITECTURE §3).

Vecteur plat par agent, découpé ensuite par le modèle :
    [ soi + global (SELF_DIM) | autres voitures (MAX_OTHERS x OTHER_DIM) | masque (MAX_OTHERS) ]

Tout est exprimé du point de vue de l'équipe : pour l'orange, le terrain est tourné de 180° autour de Z,
donc chaque agent attaque toujours vers +Y. Un même modèle sert en 1v1, 2v2 et 3v3 : les voitures absentes
sont masquées (masque = 0), jamais simplement mises à zéro.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import ObsBuilder
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import Car, GameState

from .common import OPP_GOAL_BACK, OWN_GOAL_BACK, ball_phys, car_phys, is_kickoff_state, team_proximity_ranks

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


def split_dims() -> Tuple[int, int, int]:
    return SELF_DIM, OTHER_DIM, MAX_OTHERS


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


def _perspective(state: GameState, cars: List[Car], scalars: np.ndarray, inverted: bool) -> Dict[str, np.ndarray]:
    """Tableaux (n voitures) dans une perspective d'équipe : positions, vitesses, rotations, états, bloc global."""
    phys = [car_phys(c, inverted) for c in cars]
    pos = np.array([p.position for p in phys], dtype=np.float32).reshape(-1, 3)
    vel = np.array([p.linear_velocity for p in phys], dtype=np.float32).reshape(-1, 3)
    ang = np.array([p.angular_velocity for p in phys], dtype=np.float32).reshape(-1, 3)
    rot = np.array([p.rotation_mtx for p in phys], dtype=np.float32).reshape(-1, 3, 3)
    car_state = np.concatenate([
        pos * POS_COEF, rot[:, :, 0], rot[:, :, 2], vel * CAR_VEL_COEF, ang * ANG_VEL_COEF, scalars,
    ], axis=1).astype(np.float32)
    ball = ball_phys(state, inverted)
    ball_pos = np.asarray(ball.position, dtype=np.float32)
    ball_vel = np.asarray(ball.linear_velocity, dtype=np.float32)
    pads = state.inverted_boost_pad_timers if inverted else state.boost_pad_timers
    glob = np.concatenate([
        ball_pos * POS_COEF,
        ball_vel * BALL_VEL_COEF,
        np.asarray(ball.angular_velocity, dtype=np.float32) * BALL_ANG_VEL_COEF,
        np.asarray(pads, dtype=np.float32) * PAD_COEF,
        (OPP_GOAL_BACK - ball_pos) * REL_COEF,
        (OWN_GOAL_BACK - ball_pos) * REL_COEF,
    ]).astype(np.float32)
    ball_dist = (np.linalg.norm(ball_pos[None, :] - pos, axis=1) * REL_COEF).astype(np.float32)
    return {"pos": pos, "vel": vel, "rot": rot, "car_state": car_state, "ball_pos": ball_pos,
            "ball_vel": ball_vel, "global": glob, "ball_dist": ball_dist}


class AttentionObs(ObsBuilder[str, np.ndarray, GameState, Tuple[str, int]]):
    def get_obs_space(self, agent: str) -> Tuple[str, int]:
        return "real", OBS_DIM

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Version optimisée : mêmes valeurs que `build_obs_reference` (test d'équivalence).

        Les états des voitures sont calculés une fois par perspective (bleu / orange) au lieu d'une fois
        par couple (agent, autre voiture), et les jetons des autres voitures sont construits en bloc.
        """
        if len(state.cars) <= 2:  # 1v1 : la version d'origine est plus rapide (moins de surcoût numpy)
            return self.build_obs_reference(agents, state, shared_info)
        ids = sorted(state.cars.keys(), key=str)
        cars = [state.cars[a] for a in ids]
        n = len(ids)
        index = {a: i for i, a in enumerate(ids)}
        teams = np.array([c.team_num for c in cars])
        ranks = team_proximity_ranks(state)
        rank_arr = np.array([ranks.get(a, 0) for a in ids], dtype=np.float32) / 2.0
        kickoff = float(is_kickoff_state(state))
        prev_actions = shared_info.get("prev_actions", {})
        team_sizes: Dict[int, int] = {}
        for c in cars:
            team_sizes[c.team_num] = team_sizes.get(c.team_num, 0) + 1
        scalars = np.array([[c.boost_amount / 100.0, float(c.on_ground), min(c.demo_respawn_timer, 3.0) / 3.0,
                             float(c.is_boosting), float(c.is_supersonic), float(c.has_flip)] for c in cars],
                           dtype=np.float32).reshape(n, 6)

        persp: Dict[bool, Dict[str, np.ndarray]] = {}
        out: Dict[str, np.ndarray] = {}
        for agent in agents:
            i = index[agent]
            car = cars[i]
            inv = car.team_num == cv.ORANGE_TEAM
            p = persp.get(inv)
            if p is None:
                p = persp[inv] = _perspective(state, cars, scalars, inv)
            pos, vel, rot = p["pos"], p["vel"], p["rot"][i]
            ball_pos, ball_vel = p["ball_pos"], p["ball_vel"]
            rel_ball = ball_pos - pos[i]
            n_mates = team_sizes.get(car.team_num, 1)
            n_opps = sum(v for k, v in team_sizes.items() if k != car.team_num)
            prev = prev_actions.get(agent)
            if prev is None:
                prev = np.zeros(8, dtype=np.float32)

            vec = np.empty(OBS_DIM, dtype=np.float32)
            o = 0
            vec[o:o + CAR_STATE_DIM] = p["car_state"][i]; o += CAR_STATE_DIM
            vec[o:o + 3] = (rot.T @ vel[i]) * CAR_VEL_COEF; o += 3
            vec[o:o + 10] = (float(car.has_jumped), float(car.is_jumping), float(car.has_flipped),
                             float(car.is_flipping), float(car.has_double_jumped), float(car.can_flip),
                             min(car.air_time_since_jump, 1.25) / 1.25, min(car.jump_time, 0.2) / 0.2,
                             float(car.handbrake), float(car.is_holding_jump)); o += 10
            vec[o:o + 3] = rel_ball * REL_COEF; o += 3
            vec[o:o + 3] = (rot.T @ rel_ball) * REL_COEF; o += 3
            vec[o:o + 3] = (ball_vel - vel[i]) * BALL_VEL_COEF; o += 3
            vec[o] = np.linalg.norm(rel_ball) * REL_COEF; o += 1
            vec[o:o + 8] = np.asarray(prev, dtype=np.float32).reshape(-1)[:8]; o += 8
            vec[o:o + 4] = (rank_arr[i], n_mates / 3.0, n_opps / 3.0, kickoff); o += 4
            vec[o:o + GLOBAL_DIM] = p["global"]; o += GLOBAL_DIM

            # autres voitures : coéquipiers d'abord, puis adversaires, chacun dans l'ordre des ids
            others = [j for j in range(n) if j != i and teams[j] == car.team_num] +                      [j for j in range(n) if j != i and teams[j] != car.team_num]
            others = others[:MAX_OTHERS]
            block = np.zeros((MAX_OTHERS, OTHER_DIM), dtype=np.float32)
            k = len(others)
            if k:
                js = np.array(others)
                is_mate = (teams[js] == car.team_num).astype(np.float32)
                rel = pos[js] - pos[i]
                block[:k, 0] = is_mate
                block[:k, 1] = 1.0 - is_mate
                c = 2
                block[:k, c:c + CAR_STATE_DIM] = p["car_state"][js]; c += CAR_STATE_DIM
                block[:k, c:c + 3] = rel * REL_COEF; c += 3
                block[:k, c:c + 3] = (vel[js] - vel[i]) * CAR_VEL_COEF; c += 3
                block[:k, c:c + 3] = (rel @ rot) * REL_COEF; c += 3
                block[:k, c] = p["ball_dist"][js]; c += 1
                block[:k, c] = rank_arr[js]
            vec[o:o + MAX_OTHERS * OTHER_DIM] = block.reshape(-1); o += MAX_OTHERS * OTHER_DIM
            vec[o:o + MAX_OTHERS] = 0.0
            vec[o:o + k] = 1.0
            out[agent] = vec
        return out

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
            np.array([np.linalg.norm(rel_ball) * REL_COEF], dtype=np.float32),
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
                np.array([np.linalg.norm(ball.position - op.position) * REL_COEF,
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
