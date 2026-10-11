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


class AttentionObs(ObsBuilder[str, np.ndarray, GameState, Tuple[str, int]]):
    def get_obs_space(self, agent: str) -> Tuple[str, int]:
        return "real", OBS_DIM

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
