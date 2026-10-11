"""États de départ (ARCHITECTURE §2 et §8) : tirage du format 1v1/2v2/3v3 puis kickoff ou position aléatoire."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator


def _weighted_choice(weights: Dict[Any, float], rng: random.Random):
    keys = list(weights.keys())
    vals = [max(0.0, float(weights[k])) for k in keys]
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"Poids invalides : {weights}")
    return rng.choices(keys, weights=vals, k=1)[0]


class RandomFormatMutator(StateMutator[GameState]):
    """Choisit le nombre de joueurs par équipe (symétrique) et crée les voitures."""

    def __init__(self, format_weights: Dict[int, float], seed: Optional[int] = None):
        for k in format_weights:
            if not 1 <= int(k) <= 3:
                raise ValueError("Formats supportés : 1, 2 ou 3 joueurs par équipe")
        self.format_weights = {int(k): float(v) for k, v in format_weights.items()}
        self.rng = random.Random(seed)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        size = _weighted_choice(self.format_weights, self.rng)
        FixedTeamSizeMutator(blue_size=size, orange_size=size).apply(state, shared_info)
        shared_info["team_size"] = size


class RandomStateMutator(StateMutator[GameState]):
    """Balle et voitures placées au hasard (voitures au sol, balle au sol ou en l'air)."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = self.rng
        margin = 300.0
        ball = state.ball
        ball_z = cv.BALL_RESTING_HEIGHT if r.random() < 0.5 else r.uniform(cv.BALL_RESTING_HEIGHT, cv.CEILING_Z - 300)
        ball.position = np.array([
            r.uniform(-cv.SIDE_WALL_X + margin, cv.SIDE_WALL_X - margin),
            r.uniform(-cv.BACK_WALL_Y + margin, cv.BACK_WALL_Y - margin),
            ball_z,
        ], dtype=np.float32)
        ball.linear_velocity = (r.normal(0, 600, 3) * np.array([1, 1, 0.5])).astype(np.float32)
        ball.angular_velocity = r.normal(0, 1, 3).astype(np.float32)

        for car in state.cars.values():
            car.physics.position = np.array([
                r.uniform(-cv.SIDE_WALL_X + margin, cv.SIDE_WALL_X - margin),
                r.uniform(-cv.BACK_WALL_Y + margin, cv.BACK_WALL_Y - margin),
                17.0,
            ], dtype=np.float32)
            yaw = r.uniform(-np.pi, np.pi)
            speed = r.uniform(0, 1500)
            car.physics.linear_velocity = np.array([np.cos(yaw) * speed, np.sin(yaw) * speed, 0], dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
            car.boost_amount = float(r.uniform(0, 100))


class StartStateMutator(StateMutator[GameState]):
    """Tire le type d'état de départ (kickoff / random) selon des poids configurables."""

    def __init__(self, start_weights: Dict[str, float], seed: Optional[int] = None):
        self.start_weights = dict(start_weights)
        self.rng = random.Random(seed)
        self.mutators = {
            "kickoff": KickoffMutator(),
            "random": RandomStateMutator(seed),
        }
        unknown = set(self.start_weights) - set(self.mutators)
        if unknown:
            raise ValueError(f"Types d'état de départ inconnus : {unknown}")

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        kind = _weighted_choice(self.start_weights, self.rng)
        self.mutators[kind].apply(state, shared_info)
        shared_info["start_kind"] = kind
