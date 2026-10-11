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


def _place_car(car, x: float, y: float, yaw: float, speed: float, boost: float) -> None:
    car.physics.position = np.array([np.clip(x, -3800, 3800), np.clip(y, -4800, 4800), 17.0], dtype=np.float32)
    car.physics.linear_velocity = np.array([np.cos(yaw) * speed, np.sin(yaw) * speed, 0.0], dtype=np.float32)
    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
    car.boost_amount = float(boost)


class AttackDefenseMutator(StateMutator[GameState]):
    """Une équipe tirée au hasard attaque : balle dans le camp adverse, avançant vers le but, attaquants derrière
    la balle ; défenseurs entre la balle et leur but. Symétrique (chaque équipe attaque une fois sur deux)."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = self.rng
        attack_dir = 1.0 if r.random() < 0.5 else -1.0          # +Y : le bleu attaque
        attacking_team = cv.BLUE_TEAM if attack_dir > 0 else cv.ORANGE_TEAM
        bx, by = r.uniform(-2500, 2500), attack_dir * r.uniform(500, 3500)
        ball = state.ball
        ball.position = np.array([bx, by, cv.BALL_RESTING_HEIGHT if r.random() < 0.7 else r.uniform(150, 600)],
                                 dtype=np.float32)
        ball.linear_velocity = np.array([r.normal(0, 300), attack_dir * r.uniform(0, 1200), 0.0], dtype=np.float32)
        ball.angular_velocity = np.zeros(3, dtype=np.float32)
        goal_y = attack_dir * cv.BACK_WALL_Y
        for car in state.cars.values():
            if car.team_num == attacking_team:
                d = r.uniform(500, 2500)
                x, y = bx + r.normal(0, 600), by - attack_dir * d
                yaw = float(np.arctan2(by - y, bx - x) + r.normal(0, 0.4))
                _place_car(car, x, y, yaw, r.uniform(0, 1500), r.uniform(0, 100))
            else:
                f = r.uniform(0.2, 0.9)                         # entre la balle et son but
                x, y = bx * (1 - f) + r.normal(0, 500), by + (goal_y - by) * f
                yaw = float(np.arctan2(by - y, bx - x) + r.normal(0, 0.8))
                _place_car(car, x, y, yaw, r.uniform(0, 1000), r.uniform(0, 100))


class AerialBallMutator(StateMutator[GameState]):
    """Balle en l'air (300-1600 uu) au-dessus d'une voiture tirée au hasard ; les autres voitures au hasard."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.random_state = RandomStateMutator(seed)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = self.rng
        self.random_state.apply(state, shared_info)
        cars = list(state.cars.values())
        car = cars[int(r.integers(len(cars)))]
        p = car.physics.position
        state.ball.position = np.array([np.clip(p[0] + r.normal(0, 400), -3500, 3500),
                                        np.clip(p[1] + r.normal(0, 400), -4500, 4500),
                                        r.uniform(300, 1600)], dtype=np.float32)
        state.ball.linear_velocity = (r.normal(0, 300, 3) * np.array([1, 1, 0.5])).astype(np.float32)
        car.boost_amount = float(r.uniform(30, 100))


class StartStateMutator(StateMutator[GameState]):
    """Tire le type d'état de départ (kickoff / random) selon des poids configurables."""

    def __init__(self, start_weights: Dict[str, float], seed: Optional[int] = None):
        self.start_weights = dict(start_weights)
        self.rng = random.Random(seed)
        self.mutators = {
            "kickoff": KickoffMutator(),
            "random": RandomStateMutator(seed),
            "attack_defense": AttackDefenseMutator(seed),
            "aerial": AerialBallMutator(seed),
        }
        unknown = set(self.start_weights) - set(self.mutators)
        if unknown:
            raise ValueError(f"Types d'état de départ inconnus : {unknown}")

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        kind = _weighted_choice(self.start_weights, self.rng)
        self.mutators[kind].apply(state, shared_info)
        shared_info["start_kind"] = kind


class EmptyGoalMutator(StateMutator[GameState]):
    """Scénario « but vide » (critère S1-20) : 1v1, balle immobile au sol dans la moitié adverse, voiture bleue
    derrière la balle, voiture orange garée loin de son but (coin du camp bleu). Elle ne bouge pas pendant l'éval.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = self.rng
        FixedTeamSizeMutator(blue_size=1, orange_size=1).apply(state, shared_info)
        shared_info["team_size"] = 1
        shared_info["start_kind"] = "empty_goal"
        ball = state.ball
        bx, by = r.uniform(-2500, 2500), r.uniform(0, 3500)
        ball.position = np.array([bx, by, cv.BALL_RESTING_HEIGHT], dtype=np.float32)
        ball.linear_velocity = np.zeros(3, dtype=np.float32)
        ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = 33.0
            if car.team_num == cv.BLUE_TEAM:
                dist, ang = r.uniform(800, 2500), r.uniform(-np.pi / 3, np.pi / 3)
                x = float(np.clip(bx - dist * np.sin(ang), -3800, 3800))
                y = float(np.clip(by - dist * np.cos(ang), -4800, 4800))
                car.physics.position = np.array([x, y, 17.0], dtype=np.float32)
                yaw = float(np.arctan2(by - y, bx - x) + r.uniform(-0.5, 0.5))
            else:
                car.physics.position = np.array([3000.0 * r.choice([-1, 1]), -4300.0, 17.0], dtype=np.float32)
                yaw = float(r.uniform(-np.pi, np.pi))
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
