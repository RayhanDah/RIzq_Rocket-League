"""Utilitaires partagés : perspective d'équipe, distances, rangs de proximité."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import Car, GameState, PhysicsObject

# Dans la perspective normalisée (terrain inversé pour l'orange), on attaque toujours vers +Y.
OPP_GOAL_BACK = np.array([0.0, cv.BACK_NET_Y, cv.GOAL_HEIGHT / 2], dtype=np.float32)
OWN_GOAL_BACK = np.array([0.0, -cv.BACK_NET_Y, cv.GOAL_HEIGHT / 2], dtype=np.float32)


def car_phys(car: Car, inverted: bool) -> PhysicsObject:
    return car.inverted_physics if inverted else car.physics


def ball_phys(state: GameState, inverted: bool) -> PhysicsObject:
    return state.inverted_ball if inverted else state.ball


def norm3(v) -> float:
    """Norme d'un vecteur 3D. ~10× plus rapide que np.linalg.norm sur de si petits vecteurs (même valeur)."""
    x, y, z = v[0], v[1], v[2]
    return math.sqrt(float(x * x + y * y + z * z))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm3(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


def team_proximity_ranks(state: GameState) -> Dict[str, int]:
    """Rang de chaque voiture dans son équipe selon la distance à la balle (0 = 1er homme)."""
    ranks: Dict[str, int] = {}
    ball = state.ball.position
    by_team: Dict[int, List] = {}
    for agent, car in state.cars.items():
        by_team.setdefault(car.team_num, []).append((norm3(car.physics.position - ball), agent))
    for members in by_team.values():
        members.sort(key=lambda t: (t[0], str(t[1])))
        for rank, (_, agent) in enumerate(members):
            ranks[agent] = rank
    return ranks


def is_kickoff_state(state: GameState) -> bool:
    b = state.ball
    return bool(abs(b.position[0]) < 1 and abs(b.position[1]) < 1 and norm3(b.linear_velocity) < 1)
