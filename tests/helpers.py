"""Construction d'états de jeu synthétiques (sans RocketSim) pour les tests."""
from __future__ import annotations

import random

import numpy as np
from rlgym.rocket_league.api import GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator


def make_state(team_size: int = 1, kickoff: bool = True, seed: int = 0) -> GameState:
    random.seed(seed)
    s = GameState()
    s.tick_count = 0
    s.goal_scored = False
    s.config = GameConfig()
    s.config.gravity = 1
    s.config.boost_consumption = 1
    s.config.dodge_deadzone = 0.5
    s.ball = PhysicsObject()
    s.cars = {}
    s.boost_pad_timers = np.zeros(34, dtype=np.float32)
    FixedTeamSizeMutator(team_size, team_size).apply(s, {})
    if kickoff:
        KickoffMutator().apply(s, {})
    return s


def mirror_car_into(src_state: GameState, src_id: str, dst_state: GameState, dst_id: str) -> None:
    """Place dst_id dans dst_state comme l'image de src_id par rotation de 180° autour de Z."""
    src = src_state.cars[src_id]
    dst = dst_state.cars[dst_id]
    inv = src.physics.inverted()
    dst.physics = PhysicsObject()
    dst.physics.position = inv.position.copy()
    dst.physics.linear_velocity = inv.linear_velocity.copy()
    dst.physics.angular_velocity = inv.angular_velocity.copy()
    dst.physics.rotation_mtx = inv.rotation_mtx.copy()
    for attr in ("boost_amount", "demo_respawn_timer", "wheels_with_contact", "supersonic_time", "boost_active_time",
                 "handbrake", "has_jumped", "is_holding_jump", "is_jumping", "jump_time", "has_flipped",
                 "has_double_jumped", "air_time_since_jump", "flip_time", "flip_torque", "is_autoflipping",
                 "autoflip_timer", "autoflip_direction", "ball_touches", "bump_victim_id", "hitbox_type"):
        setattr(dst, attr, getattr(src, attr))
