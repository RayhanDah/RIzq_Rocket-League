"""Moteur de substitution très simplifié (cinématique 2D), SANS RocketSim ni collision meshes.

Uniquement pour les tests : il permet de faire tourner toute la chaîne (mutators, obs, rewards, workers,
collecteur, PPO/SPO, checkpoints) n'importe où, et de vérifier qu'un agent APPREND (aller à la balle, marquer).
Il ne remplace évidemment pas RocketSim pour le vrai entraînement.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from rlgym.api import RLGym, TransitionEngine
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.state_mutators import MutatorSequence

DT = 1 / 120


class FakeEngine(TransitionEngine[str, GameState, np.ndarray]):
    def __init__(self):
        self._state: GameState = None
        self._agents: List[str] = []

    @property
    def agents(self) -> List[str]:
        return list(self._agents)

    @property
    def max_num_agents(self) -> int:
        return 6

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return {}

    @config.setter
    def config(self, value: Dict[str, Any]):
        pass

    def create_base_state(self) -> GameState:
        s = GameState()
        s.tick_count = 0
        s.goal_scored = False
        s.config = GameConfig()
        s.config.gravity = 1
        s.config.boost_consumption = 1
        s.config.dodge_deadzone = 0.5
        s.ball = PhysicsObject()
        s.cars = {}
        s.boost_pad_timers = np.zeros(len(cv.BOOST_LOCATIONS), dtype=np.float32)
        return s

    def set_state(self, desired_state: GameState, shared_info: Dict[str, Any]) -> GameState:
        s = desired_state
        s.ball.position = np.asarray(s.ball.position, dtype=np.float32).copy()
        s.ball.position[2] = cv.BALL_RESTING_HEIGHT
        s.ball.linear_velocity = np.asarray(s.ball.linear_velocity, dtype=np.float32).copy()
        s.ball.linear_velocity[2] = 0
        s.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in s.cars.values():
            car.physics.position = np.asarray(car.physics.position, dtype=np.float32).copy()
            car.physics.linear_velocity = np.asarray(car.physics.linear_velocity, dtype=np.float32).copy()
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            yaw = float(car.physics.euler_angles[1])
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
            car.ball_touches = 0
            car.bump_victim_id = None
        self._state = s
        self._agents = list(s.cars.keys())
        return s

    def step(self, actions: Dict[str, np.ndarray], shared_info: Dict[str, Any]) -> GameState:
        s = self._state
        n_ticks = next(iter(actions.values())).shape[0] if actions else 1
        touches = {a: 0 for a in s.cars}
        for t in range(n_ticks):
            for a, car in s.cars.items():
                throttle, steer, boost = float(actions[a][t, 0]), float(actions[a][t, 1]), float(actions[a][t, 6])
                yaw = float(car.physics.euler_angles[1]) + steer * 3.0 * DT
                speed = float(np.dot(car.physics.linear_velocity[:2], [np.cos(yaw), np.sin(yaw)]))
                accel = throttle * 1600 + (1000 if boost > 0 and car.boost_amount > 0 else 0) - 0.5 * speed
                speed = float(np.clip(speed + accel * DT, -1400, cv.CAR_MAX_SPEED))
                if boost > 0:
                    car.boost_amount = max(0.0, car.boost_amount - 33.3 * DT)
                vel = np.array([np.cos(yaw) * speed, np.sin(yaw) * speed, 0.0], dtype=np.float32)
                pos = car.physics.position + vel * DT
                pos[0] = np.clip(pos[0], -cv.SIDE_WALL_X + 50, cv.SIDE_WALL_X - 50)
                pos[1] = np.clip(pos[1], -cv.BACK_WALL_Y + 50, cv.BACK_WALL_Y - 50)
                car.physics.position = pos.astype(np.float32)
                car.physics.linear_velocity = vel
                car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
                rel = s.ball.position[:2] - car.physics.position[:2]
                if np.linalg.norm(rel) < 160:
                    touches[a] += 1
                    d = rel / (np.linalg.norm(rel) + 1e-6)
                    s.ball.linear_velocity[:2] = d * max(500.0, abs(speed) * 1.4)
            b = s.ball
            b.linear_velocity[:2] *= (1 - 0.3 * DT)
            b.position = (b.position + b.linear_velocity * DT).astype(np.float32)
            for i, lim in ((0, cv.SIDE_WALL_X - cv.BALL_RADIUS), (1, cv.BACK_WALL_Y - cv.BALL_RADIUS)):
                if abs(b.position[i]) > lim:
                    in_goal_mouth = i == 1 and abs(b.position[0]) < cv.GOAL_CENTER_TO_POST
                    if in_goal_mouth:
                        if abs(b.position[1]) > cv.GOAL_THRESHOLD:
                            s.goal_scored = True
                    else:
                        b.position[i] = np.sign(b.position[i]) * lim
                        b.linear_velocity[i] *= -0.6
            s.tick_count += 1
            if s.goal_scored:
                break
        for a, car in s.cars.items():
            car.ball_touches = touches[a]
        s._inverted_ball = None
        s._inverted_boost_pad_timers = None
        for car in s.cars.values():
            car._inverted_physics = None
        return s

    def close(self) -> None:
        pass


def build_fake_env(cfg: Dict[str, Any], seed=None) -> RLGym:
    from .actions import make_action_parser
    from .mutators import RandomFormatMutator, StartStateMutator
    from .obs import AttentionObs, env_obs_flags
    from .rewards import make_reward
    e = cfg["env"]
    return RLGym(
        state_mutator=MutatorSequence(RandomFormatMutator(e["format_weights"], seed=seed),
                                      StartStateMutator(e["start_weights"], seed=seed)),
        obs_builder=AttentionObs(*env_obs_flags(cfg["env"])),
        action_parser=make_action_parser(e["action_repeat"]),
        reward_fn=make_reward(cfg),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(NoTouchTimeoutCondition(e["no_touch_timeout_s"]),
                                     TimeoutCondition(e["episode_timeout_s"])),
        transition_engine=FakeEngine(),
    )


_ = Car  # (import conservé pour la lisibilité des annotations)
