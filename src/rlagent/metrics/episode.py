"""Statistiques par épisode, calculées dans les workers (ARCHITECTURE §10).

Jeu d'équipe (équipes de 2+ joueurs) :
  - double_commit_rate : part des steps où ≥ 2 coéquipiers sont "engagés" sur la balle
  - teammate_bumps     : contacts entre coéquipiers (par minute)
  - teammate_dist      : distance moyenne entre coéquipiers (uu)
Kickoff : vainqueur = équipe qui marque dans les 3 s, sinon équipe dans le camp ADVERSE de laquelle se trouve la balle à 3 s.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Optional

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from ..env.rewards import DoubleCommitTerm

KICKOFF_EVAL_S = 3.0


class EpisodeTracker:
    def __init__(self, double_commit_radius: float = 900.0, double_commit_min_speed: float = 400.0):
        self.dc = DoubleCommitTerm(double_commit_radius, double_commit_min_speed)
        self.reset(None, {})

    def reset(self, state: Optional[GameState], shared_info: Dict[str, Any]) -> None:
        self.start_tick = state.tick_count if state is not None else 0
        self.last_tick = self.start_tick
        self.team_size = int(shared_info.get("team_size", 1))
        self.start_kind = shared_info.get("start_kind", "unknown")
        self.steps = 0
        self.team_steps = 0
        self.double_commit_steps = 0
        self.teammate_bumps = 0
        self.teammate_dist_sum = 0.0
        self.teammate_dist_n = 0
        self.touches = {0: 0, 1: 0}
        self.kickoff_winner: Optional[int] = None
        self.kickoff_done = self.start_kind != "kickoff"
        self.air_touches = 0

    def step(self, state: GameState) -> None:
        self.steps += 1
        self.last_tick = state.tick_count
        cars = state.cars
        for car in cars.values():
            if car.ball_touches > 0:
                self.touches[car.team_num] = self.touches.get(car.team_num, 0) + 1
                if state.ball.position[2] > 300:
                    self.air_touches += 1
        if self.team_size >= 2:
            self.team_steps += 1
            by_team: Dict[int, list] = {}
            for a, car in cars.items():
                by_team.setdefault(car.team_num, []).append(a)
            dc = False
            for team, members in by_team.items():
                if sum(self.dc.committed(state, a) for a in members) >= 2:
                    dc = True
                for a, b in combinations(members, 2):
                    self.teammate_dist_sum += float(np.linalg.norm(cars[a].physics.position - cars[b].physics.position))
                    self.teammate_dist_n += 1
                for a in members:
                    victim = cars[a].bump_victim_id
                    if victim is not None and victim in cars and cars[victim].team_num == team:
                        self.teammate_bumps += 1
            self.double_commit_steps += int(dc)
        if not self.kickoff_done:
            elapsed = (state.tick_count - self.start_tick) / cv.TICKS_PER_SECOND
            if state.goal_scored:
                self.kickoff_winner = state.scoring_team
                self.kickoff_done = True
            elif elapsed >= KICKOFF_EVAL_S:
                y = float(state.ball.position[1])
                self.kickoff_winner = 0 if y > 0 else 1  # balle dans le camp orange → avantage bleu
                self.kickoff_done = True

    def finish(self, state: GameState, terminated: bool) -> Dict[str, Any]:
        seconds = max(1e-6, (self.last_tick - self.start_tick) / cv.TICKS_PER_SECOND)
        minutes = seconds / 60.0
        out: Dict[str, Any] = {
            "team_size": self.team_size,
            "start_kind": self.start_kind,
            "length_s": seconds,
            "goal_team": (state.scoring_team if (terminated and state.goal_scored) else None),
            "touches_blue": self.touches.get(0, 0),
            "touches_orange": self.touches.get(1, 0),
            "touches_per_min": (self.touches.get(0, 0) + self.touches.get(1, 0)) / minutes,
            "air_touches_per_min": self.air_touches / minutes,
            "kickoff_winner": self.kickoff_winner,
        }
        if self.team_size >= 2 and self.team_steps > 0:
            out["double_commit_rate"] = self.double_commit_steps / self.team_steps
            out["teammate_bumps_per_min"] = self.teammate_bumps / minutes
            out["teammate_dist"] = self.teammate_dist_sum / max(1, self.teammate_dist_n)
        return out
