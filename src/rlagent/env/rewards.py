"""Rewards (ARCHITECTURE §7).

Chaque terme renvoie une valeur BRUTE par agent. `TeamReward` les pondère (avec décroissance programmée),
puis applique l'esprit d'équipe τ et la soustraction adverse :

    r_final(i) = (1 - τ) * r_i + τ * moyenne(r de l'équipe de i) - zero_sum * moyenne(r de l'équipe adverse)

La progression (nombre global de pas) est lue dans shared_info["global_step"], mise à jour par le trainer.
Les sommes par terme de l'épisode sont accumulées dans shared_info["reward_log"] pour le suivi.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from .common import OPP_GOAL_BACK, ball_phys, unit


class RewardTerm:
    name = "term"

    def reset(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get(self, agents: List[str], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class GoalTerm(RewardTerm):
    """+1 pour chaque membre de l'équipe qui marque (la soustraction adverse donne -1 à l'équipe qui encaisse)."""
    name = "goal"

    def get(self, agents, state, shared_info):
        scorer = state.scoring_team if state.goal_scored else None
        return {a: (1.0 if scorer is not None and state.cars[a].team_num == scorer else 0.0) for a in agents}


class TouchTerm(RewardTerm):
    """1 par touche, majorée quand la balle est haute (encourage les aériennes)."""
    name = "touch"

    def __init__(self, aerial_bonus: float = 1.0):
        self.aerial_bonus = aerial_bonus

    def get(self, agents, state, shared_info):
        height = float(np.clip(state.ball.position[2] / cv.CEILING_Z, 0.0, 1.0))
        mult = 1.0 + self.aerial_bonus * height
        return {a: (mult if state.cars[a].ball_touches > 0 else 0.0) for a in agents}


class VelPlayerToBallTerm(RewardTerm):
    name = "vel_player_to_ball"

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            p = state.cars[a].physics
            d = unit(state.ball.position - p.position)
            out[a] = max(0.0, float(np.dot(p.linear_velocity, d))) / cv.CAR_MAX_SPEED
        return out


class VelBallToGoalTerm(RewardTerm):
    """Vitesse de la balle vers le but adverse (signée : négatif si elle va vers notre but)."""
    name = "vel_ball_to_goal"

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            inv = state.cars[a].team_num == cv.ORANGE_TEAM
            ball = ball_phys(state, inv)
            d = unit(OPP_GOAL_BACK - ball.position)
            out[a] = float(np.dot(ball.linear_velocity, d)) / cv.BALL_MAX_SPEED
        return out


class FaceBallTerm(RewardTerm):
    name = "face_ball"

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            p = state.cars[a].physics
            out[a] = float(np.dot(p.forward, unit(state.ball.position - p.position)))
        return out


class BoostGainTerm(RewardTerm):
    """Gain de boost en racine carrée (le boost bas vaut plus cher). Dépense pénalisée par loss_coef."""
    name = "boost_gain"

    def __init__(self, loss_coef: float = 0.0):
        self.loss_coef = loss_coef
        self.prev: Dict[str, float] = {}

    def reset(self, agents, state, shared_info):
        self.prev = {a: state.cars[a].boost_amount for a in agents}

    def get(self, agents, state, shared_info):
        out = {}
        for a in agents:
            b = state.cars[a].boost_amount
            prev = self.prev.get(a, b)
            delta = math.sqrt(max(b, 0.0) / 100.0) - math.sqrt(max(prev, 0.0) / 100.0)
            out[a] = delta if delta >= 0 else self.loss_coef * delta
            self.prev[a] = b
        return out


class DoubleCommitTerm(RewardTerm):
    """-1 pour chaque agent "engagé" sur la balle quand au moins un coéquipier l'est aussi (2v2/3v3)."""
    name = "double_commit"

    def __init__(self, radius: float = 900.0, min_speed: float = 400.0):
        self.radius = radius
        self.min_speed = min_speed

    def committed(self, state: GameState, agent: str) -> bool:
        p = state.cars[agent].physics
        rel = state.ball.position - p.position
        dist = float(np.linalg.norm(rel))
        if dist > self.radius:
            return False
        return float(np.dot(p.linear_velocity, unit(rel))) > self.min_speed

    def get(self, agents, state, shared_info):
        committed = {a: self.committed(state, a) for a in state.cars}
        per_team: Dict[int, int] = {}
        for a, c in committed.items():
            if c:
                t = state.cars[a].team_num
                per_team[t] = per_team.get(t, 0) + 1
        return {a: (-1.0 if committed[a] and per_team.get(state.cars[a].team_num, 0) >= 2 else 0.0)
                for a in agents}


def linear_schedule(start: float, end: float, steps: float, t: float) -> float:
    if steps <= 0:
        return end
    frac = min(max(t / steps, 0.0), 1.0)
    return start + (end - start) * frac


class TeamReward(RewardFunction[str, GameState, float]):
    def __init__(self, terms: List[RewardTerm], weights: Dict[str, float],
                 anneal: Optional[Dict[str, Dict[str, float]]] = None,
                 team_spirit: Optional[Dict[str, float]] = None, zero_sum: float = 1.0):
        self.terms = [t for t in terms if weights.get(t.name, 0.0) != 0.0]
        self.weights = dict(weights)
        self.anneal = anneal or {}
        self.team_spirit = team_spirit or {"start": 0.0, "end": 0.0, "steps": 1}
        self.zero_sum = zero_sum

    def current_weight(self, name: str, step: float) -> float:
        w = self.weights.get(name, 0.0)
        sched = self.anneal.get(name)
        if sched:
            w *= linear_schedule(1.0, float(sched["end"]), float(sched["steps"]), step)
        return w

    def current_tau(self, step: float) -> float:
        ts = self.team_spirit
        return linear_schedule(float(ts["start"]), float(ts["end"]), float(ts["steps"]), step)

    def reset(self, agents, initial_state, shared_info):
        for t in self.terms:
            t.reset(agents, initial_state, shared_info)
        shared_info["reward_log"] = {t.name: 0.0 for t in self.terms}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> Dict[str, float]:
        step = float(shared_info.get("global_step", 0))
        raw = {a: 0.0 for a in agents}
        log = shared_info.setdefault("reward_log", {})
        for term in self.terms:
            w = self.current_weight(term.name, step)
            vals = term.get(agents, state, shared_info)
            for a in agents:
                v = w * vals[a]
                raw[a] += v
            log[term.name] = log.get(term.name, 0.0) + sum(w * vals[a] for a in agents) / max(1, len(agents))
        return combine_team_rewards(raw, {a: state.cars[a].team_num for a in agents},
                                    self.current_tau(step), self.zero_sum)


def combine_team_rewards(raw: Dict[str, float], teams: Dict[str, int], tau: float, zero_sum: float) -> Dict[str, float]:
    team_vals: Dict[int, List[float]] = {}
    for a, r in raw.items():
        team_vals.setdefault(teams[a], []).append(r)
    team_mean = {t: float(np.mean(v)) for t, v in team_vals.items()}
    out = {}
    for a, r in raw.items():
        t = teams[a]
        opp = [m for tt, m in team_mean.items() if tt != t]
        opp_mean = float(np.mean(opp)) if opp else 0.0
        out[a] = (1.0 - tau) * r + tau * team_mean[t] - zero_sum * opp_mean
    return out


def make_reward(cfg: Dict[str, Any]) -> TeamReward:
    rc = cfg["rewards"]
    terms = [
        GoalTerm(),
        TouchTerm(aerial_bonus=rc.get("touch_aerial_bonus", 1.0)),
        VelPlayerToBallTerm(),
        VelBallToGoalTerm(),
        FaceBallTerm(),
        BoostGainTerm(loss_coef=rc.get("boost_loss_coef", 0.0)),
        DoubleCommitTerm(radius=rc.get("double_commit_radius", 900), min_speed=rc.get("double_commit_min_speed", 400)),
    ]
    return TeamReward(terms, rc["weights"], rc.get("anneal"), rc.get("team_spirit"), rc.get("zero_sum", 1.0))
