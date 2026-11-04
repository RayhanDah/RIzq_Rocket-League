"""Statistiques par épisode, calculées dans les workers (ARCHITECTURE §10).

Jeu d'équipe (équipes de 2+ joueurs) :
  - double_commit_rate : part des steps où ≥ 2 coéquipiers sont "engagés" sur la balle
  - teammate_bumps     : contacts entre coéquipiers (par minute)
  - teammate_dist      : distance moyenne entre coéquipiers (uu)
Kickoff : vainqueur = équipe qui marque dans les 3 s, sinon équipe dans le camp ADVERSE de laquelle se trouve la balle à 3 s.
Tirs / arrêts (env/predict.py) : tir = touche après laquelle la trajectoire entre dans le cadre adverse (et n'y entrait
pas avant) ; arrêt = touche après laquelle une trajectoire qui entrait dans NOTRE cadre n'y entre plus.
Style (palier 5, mêmes définitions que scripts/style_audit.py) : déviation = touche qui fait sortir du cadre SA propre
balle cadrée ; hors cadre = frappe qui atteint la ligne de but hors du cadre ; vraie aérienne = touche en l'air, balle
au-dessus de 500 uu, après 0,25 s de boost depuis le décollage.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Optional

from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState

from ..env.common import norm3
from ..env.predict import goal_line_time, goal_threat
from ..env.rewards import DoubleCommitTerm

KICKOFF_EVAL_S = 3.0

# Réussite des exercices (env/mutators.py, `drill_*`) pour l'équipe testée, dans une fenêtre en secondes.
# "defense" réussit si aucun but n'est encaissé dans la fenêtre ; les autres si la condition est vue dans la fenêtre.
DRILL_WINDOW_S = {"defense": 6.0, "shot": 6.0, "aerial": 4.0, "wall": 5.0, "low_boost": 6.0, "recovery": 2.0,
                  "aerial_defense": 6.0}
NO_GOAL_DRILLS = ("defense", "aerial_defense")   # réussite = aucun but encaissé dans la fenêtre
AERIAL_TOUCH_Z = 400.0                          # exercice aérien sans niveau (s6)
AERIAL_TOUCH_Z_LEVEL = (200.0, 700.0)           # avec niveau : hauteur de touche exigée au niveau 0 → 1
AERIAL_MIN_Z = 220.0                            # touche « aérienne » : voiture en l'air, balle au-dessus
TRUE_AERIAL_Z = 500.0                           # vraie aérienne : balle au-dessus (comme rewards.aerial_min_z)…
TRUE_AERIAL_BOOST_S = 0.25                      # … et boost en l'air depuis le décollage
WALL_TAKEOFF_Z = 300.0                          # décollage au-dessus : la voiture quitte un mur
STRONG_HIT = 556.0                              # |Δv balle| d'une frappe (uu/s), seuil de strong_touch
LOW_BOOST = 10.0


def aerial_drill_min_z(level) -> float:
    if level is None:
        return AERIAL_TOUCH_Z
    lo, hi = AERIAL_TOUCH_Z_LEVEL
    return lo + (hi - lo) * float(level)
WALL_ZONE_X = cv.SIDE_WALL_X - 600.0


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
        self.aerial_touches = 0
        self.shots = 0
        self.saves = 0
        self.off_target = 0
        self.spoils = 0
        self.true_aerials = 0
        self.wall_aerials = 0
        self.air_boost_ticks: Dict[Any, int] = {}
        self.takeoff_z: Dict[Any, float] = {}
        # mécaniques (toutes les voitures) : sommes sur (voitures × steps)
        self.car_steps = 0
        self.boost_sum = 0.0
        self.boost_pickups = 0
        self.jumps = 0
        self.air_steps = 0
        self.supersonic_steps = 0
        self.boost_waste_steps = 0
        self.flips = 0
        self.low_boost_steps = 0
        self.speed_sum = 0.0
        self.air_boost_steps = 0
        self.ball_dist_sum = 0.0
        self.opp_half_steps = 0
        self.bumps = 0
        self.prev_victim: Dict[Any, Any] = {}
        self.prev_car: Dict[Any, tuple] = {}
        self.prev_ball = None
        if state is not None:
            self.prev_car = {a: (float(c.boost_amount), bool(c.has_jumped), float(c.demo_respawn_timer),
                                 bool(c.has_flipped)) for a, c in state.cars.items()}
            self.prev_ball = (state.ball.position.tolist(), state.ball.linear_velocity.tolist())
        # exercice
        kind = str(self.start_kind)
        self.drill = kind[len("drill_"):] if kind.startswith("drill_") else None
        self.drill_team = shared_info.get("drill_team")
        self.drill_level = shared_info.get("drill_level") if self.drill is not None else None
        self.drill_result: Optional[float] = None

    def step(self, state: GameState) -> None:
        self.steps += 1
        dt_ticks = state.tick_count - self.last_tick
        self.last_tick = state.tick_count
        cars = state.cars
        touch_teams = set()
        for a, car in cars.items():
            if car.on_ground:
                self.air_boost_ticks[a] = 0
                self.takeoff_z[a] = float(car.physics.position[2])
            elif car.is_boosting:
                self.air_boost_ticks[a] = self.air_boost_ticks.get(a, 0) + dt_ticks
            if car.ball_touches > 0:
                self.touches[car.team_num] = self.touches.get(car.team_num, 0) + 1
                touch_teams.add(car.team_num)
                if state.ball.position[2] > 300:
                    self.air_touches += 1
                if not car.on_ground and state.ball.position[2] > AERIAL_MIN_Z:
                    self.aerial_touches += 1
                if not car.on_ground and state.ball.position[2] > TRUE_AERIAL_Z:
                    if self.air_boost_ticks.get(a, 0) >= TRUE_AERIAL_BOOST_S * cv.TICKS_PER_SECOND:
                        self.true_aerials += 1
                    elif self.takeoff_z.get(a, 0.0) > WALL_TAKEOFF_Z:
                        self.wall_aerials += 1
        ball_now = (state.ball.position.tolist(), state.ball.linear_velocity.tolist())
        if touch_teams and self.prev_ball is not None:
            self._shots_saves(touch_teams, self.prev_ball, ball_now)
        self.prev_ball = ball_now
        self._mechanics(state)
        if self.drill is not None and self.drill_result is None:
            self._drill_step(state)
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
                    self.teammate_dist_sum += norm3(cars[a].physics.position - cars[b].physics.position)
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

    def _shots_saves(self, teams, before, after) -> None:
        dv = [after[1][i] - before[1][i] for i in range(3)]
        strong = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] >= STRONG_HIT * STRONG_HIT
        for team in teams:
            opp_sign = 1.0 if team == cv.BLUE_TEAM else -1.0     # le bleu attaque vers +Y
            was, now = goal_threat(*before, opp_sign), goal_threat(*after, opp_sign)
            if now > 0.0 and was == 0.0:
                self.shots += 1
            elif was > 0.0 and now == 0.0:
                self.spoils += 1
            elif now == 0.0 and strong and goal_line_time(*after, opp_sign) is not None:
                self.off_target += 1
            if goal_threat(*before, -opp_sign) > 0.0 and goal_threat(*after, -opp_sign) == 0.0:
                self.saves += 1

    def _mechanics(self, state: GameState) -> None:
        ball = state.ball.position
        for a, car in state.cars.items():
            boost, jumped, demo = float(car.boost_amount), bool(car.has_jumped), float(car.demo_respawn_timer)
            flipped = bool(car.has_flipped)
            pb, pj, pdemo, pf = self.prev_car.get(a, (boost, jumped, demo, flipped))
            self.car_steps += 1
            self.boost_sum += boost
            self.speed_sum += norm3(car.physics.linear_velocity)
            self.ball_dist_sum += norm3(car.physics.position - ball)
            self.opp_half_steps += float(car.physics.position[1]) * (1.0 if car.team_num == cv.BLUE_TEAM else -1.0) > 0.0
            victim = car.bump_victim_id
            self.bumps += victim is not None and victim != self.prev_victim.get(a)
            self.prev_victim[a] = victim
            if boost < LOW_BOOST:
                self.low_boost_steps += 1
            if flipped and not pf:
                self.flips += 1
            if boost > pb + 1.0 and pdemo <= 0.0 and demo <= 0.0:   # ramassage (pas le boost de réapparition)
                self.boost_pickups += 1
            if jumped and not pj:
                self.jumps += 1
            if not car.on_ground:
                self.air_steps += 1
                self.air_boost_steps += bool(car.is_boosting)
            if car.is_supersonic:
                self.supersonic_steps += 1
                if car.is_boosting and car.on_ground:      # boost au-delà de la vitesse max au sol : gaspillé
                    self.boost_waste_steps += 1
            self.prev_car[a] = (boost, jumped, demo, flipped)

    def _drill_step(self, state: GameState) -> None:
        elapsed = (state.tick_count - self.start_tick) / cv.TICKS_PER_SECOND
        team = self.drill_team
        mine = [c for c in state.cars.values() if c.team_num == team]
        kind = self.drill
        if kind in NO_GOAL_DRILLS:
            if state.goal_scored and state.scoring_team != team:
                self.drill_result = 0.0
            elif elapsed >= DRILL_WINDOW_S[kind]:
                self.drill_result = 1.0
            return
        if elapsed > DRILL_WINDOW_S[kind]:
            self.drill_result = 0.0
            return
        ok = False
        if kind == "shot":
            ok = bool(state.goal_scored and state.scoring_team == team)
        elif kind == "aerial":
            ok = (any(c.ball_touches > 0 for c in mine)
                  and float(state.ball.position[2]) > aerial_drill_min_z(self.drill_level))
        elif kind == "wall":
            ball = state.ball.position
            ok = any(c.ball_touches > 0 for c in mine) and (abs(float(ball[0])) > WALL_ZONE_X or float(ball[2]) > 300)
        elif kind == "low_boost":
            ok = any(float(c.boost_amount) >= 50.0 for c in mine)
        elif kind == "recovery":
            ok = all(c.on_ground for c in mine)
        if ok:
            self.drill_result = 1.0

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
            "aerial_touches_per_min": self.aerial_touches / minutes,
            "shots_per_min": self.shots / minutes,
            "saves_per_min": self.saves / minutes,
            "off_target_per_min": self.off_target / minutes,
            "spoils_per_min": self.spoils / minutes,
            "true_aerials_per_min": self.true_aerials / minutes,
            "wall_aerials_per_min": self.wall_aerials / minutes,
            "kickoff_winner": self.kickoff_winner,
        }
        if self.car_steps > 0:
            car_minutes = max(1e-6, self.car_steps * (seconds / max(1, self.steps)) / 60.0)
            out["mech"] = {
                "boost_pickups_per_min": self.boost_pickups / car_minutes,
                "avg_boost": self.boost_sum / self.car_steps,
                "jumps_per_min": self.jumps / car_minutes,
                "air_frac": self.air_steps / self.car_steps,
                "supersonic_frac": self.supersonic_steps / self.car_steps,
                "boost_waste_frac": self.boost_waste_steps / self.car_steps,
                "flips_per_min": self.flips / car_minutes,
                "low_boost_frac": self.low_boost_steps / self.car_steps,
                "avg_speed": self.speed_sum / self.car_steps,
                "air_boost_frac": self.air_boost_steps / max(1, self.air_steps),
                "ball_dist": self.ball_dist_sum / self.car_steps,
                "opp_half_frac": self.opp_half_steps / self.car_steps,
                "bumps_per_min": self.bumps / car_minutes,
            }
        if self.drill is not None:
            result = self.drill_result
            if result is None:  # épisode fini avant la fin de la fenêtre
                conceded = terminated and state.goal_scored and state.scoring_team != self.drill_team
                result = (0.0 if conceded else 1.0) if self.drill in NO_GOAL_DRILLS else 0.0
            out["drill"] = self.drill
            out["drill_success"] = float(result)
            if self.drill_level is not None:
                out["drill_level"] = float(self.drill_level)
        if self.team_size >= 2 and self.team_steps > 0:
            out["double_commit_rate"] = self.double_commit_steps / self.team_steps
            out["teammate_bumps_per_min"] = self.teammate_bumps / minutes
            out["teammate_dist"] = self.teammate_dist_sum / max(1, self.teammate_dist_n)
        return out
