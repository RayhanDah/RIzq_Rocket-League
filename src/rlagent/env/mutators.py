"""États de départ (ARCHITECTURE §2 et §8) : tirage du format 1v1/2v2/3v3 puis kickoff ou position aléatoire."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from .common import norm3, unit


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


# --------------------------------------------------------------------------- exercices dédiés (drills)
# Chaque exercice est un ÉTAT DE DÉPART : le reste de l'épisode est un match normal, avec les mêmes rewards (sinon le
# bot apprendrait des réflexes qui ne servent pas en match). L'équipe testée (`shared_info["drill_team"]`) est tirée
# au hasard : symétrique bleu/orange. La réussite est mesurée par metrics/episode.py (DRILL_RULES).

def _team_cars(state: GameState, team: int):
    return [c for _, c in sorted(state.cars.items(), key=lambda kv: str(kv[0])) if c.team_num == team]


def _dir(team: int) -> float:
    """+1 si l'équipe attaque vers +Y (bleu), -1 sinon (orange)."""
    return 1.0 if team == cv.BLUE_TEAM else -1.0


# Niveau de difficulté (curriculum, `env.drill_curriculum`) : les exercices « à niveau » lisent
# shared_info["drill_levels"][kind] ∈ [0, 1], réglé par le processus principal (train.DrillCurriculum) selon la réussite
# de TOUTES les arènes, et l'écrivent dans shared_info["drill_level"] pour la mesure de réussite (metrics/episode.py).
# Sans curriculum (niveau absent) : réglages d'origine (s6), inchangés.

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class _Drill(StateMutator[GameState]):
    kind = "drill"
    leveled = False

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.random_state = RandomStateMutator(None if seed is None else seed + 17)
        self.level: Optional[float] = None

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.random_state.apply(state, shared_info)          # tout le monde au hasard, puis on place l'exercice
        team = cv.BLUE_TEAM if self.rng.random() < 0.5 else cv.ORANGE_TEAM
        shared_info["drill_team"] = team
        level = shared_info.get("drill_levels", {}).get(self.kind) if self.leveled else None
        self.level = None if level is None else float(min(max(level, 0.0), 1.0))
        if self.level is None:
            shared_info.pop("drill_level", None)
        else:
            shared_info["drill_level"] = self.level
        self.setup(state, team, self.rng)

    def setup(self, state: GameState, team: int, r) -> None:
        raise NotImplementedError


class DefenseDrill(_Drill):
    """Tir adverse en route vers NOTRE but ; nos voitures en retour défensif, les attaquants adverses derrière la balle.
    Niveau : tirs plus rapides et plus souvent en hauteur (0 : 900-1700 uu/s, surtout au sol ; 1 : 1800-3200 uu/s)."""
    kind = "defense"
    leveled = True

    def setup(self, state, team, r):
        d = _dir(team)
        own_goal_y = -d * cv.BACK_WALL_Y
        bx, by = r.uniform(-2500, 2500), -d * r.uniform(1500, 3800)
        L = self.level
        speed = r.uniform(1200, 2800) if L is None else r.uniform(_lerp(900, 1800, L), _lerp(1700, 3200, L))
        v = unit(np.array([r.uniform(-700, 700) - bx, own_goal_y - by, 0.0])) * speed
        p_high, z_max = (0.4, 700.0) if L is None else (_lerp(0.15, 0.5, L), _lerp(350, 900, L))
        z = cv.BALL_RESTING_HEIGHT if r.random() >= p_high else r.uniform(150, z_max)
        state.ball.position = np.array([bx, by, z], dtype=np.float32)
        state.ball.linear_velocity = np.array([v[0], v[1], r.uniform(0, 300)], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in _team_cars(state, team):                  # défenseurs : entre la balle et le but, ou sur le côté
            f = r.uniform(0.3, 1.0)
            x = bx * (1 - f) + r.normal(0, 900)
            y = by + (own_goal_y - by) * f + d * r.uniform(-300, 300)
            yaw = float(np.arctan2(own_goal_y - y, -x) if r.random() < 0.5 else np.arctan2(by - y, bx - x))
            _place_car(car, x, y, yaw, r.uniform(500, 1600), r.uniform(10, 100))
        for car in _team_cars(state, 1 - team):              # attaquants derrière la balle
            x, y = bx + r.normal(0, 500), by + d * r.uniform(600, 1800)
            _place_car(car, x, y, float(np.arctan2(by - y, bx - x)), r.uniform(800, 1800), r.uniform(0, 100))


class ShotDrill(_Drill):
    """Balle qui roule devant la surface adverse, nos voitures en approche ; gardien adverse une fois sur deux.
    Niveau : balle plus loin et plus excentrée, gardien plus souvent
    (0 : 3000-3900 uu du centre, gardien 15 % ; 1 : 1800-3900 uu, gardien 75 %)."""
    kind = "shot"
    leveled = True

    def setup(self, state, team, r):
        d = _dir(team)
        goal_y = d * cv.BACK_WALL_Y
        L = self.level
        y_min, p_keeper, x_span = (1800.0, 0.5, 1.0) if L is None else (_lerp(3000, 1800, L), _lerp(0.15, 0.75, L),
                                                                         _lerp(0.5, 1.0, L))
        bx, by = r.uniform(-2200, 2200) * x_span, d * r.uniform(y_min, 3900)
        z = cv.BALL_RESTING_HEIGHT if r.random() < 0.7 else r.uniform(150, 500)
        state.ball.position = np.array([bx, by, z], dtype=np.float32)
        state.ball.linear_velocity = np.array([r.normal(0, 400), d * r.uniform(-300, 700), 0.0], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in _team_cars(state, team):
            dist, ang = r.uniform(600, 1800), r.uniform(-0.9, 0.9)
            x, y = bx - dist * np.sin(ang), by - d * dist * np.cos(ang)
            _place_car(car, x, y, float(np.arctan2(by - y, bx - x) + r.normal(0, 0.3)), r.uniform(600, 1800),
                       r.uniform(20, 100))
        for car in _team_cars(state, 1 - team):
            if r.random() < p_keeper:                        # gardien
                _place_car(car, r.uniform(-600, 600), goal_y - d * r.uniform(150, 500), float(-d * np.pi / 2),
                           0.0, r.uniform(0, 100))
            else:                                            # en retard, derrière l'attaquant
                _place_car(car, r.uniform(-3000, 3000), by - d * r.uniform(1500, 3500), float(d * np.pi / 2),
                           r.uniform(500, 1500), r.uniform(0, 100))


class AerialDrill(_Drill):
    """Balle en l'air près de notre voiture, au sol avec beaucoup de boost.
    Sans niveau : balle qui retombe de 600-1800 uu (réglage s6).
    Avec niveau (s7d) : RENDEZ-VOUS. La voiture roule vers un point M ; la trajectoire de la balle est calculée pour
    qu'elle passe au-dessus de M, à la hauteur h, à l'instant T où la voiture peut y être :
        T = distance / vitesse + temps de montée ((h − 250) / 900 s ; 0 pour un simple saut).
    0 : h = 200-280 uu, voiture à 600-1000 uu (~1 s pour préparer le saut) ; 1 : h = 1150-1800 uu, voiture à
    800-1400 uu, balle qui dérive davantage (vraie aérienne). L'ancien niveau 0 (balle à 150-400 uu qui montait, voiture
    déjà lancée) était quasi irréalisable : s7c n'y réussissait que 3,5 % (runs/s7c_spo_laptop/DIAGNOSTIC_AERIEN_BOOST.md).
    Réussite : metrics/episode.py (touche au-dessus de 200 → 700 uu en 4 s)."""
    kind = "aerial"
    leveled = True
    GRAVITY = 650.0

    def setup(self, state, team, r):
        car = _team_cars(state, team)[0]
        d = _dir(team)
        L = self.level
        cx, cy = r.uniform(-3000, 3000), r.uniform(-3500, 3500)
        if L is None:
            dist, ang = r.uniform(200, 1400), r.uniform(-np.pi, np.pi)
            z, sigma, vz, yaw_noise = r.uniform(600, 1800), 400.0, r.uniform(-300, 400), r.normal(0, 0.3)
            bx = float(np.clip(cx + dist * np.cos(ang), -3600, 3600))
            by = float(np.clip(cy + dist * np.sin(ang), -4400, 4400))
            state.ball.position = np.array([bx, by, z], dtype=np.float32)
            state.ball.linear_velocity = np.array([r.normal(0, sigma), r.normal(0, sigma) + d * 200.0, vz],
                                                  dtype=np.float32)
            state.ball.angular_velocity = r.normal(0, 1, 3).astype(np.float32)
            _place_car(car, cx, cy, float(np.arctan2(by - cy, bx - cx) + yaw_noise), r.uniform(0, 1200),
                       r.uniform(60, 100))
            return
        h = r.uniform(_lerp(200, 1150, L), _lerp(280, 1800, L))
        dist = r.uniform(_lerp(600, 800, L), _lerp(1000, 1400, L))
        speed = r.uniform(_lerp(700, 600, L), _lerp(1000, 1200, L))
        yaw = r.uniform(-np.pi, np.pi)
        mx = float(np.clip(cx + dist * np.cos(yaw), -3600, 3600))      # point de rendez-vous
        my = float(np.clip(cy + dist * np.sin(yaw), -4400, 4400))
        dist = float(np.hypot(mx - cx, my - cy))
        g, top, low = self.GRAVITY, 1850.0, 150.0                        # sommet sous le plafond, départ au-dessus du sol
        climb = max(0.0, h - 250.0) / 900.0
        T = dist / speed + climb
        t_max = math.sqrt(2 * g * (top - low)) / g + math.sqrt(2 * (top - h) / g)
        if T > t_max:                                                   # balle incapable d'attendre : voiture plus rapide
            T = t_max
            speed = min(2200.0, dist / max(T - climb, 0.3))
        sigma = _lerp(30, 300, L)
        vx, vy = r.normal(0, sigma), r.normal(0, sigma) + d * _lerp(0, 200, L)
        apex = min(h + r.uniform(_lerp(40, 20, L), _lerp(250, 100, L)), top)
        t_apex = T - math.sqrt(2 * (apex - h) / g)                      # la balle redescend de `apex` à h pendant T − t_apex
        z0 = apex - 0.5 * g * t_apex * t_apex
        if z0 >= low:
            vz = g * t_apex
        else:                                                           # départ trop bas : on part de `low`, z(T) = h
            z0 = low
            vz = (h - z0 + 0.5 * g * T * T) / T
        bx = float(np.clip(mx - vx * T, -3600, 3600))
        by = float(np.clip(my - vy * T, -4400, 4400))
        state.ball.position = np.array([bx, by, z0], dtype=np.float32)
        drag = 1.0 + 0.0305 * T / 2                                     # compense le frottement de l'air de la balle
        state.ball.linear_velocity = np.array([(mx - bx) / T * drag, (my - by) / T * drag, vz * drag],
                                              dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        _place_car(car, cx, cy, float(np.arctan2(my - cy, mx - cx) + r.normal(0, _lerp(0.03, 0.3, L))), speed,
                   r.uniform(60, 100))
        self.last_meet = (mx, my, h, T)                                 # tests / diagnostic


class AirDrill(_Drill):
    """Voiture déjà en vol vers une balle haute : curriculum INVERSÉ (palier 6). Le rendez-vous (`aerial`) plafonne à 3,7 %
    de réussite : décoller au bon moment, monter et viser d'un coup, c'est trop. Ici on part de la fin du geste et on
    recule avec le niveau, jusqu'au décollage :
        0 : voiture à 300-700 uu sous la balle, lancée vers elle, nez pointé dessus, 60-100 de boost ;
        1 : voiture à 1200-2400 uu, à 50-300 uu de haut, mal orientée, balle qui file, 30-100 de boost.
    Voiture plutôt de notre côté de la balle : la frappe part vers le but adverse. Réussite : metrics/episode.py (touche
    au-dessus de 300 uu en 3 s). La balle garde toujours un peu de vitesse : immobile, elle resterait suspendue."""
    kind = "air"
    leveled = True

    def setup(self, state, team, r):
        car = _team_cars(state, team)[0]
        d = _dir(team)
        L = 0.5 if self.level is None else self.level
        ball = np.array([r.uniform(-3000, 3000), d * r.uniform(-1500, 4200),
                         r.uniform(_lerp(500, 450, L), _lerp(1100, 1800, L))])
        dist = r.uniform(_lerp(300, 1200, L), _lerp(700, 2400, L))
        z = _lerp(ball[2] - r.uniform(150, 450), r.uniform(50, 300), L)
        flat = math.sqrt(max(dist * dist - (ball[2] - z) ** 2, 0.0))
        side = r.normal(0.0, _lerp(0.5, 1.2, L))
        pos = np.array([np.clip(ball[0] - flat * math.sin(side), -3900, 3900),
                        np.clip(ball[1] - d * flat * math.cos(side), -4900, 4900), max(z, 50.0)])
        to_ball = unit(ball - pos)
        fwd = unit(to_ball + r.normal(0.0, _lerp(0.05, 0.8, L), 3))
        speed = r.uniform(_lerp(700, 300, L), _lerp(1300, 1200, L))
        sigma = _lerp(100, 700, L)
        state.ball.position = ball.astype(np.float32)
        state.ball.linear_velocity = np.array([r.normal(0, sigma), r.normal(0, sigma), r.uniform(50, 300)], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.position = pos.astype(np.float32)
        car.physics.linear_velocity = (to_ball * speed).astype(np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([math.asin(float(np.clip(fwd[2], -1.0, 1.0))), math.atan2(fwd[1], fwd[0]),
                                             r.normal(0.0, _lerp(0.1, 1.0, L))], dtype=np.float32)
        car.boost_amount = float(r.uniform(_lerp(60, 30, L), 100))
        car.on_ground = False
        car.has_jumped = True
        car.has_double_jumped = False
        car.has_flipped = False
        self.last_flight = (ball, pos, fwd)                              # tests / diagnostic


class ResetDrill(_Drill):
    """Roues face à une balle haute, flip déjà utilisé : le flip reset à portée (palier 6). Les bots ne le découvrent pas
    seuls : sans états où il est possible, le reward ne se déclenche presque jamais (Opti a ses états et son modèle dédiés).
        0 : voiture à 130-260 uu de la balle, roues bien en face, vitesse relative 200-500 uu/s ;
        1 : voiture à 300-900 uu, orientation approximative, approche plus rapide.
    Voiture plutôt sous la balle et de notre côté : le flip regagné part vers le but adverse. Réussite : metrics/episode.py
    (un flip reset en 3 s). `flip_time` élevé : sinon RocketSim croit le flip en cours et amortit la chute."""
    kind = "reset"
    leveled = True

    def setup(self, state, team, r):
        car = _team_cars(state, team)[0]
        d = _dir(team)
        L = 0.5 if self.level is None else self.level
        ball = np.array([r.uniform(-2800, 2800), d * r.uniform(-1500, 4000), r.uniform(700, 1500)])
        azimuth, elevation = r.normal(0.0, 0.8), r.uniform(-0.5, 1.3)
        to_ball = np.array([math.sin(azimuth) * math.cos(elevation), d * math.cos(azimuth) * math.cos(elevation),
                            math.sin(elevation)])
        pos = ball - to_ball * r.uniform(_lerp(130, 300, L), _lerp(260, 900, L))
        up = unit(-to_ball + r.normal(0.0, _lerp(0.03, 0.35, L), 3))
        attack = np.array([0.0, d, 0.0])
        fwd = attack - up * float(np.dot(attack, up))
        fwd = unit(fwd) if norm3(fwd) > 1e-3 else unit(np.cross(up, [1.0, 0.0, 0.0]))
        ball_vel = np.array([r.normal(0, 150), r.normal(0, 150), r.uniform(50, 250)])
        state.ball.position = ball.astype(np.float32)
        state.ball.linear_velocity = ball_vel.astype(np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.position = pos.astype(np.float32)
        car.physics.linear_velocity = (ball_vel + to_ball * r.uniform(_lerp(200, 300, L), _lerp(500, 900, L))).astype(np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.rotation_mtx = np.column_stack([fwd, np.cross(up, fwd), up]).astype(np.float32)
        car.boost_amount = float(r.uniform(30, 100))
        car.on_ground = False
        car.has_jumped = True
        car.has_double_jumped = False
        car.has_flipped = True
        car.flip_time = 1.0
        car.air_time_since_jump = 2.0


class AerialDefenseDrill(_Drill):
    """Lob adverse qui retombe DANS notre but : l'intercepter en l'air (ou au rebond) avant la ligne.
    Réussite : pas de but encaissé en 6 s (metrics/episode.py).

    Avec niveau (s7e) : RENDEZ-VOUS, comme l'exercice aérien. La balle passe au-dessus d'un point M, devant notre but, à
    la hauteur h et à l'instant T, puis continue dans le but (sous la barre, au rebond éventuellement). Notre voiture
    attend entre M et le but, tournée vers la balle :
        0 : h = 140-260 uu (un saut, ou même la voiture au sol), M à 500-1000 uu de la ligne, voiture à 250-600 uu de M ;
        1 : h = 500-1100 uu (vraie aérienne), M à 1200-2500 uu, voiture à 800-2000 uu de M, décalée.
    L'ancien niveau 0 (lob de 350-700 uu, voiture collée à la ligne) passait à 450-600 uu au-dessus d'elle : 0 % de
    réussite et 0 % de touche pour s7c sur 37 M steps, comme pour une voiture immobile (22/10/2026).
    Sans niveau : ancien réglage (niveau 0,5), inchangé."""
    kind = "aerial_defense"
    leveled = True
    GRAVITY = 650.0
    BAR_Z = 550.0          # centre de la balle sous la barre (642 − rayon)

    def setup(self, state, team, r):
        if self.level is None:
            self._setup_legacy(state, team, r)
            return
        d = _dir(team)
        L = self.level
        g = self.GRAVITY
        own_y = -d * cv.BACK_WALL_Y
        xt = r.uniform(-650, 650)                              # point d'entrée dans le but
        s_m = r.uniform(_lerp(500, 1200, L), _lerp(1000, 2500, L))     # M : distance à la ligne
        my = own_y + d * s_m
        mx = float(np.clip(xt * (1 - s_m / 3000) + r.normal(0, _lerp(200, 900, L)), -3000, 3000))
        h = r.uniform(_lerp(140, 500, L), _lerp(260, 1100, L))
        T = r.uniform(_lerp(1.0, 1.1, L), _lerp(1.4, 1.8, L))
        u = r.uniform(_lerp(700, 1200, L), _lerp(1200, 2000, L))      # vitesse horizontale de la balle
        ex, ey = xt - mx, own_y - my
        n = math.hypot(ex, ey)
        ux, uy = ex / n, ey / n
        t2 = n / u                                             # de M à la ligne
        z0_min = 120.0
        vz_hi = (h - z0_min - 0.5 * g * T * T) / T              # vitesse verticale en M pour partir d'au moins z0_min
        vz_bar = (self.BAR_Z - h + 0.5 * g * t2 * t2) / t2      # au-delà, la balle passe au-dessus de la barre
        vz_m = min(vz_hi, vz_bar) - r.uniform(0, _lerp(150, 400, L))
        vz0 = vz_m + g * T                                     # z0 = z0_min + (marge tirée) × T ; sommet < 1400 uu
        z0 = h - vz0 * T + 0.5 * g * T * T
        bx = float(np.clip(mx - ux * u * T, -3600, 3600))
        by = float(np.clip(my - uy * u * T, -4400, 4400))
        state.ball.position = np.array([bx, by, z0], dtype=np.float32)
        drag = 1.0 + 0.0305 * T / 2                            # compense le frottement de l'air de la balle
        state.ball.linear_velocity = np.array([(mx - bx) / T * drag, (my - by) / T * drag, vz0 * drag],
                                              dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in _team_cars(state, team):                   # défenseurs : entre M et le but, tournés vers la balle
            dist = r.uniform(_lerp(250, 800, L), _lerp(600, 2000, L))
            # à côté de la trajectoire (immobile, elle ne bloquerait la balle que par hasard)
            side = (1.0 if r.random() < 0.5 else -1.0) * r.uniform(_lerp(250, 400, L), _lerp(450, 1200, L))
            x = mx + ux * dist - uy * side
            y = my + uy * dist + ux * side
            y = own_y + d * max(d * (y - own_y), 150.0)        # devant la ligne
            yaw = float(np.arctan2(my - y, mx - x) + r.normal(0, _lerp(0.2, 1.0, L)))
            _place_car(car, x, y, yaw, r.uniform(0, _lerp(300, 1200, L)), r.uniform(_lerp(60, 30, L), 100))
        for car in _team_cars(state, 1 - team):               # attaquants qui suivent le lob
            x, y = bx + r.normal(0, 600), by + d * r.uniform(800, 2000)
            _place_car(car, x, y, float(np.arctan2(by - y, bx - x)), r.uniform(800, 1600), r.uniform(0, 100))
        self.last_meet = (mx, my, h, T)                        # tests / diagnostic

    def _setup_legacy(self, state, team, r):
        d = _dir(team)
        L = 0.5
        own_y = -d * cv.BACK_WALL_Y
        bx, by = r.uniform(-2000, 2000), -d * r.uniform(1000, 3500)
        z0 = r.uniform(150, 500)
        zt = r.uniform(150, 500)                               # hauteur au passage de la ligne de but
        apex = max(z0, zt) + max(50.0, r.uniform(_lerp(350, 1200, L), _lerp(700, 1800, L)) - max(z0, zt))
        g = 650.0
        vz = float(np.sqrt(2 * g * (apex - z0)))
        t = (vz + np.sqrt(vz * vz + 2 * g * (z0 - zt))) / g    # temps de vol jusqu'à la ligne (branche descendante)
        xt = r.uniform(-650, 650)
        state.ball.position = np.array([bx, by, z0], dtype=np.float32)
        # ×1,04 : compense le frottement de l'air de la balle
        state.ball.linear_velocity = np.array([(xt - bx) / t, (own_y - by) / t * 1.04, vz], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in _team_cars(state, team):                   # défenseurs : près du but (bas niveau) → loin, décalés
            f = r.uniform(_lerp(0.75, 0.2, L), _lerp(0.95, 0.6, L))
            x = bx * (1 - f) + xt * f + r.normal(0, _lerp(300, 1200, L))
            y = by + (own_y - by) * f
            yaw = float(np.arctan2(by - y, bx - x) + r.normal(0, _lerp(0.3, 1.2, L)))
            _place_car(car, x, y, yaw, r.uniform(0, 1200), r.uniform(_lerp(60, 30, L), 100))
        for car in _team_cars(state, 1 - team):               # attaquants qui suivent le lob
            x, y = bx + r.normal(0, 600), by + d * r.uniform(800, 2000)
            _place_car(car, x, y, float(np.arctan2(by - y, bx - x)), r.uniform(800, 1600), r.uniform(0, 100))


class WallDrill(_Drill):
    """Balle le long d'un mur latéral (au sol contre le mur ou en hauteur), nos voitures qui arrivent."""
    kind = "wall"

    def setup(self, state, team, r):
        side = 1.0 if r.random() < 0.5 else -1.0
        bx = side * r.uniform(cv.SIDE_WALL_X - 400, cv.SIDE_WALL_X - 150)
        by = r.uniform(-3500, 3500)
        state.ball.position = np.array([bx, by, r.uniform(cv.BALL_RESTING_HEIGHT, 1200)], dtype=np.float32)
        state.ball.linear_velocity = np.array([side * r.uniform(0, 600), r.normal(0, 900), r.uniform(-200, 500)],
                                              dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in _team_cars(state, team):
            x, y = bx - side * r.uniform(800, 2200), by + r.normal(0, 1200)
            _place_car(car, x, y, float(np.arctan2(by - y, bx - x)), r.uniform(500, 1600), r.uniform(30, 100))


class LowBoostDrill(_Drill):
    """Nos voitures presque à sec (0-15 de boost), balle loin et neutre : aller ramasser avant de s'engager."""
    kind = "low_boost"

    def setup(self, state, team, r):
        state.ball.position = np.array([r.uniform(-2000, 2000), r.uniform(-1500, 1500), cv.BALL_RESTING_HEIGHT],
                                       dtype=np.float32)
        state.ball.linear_velocity = np.array([r.normal(0, 300), r.normal(0, 300), 0.0], dtype=np.float32)
        for car in _team_cars(state, team):
            car.boost_amount = float(r.uniform(0, 15))


class RecoveryDrill(_Drill):
    """Nos voitures en l'air, mal orientées et en rotation : retomber sur ses roues vite."""
    kind = "recovery"

    def setup(self, state, team, r):
        for car in _team_cars(state, team):
            ph = car.physics
            ph.position = np.array([r.uniform(-3000, 3000), r.uniform(-4000, 4000), r.uniform(300, 1000)],
                                   dtype=np.float32)
            ph.euler_angles = np.array([r.uniform(-np.pi, np.pi), r.uniform(-np.pi, np.pi), r.uniform(-np.pi, np.pi)],
                                       dtype=np.float32)
            ph.angular_velocity = r.normal(0, 2.5, 3).astype(np.float32)
            ph.linear_velocity = (r.normal(0, 500, 3) * np.array([1, 1, 0.5])).astype(np.float32)
            car.boost_amount = float(r.uniform(0, 100))


DRILLS = {cls.kind: cls for cls in (DefenseDrill, ShotDrill, AerialDrill, WallDrill, LowBoostDrill, RecoveryDrill,
                                    AerialDefenseDrill, AirDrill, ResetDrill)}
LEVELED_DRILLS = tuple(k for k, cls in DRILLS.items() if cls.leveled)


class StartStateMutator(StateMutator[GameState]):
    """Tire le type d'état de départ selon des poids configurables (kickoff, random, exercices `drill_*`...).

    Dosage automatique (`adaptive`, exercices seulement) : chaque exercice garde une moyenne glissante de réussite
    (`report`, appelé par le worker en fin d'épisode) et son poids est multiplié par (1,2 - réussite), borné à
    [0,25 ; 1,2] : ce qui est raté revient plus souvent, ce qui est maîtrisé moins souvent. La part totale des
    exercices reste celle de la config (les matchs normaux ne sont pas grignotés)."""

    def __init__(self, start_weights: Dict[str, float], seed: Optional[int] = None, adaptive: bool = False,
                 ema: float = 0.02):
        self.start_weights = dict(start_weights)
        self.rng = random.Random(seed)
        self.mutators = {
            "kickoff": KickoffMutator(),
            "random": RandomStateMutator(seed),
            "attack_defense": AttackDefenseMutator(seed),
            "aerial": AerialBallMutator(seed),
        }
        for i, (kind, cls) in enumerate(DRILLS.items()):
            self.mutators["drill_" + kind] = cls(None if seed is None else seed + 101 * (i + 1))
        unknown = set(self.start_weights) - set(self.mutators)
        if unknown:
            raise ValueError(f"Types d'état de départ inconnus : {unknown}")
        self.adaptive = adaptive
        self.ema = ema
        self.success = {k: 0.5 for k in self.start_weights if k.startswith("drill_")}

    def report(self, kind: str, success: float) -> None:
        if kind in self.success:
            self.success[kind] += self.ema * (float(success) - self.success[kind])

    def current_weights(self) -> Dict[str, float]:
        if not self.adaptive or not self.success:
            return self.start_weights
        w = dict(self.start_weights)
        drill_total = sum(w[k] for k in self.success)
        raw = {k: w[k] * min(1.2, max(0.25, 1.2 - self.success[k])) for k in self.success}
        norm = sum(raw.values())
        if norm > 0:
            for k, v in raw.items():
                w[k] = v * drill_total / norm
        return w

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info.pop("drill_team", None)
        shared_info.pop("drill_level", None)
        kind = _weighted_choice(self.current_weights(), self.rng)
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
