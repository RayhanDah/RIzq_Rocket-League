"""Audit de style : mesure comment joue un agent, pas seulement s'il gagne. Matchs 1v1 de 5 min en RocketSim.

    python scripts/style_audit.py --a runs/s10_ppo_kaggle/best_actor.pt --b runs/s10_ppo_kaggle/best_actor.pt --det-a
    python scripts/style_audit.py --a <nouvel agent> --b runs/s10_ppo_kaggle/best_actor.pt --det-a --games 40
    python scripts/style_audit.py --a <agent> --b necto --det-a --games 40       # Necto (NectoFamily), argmax

`necto` / `necto:<beta>` : le vrai Necto via l'adaptateur `rlagent/opponents/necto.py` (fidélité mesurée par
`scripts/necto_check.py`, rappelée dans le rapport). Jamais dans le pool d'entraînement.

Toujours le même adversaire pour comparer deux agents. Trois blocs :
  - rush : vitesse, pression, contacts : ce qu'il ne faut pas perdre ;
  - finition : tirs cadrés, balles cadrées déviées par soi-même ;
  - aérien : vraies aériennes (boost en l'air avant la touche) et touches en sautant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

GAME_S = 300.0
CHUNK = 2                      # parties par tâche : chaque camp joue une fois de chaque couleur
AERIAL_Z = 500.0               # hauteur de balle d'une touche aérienne (comme le reward)
AIR_BOOST_S = 0.25             # boost en l'air avant la touche pour parler de vraie aérienne
WALL_TAKEOFF_Z = 300.0         # décollage au-dessus : la voiture quitte un mur
STRONG_DV = 556.0              # frappe (uu/s), seuil de strong_touch

COUNTERS = ("goals", "touches", "strong_touches", "on_target", "off_target", "spoils", "on_target_balls", "saves",
            "aerial_touches", "true_aerials", "wall_aerials", "aerial_goals", "flip_resets", "bumps", "demos", "flips",
            "pickups", "kickoffs", "kickoff_first_touch", "kickoff_won", "kickoff_goals", "own_goals")
SUMS = ("speed", "boost", "supersonic", "air", "air_boost", "low_boost", "own_half", "ball_dist", "closer", "steps")

_env = None
_tick_s = 0.0
_actors = {}


def setup(config: str) -> None:
    global _env, _tick_s
    if _env is not None:
        return
    import torch
    import yaml

    from rlagent.env.factory import build_env
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    cfg["paths"]["collision_meshes"] = str(ROOT / "collision_meshes")
    cfg["env"]["format_weights"] = {1: 1.0}
    cfg["env"]["start_weights"] = {"kickoff": 1.0}
    cfg["env"]["episode_timeout_s"] = GAME_S
    cfg["env"]["no_touch_timeout_s"] = GAME_S
    cfg["env"]["adaptive_drills"] = False
    _tick_s = cfg["env"]["action_repeat"] / 120.0
    _env = build_env(cfg, seed=int(time.time_ns() % 2**31))
    from rlagent.opponents.necto import MixedRepeatAction
    _env.action_parser = MixedRepeatAction(cfg["env"]["action_repeat"])    # indices : inchangé ; Necto : commandes


def actor(spec: str):
    from evaluate import load_actor

    from rlagent.opponents.necto import NectoOpponent, is_necto_spec
    if spec not in _actors:
        seed = int(time.time_ns() % 2**31)
        _actors[spec] = NectoOpponent.from_spec(spec, seed=seed) if is_necto_spec(spec) else load_actor(spec)
    return _actors[spec]


def play_game(spec_a: str, spec_b: str, a_team: int, det_a: bool, det_b: bool = False, stale: float = 0.0):
    import torch

    from rlagent.env.common import is_flip_reset
    from rlagent.env.predict import goal_line_time, goal_threat
    from rlagent.opponents.necto import NectoOpponent, pad_locations_of
    env = _env
    pol = {"A": actor(spec_a), "B": actor(spec_b)}
    necto = [p for p in pol.values() if isinstance(p, NectoOpponent)]
    for p in necto:
        p.new_game(pad_locations_of(env))
    st = {s: {k: 0 for k in COUNTERS} | {k: 0.0 for k in SUMS} for s in "AB"}
    side_of = {a_team: "A", 1 - a_team: "B"}

    def fresh():
        s = env.state
        ball = (s.ball.position.tolist(), s.ball.linear_velocity.tolist())
        cars = {a: {"flip": bool(c.has_flipped), "demo": float(c.demo_respawn_timer), "boost": float(c.boost_amount),
                    "victim": c.bump_victim_id, "air_boost": 0.0, "takeoff_z": 17.0, "ground": bool(c.on_ground),
                    "jumped": bool(c.has_jumped)}
                for a, c in s.cars.items()}
        return ball, cars

    obs = env.reset()
    for p in necto:
        p.on_reset(env.state)
    prev_ball, prev = fresh()
    last_touch = None
    first_touch, kickoff_judged = None, False
    KICKOFF_EVAL_S, KICKOFF_GOAL_S = 3.0, 5.0      # comme le reward kickoff_win ; but « sur kickoff » avant 5 s
    t = 0.0
    spawn = lambda: tuple(tuple(np.round(c.physics.position[:2]).tolist()) for c in env.state.cars.values())
    episodes, ep_start, ep_steps = [], spawn(), 0
    last_actions = {}                      # --stale : action précédente rejouée, comme un paquet RLBot en retard
    while t < GAME_S:
        actions = {}
        for side in "AB":
            team = a_team if side == "A" else 1 - a_team
            agents = [ag for ag in obs if env.state.cars[ag].team_num == team]
            if isinstance(pol[side], NectoOpponent):
                actions.update(pol[side].act(agents, env.state))
                continue
            x = torch.as_tensor(np.stack([obs[ag] for ag in agents]), dtype=torch.float32)
            with torch.no_grad():
                act, _ = pol[side].act(x, deterministic=(det_a if side == "A" else det_b))
            for ag, a in zip(agents, act.numpy()):
                actions[ag] = np.array([a])
        if stale > 0.0:
            for ag in actions:
                if ag in last_actions and np.random.random() < stale:
                    actions[ag] = last_actions[ag]
            last_actions = dict(actions)
        obs, _, term, trunc = env.step(actions)
        t += _tick_s
        state = env.state
        pos, vel = state.ball.position.tolist(), state.ball.linear_velocity.tolist()
        dv = float(np.linalg.norm(np.array(vel) - np.array(prev_ball[1])))
        dists = {a: float(np.linalg.norm(c.physics.position - state.ball.position)) for a, c in state.cars.items()}
        for ag, car in state.cars.items():
            d, p = st[side_of[car.team_num]], prev[ag]
            sign = 1.0 if car.team_num == 0 else -1.0
            v = car.physics.linear_velocity
            boost, demo, ground = float(car.boost_amount), float(car.demo_respawn_timer), bool(car.on_ground)
            d["steps"] += 1
            d["speed"] += float(np.sqrt((v * v).sum()))
            d["boost"] += boost
            d["low_boost"] += boost < 10.0
            d["supersonic"] += bool(car.is_supersonic)
            d["air"] += not ground
            d["air_boost"] += (not ground) and bool(car.is_boosting)
            d["own_half"] += float(car.physics.position[1]) * sign < 0
            d["ball_dist"] += dists[ag]
            d["closer"] += dists[ag] <= min(dists.values())
            d["flips"] += bool(car.has_flipped) and not p["flip"]
            d["pickups"] += boost > p["boost"] + 1.0 and demo <= 0.0 and p["demo"] <= 0.0
            d["demos"] += sum(1 for o, c in state.cars.items() if c.team_num != car.team_num
                              and float(c.demo_respawn_timer) > 0.0 and prev[o]["demo"] <= 0.0
                              and car.bump_victim_id == o)
            d["bumps"] += car.bump_victim_id is not None and car.bump_victim_id != p["victim"]
            if ground:
                p["air_boost"], p["takeoff_z"] = 0.0, float(car.physics.position[2])
            elif car.is_boosting:
                p["air_boost"] += _tick_s
            if goal_threat(*prev_ball, sign) > 0.0:
                d["on_target_balls"] += car.ball_touches > 0
            if car.ball_touches > 0 and first_touch is None:
                first_touch = side_of[car.team_num]
            if car.ball_touches > 0:
                last_touch = (side_of[car.team_num], (not ground) and pos[2] > AERIAL_Z
                              and p["air_boost"] >= AIR_BOOST_S)
                d["touches"] += 1
                d["strong_touches"] += dv >= STRONG_DV
                before, after = goal_threat(*prev_ball, sign), goal_threat(pos, vel, sign)
                if after > 0.0 and before == 0.0:
                    d["on_target"] += 1
                elif after == 0.0 and before == 0.0 and dv >= STRONG_DV and goal_line_time(pos, vel, sign) is not None:
                    d["off_target"] += 1            # frappe qui atteint la ligne de but hors du cadre
                if before > 0.0 and after == 0.0:
                    d["spoils"] += 1                # sa propre balle cadrée, qui ne l'est plus après sa touche
                if goal_threat(*prev_ball, -sign) > 0.0 and goal_threat(pos, vel, -sign) == 0.0:
                    d["saves"] += 1
                if not ground and pos[2] > AERIAL_Z:
                    d["aerial_touches"] += 1
                    if p["air_boost"] >= AIR_BOOST_S:
                        d["true_aerials"] += 1
                    elif p["takeoff_z"] > WALL_TAKEOFF_Z:
                        d["wall_aerials"] += 1
            d["flip_resets"] += is_flip_reset(car, state.ball.position, p["ground"], p["jumped"])
            p.update(flip=bool(car.has_flipped), demo=demo, boost=boost, victim=car.bump_victim_id, ground=ground,
                     jumped=bool(car.has_jumped))
        prev_ball = (pos, vel)
        ep_steps += 1
        scored = any(term.values())
        if not kickoff_judged and (scored or ep_steps * _tick_s >= KICKOFF_EVAL_S):
            kickoff_judged = True                   # balle dans le camp adverse à 3 s (ou but avant) : kickoff gagné
            for s_ in "AB":
                st[s_]["kickoffs"] += 1
            if first_touch is not None:             # premier contact dans les 3 s (ou avant le but)
                st[first_touch]["kickoff_first_touch"] += 1
            if scored:
                st[side_of[state.scoring_team]]["kickoff_won"] += 1
            elif abs(pos[1]) > 0.0:
                st[side_of[0] if pos[1] > 0 else side_of[1]]["kickoff_won"] += 1
        if scored:
            st[side_of[state.scoring_team]]["goals"] += 1
            if ep_steps * _tick_s <= KICKOFF_GOAL_S:
                st[side_of[state.scoring_team]]["kickoff_goals"] += 1
            if last_touch and last_touch[0] != side_of[state.scoring_team]:
                st[last_touch[0]]["own_goals"] += 1
            if last_touch and last_touch[1] and last_touch[0] == side_of[state.scoring_team]:
                st[last_touch[0]]["aerial_goals"] += 1
        if scored or any(trunc.values()):
            episodes.append((ep_start, side_of[state.scoring_team] if scored else "-", ep_steps))
            obs = env.reset()
            for p in necto:
                p.on_reset(env.state)
            prev_ball, prev = fresh()
            last_touch = None
            ep_start, ep_steps = spawn(), 0
            first_touch, kickoff_judged = None, False
    st["episodes"] = episodes       # (placement au kickoff, buteur, durée) : diversité des parties
    return st


def task(config, spec_a, spec_b, det_a, seed, n_games, det_b=False, stale=0.0):
    import torch
    setup(config)
    torch.manual_seed(seed)
    np.random.seed(seed % 2**31)
    return [play_game(spec_a, spec_b, g % 2, det_a, det_b, stale) for g in range(n_games)]


def summarize(d: dict) -> dict:
    n = max(1.0, d["steps"])
    minutes = n * _STEP_S / 60.0
    per_min = lambda k: d[k] / minutes
    shots = d["on_target"] + d["off_target"]
    return {
        "rush": {"vitesse": d["speed"] / n, "supersonique_%": 100 * d["supersonic"] / n, "flips/min": per_min("flips"),
                 "touches/min": per_min("touches"), "frappes/min": per_min("strong_touches"),
                 "distance_balle": d["ball_dist"] / n, "plus_proche_%": 100 * d["closer"] / n,
                 "camp_adverse_%": 100 - 100 * d["own_half"] / n, "contacts/min": per_min("bumps"),
                 "demos/min": per_min("demos")},
        "finition": {"buts/min": per_min("goals"), "cadres/min": per_min("on_target"),
                     "hors_cadre/min": per_min("off_target"),
                     "precision_%": 100 * d["on_target"] / shots if shots else 0.0,
                     "deviations/min": per_min("spoils"),
                     "deviations_%_balles_cadrees": 100 * d["spoils"] / max(1, d["on_target_balls"]),
                     "arrets/min": per_min("saves")},
        "kickoff": {"kickoffs": d["kickoffs"], "premier_contact_%": 100 * d["kickoff_first_touch"] / max(1, d["kickoffs"]),
                    "gagnes_a_3s_%": 100 * d["kickoff_won"] / max(1, d["kickoffs"]),
                    "buts_sur_kickoff_%": 100 * d["kickoff_goals"] / max(1, d["kickoffs"]),
                    "csc/min": per_min("own_goals")},
        "aerien": {"touches_aeriennes/min": per_min("aerial_touches"), "vraies_aeriennes/min": per_min("true_aerials"),
                   "depuis_mur/min": per_min("wall_aerials"), "buts_aeriens/min": per_min("aerial_goals"),
                   "flip_resets/min": per_min("flip_resets"),
                   "temps_en_l_air_%": 100 * d["air"] / n, "boost_en_l_air_%": 100 * d["air_boost"] / max(1.0, d["air"]),
                   "boost_moyen": d["boost"] / n, "moins_de_10_boost_%": 100 * d["low_boost"] / n,
                   "ramassages/min": per_min("pickups")},
    }


_STEP_S = 8 / 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--det-a", action="store_true",
                    help="A en argmax (comme le bot) ; B reste échantillonné (sauf necto : argmax, necto:<beta> sinon)")
    ap.add_argument("--det-b", action="store_true", help="B aussi en argmax (deux bots comme dans le vrai jeu)")
    ap.add_argument("--stale", type=float, default=0.0,
                    help="probabilité par step et par voiture de rejouer l'action précédente (bruit du vrai jeu)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--config", default=str(ROOT / "runs/s10_ppo_kaggle/config.yaml"))
    ap.add_argument("--out", default=None, help="résumé JSON")
    args = ap.parse_args()
    tasks = [(args.config, args.a, args.b, args.det_a, 1000 + c, min(CHUNK, args.games - c), args.det_b, args.stale)
             for c in range(0, args.games, CHUNK)]
    total = {s: {k: 0.0 for k in COUNTERS + SUMS} for s in "AB"}
    wins = draws = 0
    kick_eps = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(task, *t) for t in tasks]):
            for game in f.result():
                kick_eps += [(str(e[0]), e[1], e[2]) for e in game.pop("episodes")]
                for s in "AB":
                    for k, v in game[s].items():
                        total[s][k] += v
                wins += game["A"]["goals"] > game["B"]["goals"]
                draws += game["A"]["goals"] == game["B"]["goals"]
    res = {"a": args.a, "b": args.b, "det_a": args.det_a, "det_b": args.det_b, "stale": args.stale, "games": args.games, "wins": wins, "draws": draws,
           "goals": [total["A"]["goals"], total["B"]["goals"]], "A": summarize(total["A"]), "B": summarize(total["B"])}
    n_starts = len({e[0] for e in kick_eps})
    res["episodes"] = {"total": len(kick_eps), "debuts_distincts": n_starts, "issues_distinctes": len(set(kick_eps))}
    print(f"{args.games} matchs en {time.time() - t0:.0f} s | A : {wins} V, {draws} N, {args.games - wins - draws} D | "
          f"buts {total['A']['goals']:.0f}-{total['B']['goals']:.0f}")
    print(f"remises au centre : {len(kick_eps)}, débuts distincts {n_starts}, (début, buteur, durée) distincts "
          f"{len(set(kick_eps))}" + ("  <- deux politiques déterministes : peu de parties vraiment différentes"
                                     if len(set(kick_eps)) < len(kick_eps) / 4 else ""))
    from rlagent.opponents.necto import fidelity_report, is_necto_spec
    if is_necto_spec(args.a) or is_necto_spec(args.b):
        res["necto_fidelite"] = fidelity_report()
        print("--- Necto (adaptateur RocketSim)\n" + res["necto_fidelite"]["texte"])
    for block in ("rush", "finition", "kickoff", "aerien"):
        print(f"--- {block}" + " " * 22 + "A        B")
        for k in res["A"][block]:
            print(f"{k:30s} {res['A'][block][k]:8.2f} {res['B'][block][k]:8.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
