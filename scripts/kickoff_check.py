"""Kickoffs du vrai jeu contre RocketSim, à partir d'un journal de bot.py (2e argument, avec décisions et balle).

Pour chaque kickoff du match réel : placement des voitures, décisions du bot, première touche (qui, quand), suite
(balle chez qui à +3 s, but dans les 6 s). Puis le même kickoff rejoué dans RocketSim : même placement, même bot
(argmax), même adversaire (adaptateur Necto, kickoff codé en dur), comme à l'entraînement et dans style_audit. On
compare décision par décision, et l'observation de la 1re décision champ par champ.

    python scripts/kickoff_check.py runs/s10_ppo_kaggle/real_2026-11-24b/trace_v30160_necto.jsonl \\
        --actor runs/s10_ppo_kaggle/pool/actor_0030160068410.pt --team 0 --opponent necto
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

KICKOFF, GOAL = 2, 4                   # RLBot v5 MatchPhase
TICK_S = 1 / 120
WINDOW_S = 3.5                         # suivi après le top départ
TOUCH_SPEED = 30.0                     # la balle bouge : 1re touche


def obs_names() -> list:
    car = [f"{k}{a}" for k in ("pos", "fwd", "up", "vel", "angvel") for a in "xyz"] + [
        "boost", "on_ground", "demo", "is_boosting", "supersonic", "has_flip"]
    self_ = (["soi." + n for n in car] + [f"soi.vel_locale{a}" for a in "xyz"]
             + ["soi." + n for n in ("has_jumped", "is_jumping", "has_flipped", "is_flipping", "has_double_jumped",
                                     "can_flip", "air_time", "jump_time", "handbrake", "holding_jump")]
             + [f"soi.balle_rel{a}" for a in "xyz"] + [f"soi.balle_rel_locale{a}" for a in "xyz"]
             + [f"soi.balle_vel_rel{a}" for a in "xyz"] + ["soi.dist_balle"]
             + [f"soi.action_prec.{k}" for k in ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost",
                                                 "handbrake")]
             + ["ctx.rang", "ctx.coequipiers", "ctx.adversaires", "ctx.kickoff"]
             + [f"balle.pos{a}" for a in "xyz"] + [f"balle.vel{a}" for a in "xyz"] + [f"balle.ang{a}" for a in "xyz"]
             + [f"pad{i:02d}" for i in range(34)] + [f"balle_but_adv{a}" for a in "xyz"]
             + [f"balle_but_nous{a}" for a in "xyz"] + [f"pred{i:02d}" for i in range(14)])
    other = (["adv.role_mate", "adv.role_opp"] + ["adv." + n for n in car] + [f"adv.rel{a}" for a in "xyz"]
             + [f"adv.vel_rel{a}" for a in "xyz"] + [f"adv.rel_locale{a}" for a in "xyz"] + ["adv.dist_balle", "adv.rang"])
    return self_ + other + [f"autre{i}.{n}" for i in range(1, 5) for n in other] + [f"masque{i}" for i in range(5)]


def load(path: Path):
    ticks, steps = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if "step" in r:
                steps.append(r)
            elif r["f"] not in ticks:
                ticks[r["f"]] = r
    return [ticks[k] for k in sorted(ticks)], sorted(steps, key=lambda s: s["step"])


def player(r, team):
    return next(p for p in r["p"] if p["team"] == team)


def real_kickoffs(rows, steps, team: int) -> list:
    out = []
    for i in range(1, len(rows)):
        if rows[i]["phase"] != KICKOFF or rows[i - 1]["phase"] == KICKOFF or rows[i]["b"] is None:
            continue
        go = rows[i]["f"]
        end = go + int(WINDOW_S / TICK_S)
        seg = [r for r in rows[i:] if r["f"] <= end]
        me, opp = player(rows[i], team), player(rows[i], 1 - team)
        touch = None
        for r in seg:
            v = np.array(r["b"][1])
            if np.linalg.norm(v) > TOUCH_SPEED:
                b = np.array(r["b"][0])
                d = {t: float(np.linalg.norm(np.array(player(r, t)["pos"]) - b)) for t in (0, 1)}
                touch = {"t": round((r["f"] - go) * TICK_S, 3), "by": "nous" if min(d, key=d.get) == team else "adv"}
                break
        at3 = next((r for r in seg if r["f"] >= go + 360), seg[-1])
        sign = 1.0 if team == 0 else -1.0
        goal = None
        for r in rows[i:]:
            if r["f"] > go + 720:
                break
            if r["phase"] == GOAL:
                goal = {"t": round((r["f"] - go) * TICK_S, 2),
                        "pour": "nous" if r["b"] and r["b"][0][1] * sign > 0 else "adv"}
                break
        j = i - 1                                 # début du compte à rebours : la 1re décision y est prise
        while j > 0 and rows[j - 1]["phase"] == rows[i - 1]["phase"]:
            j -= 1
        decs = [s for s in steps if rows[j]["f"] <= s["step"] <= end]
        out.append({"go": go, "spawn_nous": [round(x) for x in me["pos"][:2]],
                    "spawn_adv": [round(x) for x in opp["pos"][:2]], "touche": touch,
                    "balle_y_3s": round(at3["b"][0][1] * sign), "but_6s": goal,
                    "actions": [s["a"] for s in decs], "obs0": decs[0]["o"] if decs else None,
                    "entrees_nous": [player(r, team)["in"] for r in seg[:120]]})
    return out


class SimKickoff:
    """L'environnement d'entraînement (env/factory.py), départs au kickoff seulement, 1v1."""

    def __init__(self, config: Path, actor_path: str, opponent: str):
        import yaml

        from evaluate import load_actor
        from rlagent.env.factory import build_env
        from rlagent.opponents.necto import MixedRepeatAction
        from rlagent.opponents.nexto import bot_from_spec
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        cfg["paths"]["collision_meshes"] = str(ROOT / "collision_meshes")
        cfg["env"].update({"format_weights": {1: 1.0}, "start_weights": {"kickoff": 1.0}, "episode_timeout_s": 30,
                           "no_touch_timeout_s": 30, "adaptive_drills": False})
        self.env = build_env(cfg, seed=int(time.time_ns() % 2**31))
        self.env.action_parser = MixedRepeatAction(cfg["env"]["action_repeat"])
        self.repeat = int(cfg["env"]["action_repeat"])
        self.actor = load_actor(actor_path)
        self.opp = bot_from_spec(opponent, seed=0)
        from rlagent.opponents.necto import pad_locations_of
        self.opp.new_game(pad_locations_of(self.env))

    def play(self, team: int, spawn_me, tries: int = 60) -> dict | None:
        import torch
        env = self.env
        for _ in range(tries):                        # le placement est tiré au hasard : on retire jusqu'au bon
            obs = env.reset()
            me = next(a for a, c in env.state.cars.items() if c.team_num == team)
            if np.linalg.norm(env.state.cars[me].physics.position[:2] - np.array(spawn_me)) < 60:
                break
        else:
            return None
        self.opp.on_reset(env.state)
        obs0 = obs[me].tolist()
        actions, touch, t = [], None, 0.0
        sign = 1.0 if team == 0 else -1.0
        ball_y_3s, goal = None, None
        while t < 6.0:
            acts = {}
            others = [a for a in obs if a != me]
            acts.update(self.opp.act(others, env.state))
            with torch.no_grad():
                a, _ = self.actor.act(torch.as_tensor(obs[me][None], dtype=torch.float32), deterministic=True)
            if t < WINDOW_S:
                actions.append(int(a.item()))
            acts[me] = np.array([int(a.item())])
            obs, _, term, trunc = env.step(acts)
            t += self.repeat * TICK_S
            st = env.state
            if touch is None and np.linalg.norm(st.ball.linear_velocity) > TOUCH_SPEED:
                d = {c.team_num: float(np.linalg.norm(c.physics.position - st.ball.position)) for c in st.cars.values()}
                touch = {"t": round(t, 3), "by": "nous" if min(d, key=d.get) == team else "adv"}
            if ball_y_3s is None and t >= 3.0:
                ball_y_3s = round(float(st.ball.position[1]) * sign)
            if any(term.values()):
                goal = {"t": round(t, 2), "pour": "nous" if float(st.ball.position[1]) * sign > 0 else "adv"}
                break
            if any(trunc.values()):
                break
        return {"touche": touch, "balle_y_3s": ball_y_3s, "but_6s": goal, "actions": actions, "obs0": obs0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--actor", required=True)
    ap.add_argument("--team", type=int, default=0, help="équipe de notre bot dans le match réel")
    ap.add_argument("--opponent", default="necto")
    ap.add_argument("--config", default=str(ROOT / "runs/s10_ppo_kaggle/config.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows, steps = load(Path(args.trace))
    kicks = real_kickoffs(rows, steps, args.team)
    print(f"{len(rows)} paquets, {len(steps)} décisions, {len(kicks)} kickoffs")
    sim = SimKickoff(Path(args.config), args.actor, args.opponent)
    names = obs_names()
    report = []
    for k in kicks:
        s = sim.play(args.team, k["spawn_nous"])
        if s is None:
            print(f"kickoff {k['go']} : placement {k['spawn_nous']} introuvable en simulation")
            continue
        n_same = next((i for i, (a, b) in enumerate(zip(k["actions"], s["actions"])) if a != b),
                      min(len(k["actions"]), len(s["actions"])))
        diffs = []
        if k["obs0"] is not None:
            d = np.abs(np.array(k["obs0"]) - np.array(s["obs0"]))
            diffs = [(names[i] if i < len(names) else str(i), round(k["obs0"][i], 4), round(s["obs0"][i], 4))
                     for i in np.argsort(-d)[:8] if d[i] > 1e-3]
        rep = {"go": k["go"], "spawn": k["spawn_nous"], "reel": {x: k[x] for x in ("touche", "balle_y_3s", "but_6s")},
               "sim": {x: s[x] for x in ("touche", "balle_y_3s", "but_6s")}, "decisions_identiques": n_same,
               "actions_reel": k["actions"][:12], "actions_sim": s["actions"][:12], "obs0_ecarts": diffs}
        report.append(rep)
        tr, ts = k["touche"] or {}, s["touche"] or {}
        print(f"\nkickoff {k['go']} placement {k['spawn_nous']} : 1re touche réel {tr.get('by')} à {tr.get('t')} s, "
              f"sim {ts.get('by')} à {ts.get('t')} s ; balle à +3 s réel {k['balle_y_3s']}, sim {s['balle_y_3s']} ; "
              f"but < 6 s réel {k['but_6s']}, sim {s['but_6s']}")
        print(f"  décisions identiques au départ : {n_same} ; réel {k['actions'][:10]} ; sim {s['actions'][:10]}")
        if diffs:
            print("  1re observation, plus grands écarts (nom, réel, sim) : " + "; ".join(map(str, diffs)))
    if report:
        same = [r["decisions_identiques"] for r in report]
        win_r = sum((r["reel"]["touche"] or {}).get("by") == "nous" for r in report)
        win_s = sum((r["sim"]["touche"] or {}).get("by") == "nous" for r in report)
        g_r = sum(1 for r in report if (r["reel"]["but_6s"] or {}).get("pour") == "adv")
        g_s = sum(1 for r in report if (r["sim"]["but_6s"] or {}).get("pour") == "adv")
        print(f"\n{len(report)} kickoffs : 1re touche pour nous réel {win_r}, sim {win_s} ; but adverse < 6 s réel {g_r}, "
              f"sim {g_s} ; décisions identiques au départ (médiane) {int(np.median(same))}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
