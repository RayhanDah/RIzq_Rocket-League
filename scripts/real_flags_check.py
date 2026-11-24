"""Indicateurs de la voiture dans le vrai jeu contre RocketSim, à partir d'un journal de bot.py (2e argument).

Le bot ne voit pas l'état interne du jeu : rlgym_compat le reconstruit à partir du paquet RLBot (air_state,
has_jumped, dodge_timeout, entrées…). À l'entraînement, ces indicateurs viennent de RocketSim. Ce script rejoue le
match tick par tick dans RocketSim : à chaque tick, la physique de la voiture est recalée sur le paquet du vrai jeu,
on applique les entrées réellement jouées, et on compare les indicateurs internes de RocketSim à ceux du bot.

    python scripts/real_flags_check.py runs/s10_ppo_kaggle/real_2026-11-14/trace_v13720.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLAGS = ("on_ground", "is_jumping", "has_jumped", "has_flipped", "is_flipping", "has_double_jumped", "can_flip",
         "has_flip", "air_time_since_jump", "jump_time", "handbrake", "is_holding_jump", "is_boosting", "flip_time")
IN_OBS = ("on_ground", "is_jumping", "has_jumped", "has_flipped", "is_flipping", "has_double_jumped", "can_flip",
          "has_flip", "air_time_since_jump", "jump_time", "handbrake", "is_holding_jump", "is_boosting")
DOUBLEJUMP_MAX_DELAY, FLIP_TORQUE_TIME = 1.25, 0.65
ACTIVE_PHASES = (2, 3)            # Kickoff, Active (RLBot v5 MatchPhase)
SETTLE_TICKS = 30                 # ticks ignorés après un recalage complet (remise en jeu, démolition)


def obs_scale(name: str, v: float) -> float:
    """Valeur telle que le réseau la voit (env/obs.py)."""
    if name == "air_time_since_jump":
        return min(v, 1.25) / 1.25
    if name == "jump_time":
        return min(v, 0.2) / 0.2
    if name == "flip_time":
        return v
    return float(v)


def sim_flags(st) -> dict:
    """Mêmes définitions que rlgym (rocketsim_engine.py et api/car.py)."""
    on_ground = sum(st.wheels_with_contact) >= 3
    has_flip = (not st.has_double_jumped and not st.has_flipped and st.air_time_since_jump < DOUBLEJUMP_MAX_DELAY)
    hold = bool(st.last_controls.jump)
    return {"on_ground": on_ground, "is_jumping": st.is_jumping, "has_jumped": st.has_jumped,
            "has_flipped": st.has_flipped, "is_flipping": st.has_flipped and st.flip_time < FLIP_TORQUE_TIME,
            "has_double_jumped": st.has_double_jumped, "can_flip": (not on_ground) and (not hold) and has_flip,
            "has_flip": has_flip, "air_time_since_jump": st.air_time_since_jump, "jump_time": st.jump_time,
            "handbrake": st.handbrake_val, "is_holding_jump": hold, "is_boosting": st.time_spent_boosting > 0,
            "flip_time": st.flip_time}


def load(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break                     # dernière ligne coupée (jeu fermé)
            if "f" in r:                  # les lignes « step » (décisions du bot) sont lues par kickoff_check.py
                rows.append(r)
    rows.sort(key=lambda r: r["f"])
    out, seen = [], set()
    for r in rows:                        # un paquet peut être vu deux fois : on garde le premier
        if r["f"] not in seen:
            seen.add(r["f"])
            out.append(r)
    return out


def replay(rows, team: int, offset: int):
    import RocketSim as rs

    from rlagent.config import load_config
    from rlagent.env.factory import init_rocketsim

    init_rocketsim(load_config([]))
    arena = rs.Arena(rs.GameMode.SOCCAR)
    arena.set_car_ball_collision(False)
    car = arena.add_car(team, rs.CarConfig.OCTANE)
    stats = {k: defaultdict(float) for k in IN_OBS}
    examples = defaultdict(list)
    settle, prev_pos, n = 0, None, 0

    def player(r):
        return next(p for p in r["p"] if p["team"] == team)

    def sync(st, p):
        st.pos = rs.Vec(*p["pos"])
        st.vel = rs.Vec(*p["vel"])
        st.ang_vel = rs.Vec(*p["ang"])
        pitch, yaw, roll = p["rot"]
        st.rot_mat = rs.Angle(yaw=yaw, pitch=pitch, roll=roll).as_rot_mat()
        st.boost = float(p["boost"])

    for i in range(len(rows) - 1 - max(offset, 0)):
        r, nxt = rows[i], rows[i + 1]
        if r["phase"] not in ACTIVE_PHASES or nxt["phase"] not in ACTIVE_PHASES:
            prev_pos = None
            continue
        p, pn = player(r), player(nxt)
        if p["demo"] > 0 or pn["demo"] > 0:
            prev_pos = None
            continue
        jump = prev_pos is None or sum((a - b) ** 2 for a, b in zip(p["pos"], prev_pos)) > 300.0 ** 2
        prev_pos = pn["pos"]
        if jump:                          # recalage complet : état interne inconnu, on repart d'un état neuf
            st = rs.CarState()
            sync(st, p)
            car.set_state(st)
            settle = SETTLE_TICKS
        st = car.get_state()
        sync(st, p)
        car.set_state(st)
        a = rows[i + offset]["p"]
        li = next(q for q in a if q["team"] == team)["in"]
        car.set_controls(rs.CarControls(throttle=li[0], steer=li[1], pitch=li[2], yaw=li[3], roll=li[4],
                                        jump=bool(li[5]), boost=bool(li[6]), handbrake=bool(li[7])))
        gap = max(1, nxt["f"] - r["f"])
        arena.step(gap)
        if settle > 0:
            settle -= 1
            continue
        sim = sim_flags(car.get_state())
        bot = dict(zip(FLAGS, nxt["c"][str(team)]))
        n += 1
        for k in IN_OBS:
            s, b = obs_scale(k, float(sim[k])), obs_scale(k, float(bot[k]))
            d = abs(s - b)
            stats[k]["abs"] += d
            if d > 0.05:
                stats[k]["bad"] += 1
                stats[k]["bot>sim" if b > s else "sim>bot"] += 1
                stats[k][f"air_state_{pn['air']}"] += 1          # 0 sol, 1 saut, 2 double saut, 3 flip, 4 en l'air
                if pn["boost"] < 1.0:
                    stats[k]["empty_tank"] += 1
                if len(examples[k]) < 6:
                    examples[k].append({"f": nxt["f"], "air": pn["air"], "hj": pn["hj"], "dto": round(pn["dto"], 3),
                                        "boost": round(pn["boost"], 1), "in_boost": li[6], "in_jump": li[5],
                                        "sim_ground": sim["on_ground"], "sim": round(float(sim[k]), 3),
                                        "bot": round(float(bot[k]), 3)})
    return n, stats, examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1],
                    help="entrée du paquet t+k appliquée entre t et t+1 (on garde le meilleur)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(Path(args.trace))
    print(f"{len(rows)} paquets, frames {rows[0]['f']}-{rows[-1]['f']}")
    report = {}
    for team in sorted({p["team"] for p in rows[0]["p"]}):
        best = None
        for off in args.offsets:
            n, stats, ex = replay(rows, team, off)
            total_bad = sum(stats[k]["bad"] for k in ("on_ground", "is_jumping", "has_jumped"))
            print(f"équipe {team}, décalage {off} : {n} ticks comparés, écarts (sol + saut) {total_bad}")
            if best is None or total_bad < best[0]:
                best = (total_bad, off, n, stats, ex)
        _, off, n, stats, ex = best
        print(f"\n--- équipe {team} (décalage {off}, {n} ticks) : % de ticks en écart > 0,05 (échelle de l'obs)")
        print(f"{'indicateur':22}{'écart %':>9}{'moy |d|':>9}{'bot>sim':>9}{'sim>bot':>9}")
        rep = {}
        for k in IN_OBS:
            s = stats[k]
            rep[k] = {"bad_pct": 100 * s["bad"] / max(n, 1), "mean_abs": s["abs"] / max(n, 1),
                      "bot_gt_sim": int(s["bot>sim"]), "sim_gt_bot": int(s["sim>bot"]),
                      "by_air_state": {a: int(v) for a, v in s.items() if a.startswith("air_state_")},
                      "empty_tank": int(s["empty_tank"]), "examples": ex[k]}
            print(f"{k:22}{rep[k]['bad_pct']:9.2f}{rep[k]['mean_abs']:9.4f}{int(s['bot>sim']):9d}{int(s['sim>bot']):9d}")
        report[str(team)] = {"offset": off, "ticks": n, "flags": rep}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
