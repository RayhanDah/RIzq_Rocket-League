"""Fidélité de l'adaptateur Necto (`rlagent/opponents/necto.py`) au vrai bot NectoFamily.

    .venv\\Scripts\\python scripts/necto_check.py dump --steps 1500             # parties A vs Necto, états + obs adaptateur
    .venv-bot\\Scripts\\python scripts/necto_check.py compare                   # code du clone sur paquets RLBot équivalents

`dump` (venv d'entraînement) joue A contre Necto en RocketSim (Necto bleu puis orange) et enregistre à chaque step
l'état brut, l'observation et l'action de l'adaptateur. `compare` (venv du bot : rlbot + rlgym_compat) refait des
`GamePacket` RLBot avec ces états, les passe à `V1GameState` puis au `NectoObsBuilder` et à l'`Agent` du clone
(importés depuis dist/opponents/NectoFamily/necto), et mesure l'écart d'observation par groupe de variables et
l'accord des actions argmax. Résumé : dist/opponents/necto_check.json, relu par style_audit.py dans son rapport.

Ce que la comparaison ne voit pas (pas de jeu ici) : la physique du vrai jeu, le retard RLBot, et l'`air_state` du
paquet. Pour `on_ground`, `dump` estime à part combien de steps le jeu aurait dit autre chose que RocketSim
(règle de rlgym_compat : en saut, au sol pendant les 6 premiers ticks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = ROOT / "dist" / "opponents" / "necto_check.npz"
SUMMARY = ROOT / "dist" / "opponents" / "necto_check.json"
GROUPS = {"drapeaux": slice(0, 5), "position": slice(5, 8), "vitesse": slice(8, 11), "avant": slice(11, 14),
          "haut": slice(14, 17), "vit_angulaire": slice(17, 20), "boost": slice(20, 21), "minuteur": slice(21, 22),
          "au_sol": slice(22, 23), "flip_dispo": slice(23, 24)}


def dump(args) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import style_audit as sa
    import torch

    from rlagent.opponents.necto import NectoOpponent, pad_locations_of
    sa.setup(args.config)
    env = sa._env
    ours = sa.actor(args.a)
    necto = NectoOpponent(beta=1.0)
    rec = {k: [] for k in ("ball", "cars", "pads", "me", "prev", "q", "kv", "act", "game", "ground_gap")}
    for game, necto_team in enumerate((0, 1)):
        necto.new_game(pad_locations_of(env))
        obs = env.reset()
        necto.on_reset(env.state)
        for _ in range(args.steps):
            s = env.state
            agents = list(s.cars)
            me = next(a for a in agents if s.cars[a].team_num == necto_team)
            rec["ball"].append(np.concatenate([s.ball.position, s.ball.linear_velocity, s.ball.angular_velocity]))
            rec["cars"].append([[c.team_num, *c.physics.position, *c.physics.linear_velocity,
                                 *c.physics.angular_velocity, *c.physics.euler_angles, c.boost_amount, c.on_ground,
                                 c.is_jumping, c.has_jumped, c.has_double_jumped, c.has_flipped, c.is_flipping,
                                 c.air_time_since_jump, c.flip_time, c.demo_respawn_timer, c.jump_time]
                                for c in s.cars.values()])
            rec["pads"].append(np.array(s.boost_pad_timers))
            rec["me"].append(agents.index(me))
            rec["prev"].append(necto.prev.get(me, np.zeros(8)).copy())
            q, kv, mask = necto.build_obs(me, s)
            rec["q"].append(q[0, 0])
            rec["kv"].append(kv[0])
            rec["act"].append(necto.policy(q, kv, mask))
            rec["game"].append(game)
            c = s.cars[me]
            game_ground = c.on_ground if not c.is_jumping else c.jump_time <= 6 / 120
            rec["ground_gap"].append(bool(game_ground) != bool(c.on_ground))
            rows = necto._kickoff_rows(me, s)
            if rows is None:
                rows = np.repeat(rec["act"][-1][None], 8, axis=0)
            necto.prev[me] = rows[-1].astype(np.float64)
            actions = {me: rows}
            for ag in agents:
                if ag != me:
                    x = torch.as_tensor(obs[ag][None], dtype=torch.float32)
                    with torch.no_grad():
                        actions[ag] = np.array([int(ours.act(x, deterministic=True)[0][0])])
            obs, _, term, trunc = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                obs = env.reset()
                necto.on_reset(env.state)
    out = {k: np.array(v) for k, v in rec.items()}
    out["pad_locations"] = pad_locations_of(env)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    kv = out["kv"]
    print(f"{len(out['q'])} steps -> {args.out}")
    print(f"plages kv : min {kv.min():.2f} max {kv.max():.2f} ; |pos| max {np.abs(kv[:, :, 5:8]).max():.2f} ; "
          f"q drapeaux {np.unique(out['q'][:, :5], axis=0).tolist()}")
    print(f"on_ground du jeu (air_state, règle rlgym_compat) != RocketSim : {100 * out['ground_gap'].mean():.2f} % des steps")


def compare(args) -> None:
    necto_dir = ROOT / "dist" / "opponents" / "NectoFamily" / "necto"
    sys.path.insert(0, str(necto_dir))
    from agent import Agent
    from necto_obs import NectoObsBuilder
    from rlbot import flat
    from rlgym_compat import V1GameState
    with np.load(args.dump) as z:
        d = {k: z[k] for k in z.files}          # NpzFile relit et décompresse tout à chaque accès
    pads = d["pad_locations"]
    field = flat.FieldInfo(boost_pads=[flat.BoostPad(location=flat.Vector3(*p), is_full_boost=bool(p[2] > 72))
                                       for p in pads], goals=[])
    agent = Agent()
    v3 = lambda a: flat.Vector3(float(a[0]), float(a[1]), float(a[2]))

    def packet(i: int, frame: int):
        players = []
        for k, c in enumerate(d["cars"][i]):
            (team, px, py, pz, vx, vy, vz, wx, wy, wz, pitch, yaw, roll, boost, ground, _jumping, jumped, dj, flipped,
             flipping, air_t, flip_t, demo, _jt) = c
            air = flat.AirState.OnGround if ground else (flat.AirState.Dodging if flipping else flat.AirState.InAir)
            players.append(flat.PlayerInfo(
                physics=flat.Physics(location=v3((px, py, pz)), rotation=flat.Rotator(float(pitch), float(yaw),
                                                                                      float(roll)),
                                     velocity=v3((vx, vy, vz)), angular_velocity=v3((wx, wy, wz))),
                score_info=flat.ScoreInfo(), hitbox=flat.BoxShape(118.0074, 84.19941, 36.15907),
                hitbox_offset=flat.Vector3(13.87566, 0, 20.75499), latest_touch=None, air_state=air,
                dodge_timeout=float(1.25 - air_t) if (jumped and not ground) else -1.0,
                demolished_timeout=float(demo) if demo > 0 else -1.0, team=int(team), boost=float(boost),
                player_id=k + 1, name=f"p{k}", has_jumped=bool(jumped), has_double_jumped=bool(dj),
                has_dodged=bool(flipped), dodge_elapsed=float(flip_t), last_input=flat.ControllerState()))
        b = d["ball"][i]
        ball = flat.BallInfo(physics=flat.Physics(location=v3(b[0:3]), rotation=flat.Rotator(0, 0, 0),
                                                  velocity=v3(b[3:6]), angular_velocity=v3(b[6:9])),
                             shape=flat.SphereShape(182.5))
        return flat.GamePacket(
            players=players, balls=[ball], teams=[flat.TeamInfo(team_index=0), flat.TeamInfo(team_index=1)],
            boost_pads=[flat.BoostPadState(is_active=bool(t == 0), timer=float(t)) for t in d["pads"][i]],
            match_info=flat.MatchInfo(seconds_elapsed=frame / 120, match_phase=flat.MatchPhase.Active,
                                      frame_num=frame, world_gravity_z=-650.0, game_speed=1.0))

    n = len(d["q"])
    diff_q, diff_kv, agree, agree_same_obs = [], [], [], []
    cur_game, gs, builder = None, None, None
    for i in range(n):
        if i % 500 == 0:
            print(f"{i}/{n}", flush=True)
        if d["game"][i] != cur_game:                  # nouvelle partie : bot relancé (minuteurs à zéro)
            cur_game = d["game"][i]
            NectoObsBuilder.demo_timers = __import__("collections").Counter()
            builder = NectoObsBuilder(field)
            gs = V1GameState(field, tick_skip=8)
        frame = 1000 + 8 * i
        gs.update(packet(i, frame))
        me = int(d["me"][i])
        player = gs.players[me]
        team = player.team_num
        mates = [p for p in gs.players if p.team_num == team and p is not player]
        opps = [p for p in gs.players if p.team_num != team]
        gs.players = [player] + mates + opps
        q, kv, _mask = builder.build_obs(player, gs, d["prev"][i].copy())
        diff_q.append(np.abs(q[0, 0] - d["q"][i]))
        diff_kv.append(np.abs(kv[0] - d["kv"][i]))
        a_ref, _ = agent.act((q, kv, _mask), 1)
        agree.append(np.array_equal(np.asarray(a_ref, dtype=np.float32), d["act"][i].astype(np.float32)))
        a_same, _ = agent.act((d["q"][i][None, None], d["kv"][i][None], np.zeros((1, kv.shape[1]))), 1)
        agree_same_obs.append(np.array_equal(np.asarray(a_same, dtype=np.float32), d["act"][i].astype(np.float32)))
    dq, dkv = np.array(diff_q), np.array(diff_kv)
    groups = {}
    for name, sl in GROUPS.items():
        e = dkv[:, :, sl]
        groups[name] = {"max": float(e.max()), "steps_ecart_%": float(100 * (e.max(axis=(1, 2)) > 1e-3).mean())}
    res = {"steps": n, "q_prev_action_max": float(dq[:, 24:].max()), "q_max": float(dq.max()),
           "kv_max": float(dkv.max()), "groupes": groups,
           "obs_identique_%": float(100 * ((dq.max(axis=1) < 1e-3) & (dkv.max(axis=(1, 2)) < 1e-3)).mean()),
           "actions_identiques_%": float(100 * np.mean(agree)),
           "actions_identiques_meme_obs_%": float(100 * np.mean(agree_same_obs)),
           "on_ground_jeu_different_%": float(100 * d["ground_gap"].mean())}
    print(json.dumps(res, indent=1))
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--a", default=str(ROOT / "runs/s10_ppo_kaggle/pool/actor_0013535043833.pt"))
    p.add_argument("--steps", type=int, default=1500, help="steps par couleur de Necto")
    p.add_argument("--config", default=str(ROOT / "runs/s10_ppo_kaggle/config.yaml"))
    p.add_argument("--out", default=str(DEFAULT_DUMP))
    p = sub.add_parser("compare")
    p.add_argument("--dump", default=str(DEFAULT_DUMP))
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
