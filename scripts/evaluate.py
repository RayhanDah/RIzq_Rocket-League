"""Matchs simulés entre deux politiques (checkpoint vs checkpoint, ou vs politique aléatoire).

    python scripts/evaluate.py --a runs/<run> --b random --format 1 --games 50
    python scripts/evaluate.py --a runs/<runA> --b runs/<runB>/checkpoints/ckpt_XXX.pt --format 2

Un "match" = parties de 5 minutes de jeu simulé (buts comptés, remise au centre après chaque but).
Les politiques jouent en mode stochastique (comme à l'entraînement) sauf --deterministic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_actor(spec: str):
    import torch
    from rlagent.train import build_actor_from_model_cfg, find_latest_checkpoint
    if spec == "random":
        return None
    p = Path(spec)
    if p.is_dir():
        p = find_latest_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model_cfg = ck["cfg"]["model"] if "cfg" in ck else ck["model_cfg"]
    actor = build_actor_from_model_cfg(model_cfg)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--format", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--game-seconds", type=float, default=300)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    import torch
    from rlagent.config import load_config
    from rlagent.env.factory import build_env

    cfg = load_config()
    cfg["env"]["format_weights"] = {args.format: 1.0}
    cfg["env"]["start_weights"] = {"kickoff": 1.0}
    cfg["env"]["episode_timeout_s"] = args.game_seconds
    cfg["env"]["no_touch_timeout_s"] = args.game_seconds
    env = build_env(cfg, seed=123)
    pol = {"A": load_actor(args.a), "B": load_actor(args.b)}
    tick_s = cfg["env"]["action_repeat"] / 120.0

    wins = {"A": 0, "B": 0, "draw": 0}
    goals_total = {"A": 0, "B": 0}
    for g in range(args.games):
        a_team = g % 2  # alterne les couleurs
        goals = {"A": 0, "B": 0}
        t = 0.0
        obs = env.reset()
        while t < args.game_seconds:
            actions = {}
            for side in ("A", "B"):
                team = a_team if side == "A" else 1 - a_team
                agents = [ag for ag in obs if env.state.cars[ag].team_num == team]
                if pol[side] is None:
                    for ag in agents:
                        actions[ag] = np.random.randint(0, 90, size=(1,))
                    continue
                x = torch.as_tensor(np.stack([obs[ag] for ag in agents]), dtype=torch.float32)
                act, _ = pol[side].act(x, deterministic=args.deterministic)
                for ag, a in zip(agents, act.numpy()):
                    actions[ag] = np.array([a])
            obs, _, term, trunc = env.step(actions)
            t += tick_s
            if any(term.values()):
                scorer = env.state.scoring_team
                goals["A" if scorer == a_team else "B"] += 1
                obs = env.reset()
            elif any(trunc.values()):
                obs = env.reset()
        for s in goals:
            goals_total[s] += goals[s]
        res = "A" if goals["A"] > goals["B"] else "B" if goals["B"] > goals["A"] else "draw"
        wins[res] += 1
        print(f"match {g + 1:3d} : A {goals['A']} - {goals['B']} B")
    n = args.games
    print(f"\nBilan {args.format}v{args.format} sur {n} matchs : A gagne {wins['A']} ({wins['A'] / n:.0%}), "
          f"B gagne {wins['B']} ({wins['B'] / n:.0%}), nuls {wins['draw']} | buts A {goals_total['A']} - {goals_total['B']} B")


if __name__ == "__main__":
    main()
