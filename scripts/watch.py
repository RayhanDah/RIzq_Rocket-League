"""Regarder le bot jouer dans RLViser (visualiseur 3D de RocketSim), à vitesse réelle.

    pip install "rlgym[rl-rlviser]==2.0.1"
    python scripts/watch.py --policy runs/<run> --format 1

RLViser a besoin de son exécutable (téléchargé/lancé par rlviser-py ; voir https://github.com/VirxEC/rlviser).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="dossier de run, checkpoint .pt ou 'random'")
    ap.add_argument("--opponent", default=None, help="par défaut : la même politique")
    ap.add_argument("--format", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--start", default="kickoff", choices=["kickoff", "random"])
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from evaluate import load_actor
    from rlagent.config import load_config
    from rlagent.env.factory import build_env
    from rlgym.rocket_league.rlviser import RLViserRenderer

    # 1 thread suffit en temps réel ; sinon PyTorch prend tous les cœurs et ralentit un entraînement en cours
    torch.set_num_threads(1)
    cfg = load_config(overrides=["env.ball_prediction=true", "env.race_features=true", "env.boost_features=true"])  # sur-ensemble : anciens agents compatibles (fit_obs)
    cfg["env"]["format_weights"] = {args.format: 1.0}
    cfg["env"]["start_weights"] = {args.start: 1.0}
    env = build_env(cfg, seed=7)
    env.renderer = RLViserRenderer(tick_rate=120 / cfg["env"]["action_repeat"])
    blue = load_actor(args.policy)
    orange = load_actor(args.opponent) if args.opponent else blue
    dt = cfg["env"]["action_repeat"] / 120.0 / args.speed

    obs = env.reset()
    try:
        while True:
            t0 = time.perf_counter()
            actions = {}
            for team, pol in ((0, blue), (1, orange)):
                agents = [a for a in obs if env.state.cars[a].team_num == team]
                if pol is None:
                    for a in agents:
                        actions[a] = np.random.randint(0, 90, size=(1,))
                    continue
                act, _ = pol.act(torch.as_tensor(np.stack([obs[a] for a in agents]), dtype=torch.float32))
                for a, x in zip(agents, act.numpy()):
                    actions[a] = np.array([x])
            obs, _, term, trunc = env.step(actions)
            env.render()
            if any(term.values()) or any(trunc.values()):
                obs = env.reset()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
