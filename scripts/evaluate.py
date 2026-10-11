"""Matchs simulés entre deux politiques (checkpoint vs checkpoint, ou vs politique aléatoire).

    python scripts/evaluate.py --a runs/<run> --b random --format 1 --games 50
    python scripts/evaluate.py --a runs/<runA> --b runs/<runB>/checkpoints/ckpt_XXX.pt --format 2
    python scripts/evaluate.py --a runs/<run> --scenario empty_goal --trials 200    # critère S1-20

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


def noop_action_index() -> int:
    from rlgym.rocket_league.action_parsers import LookupTableAction
    table = LookupTableAction.make_lookup_table()
    return int(np.argmin(np.abs(table).sum(axis=1)))


def empty_goal_eval(spec: str, trials: int, seconds: float, deterministic: bool, seed: int = 0) -> dict:
    """S1-20 : taux de buts du bleu en 1v1 face à un but vide (adversaire garé loin et immobile)."""
    import torch
    from rlagent.config import load_config
    from rlagent.env.factory import build_env
    from rlagent.env.mutators import EmptyGoalMutator

    cfg = load_config()
    cfg["env"]["episode_timeout_s"] = seconds
    cfg["env"]["no_touch_timeout_s"] = seconds
    env = build_env(cfg, seed=seed, state_mutator=EmptyGoalMutator(seed))
    actor = load_actor(spec)
    noop = noop_action_index()
    rng = np.random.default_rng(seed)
    goals = own_goals = 0
    times = []
    for _ in range(trials):
        obs = env.reset()
        t = 0.0
        while True:
            actions = {}
            for ag in obs:
                if env.state.cars[ag].team_num == 1:
                    actions[ag] = np.array([noop])
                elif actor is None:
                    actions[ag] = np.array([rng.integers(0, 90)])
                else:
                    x = torch.as_tensor(obs[ag][None], dtype=torch.float32)
                    actions[ag] = np.array([int(actor.act(x, deterministic=deterministic)[0])])
            obs, _, term, trunc = env.step(actions)
            t += cfg["env"]["action_repeat"] / 120.0
            if any(term.values()):
                if env.state.scoring_team == 0:
                    goals += 1
                    times.append(t)
                else:
                    own_goals += 1
                break
            if any(trunc.values()):
                break
    env.close()
    return {"trials": trials, "goal_rate": goals / trials, "own_goal_rate": own_goals / trials,
            "mean_time_s": float(np.mean(times)) if times else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="random")
    ap.add_argument("--scenario", default="match", choices=["match", "empty_goal"])
    ap.add_argument("--trials", type=int, default=200, help="empty_goal : nombre d'essais")
    ap.add_argument("--seconds", type=float, default=10.0, help="empty_goal : durée max d'un essai")
    ap.add_argument("--format", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--game-seconds", type=float, default=300)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    if args.scenario == "empty_goal":
        res = empty_goal_eval(args.a, args.trials, args.seconds, args.deterministic)
        print(f"But vide ({args.trials} essais de {args.seconds:.0f} s) : buts {res['goal_rate']:.0%}, "
              f"contre son camp {res['own_goal_rate']:.0%}, temps moyen {res['mean_time_s']:.1f} s")
        return

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
