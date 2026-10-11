"""Mesure le débit de simulation (steps d'agents / s) selon le nombre de workers, pour régler train.n_workers.

    python scripts/benchmark_sps.py --workers 1 2 4 8 --steps 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--envs", type=int, nargs="+", default=[1], help="arènes par worker (train.envs_per_worker)")
    ap.add_argument("--profile", action="append", default=[])
    args = ap.parse_args()

    import torch
    from rlagent.algo.collector import Collector
    from rlagent.config import load_config
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models

    cfg = load_config(args.profile)
    cfg["selfplay"]["hist_prob"] = 0.0
    actor, _ = build_models(cfg, N_ACTIONS)
    torch.set_num_threads(int(cfg["train"]["torch_threads"]))
    results = []
    for w, e in [(w, e) for w in args.workers for e in args.envs]:
        col = Collector(cfg, w, seed=0, envs_per_worker=e)
        try:
            col.collect(actor, 1_000, torch.device("cpu"))  # chauffe
            t = time.perf_counter()
            _, _, n, _ = col.collect(actor, args.steps, torch.device("cpu"))
            sps = n / (time.perf_counter() - t)
        finally:
            col.close()
        results.append((w, e, sps))
        print(f"workers={w:3d} arènes/worker={e:2d} → {sps:,.0f} steps/s (inférence acteur incluse, CPU)",
              flush=True)
    best = max(results, key=lambda r: r[2])
    print(f"\nMeilleur : {best[0]} workers x {best[1]} arènes ({best[2]:,.0f} steps/s). "
          f"À reporter dans configs/<profil>.yaml → train.n_workers / train.envs_per_worker")


if __name__ == "__main__":
    main()
