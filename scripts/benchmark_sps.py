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
    for w in args.workers:
        col = Collector(cfg, w, seed=0)
        try:
            col.collect(actor, 1_000, torch.device("cpu"))  # chauffe
            t = time.perf_counter()
            _, _, n, _ = col.collect(actor, args.steps, torch.device("cpu"))
            sps = n / (time.perf_counter() - t)
        finally:
            col.close()
        results.append((w, sps))
        print(f"workers={w:3d} → {sps:,.0f} steps/s (inférence acteur incluse, CPU)")
    best = max(results, key=lambda r: r[1])
    print(f"\nMeilleur : {best[0]} workers ({best[1]:,.0f} steps/s). "
          f"À reporter dans configs/<profil>.yaml → train.n_workers")


if __name__ == "__main__":
    main()
