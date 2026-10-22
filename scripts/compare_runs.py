"""Compare des runs AU MÊME global_step (moyennes sur une fenêtre de ±0,5 M steps), à partir des metrics.jsonl.

    python scripts/compare_runs.py s2_spo_laptop s4_spo_laptop --at 1 2.5 5 10
    python scripts/compare_runs.py s2_spo_laptop s4_spo_laptop --at 5 --keys learn/entropy touches_per_min

C'est la bonne façon de juger un changement d'apprentissage : à compute égal (steps), pas à durée égale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYS = ["learn/entropy", "learn/kl_final", "learn/lr_actor", "learn/explained_variance", "touches_per_min",
                "goals_per_min", "by_format/1v1/touches_per_min", "by_format/1v1/goals_per_min", "ep_length_s",
                "elo", "vs_hist/win_rate", "sps_total"]


def load(run: str):
    path = Path(run)
    if not path.exists():
        path = ROOT / "runs" / run
    return [json.loads(line) for line in open(path / "metrics.jsonl", encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--at", type=float, nargs="+", default=[1, 2.5, 5], help="global_step en millions")
    ap.add_argument("--window", type=float, default=0.5, help="demi-fenêtre en millions de steps")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    args = ap.parse_args()
    data = {r: load(r) for r in args.runs}
    width = max(len(r) for r in args.runs)
    for key in args.keys:
        print(f"\n{key}")
        print(" " * (width + 2) + "".join(f"{m:>10g}M" for m in args.at))
        for r, rows in data.items():
            cells = []
            for m in args.at:
                sel = [x[key] for x in rows if abs(x["global_step"] / 1e6 - m) <= args.window
                       and key in x and x[key] == x[key]]
                cells.append(f"{np.mean(sel):>11.4g}" if sel else f"{'-':>11}")
            print(f"{r:<{width}}  " + "".join(cells))


if __name__ == "__main__":
    main()
