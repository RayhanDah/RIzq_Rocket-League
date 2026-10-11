"""Exporte l'acteur d'un run vers bot/model.pt (ce que charge bot/bot.py dans RLBot).

    python scripts/export_bot.py --run runs/<run>                 # dernier checkpoint
    python scripts/export_bot.py --ckpt runs/<run>/checkpoints/ckpt_XXX.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from rlagent.train import find_latest_checkpoint

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--ckpt")
    ap.add_argument("--out", default=str(ROOT / "bot" / "model.pt"))
    args = ap.parse_args()

    path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(args.run))
    if path is None:
        sys.exit("Aucun checkpoint trouvé.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {
        "actor": ck["actor"],
        "model_cfg": ck["cfg"]["model"],
        "action_repeat": ck["cfg"]["env"]["action_repeat"],
        "obs_dim": ck["obs_dim"],
        "source": str(path),
        "global_step": ck["global_step"],
        "elo": ck.get("elo"),
    }
    torch.save(out, args.out)
    print(f"Exporté : {path.name} (step {ck['global_step']:,}) → {args.out}")


if __name__ == "__main__":
    main()
