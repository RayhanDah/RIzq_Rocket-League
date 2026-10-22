"""Journalisation : toujours en local (metrics.jsonl + console), W&B en plus si disponible."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict


class MetricsLogger:
    def __init__(self, run_dir: Path, cfg: Dict[str, Any]):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.f = open(self.path, "a", encoding="utf-8")
        self.wandb = None
        mode = str(cfg.get("logging", {}).get("wandb", "auto")).lower()
        if mode in ("off", "false", "disabled", "none"):
            return
        try:
            import wandb
        except ImportError:
            print("[log] wandb non installé : journal local uniquement (metrics.jsonl).")
            return
        if mode == "auto":
            has_key = bool(os.environ.get("WANDB_API_KEY")) or _netrc_has_wandb()
            mode = "online" if has_key else "offline"
        try:
            self.wandb = wandb.init(
                project=cfg["logging"].get("wandb_project", "rocket-league-agent"),
                entity=cfg["logging"].get("wandb_entity"),
                name=cfg.get("run_name"), id=cfg.get("run_name"), resume="allow",
                config=cfg, dir=str(run_dir), mode=mode,
            )
            print(f"[log] wandb actif (mode {mode}).")
        except Exception as err:  # réseau, quota… : on n'arrête pas l'entraînement pour ça
            print(f"[log] wandb indisponible ({err}) : journal local uniquement.")
            self.wandb = None

    def log(self, metrics: Dict[str, Any]) -> None:
        clean = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        self.f.write(json.dumps(clean, default=str) + "\n")
        self.f.flush()
        print(_console_line(clean), flush=True)
        if self.wandb is not None:
            try:
                self.wandb.log(clean, step=int(clean["global_step"]))
            except Exception as err:
                print(f"[log] wandb.log a échoué : {err}")

    def close(self) -> None:
        try:
            self.f.close()
        finally:
            if self.wandb is not None:
                try:
                    self.wandb.finish()
                except Exception:
                    pass


def _netrc_has_wandb() -> bool:
    try:
        import netrc
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except Exception:
        return False


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{nd}f}"
    return str(v)


def _console_line(m: Dict[str, Any]) -> str:
    parts = [
        f"it {int(m.get('iteration', 0))}",
        f"steps {int(m.get('global_step', 0)):,}",
        f"sps {m.get('sps_total', 0):,.0f}",
        f"elo {m.get('elo', 0):.0f}",
        f"r/step {_fmt(m.get('reward/mean_step', float('nan')), 4)}",
        f"ent {_fmt(m.get('learn/entropy', float('nan')))}",
        f"kl {_fmt(m.get('learn/kl_final', m.get('learn/approx_kl', float('nan'))), 4)}",
        f"lr {m.get('learn/lr_actor', float('nan')):.1e}",
        f"ev {_fmt(m.get('learn/explained_variance', float('nan')), 2)}",
        f"buts/min {_fmt(m.get('goals_per_min', float('nan')), 2)}",
        f"touches/min {_fmt(m.get('touches_per_min', float('nan')), 1)}",
    ]
    if "vs_hist/win_rate" in m:
        parts.append(f"win_vs_hist {_fmt(m['vs_hist/win_rate'], 2)}")
    if "team/double_commit_rate" in m:
        parts.append(f"double_commit {_fmt(m['team/double_commit_rate'], 3)}")
    return " | ".join(parts)
