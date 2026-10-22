"""Chargement de configuration YAML : default.yaml + profils superposés + surcharges en ligne de commande."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Surcharges du type `ppo.policy_loss=ppo` ou `env.format_weights={1: 1.0}`."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Surcharge invalide (attendu clé=valeur) : {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return cfg


def load_config(profiles: Optional[Iterable[str]] = None, overrides: Optional[Iterable[str]] = None,
                config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    with open(config_dir / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for profile in profiles or []:
        path = Path(profile)
        if not path.suffix:
            path = config_dir / f"{profile}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides or [])
    _normalize(cfg)
    return cfg


def _normalize(cfg: Dict[str, Any]) -> None:
    env = cfg["env"]
    env["format_weights"] = {int(k): float(v) for k, v in env["format_weights"].items()}
    env["start_weights"] = {str(k): float(v) for k, v in env["start_weights"].items()}
    from .env.obs import self_dim  # import tardif : config.py reste léger
    env["ball_prediction"] = bool(env.get("ball_prediction", False))
    env["race_features"] = bool(env.get("race_features", False))
    env["boost_features"] = bool(env.get("boost_features", False))
    cfg.setdefault("model", {})["self_dim"] = self_dim(env["ball_prediction"], env["race_features"], env["boost_features"])
    train = cfg["train"]
    if train.get("n_workers") in (None, "auto"):
        train["n_workers"] = max(1, (os.cpu_count() or 2) - 1)
    elif train.get("n_workers") == "all":       # collecte locale : le processus principal attend presque tout le temps
        train["n_workers"] = max(1, os.cpu_count() or 1)
    if train.get("torch_threads") in (None, "auto"):
        train["torch_threads"] = max(1, min(4, (os.cpu_count() or 2) // 2))
