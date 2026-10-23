"""Curriculum par paliers de compétence (`curriculum.stages`, profil s10, refonte du 23/10/2026).

Le bot reste à un palier tant qu'il n'en maîtrise pas les compétences, mesurées sur ses propres métriques
(moyennes glissantes) : il avance « à son rythme », sans calendrier en steps fixé à l'avance. Chaque palier ne liste
que les poids de rewards qui CHANGENT par rapport au palier précédent (cumulatif, à partir de `rewards.weights`).

    curriculum:
      ema: 0.05               # moyenne glissante des métriques (≈ 20 itérations)
      patience: 20            # itérations consécutives critères remplis avant de passer au palier suivant
      blend_steps: 10_000_000 # passage progressif des anciens aux nouveaux poids
      stages:
        - name: bases
          advance: {touches_per_min: 16, mech/flips_per_min: 3}   # nombre = minimum ; {min: , max: } possible
          min_steps: 40_000_000
        - name: jeu
          rewards: {touch: 0.0, face_ball: 0.0}
          ppo: {lr_actor: 1.5e-4}       # lr_actor, lr_critic, ent_coef (pris en compte au changement de palier)

Le dernier palier n'a pas de critère de sortie. Les poids du moment sont envoyés aux workers
(shared_info["reward_weights"], cf. env/rewards.py TeamReward) ; l'état est sauvegardé avec les checkpoints.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

PPO_KEYS = ("lr_actor", "lr_critic", "ent_coef")


def _bounds(spec: Any):
    if isinstance(spec, dict):
        return float(spec.get("min", -math.inf)), float(spec.get("max", math.inf))
    return float(spec), math.inf


class SkillCurriculum:
    def __init__(self, base_weights: Dict[str, float], stages: List[Dict[str, Any]], ema: float = 0.05,
                 patience: int = 20, blend_steps: float = 10_000_000):
        if not stages:
            raise ValueError("curriculum.stages : au moins un palier")
        self.stages = stages
        self.names = [str(s.get("name", f"palier{i}")) for i, s in enumerate(stages)]
        # poids cumulés de chaque palier
        self.stage_weights: List[Dict[str, float]] = []
        w = {k: float(v) for k, v in base_weights.items()}
        for s in stages:
            w = {**w, **{k: float(v) for k, v in (s.get("rewards") or {}).items()}}
            self.stage_weights.append(dict(w))
        self.ema_rate, self.patience, self.blend_steps = float(ema), int(patience), float(blend_steps)
        self.index = 0
        self.stage_start = 0
        self.prev_weights: Optional[Dict[str, float]] = None
        self.ema: Dict[str, float] = {}
        self.streak = 0

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> Optional["SkillCurriculum"]:
        c = cfg.get("curriculum") or {}
        if not c.get("stages"):
            return None
        return cls(cfg["rewards"]["weights"], c["stages"], ema=c.get("ema", 0.05), patience=c.get("patience", 20),
                   blend_steps=c.get("blend_steps", 10_000_000))

    @property
    def name(self) -> str:
        return self.names[self.index]

    def blend(self, step: int) -> float:
        if self.prev_weights is None or self.blend_steps <= 0:
            return 1.0
        return min(max((step - self.stage_start) / self.blend_steps, 0.0), 1.0)

    def weights(self, step: int) -> Dict[str, float]:
        """Poids de rewards à appliquer au step donné (transition linéaire après un changement de palier)."""
        cur = self.stage_weights[self.index]
        f = self.blend(step)
        if f >= 1.0:
            return dict(cur)
        prev = self.prev_weights or {}
        return {k: (1.0 - f) * prev.get(k, 0.0) + f * cur.get(k, 0.0) for k in set(prev) | set(cur)}

    def ppo_overrides(self) -> Dict[str, float]:
        """Réglages PPO du palier courant (hérités des paliers précédents)."""
        out: Dict[str, float] = {}
        for s in self.stages[:self.index + 1]:
            out.update({k: float(v) for k, v in (s.get("ppo") or {}).items() if k in PPO_KEYS})
        return out

    def criteria(self) -> Dict[str, Any]:
        return dict(self.stages[self.index].get("advance") or {})

    def update(self, metrics: Dict[str, Any], step: int) -> Optional[str]:
        """Met à jour les moyennes glissantes ; renvoie un message si le palier change."""
        crit = self.criteria()
        for key in crit:
            v = metrics.get(key)
            if v is None or not math.isfinite(float(v)):
                continue
            old = self.ema.get(key)
            self.ema[key] = float(v) if old is None else old + self.ema_rate * (float(v) - old)
        if self.index >= len(self.stages) - 1 or not crit:
            return None
        min_steps = float(self.stages[self.index].get("min_steps", 0))
        ok = step - self.stage_start >= min_steps and all(
            key in self.ema and _bounds(spec)[0] <= self.ema[key] <= _bounds(spec)[1] for key, spec in crit.items())
        self.streak = self.streak + 1 if ok else 0
        if self.streak < self.patience:
            return None
        self.prev_weights = self.weights(step)
        old_name = self.name
        self.index += 1
        self.stage_start = int(step)
        self.streak = 0
        self.ema = {}
        return (f"palier « {old_name} » maîtrisé à {step:,} steps → palier {self.index + 1}/{len(self.stages)} "
                f"« {self.name} »")

    def metrics(self, step: int) -> Dict[str, float]:
        out = {"curriculum/stage": float(self.index), "curriculum/blend": self.blend(step),
               "curriculum/streak": float(self.streak)}
        out.update({f"curriculum/ema/{k}": v for k, v in self.ema.items()})
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "stage_start": self.stage_start, "prev_weights": self.prev_weights,
                "ema": dict(self.ema), "streak": self.streak, "names": list(self.names)}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.index = min(int(d.get("index", 0)), len(self.stages) - 1)
        self.stage_start = int(d.get("stage_start", 0))
        self.prev_weights = d.get("prev_weights")
        self.ema = {k: float(v) for k, v in (d.get("ema") or {}).items()}
        self.streak = int(d.get("streak", 0))
