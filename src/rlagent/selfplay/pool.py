"""Pool d'adversaires historiques + Elo (ARCHITECTURE §9).

- Une version de l'acteur est figée dans le pool tous les `pool_every_steps` pas.
- Une part `hist_prob` des épisodes oppose une équipe "politique courante" à une équipe "version historique".
- L'Elo de la politique courante est mis à jour après chaque épisode contre l'historique ; les versions du pool
  gardent l'Elo qu'elles avaient au moment de leur sauvegarde (ancres).
"""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> float:
    return r_a + k * (score_a - elo_expected(r_a, r_b))


class OpponentPool:
    def __init__(self, pool_dir: Path, recent: int = 20, cache_size: int = 4, seed: Optional[int] = None):
        self.dir = Path(pool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recent = recent
        self.cache_size = cache_size
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self.entries: List[Dict] = []
        self._scan()

    def _scan(self) -> None:
        import torch
        self.entries = []
        for path in sorted(self.dir.glob("actor_*.pt")):
            try:
                meta = torch.load(path, map_location="cpu", weights_only=False)
                self.entries.append({"id": path.stem, "path": path, "elo": float(meta.get("elo", 1000.0)),
                                     "step": int(meta.get("global_step", 0))})
            except Exception as err:  # fichier corrompu (arrêt brutal pendant l'écriture)
                print(f"[pool] ignoré {path.name}: {err}")
        self.entries.sort(key=lambda e: e["step"])

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, actor_state: dict, model_cfg: dict, global_step: int, elo: float) -> None:
        import torch
        path = self.dir / f"actor_{global_step:013d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"actor": actor_state, "model_cfg": model_cfg, "global_step": global_step, "elo": elo}, tmp)
        tmp.replace(path)
        self.entries.append({"id": path.stem, "path": path, "elo": elo, "step": global_step})
        self.entries.sort(key=lambda e: e["step"])

    def sample(self) -> Optional[Dict]:
        if not self.entries:
            return None
        return self.rng.choice(self.entries[-self.recent:])

    def get_actor(self, entry: Dict, build_fn: Callable):
        import torch
        key = entry["id"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = torch.load(entry["path"], map_location="cpu", weights_only=False)
        actor = build_fn(data["model_cfg"])
        actor.load_state_dict(data["actor"])
        actor.eval()
        self._cache[key] = actor
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return actor
