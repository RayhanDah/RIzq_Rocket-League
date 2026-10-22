"""Pool d'adversaires historiques + Elo (ARCHITECTURE §9).

- Une version de l'acteur est figée dans le pool tous les `pool_every_steps` pas.
- Une part `hist_prob` des épisodes oppose une équipe "politique courante" à une équipe "version historique".
- L'Elo de la politique courante est mis à jour après chaque épisode contre l'historique ; les versions du pool
  gardent l'Elo qu'elles avaient au moment de leur sauvegarde (ancres).

Options (désactivées par défaut, `selfplay.*`) :
- `pfsp` : self-play priorisé « hard » (AlphaStar). Chaque version a un score glissant (victoires / nuls / défaites
  de la politique courante contre elle, fenêtre ≈ `pfsp_window` épisodes) ; elle est tirée avec un poids
  (1 − p_victoire)², mélangé à une part uniforme `pfsp_uniform` pour ne pas oublier les versions déjà battues.
  Scores sauvegardés dans `pool/pfsp_stats.json`.
- `anchors` : adversaires FIXES (anciens runs ou fichiers .pt), toujours tirables (hors fenêtre `pool_recent`),
  pour une part `anchor_prob` des épisodes contre l'historique. Figés à la 1re utilisation dans `pool/anchors/`
  (le run source peut continuer, tourner ses checkpoints ou être absent de la machine). Un ancien agent au bloc
  « soi » plus court joue via `fit_obs` (models/attention.py). Pas d'Elo pour eux (échelle d'un autre run).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

ANCHOR_PREFIX = "anchor:"
PFSP_PRIOR = 2.0          # pseudo-épisodes à 0,5 : une version jamais jouée a p = 0,5 ; 1 victoire → 0,67, pas 1
STATS_FILE = "pfsp_stats.json"


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> float:
    return r_a + k * (score_a - elo_expected(r_a, r_b))


def anchor_name(hist_id: str) -> Optional[str]:
    """Nom de l'ancre si `hist_id` (e["vs_hist"]["hist_id"]) en désigne une, sinon None."""
    return hist_id[len(ANCHOR_PREFIX):] if str(hist_id).startswith(ANCHOR_PREFIX) else None


def pfsp_weight(p_win: float) -> float:
    """PFSP « hard » : les adversaires qu'on bat déjà ne coûtent presque plus de parties."""
    return (1.0 - p_win) ** 2


def resolve_anchor_file(path: Path) -> Optional[Path]:
    """Dossier de run → son dernier checkpoint (latest.txt, sinon le plus récent) ; fichier .pt → lui-même."""
    path = Path(path)
    if path.is_dir():
        marker = path / "latest.txt"
        if marker.exists():
            p = path / "checkpoints" / marker.read_text(encoding="utf-8").strip()
            if p.exists():
                return p
        ckpts = sorted((path / "checkpoints").glob("ckpt_*.pt"))
        return ckpts[-1] if ckpts else None
    return path if path.is_file() else None


def _anchor_default_name(spec: str) -> str:
    p = Path(spec)
    if p.suffix == ".pt" and p.parent.name in ("checkpoints", "pool"):
        name = p.parent.parent.name if p.parent.name == "checkpoints" else f"{p.parent.parent.name}_{p.stem}"
    else:
        name = p.stem if p.suffix == ".pt" else p.name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "anchor"


class OpponentPool:
    def __init__(self, pool_dir: Path, recent: int = 20, cache_size: Optional[int] = None, seed: Optional[int] = None,
                 pfsp: bool = False, pfsp_uniform: float = 0.3, pfsp_window: float = 50.0,
                 anchors: Optional[Sequence[Union[str, Path]]] = None, anchor_prob: float = 0.2,
                 anchor_root: Optional[Path] = None):
        self.dir = Path(pool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recent = recent
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        # sample() tourne dans le thread de collecte (train.async_collect) pendant que add()/record_results()
        # tournent dans le thread principal
        self._lock = threading.Lock()
        self.entries: List[Dict] = []
        self._scan()
        self.pfsp = bool(pfsp)
        self.pfsp_uniform = min(1.0, max(0.0, float(pfsp_uniform)))
        self.pfsp_window = max(1.0, float(pfsp_window))
        self.pfsp_decay = 1.0 - 1.0 / self.pfsp_window
        self.anchor_prob = min(1.0, max(0.0, float(anchor_prob)))
        refrozen: List[str] = []
        self.anchors: List[Dict] = self._load_anchors(anchors or [], anchor_root, refrozen)
        # Le cache doit contenir TOUTES les versions tirables (`recent` + ancres) : sinon, avec plusieurs arènes contre
        # des versions différentes, chaque step recharge des acteurs depuis le disque (débit ÷ 3 à 5 mesuré le
        # 18/10/2026). ~2 Mo par acteur : sans souci pour la mémoire.
        self.cache_size = max(int(cache_size or 0), recent + 2 + len(self.anchors))
        # Scores glissants (politique courante contre chaque version), seulement si une option s'en sert
        self.track = self.pfsp or bool(self.anchors)
        self.stats: Dict[str, Dict[str, float]] = self._load_stats() if self.track else {}
        for hist_id in refrozen:  # ancre (re)copiée : les scores d'une ancienne ancre du même nom ne valent plus
            self.stats.pop(hist_id, None)

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
        entry = {"id": path.stem, "path": path, "elo": elo, "step": global_step}
        with self._lock:  # nouvelle liste : un sample() concurrent ne voit jamais une liste en cours de tri
            self.entries = sorted(self.entries + [entry], key=lambda e: e["step"])

    # ------------------------------------------------------------------ ancres
    def _load_anchors(self, specs: Iterable[Union[str, Path]], root: Optional[Path], refrozen: List[str]) -> List[Dict]:
        import torch
        out: List[Dict] = []
        names = set()
        for spec in specs:
            spec = str(spec).replace("\\", "/").rstrip("/")  # même écriture sous Windows et Linux
            name = base = _anchor_default_name(spec)
            i = 2
            while name in names:
                name, i = f"{base}_{i}", i + 1
            names.add(name)
            frozen = self.dir / "anchors" / f"{name}.pt"
            data = None
            if frozen.exists():
                try:
                    data = torch.load(frozen, map_location="cpu", weights_only=False)
                except Exception as err:  # copie corrompue : on la refait depuis la source
                    print(f"[pool] ancre {name} : copie illisible ({err}), nouvelle copie")
                if data is not None and data.get("anchor", {}).get("spec") != spec:
                    data = None  # même nom, autre source : l'utilisateur a changé d'ancre
            if data is None:
                src = Path(spec)
                if not src.is_absolute() and root is not None:
                    src = Path(root) / src
                src_file = resolve_anchor_file(src)
                if src_file is None:
                    raise FileNotFoundError(f"selfplay.anchors : aucun checkpoint pour {spec} ({src}) et pas de copie "
                                            f"figée dans {frozen}")
                data = self._freeze_anchor(src_file, spec, frozen)
                refrozen.append(ANCHOR_PREFIX + name)
                print(f"[pool] ancre {name} figée depuis {src_file} -> {frozen}")
            out.append({"id": ANCHOR_PREFIX + name, "name": name, "path": frozen, "anchor": True,
                        "elo": float(data.get("elo", 1000.0)), "step": int(data.get("global_step", 0))})
        return out

    @staticmethod
    def _freeze_anchor(src_file: Path, spec: str, frozen: Path) -> Dict:
        """Copie au format du pool (acteur seul) : checkpoint de run (`cfg`) ou version de pool (`model_cfg`)."""
        import torch
        src = torch.load(src_file, map_location="cpu", weights_only=False)
        model_cfg = dict(src.get("model_cfg") or src["cfg"]["model"])
        # Les anciens runs (≤ s5) n'enregistraient pas self_dim : on le lit sur la 1re couche du jeton « soi »
        model_cfg["self_dim"] = int(src["actor"]["encoder.self_embed.0.weight"].shape[1])
        data = {"actor": src["actor"], "model_cfg": model_cfg, "global_step": int(src.get("global_step", 0)),
                "elo": float(src.get("elo", 1000.0)),
                "anchor": {"spec": spec, "source": str(src_file), "frozen_at": datetime.now().isoformat()}}
        frozen.parent.mkdir(parents=True, exist_ok=True)
        tmp = frozen.with_suffix(".tmp")
        torch.save(data, tmp)
        tmp.replace(frozen)
        return data

    # ------------------------------------------------------------------ scores glissants (PFSP)
    def _load_stats(self) -> Dict[str, Dict[str, float]]:
        path = self.dir / STATS_FILE
        if not path.exists():
            return {}
        try:
            return {str(k): {f: float(v.get(f, 0.0)) for f in ("w", "d", "l", "games")}
                    for k, v in json.loads(path.read_text(encoding="utf-8"))["stats"].items()}
        except Exception as err:  # fichier abîmé : les scores se reconstruisent en quelques itérations
            print(f"[pool] {STATS_FILE} illisible ({err}) : scores PFSP remis à zéro")
            return {}

    def save_stats(self) -> None:
        with self._lock:
            payload = {"window": self.pfsp_window, "stats": {k: dict(v) for k, v in self.stats.items()}}
        path = self.dir / STATS_FILE
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as err:  # fichier verrouillé (antivirus, éditeur) : on réessaiera à l'itération suivante
            print(f"[pool] écriture de {STATS_FILE} impossible : {err}")

    def record_results(self, results: Iterable[Tuple[str, float]]) -> None:
        """Résultats de la politique courante (1 victoire, 0,5 nul, 0 défaite) contre des versions/ancres, puis
        sauvegarde. Sans effet si ni `pfsp` ni `anchors`."""
        if not self.track:
            return
        results = list(results)
        if not results:
            return
        with self._lock:
            for hist_id, score in results:
                s = self.stats.setdefault(str(hist_id), {"w": 0.0, "d": 0.0, "l": 0.0, "games": 0.0})
                for f in ("w", "d", "l"):
                    s[f] *= self.pfsp_decay
                s["w" if score > 0.75 else "l" if score < 0.25 else "d"] += 1.0
                s["games"] += 1.0
        self.save_stats()

    def win_prob(self, hist_id: str) -> float:
        """Probabilité de victoire glissante (nul = ½), tirée vers 0,5 par PFSP_PRIOR pseudo-épisodes."""
        s = self.stats.get(hist_id)
        if s is None:
            return 0.5
        n = s["w"] + s["d"] + s["l"]
        return (s["w"] + 0.5 * s["d"] + 0.5 * PFSP_PRIOR) / (n + PFSP_PRIOR)

    # ------------------------------------------------------------------ tirage
    def _group_probs(self, cands: List[Dict]) -> List[float]:
        n = len(cands)
        if not self.pfsp:
            return [1.0 / n] * n
        w = [pfsp_weight(self.win_prob(e["id"])) for e in cands]
        total = sum(w)
        if total <= 0.0:
            return [1.0 / n] * n
        u = self.pfsp_uniform
        return [(1.0 - u) * x / total + u / n for x in w]

    def sampling_distribution(self) -> List[Tuple[Dict, float]]:
        """(adversaire, probabilité) de chaque adversaire tirable : fenêtre `recent` puis ancres."""
        with self._lock:
            recent = self.entries[-self.recent:]
            anchors = list(self.anchors)
        a = 0.0 if not anchors else (self.anchor_prob if recent else 1.0)
        out = [(e, (1.0 - a) * p) for e, p in zip(recent, self._group_probs(recent))] if recent else []
        if anchors:
            out += [(e, a * p) for e, p in zip(anchors, self._group_probs(anchors))]
        return [(e, p) for e, p in out if p > 0.0]

    def sample(self) -> Optional[Dict]:
        if not self.pfsp and not self.anchors:  # comportement d'origine (même suite de tirages pour une graine)
            with self._lock:
                recent = self.entries[-self.recent:]
            return self.rng.choice(recent) if recent else None
        dist = self.sampling_distribution()
        if not dist:
            return None
        return self.rng.choices([e for e, _ in dist], weights=[p for _, p in dist])[0]

    def metrics(self) -> Dict[str, float]:
        """Entropie (nats) de la loi de tirage des adversaires (ln N si uniforme sur N) et score glissant par ancre."""
        out: Dict[str, float] = {}
        if not self.track:
            return out
        dist = self.sampling_distribution() if len(self) > 0 else []
        if dist:
            out["vs_hist/pfsp_entropy"] = float(-sum(p * math.log(p) for _, p in dist))
        for a in self.anchors:
            out[f"vs_anchor/{a['name']}/p_win"] = self.win_prob(a["id"])
        return out

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
