"""Échelle Elo absolue (`selfplay.ladder`, profil s10, refonte du 23/10/2026).

L'Elo historique (`elo`, pool.py) est mis à jour en continu contre des adversaires dont l'Elo date de leur
sauvegarde : il dérive et oscille (±150 d'une itération à l'autre dans s9). Ici, on classe des versions FIGÉES :

- chaque version du pool est un joueur fixe ; la politique en cours d'apprentissage est le joueur « current » ;
  quand elle entre dans le pool, ses parties lui sont réattribuées et un nouveau « current » commence ;
- les ancres (selfplay.anchors, ex. s6) sont des joueurs fixes aussi : leur note situe le bot par rapport à eux ;
- classement par maximum de vraisemblance (Bradley-Terry, nul = ½ victoire, algorithme MM de Hunter 2004) sur
  TOUTES les parties jouées depuis le début, avec 1 nul fictif par paire jouée et entre versions successives
  (évite les notes infinies) ; la 1re version (réseau aléatoire du départ) vaut 0.

Aucune partie en plus : on réutilise les parties contre le pool (selfplay.hist_prob). Fichier : <run>/ladder.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

CURRENT = "current"


class Ladder:
    def __init__(self, path: Path, min_games: int = 30):
        self.path = Path(path)
        self.min_games = int(min_games)
        self.players: Dict[str, int] = {}            # version figée → step (ancres : -1)
        self.games: Dict[Tuple[str, str], List[float]] = {}   # (a, b) → [score de a, nombre de parties]
        self.ratings: Dict[str, float] = {}
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.players = {k: int(v) for k, v in d.get("players", {}).items()}
            self.games = {tuple(k.split("|", 1)): [float(s), float(n)] for k, (s, n) in d.get("games", {}).items()}
            self.ratings = {k: float(v) for k, v in d.get("ratings", {}).items()}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"players": self.players, "ratings": self.ratings,
                                   "games": {f"{a}|{b}": v for (a, b), v in self.games.items()}}), encoding="utf-8")
        tmp.replace(self.path)

    def add_version(self, version_id: str, step: int) -> None:
        """Nouvelle version figée (entrée dans le pool) : les parties de « current » deviennent les siennes."""
        moved: Dict[Tuple[str, str], List[float]] = {}
        for (a, b), v in self.games.items():
            key = (version_id if a == CURRENT else a, version_id if b == CURRENT else b)
            acc = moved.setdefault(key, [0.0, 0.0])
            acc[0] += v[0]
            acc[1] += v[1]
        self.games = moved
        self.players[version_id] = int(step)

    def record(self, results: Iterable[Tuple[str, float]], anchors: Iterable[str] = ()) -> None:
        """Parties de la politique courante : (id de l'adversaire, score 1 / ½ / 0)."""
        anchors = set(anchors)
        for opp, score in results:
            if opp not in self.players:
                if opp not in anchors:
                    continue             # version hors de l'échelle (ex. héritée d'un autre run)
                self.players[opp] = -1
            acc = self.games.setdefault((CURRENT, opp), [0.0, 0.0])
            acc[0] += float(score)
            acc[1] += 1.0

    def _versions(self) -> List[str]:
        return sorted((p for p, s in self.players.items() if s >= 0), key=lambda p: self.players[p])

    def fit(self, iters: int = 500) -> Dict[str, float]:
        versions = self._versions()
        if not versions:
            return {}
        names = list(self.players)
        if any(CURRENT in k for k in self.games):
            names.append(CURRENT)
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        wins = np.zeros((n, n))                       # wins[i, j] = score de i contre j (nuls = ½)
        for (a, b), (s, g) in self.games.items():
            if a in idx and b in idx:
                wins[idx[a], idx[b]] += s
                wins[idx[b], idx[a]] += g - s
        prior = np.zeros((n, n))
        prior[(wins + wins.T) > 0] = 1.0              # 1 nul fictif par paire jouée…
        chain = versions + ([CURRENT] if CURRENT in idx else [])
        for a, b in zip(chain, chain[1:]):            # … et entre versions successives
            prior[idx[a], idx[b]] = prior[idx[b], idx[a]] = 1.0
        wins = wins + prior / 2.0
        games = wins + wins.T
        w_tot = wins.sum(axis=1)
        p = np.ones(n)
        active = w_tot > 0
        for _ in range(iters):
            denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
            new = np.where(active & (denom > 0), w_tot / np.maximum(denom, 1e-300), p)
            new /= math.exp(np.mean(np.log(new[active]))) if active.any() else 1.0
            if np.max(np.abs(np.log(new[active]) - np.log(p[active]))) < 1e-7:
                p = new
                break
            p = new
        ref = p[idx[versions[0]]]
        self.ratings = {name: float(400.0 * math.log10(p[i] / ref)) for name, i in idx.items()
                        if active[i] and (games[i].sum() > 0)}
        return self.ratings

    def n_games(self, player: str) -> float:
        return sum(v[1] for k, v in self.games.items() if player in k)

    def metrics(self, anchor_names: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """ladder/elo : note de la dernière version figée (stable, palier par palier) ;
        ladder/elo_live : note provisoire de la politique courante (après `min_games` parties) ;
        ladder/anchor/<nom> : note d'une ancre sur la même échelle ;
        ladder/gap/<nom> : ladder/elo − note de l'ancre (> 0 : la dernière version est au-dessus de l'ancre)."""
        out: Dict[str, float] = {}
        versions = self._versions()
        latest = self.ratings.get(versions[-1]) if versions else None
        if latest is not None:
            out["ladder/elo"] = latest
            out["ladder/versions"] = float(len(versions))
        if CURRENT in self.ratings and self.n_games(CURRENT) >= self.min_games:
            out["ladder/elo_live"] = self.ratings[CURRENT]
        for pid, name in (anchor_names or {}).items():
            if pid in self.ratings:
                out[f"ladder/anchor/{name}"] = self.ratings[pid]
                if latest is not None:
                    out[f"ladder/gap/{name}"] = latest - self.ratings[pid]
        best = self.best_version()
        if best is not None and latest is not None:
            out["ladder/best_elo"] = self.ratings[best]
            out["ladder/best_step"] = float(self.players[best])
            out["ladder/drop"] = self.ratings[best] - latest      # > 0 : la dernière version est sous la meilleure
        return out

    def best_version(self) -> Optional[str]:
        """Version figée la mieux notée (garde-fou contre le surentraînement : train.py la copie dans
        <run>/best_actor.pt, au format du pool, jouable par evaluate.py / watch.py / le bot)."""
        rated = [v for v in self._versions() if v in self.ratings]
        return max(rated, key=lambda v: self.ratings[v]) if rated else None
