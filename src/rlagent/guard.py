"""Garde-fou d'entraînement (`train.guard`, 08/11/2026).

Session 4 (07/11) : l'agent a trouvé une faille de `aerial_touch` à 8,7 G. Rien n'a arrêté le run : 2,2 G steps
(~6 h de TPU, après 35 h de file d'attente Kaggle) ont produit un agent qui ne marque plus. Le garde-fou surveille trois
signaux, chacun maintenu `patience` itérations de suite avant de compter :

- `ladder/drop` > `max_drop` : la dernière version est loin sous la meilleure de l'échelle Elo ;
- la part d'un terme surveillé dans le signal (`reward_abs/<terme>`, moyenne glissante) > `max_share` ;
- `learn/entropy` < `min_entropy`.

Quand l'un se déclenche, train.py revient au jalon sain (checkpoint complet gardé tous les `milestone_every_steps`),
écarte les versions du pool postérieures, et coupe le terme fautif s'il est identifié (poids 0 jusqu'à la fin du run,
décision notée dans guard.json pour le PO). Au-delà de `max_rollbacks` retours dans la même session (le même processus),
arrêt propre. L'état est dans <run>/guard.json : il survit aux reprises (sessions Kaggle successives), mais le compte des
retours repart de zéro à chaque session (11/11 : la session 5 en a fait 2 ; compté sur tout le run, le premier
déclenchement de la session 6 aurait arrêté le run et laissé la session inutilisée).

Ajouts du 14/11 (paliers 8-10, décision PO : introduire lentement kickoff, démos et aériennes sans perdre la session) :
- `term_max_share` : plafond de part propre à certains termes (les nouveaux, 5 %). Le seuil général de 25 % n'a jamais
  vu les dérives lentes de la session 5 : `aerial_touch` n'a pas dépassé ~6 % du signal ;
- `min_goals_ratio` : buts par minute (moyenne glissante) sous cette fraction du meilleur des 4 derniers jalons. Les
  deux dérives de la session 5 sont passées de 2,4 à 0,5 but / min, bien avant la chute d'Elo ;
- `fallback_weights` : un terme fautif revient à ce poids au lieu de 0 (un palier qui renforce les démos ne doit pas,
  s'il dérive, supprimer les démos que l'agent payait déjà) ;
- `reenable_terms` : termes coupés par une session précédente que le PO remet en jeu, une seule fois par run (noté
  dans guard.json, `reenabled`) ;
- le coupable se cherche d'abord parmi les termes de `term_max_share` (part ×2 depuis le jalon et > 1 %).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_WATCH = ["aerial_touch", "air_approach", "flip_reset", "strong_touch", "boost_gain", "boost_keep", "boost_spend",
                 "in_air", "flip_speed", "face_ball", "kickoff_rush", "kickoff_win", "demo", "bump", "touch"]
STATE_FILE = "guard.json"


class TrainingGuard:
    def __init__(self, cfg: Optional[Dict[str, Any]], run_dir: Path):
        g = dict(cfg or {})
        self.enabled = bool(g.get("enabled", False))
        self.max_drop = float(g.get("max_drop", 120.0))
        self.max_share = float(g.get("max_share", 0.25))
        self.min_entropy = float(g.get("min_entropy", 1.5))
        self.patience = max(1, int(g.get("patience", 60)))
        self.warmup = max(0, int(g.get("warmup_iters", 100)))
        self.max_rollbacks = int(g.get("max_rollbacks", 2))
        self.watch = list(g.get("watch_terms") or DEFAULT_WATCH)
        self.ema_rate = float(g.get("ema", 0.05))
        self.term_max_share = {str(k): float(v) for k, v in (g.get("term_max_share") or {}).items()}
        self.min_goals_ratio = float(g.get("min_goals_ratio") or 0.0)
        self.fallback = {str(k): float(v) for k, v in (g.get("fallback_weights") or {}).items()}
        for t in self.term_max_share:
            if t not in self.watch:
                self.watch.append(t)
        self.path = Path(run_dir) / STATE_FILE
        self.state: Dict[str, Any] = {"rollbacks": [], "disabled_terms": [], "manual_rollbacks": [], "milestones": {}}
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text(encoding="utf-8")))
        self.session_start = len(self.state.get("rollbacks", []))  # retours des sessions précédentes : hors quota
        self.reenabled_now: List[str] = []
        for t in g.get("reenable_terms") or []:
            done = self.state.setdefault("reenabled", [])
            if t in done:
                continue                                # une seule fois par run : un nouveau retour le recoupe
            done.append(t)
            if t in self.state.get("disabled_terms", []):
                self.state["disabled_terms"].remove(t)
                self.reenabled_now.append(t)
            self.save()
        self.goals_ema: Optional[float] = None
        self.shares: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.onset: Dict[str, float] = {}
        self.quiet = self.warmup

    # ------------------------------------------------------------------ état
    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def disabled_terms(self) -> List[str]:
        return list(self.state.get("disabled_terms", []))

    def apply(self, weights: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Poids envoyés aux workers, avec les termes coupés par un retour en arrière : 0, ou `fallback_weights` (sans
        jamais dépasser le poids du palier)."""
        off = self.disabled_terms
        if not off or weights is None:
            return weights
        return {**weights, **{t: min(self.fallback.get(t, 0.0), float(weights.get(t, 0.0))) for t in off}}

    def note_milestone(self, step: int) -> None:
        """Parts du signal (et buts / min, clé `_goals_per_min`) au jalon : la référence pour désigner le terme qui a
        dérivé ensuite, et pour mesurer une chute des buts."""
        ref = dict(self.shares)
        if self.goals_ema is not None:
            ref["_goals_per_min"] = self.goals_ema
        self.state.setdefault("milestones", {})[str(int(step))] = ref
        self.save()

    def goals_reference(self, window: int = 4) -> Optional[float]:
        """Meilleur niveau de buts / min des `window` derniers jalons : une dérive lente fait baisser chaque jalon, le
        dernier seul ne servirait pas de référence."""
        ms = self.state.get("milestones") or {}
        vals = [ms[k]["_goals_per_min"] for k in sorted(ms, key=int)[-window:]
                if isinstance(ms[k], dict) and "_goals_per_min" in ms[k]]
        return max(vals) if vals else None

    def restart(self) -> None:
        """Après un retour en arrière : compteurs remis à zéro, pas de contrôle pendant `warmup_iters` itérations."""
        self.counters, self.onset = {}, {}
        self.goals_ema = None                  # l'acteur rechargé ne joue plus comme celui de la dérive
        self.quiet = self.warmup

    # ------------------------------------------------------------------ surveillance
    def _shares_update(self, metrics: Dict[str, float]) -> None:
        vals = {k.split("/", 1)[1]: float(v) for k, v in metrics.items()
                if k.startswith("reward_abs/") and isinstance(v, (int, float))}
        total = sum(abs(v) for v in vals.values())
        if total <= 0.0:
            return
        for t, v in vals.items():
            s = abs(v) / total
            prev = self.shares.get(t)
            self.shares[t] = s if prev is None else prev + self.ema_rate * (s - prev)

    def _hold(self, key: str, cond: bool, step: float) -> bool:
        if not cond:
            self.counters.pop(key, None)
            self.onset.pop(key, None)
            return False
        self.counters[key] = self.counters.get(key, 0) + 1
        self.onset.setdefault(key, step)
        return self.counters[key] >= self.patience

    def culprit(self, reference: Optional[Dict[str, float]]) -> Optional[str]:
        """Terme surveillé dont la part a le plus grossi depuis le jalon (×3 au moins, et > 5 % du signal). Sans parts
        notées au jalon, aucun terme n'est désigné : on ne coupe pas un terme sans point de comparaison."""
        if not reference:
            return None
        ref = reference
        best, ratio = None, 2.0
        for t in self.term_max_share:          # termes nouveaux ou renforcés : soupçonnés en premier
            now = self.shares.get(t, 0.0)
            if now <= 0.01:
                continue
            r = now / max(ref.get(t, 0.0), 0.005)
            if r >= ratio:
                best, ratio = t, r
        if best is not None:
            return best
        best, ratio = None, 3.0
        for t in self.watch:
            now = self.shares.get(t, 0.0)
            if now <= 0.05:
                continue
            r = now / max(ref.get(t, 0.0), 0.005)
            if r >= ratio:
                best, ratio = t, r
        return best

    def update(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Une itération. Renvoie le déclenchement {reason, term, onset_step} ou None."""
        self._shares_update(metrics)
        gpm = metrics.get("goals_per_min")
        if isinstance(gpm, (int, float)):
            prev = self.goals_ema
            self.goals_ema = float(gpm) if prev is None else prev + self.ema_rate * (float(gpm) - prev)
        if not self.enabled:
            return None
        if self.quiet > 0:
            self.quiet -= 1
            return None
        step = float(metrics.get("global_step", 0.0))
        drop = metrics.get("ladder/drop")
        ent = metrics.get("learn/entropy")
        fired: Optional[Dict[str, Any]] = None
        for t in self.watch:
            cap = self.term_max_share.get(t, self.max_share)
            if self._hold(f"share:{t}", self.shares.get(t, 0.0) > cap, step) and fired is None:
                fired = {"reason": f"part de {t} {100 * self.shares[t]:.1f} % > {100 * cap:.0f} %",
                         "term": t, "onset_step": self.onset[f"share:{t}"]}
        ref_goals = self.goals_reference() if self.min_goals_ratio > 0 else None
        low = ref_goals is not None and self.goals_ema is not None and self.goals_ema < self.min_goals_ratio * ref_goals
        if self._hold("goals", low, step) and fired is None:
            fired = {"reason": f"buts / min {self.goals_ema:.2f} < {self.min_goals_ratio:.0%} de {ref_goals:.2f}",
                     "term": None, "onset_step": self.onset["goals"]}
        if self._hold("drop", drop is not None and float(drop) > self.max_drop, step) and fired is None:
            fired = {"reason": f"ladder/drop {float(drop):.0f} > {self.max_drop:.0f}", "term": None,
                     "onset_step": self.onset["drop"]}
        if self._hold("entropy", ent is not None and float(ent) < self.min_entropy, step) and fired is None:
            fired = {"reason": f"entropie {float(ent):.2f} < {self.min_entropy:.2f}", "term": None,
                     "onset_step": self.onset["entropy"]}
        return fired

    def reference_shares(self, milestone_step: int) -> Optional[Dict[str, float]]:
        return (self.state.get("milestones") or {}).get(str(int(milestone_step)))

    @property
    def session_rollbacks(self) -> int:
        return len(self.state.get("rollbacks", [])) - self.session_start

    def can_roll_back(self) -> bool:
        return self.session_rollbacks < self.max_rollbacks

    def record_rollback(self, at_step: int, to_step: int, reason: str, term: Optional[str], rejected: int) -> None:
        self.state.setdefault("rollbacks", []).append(
            {"at_step": int(at_step), "to_step": int(to_step), "reason": reason, "term_disabled": term,
             "rejected_versions": int(rejected)})
        if term and term not in self.state.setdefault("disabled_terms", []):
            self.state["disabled_terms"].append(term)
        self.save()
