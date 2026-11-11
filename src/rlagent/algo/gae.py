"""GAE et statistiques courantes (numpy pur, testable sans PyTorch)."""
from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, bootstrap_value: float,
                gamma: float, lam: float) -> np.ndarray:
    """Avantages GAE pour UNE trajectoire.

    rewards, values : (T,). bootstrap_value = V(s_T) si la trajectoire est coupée/tronquée, 0 si terminée (but).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    next_value = bootstrap_value
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_value = values[t]
    return adv.astype(np.float32)


def discounted_returns(rewards: np.ndarray, gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float64)
    running = bootstrap
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out.astype(np.float32)


def segment_discount(x: np.ndarray, lengths: np.ndarray, coef: float) -> np.ndarray:
    """y[t] = x[t] + coef · y[t+1] à l'intérieur de chaque segment (segments mis bout à bout dans `x`, longueurs
    `lengths`), 0 après la fin de chaque segment. Même résultat que la boucle par trajectoire (compute_gae,
    discounted_returns), mais une boucle Python par pas de temps de la PLUS LONGUE trajectoire, vectorisée sur toutes
    les trajectoires : ~5× plus rapide sur un lot de 400 k pas (3 400 trajectoires)."""
    x = np.asarray(x, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    out = np.zeros_like(x)
    if len(lengths) == 0:
        return out.astype(np.float32)
    ends = np.cumsum(lengths)
    order = np.argsort(-lengths, kind="stable")          # plus longues d'abord : les actives au pas k sont un préfixe
    ends_sorted, len_sorted = ends[order] - 1, lengths[order]
    n_active = np.searchsorted(-len_sorted, -np.arange(1, int(len_sorted[0]) + 1), side="right")
    running = np.zeros(len(lengths))
    for k, m in enumerate(n_active):                      # k = distance à la fin du segment
        idx = ends_sorted[:m] - k
        running[:m] = x[idx] + coef * running[:m]
        out[idx] = running[:m]
    return out.astype(np.float32)


def batch_gae(rewards: np.ndarray, values: np.ndarray, lengths: np.ndarray, bootstrap: np.ndarray,
              gamma: float, lam: float) -> np.ndarray:
    """compute_gae sur toutes les trajectoires d'un lot d'un coup (tableaux mis bout à bout, `bootstrap` = V(s_T) par
    trajectoire, 0 si terminée)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    next_v = np.empty_like(values)
    next_v[:-1] = values[1:]
    if len(lengths):
        next_v[np.cumsum(lengths) - 1] = np.asarray(bootstrap, dtype=np.float64)
    delta = rewards + gamma * next_v - values
    return segment_discount(delta, lengths, gamma * lam)


class RunningStats:
    """Moyenne/variance courantes (Welford parallèle)."""

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        b_mean, b_var, b_count = float(x.mean()), float(x.var()), x.size
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
