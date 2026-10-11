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
