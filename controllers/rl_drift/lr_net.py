"""Numpy-only inference for the drift policy (no torch, no GPU).

``policy.npz`` holds: obs_mean, obs_std, and layers ``W0 b0 W1 b1 ...`` (tanh hidden units, tanh
output). The output is two residuals, pedal then steer, each in [-1, 1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MLPPolicy:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray],
                 obs_mean: np.ndarray, obs_std: np.ndarray):
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.obs_mean = np.asarray(obs_mean, dtype=float)
        self.obs_std = np.asarray(obs_std, dtype=float)
        self.obs_dim = int(self.weights[0].shape[0])
        self.act_dim = int(self.weights[-1].shape[1])

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        data = np.load(path)
        n = int(data["n_layers"])
        return cls([data[f"W{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)],
                   data["obs_mean"], data["obs_std"])

    def save(self, path: str | Path) -> None:
        arrays = {"n_layers": np.array(len(self.weights)), "obs_mean": self.obs_mean, "obs_std": self.obs_std}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"], arrays[f"b{i}"] = w, b
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def act(self, features: np.ndarray) -> np.ndarray:
        """Deterministic residuals in [-1, 1], shape (act_dim,)."""
        x = np.clip((np.asarray(features, dtype=float) - self.obs_mean) / self.obs_std, -6.0, 6.0)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        y = np.tanh(x @ self.weights[-1] + self.biases[-1]).reshape(-1)
        return y

    @classmethod
    def random(cls, obs_dim: int, hidden: tuple[int, ...] = (64, 64), act_dim: int = 2,
               rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        sizes = [obs_dim, *hidden, act_dim]
        ws = [rng.normal(0.0, 1.0 / np.sqrt(a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        bs = [np.zeros(b) for b in sizes[1:]]
        return cls(ws, bs, np.zeros(obs_dim), np.ones(obs_dim))
