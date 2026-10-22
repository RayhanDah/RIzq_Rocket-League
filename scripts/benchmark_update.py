"""Mesure la durée d'une mise à jour PPO/SPO (le goulot sur CPU) selon le device et le nombre de threads.

Lot synthétique aux dimensions réelles : obs aléatoires, masques 1v1/2v2/3v3 dans les proportions des formats
de la config (pondérées par le nombre d'agents). Aucun simulateur nécessaire.

Exemples :
    python scripts/benchmark_update.py --profile laptop                      # CPU, threads de la config
    python scripts/benchmark_update.py --profile laptop --device xpu         # iGPU Intel (torch « xpu »)
    python scripts/benchmark_update.py --profile bigpc --device cuda --steps 200000
    python scripts/benchmark_update.py --profile kaggle_tpu --device xla        # TPU (torch_xla), 1er passage = compilation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlagent import accel  # noqa: E402
from rlagent.algo.learner import make_optimizer, ppo_update  # noqa: E402
from rlagent.config import load_config  # noqa: E402
from rlagent.env.actions import N_ACTIONS  # noqa: E402
from rlagent.env.obs import MAX_OTHERS, env_obs_flags, obs_dim  # noqa: E402
from rlagent.models.attention import build_models, count_params  # noqa: E402


def synthetic_batch(cfg, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    fw = cfg["env"]["format_weights"]
    sizes = sorted(fw)
    share = np.array([fw[k] * 2 * k for k in sizes], dtype=np.float64)   # transitions ∝ poids × nb d'agents
    share /= share.sum()
    team = rng.choice(sizes, size=n, p=share)
    dim = obs_dim(*env_obs_flags(cfg["env"]))
    obs = rng.normal(0, 0.5, size=(n, dim)).astype(np.float32)
    n_others = 2 * team - 1
    obs[:, dim - MAX_OTHERS:] = (np.arange(MAX_OTHERS)[None, :] < n_others[:, None]).astype(np.float32)
    return {
        "obs": obs,
        "actions": rng.integers(0, N_ACTIONS, n).astype(np.int64),
        "log_probs": np.full(n, -np.log(N_ACTIONS), dtype=np.float32),
        "advantages": rng.normal(0, 1, n).astype(np.float32),
        "returns": rng.normal(0, 1, n).astype(np.float32),
        "values": rng.normal(0, 1, n).astype(np.float32),
    }


def sync(device: torch.device) -> None:
    accel.synchronize(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, nargs="*", default=None, help="threads CPU à tester (défaut : config)")
    ap.add_argument("--steps", type=int, default=None, help="taille du lot (défaut : ppo.batch_steps)")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.profile, args.overrides)
    n = args.steps or int(cfg["ppo"]["batch_steps"])
    device = accel.resolve_device(args.device)          # cpu | cuda | xpu | xla (TPU)
    batch = synthetic_batch(cfg, n)
    threads = args.threads or [int(cfg["train"]["torch_threads"])]
    print(f"torch {torch.__version__} | device {device} | lot {n:,} | epochs {cfg['ppo']['epochs']} | "
          f"minibatch {cfg['ppo']['minibatch_size']:,} | chunk {cfg['ppo']['device_chunk']:,}")
    for th in threads:
        torch.set_num_threads(th)
        torch.manual_seed(0)
        actor, critic = build_models(cfg, N_ACTIONS)
        actor.to(device)
        critic.to(device)
        accel.prepare_models(device, actor, critic)
        opt = make_optimizer(actor, critic, cfg)
        if th == threads[0]:
            print(f"acteur {count_params(actor):,} params | critique {count_params(critic):,} params")
        # échauffement (compilation des noyaux GPU, allocations)
        warm = {k: v[: min(n, 2 * int(cfg["ppo"]["device_chunk"]))] for k, v in batch.items()}
        ppo_update(actor, critic, opt, warm, cfg, device)
        sync(device)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            ppo_update(actor, critic, opt, batch, cfg, device)
            sync(device)
            times.append(time.perf_counter() - t0)
        best = min(times)
        print(f"threads {th:>2} : update {best:6.2f} s  ->  {n / best:,.0f} transitions/s apprises")


if __name__ == "__main__":
    main()
