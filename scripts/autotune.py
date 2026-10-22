"""Autoréglage de l'infrastructure sur la machine courante (Kaggle GPU / TPU, Colab, portable) : quelques minutes.

Mesure 1) la durée d'une mise à jour PPO/SPO sur chaque device d'apprentissage disponible (GPU, TPU via torch_xla,
CPU) et 2) le débit de collecte (vrai RocketSim) de quelques configurations candidates : collecte centrale avec
inférence sur GPU et double tampon, collecte locale (chaque worker a son acteur), nombre de workers et d'arènes.
Écrit un profil YAML (infra seulement : rien de ce que l'agent apprend ne change) à ajouter en DERNIER :

    python scripts/autotune.py --profile kaggle_gpu --profile s7 --profile s7b --pool runs/<run>/pool --out autotune.yaml
    python -m rlagent.train --profile kaggle_gpu --profile s7 --profile s7b --profile autotune.yaml --run-name ...

Débit attendu de bout en bout (collecte asynchrone) ≈ min(collecte, lot / durée de l'update).
`--candidates` permet de mesurer une liste explicite (banc d'essai), ex. "central:3x16:p2:cuda:1,local:4x16".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402

from rlagent import accel  # noqa: E402
from rlagent.config import load_config  # noqa: E402


@dataclass
class Candidate:
    mode: str                 # central | local
    workers: int
    envs: int
    pipeline: int = 1
    device: str = "cpu"       # inférence de collecte (mode central)
    threads: int = 1          # threads PyTorch du processus principal (inférence CPU)
    sps: float = 0.0
    error: str = ""

    @property
    def label(self) -> str:
        if self.mode == "local":
            return f"local {self.workers}x{self.envs}"
        return f"central {self.workers}x{self.envs} p{self.pipeline} {self.device} th{self.threads}"

    @classmethod
    def parse(cls, text: str) -> "Candidate":
        """central:3x16:p2:cuda:1[:th2] | local:4x16"""
        parts = text.strip().split(":")
        w, e = (int(x) for x in parts[1].split("x"))
        if parts[0] == "local":
            return cls("local", w, e)
        rest = parts[2:]
        pipe = int(rest.pop(0)[1:]) if rest and rest[0].startswith("p") else 1
        threads = int(rest.pop()[2:]) if rest and rest[-1].startswith("th") else 1
        return cls("central", w, e, pipe, ":".join(rest) or "cpu", threads)


def agents_per_arena(cfg: Dict[str, Any]) -> float:
    fw = cfg["env"]["format_weights"]
    return 2.0 * sum(k * w for k, w in fw.items()) / max(1e-9, sum(fw.values()))


def max_arenas(cfg: Dict[str, Any], batch: int) -> int:
    """Arènes maximum pour garder ≥ min_segment_steps steps par agent et par lot (cf. train.check_segment_length)."""
    seg = float(cfg["train"].get("min_segment_steps", 32))
    return max(1, int(batch / (seg * agents_per_arena(cfg))))


def default_candidates(cfg: Dict[str, Any], batch: int, ncpu: int) -> List[Candidate]:
    cap = max_arenas(cfg, batch)
    out: List[Candidate] = []
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpus:
        dev = "cuda:1" if gpus >= 2 else "cuda:0"
        for w in sorted({max(1, ncpu - 1), ncpu}):
            for e in (16, 24):
                if w * e <= cap:
                    out.append(Candidate("central", w, e, 2, dev, 1))
        out.append(Candidate("central", max(1, ncpu - 1), 16, 2, "cpu", 1))
        out.append(Candidate("local", ncpu, min(16, max(1, cap // ncpu))))
    else:
        # beaucoup de cœurs (session TPU) : collecte locale, quelques tailles ; plus une centrale sur CPU
        for w in sorted({max(1, ncpu - 2), max(1, (3 * ncpu) // 4), max(1, ncpu // 2)}, reverse=True):
            out.append(Candidate("local", w, max(1, min(8, cap // w))))
        th = max(1, min(8, ncpu // 8))
        w = max(1, ncpu - th - 1)
        out.append(Candidate("central", w, max(2, min(16, cap // w)), 2, "cpu", th))
    return out


def measure_collect(cfg: Dict[str, Any], cand: Candidate, seconds: float, pool_dir: Optional[str]) -> float:
    from rlagent.algo.collector import Collector, LocalCollector
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    from rlagent.train import build_actor_from_model_cfg

    pool = None
    if pool_dir and Path(pool_dir).is_dir() and any(Path(pool_dir).glob("actor_*.pt")):
        from rlagent.selfplay.pool import OpponentPool
        sp = cfg["selfplay"]
        pool = OpponentPool(Path(pool_dir), recent=int(sp["pool_recent"]), seed=0)   # lecture seule, sans ancres
    torch.set_num_threads(cand.threads)
    actor, _ = build_models(cfg, N_ACTIONS)
    device = torch.device("cpu")
    if cand.mode == "local":
        col = LocalCollector(cfg, cand.workers, seed=0, pool=pool, envs_per_worker=cand.envs)
    else:
        device = accel.resolve_collect_device(cand.device, torch.device("cpu"))
        actor = actor.to(device)
        col = Collector(cfg, cand.workers, seed=0, pool=pool, build_actor_fn=build_actor_from_model_cfg,
                        envs_per_worker=cand.envs, pipeline=cand.pipeline)
    try:
        warm = max(2_000, int(cand.workers * cand.envs * agents_per_arena(cfg) * 8))
        _, _, n, dt = col.collect(actor, warm, device)          # chauffe (+ chargement des versions du pool)
        _, _, n, dt = col.collect(actor, warm, device)
        target = int(max(warm, seconds * n / max(dt, 1e-6)))
        _, _, n, dt = col.collect(actor, target, device)
        return n / max(dt, 1e-6)
    finally:
        col.close()


def measure_update(cfg: Dict[str, Any], device_name: str, batch: int, threads: int, cap: int = 50_000) -> float:
    """Durée (s) d'une mise à jour sur un lot synthétique de `batch` steps, mesurée sur au plus `cap` steps puis
    extrapolée (durée ∝ taille du lot). GPU : 1 passage de chauffe ; XLA : 2 (compilations) ; CPU : aucun."""
    from benchmark_update import synthetic_batch

    from rlagent.algo.learner import make_optimizer, ppo_update
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import build_models
    torch.set_num_threads(threads)
    device = accel.resolve_device(device_name)
    torch.manual_seed(0)
    actor, critic = build_models(cfg, N_ACTIONS)
    actor.to(device)
    critic.to(device)
    accel.prepare_models(device, actor, critic)
    opt = make_optimizer(actor, critic, cfg)
    n = min(batch, cap)
    data = synthetic_batch(cfg, n)
    runs = {"cpu": 1, "xla": 3}.get(device.type, 2)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        ppo_update(actor, critic, opt, data, cfg, device, decoupled=True)
        accel.synchronize(device)
        times.append(time.perf_counter() - t0)
    return min(times[-max(1, runs - 1):]) * batch / n


def learner_devices() -> List[str]:
    if torch.cuda.is_available():
        return ["cuda:0"]
    if accel.tpu_machine() and accel.xla_available():
        return ["xla", "cpu"]
    return ["cpu"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--pool", default=None, help="dossier pool/ (parties contre l'historique comprises dans la mesure)")
    ap.add_argument("--step", type=int, default=0, help="step du run (lot de ppo.batch_schedule à ce step)")
    ap.add_argument("--seconds", type=float, default=20.0, help="durée de mesure par configuration de collecte")
    ap.add_argument("--candidates", default=None, help="liste explicite, ex. central:3x16:p2:cuda:1,local:4x16")
    ap.add_argument("--learner", default=None, help="devices d'apprentissage à tester, ex. xla,cpu (défaut : auto)")
    ap.add_argument("--learner-threads", type=int, default=None)
    ap.add_argument("--skip-learner", action="store_true")
    ap.add_argument("--out", default=None, help="profil YAML à écrire (le rapport JSON est écrit à côté)")
    args = ap.parse_args(argv)

    from rlagent.train import scheduled_batch
    cfg = load_config(args.profile, args.overrides)
    cfg["selfplay"]["anchors"] = []
    batch = scheduled_batch(cfg, args.step)[0]
    ncpu = os.cpu_count() or 2
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[autotune] {ncpu} vCPU | {gpus} GPU | torch_xla {'oui' if accel.xla_available() else 'non'} | "
          f"lot {batch:,} | arènes max {max_arenas(cfg, batch)} (≥ {cfg['train'].get('min_segment_steps', 32)} "
          f"steps/agent/lot)", flush=True)

    # 1) apprentissage
    updates: Dict[str, float] = {}
    lthreads = args.learner_threads or max(1, min(32, ncpu // 3))
    if not args.skip_learner:
        for dev in (args.learner.split(",") if args.learner else learner_devices()):
            try:
                t = measure_update(cfg, dev, batch, lthreads if dev == "cpu" else 2)
                updates[dev] = t
                print(f"[autotune] update {dev:7s} : {t:6.2f} s pour {batch:,} steps "
                      f"→ {batch / t:,.0f} steps/s apprises", flush=True)
            except Exception as err:  # noqa: BLE001 — device absent ou en panne : on continue sans lui
                print(f"[autotune] update {dev} impossible : {type(err).__name__}: {err}", flush=True)
    learner = min(updates, key=updates.get) if updates else str(cfg["ppo"]["device"])
    t_update = updates.get(learner)

    # 2) collecte
    cands = ([Candidate.parse(c) for c in args.candidates.split(",")] if args.candidates
             else default_candidates(cfg, batch, ncpu))
    if learner == "cpu" and not args.candidates:
        # l'update tournera sur CPU en même temps que la collecte : on lui laisse ses cœurs
        for c in cands:
            c.workers = max(1, min(c.workers, ncpu - lthreads - 1))
        seen, unique = set(), []
        for c in cands:                      # le plafond peut rendre deux candidats identiques
            if c.label not in seen:
                seen.add(c.label)
                unique.append(c)
        cands = unique
    for c in cands:
        try:
            c.sps = measure_collect(cfg, c, args.seconds, args.pool)
            print(f"[autotune] collecte {c.label:38s} → {c.sps:8,.0f} steps/s", flush=True)
        except Exception as err:  # noqa: BLE001
            c.error = f"{type(err).__name__}: {err}"
            print(f"[autotune] collecte {c.label} impossible : {c.error}", flush=True)
    ok = [c for c in cands if c.sps > 0]
    if not ok:
        sys.exit("[autotune] aucune configuration de collecte n'a fonctionné")
    best = max(ok, key=lambda c: c.sps)
    expected = min(best.sps, batch / t_update) if t_update else best.sps
    print(f"\n[autotune] retenu : apprentissage {learner}, collecte {best.label} "
          f"→ ~{expected:,.0f} steps/s attendus de bout en bout", flush=True)
    if t_update and batch / t_update < best.sps:
        print("[autotune] l'apprentissage est le goulot (collecte plus rapide que l'update)", flush=True)

    profile: Dict[str, Any] = {
        "ppo": {"device": "xla" if learner.startswith("xla") else learner},
        "train": {"collect_mode": best.mode, "n_workers": best.workers, "envs_per_worker": best.envs,
                  "async_collect": True},
    }
    if best.mode == "central":
        profile["train"].update(collect_pipeline=best.pipeline, collect_device=best.device,
                                torch_threads=max(best.threads, lthreads if learner == "cpu" else 1))
    else:
        profile["train"]["torch_threads"] = lthreads if learner == "cpu" else 2
    if args.out:
        out = Path(args.out)
        header = (f"# Généré par scripts/autotune.py le {datetime.now():%d/%m/%Y %H:%M} ({ncpu} vCPU, {gpus} GPU).\n"
                  f"# Infra seulement. Débit attendu ~{expected:,.0f} steps/s (collecte {best.sps:,.0f}"
                  + (f", update {t_update:.1f} s / {batch:,} steps" if t_update else "") + ").\n")
        out.write_text(header + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
        report = {"date": datetime.now().isoformat(), "ncpu": ncpu, "gpus": gpus, "batch": batch,
                  "updates_s": updates, "learner": learner, "candidates": [asdict(c) | {"label": c.label} for c in cands],
                  "best": best.label, "expected_sps": expected}
        out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[autotune] profil écrit : {out}", flush=True)
    else:
        print(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
