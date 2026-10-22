"""Importe les runs (runs/*/metrics.jsonl + config.yaml) dans MLflow pour les comparer dans l'UI.

    pip install mlflow
    python scripts/mlflow_sync.py                      # tous les runs
    python scripts/mlflow_sync.py s5_spo_laptop s6_spo_kaggle
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Les métriques sont indexées par global_step (comparaison à compute égal, comme compare_runs.py).
Relancer le script réimporte proprement : un run MLflow déjà présent du même nom est remplacé.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "rocket-league-agents"
BATCH = 1000  # limite MLflow par log_batch


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def sync_run(client, exp_id: str, run_dir: Path) -> None:
    from mlflow.entities import Metric, Param, RunTag

    name = run_dir.name
    for old in client.search_runs([exp_id], f"attributes.run_name = '{name}'"):
        client.delete_run(old.info.run_id)

    cfg = yaml.safe_load(open(run_dir / "config.yaml", encoding="utf-8")) if (run_dir / "config.yaml").exists() else {}
    params = [Param(k, str(v)[:6000]) for k, v in flatten(cfg).items()]
    tags = [RunTag("machine", name.rsplit("_", 1)[-1]), RunTag("algo", "spo" if "_spo_" in name else "ppo")]
    latest = run_dir / "latest.txt"
    if latest.exists():
        tags.append(RunTag("latest_checkpoint", latest.read_text().strip()))

    t0 = int(time.time() * 1000)
    metrics = []
    for i, line in enumerate(open(run_dir / "metrics.jsonl", encoding="utf-8")):
        row = json.loads(line)
        step = int(row.get("global_step", i))
        for k, v in row.items():
            if k == "global_step" or not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                continue
            metrics.append(Metric(k, float(v), t0 + i, step))

    run = client.create_run(exp_id, run_name=name)
    rid = run.info.run_id
    for j in range(0, len(params), 100):
        client.log_batch(rid, params=params[j:j + 100])
    client.log_batch(rid, tags=tags)
    for j in range(0, len(metrics), BATCH):
        client.log_batch(rid, metrics=metrics[j:j + BATCH])
    for f in ("RAPPORT.md", "config.yaml"):
        if (run_dir / f).exists():
            client.log_artifact(rid, str(run_dir / f))
    client.set_terminated(rid)
    print(f"{name}: {len(metrics)} points, {len(params)} params")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="noms de runs dans runs/ (défaut : tous)")
    ap.add_argument("--tracking-uri", default=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    args = ap.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(
        EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    dirs = [ROOT / "runs" / r for r in args.runs] if args.runs else sorted(
        p for p in (ROOT / "runs").iterdir() if (p / "metrics.jsonl").exists())
    for d in dirs:
        sync_run(client, exp_id, d)


if __name__ == "__main__":
    main()
