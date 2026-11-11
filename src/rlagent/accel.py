"""Accélérateurs : choix du device (CPU, GPU NVIDIA « cuda », iGPU Intel « xpu », TPU « xla ») et ce qui en dépend.

TPU : PyTorch/XLA (`torch_xla`, Linux seulement). XLA compile un graphe par forme de tenseurs : tout ce qui tourne
sur TPU doit garder des formes FIXES (lots découpés en morceaux de taille constante, pas de regroupement par nombre de
voitures), et chaque pas d'optimiseur se termine par `step_barrier` (exécution du graphe accumulé). Sans torch_xla
installé, tout ce module marche quand même (les fonctions XLA ne sont simplement jamais appelées).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import torch

import multiprocessing

if os.environ.get("PJRT_DEVICE") and multiprocessing.parent_process() is None:
    # Machine XLA (cloud/run_cloud.sh exporte PJRT_DEVICE=TPU) : le runtime XLA doit être initialisé AVANT le premier
    # backward du processus, sinon l'autograd a déjà dimensionné ses files par device sans XLA (« device_ready_queues_ »
    # au 1er backward sur TPU). Processus principal seulement : les workers (spawn) réimportent le module principal
    # (`python -m rlagent.train`, donc ce module) avec PJRT_DEVICE hérité ; à la session TPU 4, chacun des 222 workers
    # ouvrait ainsi le runtime TPU (223 messages « SliceBuilder port 8471 » dans le journal) pour rien.
    try:
        import torch_xla
        torch_xla.device()
    except Exception:  # noqa: BLE001 — absent ou en panne : resolve_device("xla") le signalera
        pass


# --------------------------------------------------------------------------- détection
def xpu_available() -> bool:
    return bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available())


def xla_available() -> bool:
    try:
        import torch_xla  # noqa: F401
    except Exception:  # noqa: BLE001 — absent, ou installé sans runtime TPU utilisable
        return False
    return True


def is_xla(device: Any) -> bool:
    return torch.device(device).type == "xla"


def xla_device() -> torch.device:
    import torch_xla
    if hasattr(torch_xla, "device"):          # torch_xla ≥ 2.6
        return torch_xla.device()
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


def _xla_highest_precision() -> None:
    """Produits matriciels en fp32 « complet » sur TPU (par défaut, XLA peut passer par bf16) : les ratios PPO/SPO
    comparent des log-probs calculées sur CPU (collecte) et sur TPU (apprentissage), il les faut à ~1e-6 près."""
    try:
        import torch_xla.backends
        torch_xla.backends.set_mat_mul_precision("highest")
        return
    except Exception:  # noqa: BLE001 — API absente selon la version
        pass
    try:
        import torch_xla
        torch_xla._XLAC._xla_set_mat_mul_precision("highest")
    except Exception:  # noqa: BLE001
        pass


def tpu_machine() -> bool:
    """Vraie machine TPU (PJRT_DEVICE=TPU, posé par cloud/run_cloud.sh) — pas XLA émulé sur CPU."""
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def resolve_device(name: str) -> torch.device:
    """auto = GPU NVIDIA (cuda) s'il y en a un, sinon TPU (xla) sur une machine TPU, sinon CPU.
    L'iGPU Intel (xpu) se demande explicitement : son intérêt dépend de la machine (scripts/benchmark_update.py)."""
    name = str(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if tpu_machine() and xla_available():
            name = "xla"
        else:
            return torch.device("cpu")
    if name == "xpu" and not xpu_available():
        raise RuntimeError("device xpu demandé mais torch.xpu n'est pas disponible (installer torch depuis "
                           "https://download.pytorch.org/whl/xpu, cf. INSTALL.md)")
    if name.startswith("xla"):
        if not xla_available():
            raise RuntimeError("device xla (TPU) demandé mais torch_xla n'est pas installé (cf. cloud/README.md)")
        _xla_highest_precision()
        return xla_device()
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {name} demandé mais aucun GPU NVIDIA utilisable")
    dev = torch.device(name)
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        # ex. cuda:1 (2e GPU pour l'inférence de collecte) sur une machine à un seul GPU → le seul GPU
        return torch.device("cuda", 0)
    return dev


def resolve_collect_device(name: str, learner: torch.device) -> torch.device:
    """Device de l'inférence de collecte. auto = 2e GPU s'il existe (Kaggle « T4 x2 » : un GPU apprend, l'autre
    joue), sinon le GPU de l'apprentissage, sinon CPU. Jamais XLA : l'inférence de collecte tourne dans un autre
    thread que l'apprentissage, et XLA n'est utilisé que depuis un seul thread."""
    name = str(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", 1 if torch.cuda.device_count() >= 2 else 0)
        return torch.device("cpu")
    if name.startswith("xla"):
        raise ValueError("train.collect_device=xla n'est pas pris en charge (cf. accel.resolve_collect_device)")
    return resolve_device(name)


# --------------------------------------------------------------------------- exécution
def step_barrier(device: Any) -> None:
    """Fin d'un pas : sur XLA, compile/exécute le graphe accumulé (sinon il grossit sans fin). Ailleurs : rien."""
    if not is_xla(device):
        return
    import torch_xla
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def synchronize(device: Any) -> None:
    """Attend la fin des calculs en cours (mesures de temps)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "xla":
        step_barrier(dev)
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()


def static_shapes(device: Any) -> bool:
    """Formes de tenseurs fixes obligatoires (XLA : une compilation par forme)."""
    return is_xla(device)


def prepare_models(device: Any, *modules: torch.nn.Module) -> None:
    """Réglages par device qui n'entrent pas dans les poids : sur XLA, pas de regroupement des échantillons par
    nombre de voitures (formes dépendantes des données → une recompilation par lot)."""
    if static_shapes(device):
        for m in modules:
            for sub in m.modules():
                if hasattr(sub, "skip_padding"):
                    sub.skip_padding = False


def to_cpu(obj: Any) -> Any:
    """Copie récursive des tenseurs sur CPU (checkpoints : lisibles partout, et torch.save ne gère pas XLA)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


# Métriques de torch_xla suivies par itération (time/xla_*) : temps cumulés en ns, octets pour In/OutboundData.
XLA_METRICS = {"LazyTracing": "time/xla_tracing_s", "CompileTime": "time/xla_compile_s",
               "ExecuteTime": "time/xla_execute_s", "TransferToDeviceTime": "time/xla_to_device_s",
               "TransferFromDeviceTime": "time/xla_from_device_s", "InboundData": "xla/to_device_mb",
               "OutboundData": "xla/from_device_mb"}


def xla_counters(device: Any) -> Dict[str, float]:
    """Totaux courants des métriques de torch_xla (vide hors XLA) ; xla_delta(avant, après) donne l'itération."""
    if not is_xla(device):
        return {}
    try:
        import torch_xla.debug.metrics as met
        out: Dict[str, float] = {}
        names = set(met.metric_names())
        for name in XLA_METRICS:
            if name in names:
                data = met.metric_data(name)
                out[name] = float(data[1]) if data else 0.0
        out["UncachedCompile"] = float(met.counter_value("UncachedCompile") or 0)
        return out
    except Exception:  # noqa: BLE001 — API absente selon la version : pas de métriques, pas de panne
        return {}


def xla_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, key in XLA_METRICS.items():
        if name in after:
            d = after[name] - before.get(name, 0.0)
            out[key] = d / (1e6 if key.endswith("_mb") else 1e9)
    if "UncachedCompile" in after:
        out["xla/compiles"] = after["UncachedCompile"] - before.get("UncachedCompile", 0.0)
    return out


def describe(device: Any) -> str:
    dev = torch.device(device)
    if dev.type == "cuda":
        i = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"{dev} ({torch.cuda.get_device_name(i)})"
    if dev.type == "xla":
        try:
            import torch_xla.runtime as xr
            return f"{dev} ({xr.device_type()}, {xr.global_runtime_device_count()} cœurs)"
        except Exception:  # noqa: BLE001
            return str(dev)
    return str(dev)


def padded_chunks(n: int, chunk: int, pad: bool) -> Iterable[tuple]:
    """(début, fin, taille_à_calculer) des morceaux de [0, n). pad=True : le dernier morceau est complété jusqu'à
    `chunk` (forme fixe pour XLA) ; l'appelant ne garde que les `fin - début` premières sorties."""
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        yield s, e, (chunk if pad else e - s)


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Complète `x` (lignes) avec des zéros jusqu'à `rows` lignes. Une observation nulle reste valide pour le réseau
    (jeton « soi » seul, autres voitures masquées) ; ces sorties sont jetées."""
    if x.shape[0] >= rows:
        return x
    pad = torch.zeros((rows - x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad])


def summary(cfg: Dict[str, Any], learner: torch.device, collect: Optional[torch.device]) -> str:
    tr = cfg["train"]
    mode = str(tr.get("collect_mode", "central"))
    arenas = f"{tr['n_workers']} workers × {tr.get('envs_per_worker', 1)} arènes"
    if mode == "local":
        return f"apprentissage {describe(learner)} | collecte locale (inférence dans les workers, CPU) | {arenas}"
    pipe = int(tr.get("collect_pipeline", 1) or 1)
    return (f"apprentissage {describe(learner)} | collecte centrale, inférence sur {describe(collect or 'cpu')}"
            f"{' + double tampon' if pipe > 1 else ''} | {arenas}")
