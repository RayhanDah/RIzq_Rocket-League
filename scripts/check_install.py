"""Vérifie l'installation pas à pas, avec un message clair pour chaque problème.

    python scripts/check_install.py

Code de sortie 0 = tout est prêt pour l'entraînement.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, KO = "✅", "❌"
problems = []


def check(label: str, fn):
    try:
        info = fn()
        print(f"{OK} {label}" + (f" — {info}" if info else ""))
        return True
    except Exception as err:  # noqa: BLE001
        print(f"{KO} {label} — {err}")
        problems.append(label)
        return False


def py_version():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} trop ancien (3.10+ requis)")
    return f"Python {platform.python_version()} ({platform.system()} {platform.machine()})"


def torch_info():
    import torch
    cuda = torch.cuda.is_available()
    return f"torch {torch.__version__}, CUDA {'oui : ' + torch.cuda.get_device_name(0) if cuda else 'non (CPU)'}"


def rlgym_info():
    from rlgym.rocket_league import __version__ as v
    if not v.startswith("2."):
        raise RuntimeError(f"rlgym-rocket-league {v} : la v2 est requise")
    return f"rlgym-rocket-league {v}"


def rocketsim_info():
    import RocketSim  # noqa: F401
    try:
        from importlib.metadata import version
        return f"RocketSim {version('rocketsim')}"
    except Exception:
        return "RocketSim importé"


def rlagent_info():
    importlib.import_module("rlagent.train")
    from rlagent.env.obs import OBS_DIM
    from rlagent.env.actions import N_ACTIONS
    return f"package rlagent OK (obs={OBS_DIM}, actions={N_ACTIONS})"


def meshes_info():
    from rlagent.config import load_config
    from rlagent.env.factory import check_meshes, resolve_meshes_dir, EXPECTED_SOCCAR_MESHES
    d = resolve_meshes_dir(load_config())
    n = check_meshes(d)
    warn = "" if n == EXPECTED_SOCCAR_MESHES else f" (⚠️ {EXPECTED_SOCCAR_MESHES} attendus d'habitude : vérifier que le dump vient du stade standard)"
    return f"{n} fichiers dans {d / 'soccar'}{warn}"


def _env_child(q):
    try:
        import numpy as np
        from rlagent.config import load_config
        from rlagent.env.factory import build_env
        cfg = load_config()
        cfg["env"]["format_weights"] = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
        env = build_env(cfg, seed=0)
        sizes = set()
        steps = 0
        for _ in range(5):
            obs = env.reset()
            sizes.add(len(obs) // 2)
            for _ in range(150):
                acts = {a: np.random.randint(0, 90, size=(1,)) for a in env.agents}
                obs, rew, term, trunc = env.step(acts)
                steps += 1
                for a, o in obs.items():
                    assert np.all(np.isfinite(o)), "obs non finie"
                if any(term.values()) or any(trunc.values()):
                    break
        q.put(("ok", f"{steps} steps aléatoires, formats vus : {sorted(sizes)}"))
    except Exception as err:  # noqa: BLE001
        import traceback
        q.put(("err", f"{err}\n{traceback.format_exc()}"))


def env_info():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_env_child, args=(q,))
    p.start()
    p.join(180)
    if p.is_alive():
        p.terminate()
        raise RuntimeError("délai dépassé (180 s)")
    if q.empty():
        raise RuntimeError(f"le processus RocketSim s'est arrêté brutalement (exitcode={p.exitcode}) : meshes invalides ?")
    status, msg = q.get()
    if status != "ok":
        raise RuntimeError(msg)
    return msg


def main():
    print("=== Vérification de l'installation ===")
    check("Python", py_version)
    check("PyTorch", torch_info)
    check("RLGym v2", rlgym_info)
    check("RocketSim", rocketsim_info)
    check("Package rlagent (pip install -e .)", rlagent_info)
    if check("Collision meshes", meshes_info):
        check("Environnement RocketSim (1v1/2v2/3v3, actions aléatoires)", env_info)
    else:
        print("   → étape suivante impossible sans les meshes (voir INSTALL.md §4).")
        problems.append("Environnement RocketSim")
    try:
        import wandb  # noqa: F401
        print(f"{OK} wandb installé (optionnel)")
    except ImportError:
        print("ℹ️  wandb non installé (optionnel) : les métriques restent dans runs/<run>/metrics.jsonl")
    print()
    if problems:
        print(f"{KO} {len(problems)} problème(s) : {', '.join(problems)}")
        sys.exit(1)
    print(f"{OK} Tout est prêt. Étape suivante : python -m rlagent.train --profile debug")


if __name__ == "__main__":
    main()
