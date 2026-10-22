"""Prépare les fichiers à déposer (en PRIVÉ) sur Kaggle ou Google Drive pour entraîner dans le cloud.

    python scripts/make_cloud_bundle.py --init runs/s5_spo_laptop
    python scripts/make_cloud_bundle.py --init runs/s7c_spo_kaggle --anchors-from s7 s7b s7c s7d s7e

Crée dans dist/ :
  - rl-agent-code.zip : le code (src, configs, scripts, cloud, pyproject, requirements). JAMAIS le débrief,
    runs/, les venvs, les collision meshes ni rlviser.exe.
  - rl-agent-init.zip (si --init) : ckpt.pt (dernier checkpoint du run) + pool/ (adversaires) pour le démarrage à chaud.
    --anchors-from <profils> : ajoute les adversaires fixes de ces profils (`selfplay.anchors`), déjà figés au format du
    pool (pool/anchors/<nom>.pt, même nom et même source que ceux que le run créerait) : sur Kaggle, les runs sources
    n'existent pas, le run réutilise ces copies.
Les collision meshes (ton dump) se déposent à part, tels quels : dossier collision_meshes/ (soccar/*.cmf).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CODE_ITEMS = ["src", "configs", "scripts", "cloud", "tests", "pyproject.toml", "requirements.txt",
              "requirements.lock.txt", "README.md", "INSTALL.md"]
EXCLUDED_NAMES = {"REVUE_STRATEGIE.md", "__pycache__", ".pytest_cache", "settings.txt"}


def add_tree(zf: zipfile.ZipFile, path: Path, arc_root: Path) -> int:
    n = 0
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(arc_root)
        if any(part in EXCLUDED_NAMES or part.endswith(".egg-info") for part in rel.parts) or f.suffix == ".pyc":
            continue
        if f.suffix == ".sh":   # scripts bash : fins de ligne Linux obligatoires (sinon « set: pipefail » invalide)
            zf.writestr(rel.as_posix(), f.read_bytes().replace(bytes([13, 10]), bytes([10])))
        else:
            zf.write(f, rel.as_posix())
        n += 1
    return n


def build_code_zip(code: Path) -> int:
    """Le code (CODE_ITEMS, sans EXCLUDED_NAMES) dans `code` ; renvoie le nombre de fichiers."""
    with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as zf:
        n = sum(add_tree(zf, ROOT / item, ROOT) for item in CODE_ITEMS if (ROOT / item).exists())
        assert not any(Path(i).name in EXCLUDED_NAMES for i in zf.namelist())
    return n


def build_init_zip(run: Path, anchors_from, out: Path) -> Path:
    """rl-agent-init.zip : dernier checkpoint de `run` + son pool (+ ancres figées de `anchors_from`)."""
    from rlagent.train import find_latest_checkpoint

    if not run.is_absolute():
        run = ROOT / run
    ckpt = find_latest_checkpoint(run)
    if ckpt is None:
        sys.exit(f"aucun checkpoint dans {run}")
    init = out / "rl-agent-init.zip"
    with zipfile.ZipFile(init, "w", zipfile.ZIP_STORED) as zf:
        zf.write(ckpt, "ckpt.pt")
        pool = sorted((run / "pool").glob("actor_*.pt"))
        for f in pool:
            zf.write(f, f"pool/{f.name}")
        anchors = freeze_anchors(anchors_from, out / "anchors") if anchors_from else []
        for f in anchors:
            zf.write(f, f"pool/anchors/{f.name}")
    print(f"{init} : {ckpt.name} + {len(pool)} adversaires + {len(anchors)} ancres, {init.stat().st_size / 1e6:.1f} Mo")
    return init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="run dont on prend le dernier checkpoint + le pool (démarrage à chaud)")
    ap.add_argument("--anchors-from", nargs="+", default=None, metavar="PROFIL",
                    help="profils dont on fige les selfplay.anchors dans le zip d'init (ex. s7 s7b s7c s7d s7e)")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    code = out / "rl-agent-code.zip"
    n = build_code_zip(code)
    print(f"{code} : {n} fichiers, {code.stat().st_size / 1e6:.1f} Mo")
    if args.init:
        build_init_zip(Path(args.init), args.anchors_from, out)


def freeze_anchors(profiles, tmp_dir: Path):
    """Copies figées des ancres des profils, faites par le code du pool lui-même (mêmes noms, même `spec`)."""
    import shutil

    from rlagent.config import load_config
    from rlagent.selfplay.pool import OpponentPool

    specs = load_config(profiles)["selfplay"].get("anchors") or []
    if not specs:
        sys.exit(f"--anchors-from {' '.join(profiles)} : aucune ancre (selfplay.anchors vide)")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    pool = OpponentPool(tmp_dir, anchors=specs, anchor_root=ROOT)
    files = [a["path"] for a in pool.anchors]
    for a in pool.anchors:
        print(f"  ancre {a['name']} (step {a['step']:,})")
    return files


if __name__ == "__main__":
    main()
