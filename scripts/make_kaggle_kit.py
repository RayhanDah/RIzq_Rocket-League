"""Dossier à envoyer à un camarade qui lance la prochaine session Kaggle sur son compte, à la main (sans kaggle_sync).

    python scripts/make_kaggle_kit.py --run runs/s10_ppo_kaggle --profiles s10

Produit dist/kit_kaggle_<run>/ :
  LISEZMOI.md                       instructions pas à pas
  rl-agent-<run>.ipynb              notebook prêt à importer (code embarqué, réglages remplis)
  datasets/rl-collision-meshes.zip  le dump du PO (collision_meshes/soccar/*.cmf)
  datasets/rl-agent-opponents/      modèle de Nexto + poids GigaLearn (sparring)
  datasets/rl-agent-resume.zip      le run à reprendre (dernier checkpoint, jalons, pool, échelle, garde-fou)

À relancer après chaque modification du code ou des profils : le code est embarqué dans le notebook.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# fichiers du run nécessaires à la reprise (metrics.jsonl, ~600 Mo, n'est pas relu par l'entraînement)
RUN_FILES = ["latest.txt", "ladder.json", "guard.json", "best_actor.pt", "best_actor.txt", "config.yaml"]


def resume_zip(run: Path, out: Path) -> dict:
    latest = (run / "latest.txt").read_text(encoding="utf-8").strip()
    ckpt = run / "checkpoints" / latest
    if not ckpt.exists():
        sys.exit(f"{ckpt} absent : rapatrier le run d'abord (kaggle_sync.py fetch)")
    files = [run / f for f in RUN_FILES if (run / f).exists()] + [ckpt]
    files += sorted((run / "checkpoints" / "milestones").glob("ckpt_*.pt"))
    files += sorted(p for p in (run / "pool").rglob("*") if p.is_file() and "rejected" not in p.parts)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:          # .pt déjà peu compressibles
        for f in files:
            zf.write(f, f"runs/{run.name}/{f.relative_to(run).as_posix()}")
    import torch
    step = int(torch.load(ckpt, map_location="cpu", weights_only=False)["global_step"])
    return {"latest": latest, "step": step, "n_files": len(files), "pool": sum("pool/" in f.as_posix() for f in files)}


def notebook(run_name: str, profiles: str, out: Path) -> None:
    import kaggle_sync as ks
    nb = ks.build_notebook({"PLATFORM": "auto", "MODE": "train", "RUN_NAME": run_name, "PROFILES": profiles,
                            "HOURS": "auto", "AUTOTUNE": True, "PREV_OUTPUT": "auto"}, ks.code_b64())
    nb["cells"][0]["source"] = (f"**Notebook préparé le {date.today():%d/%m/%Y}** pour le run `{run_name}` "
                                f"(profils `{profiles}`). Code embarqué : ne rien modifier, suivre LISEZMOI.md.")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def mb(p: Path) -> str:
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return f"{size / 1e3:,.0f} ko".replace(",", " ") if size < 1e6 else f"{size / 1e6:,.0f} Mo".replace(",", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/s10_ppo_kaggle")
    ap.add_argument("--profiles", default="s10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    out = Path(args.out) if args.out else ROOT / "dist" / f"kit_kaggle_{run.name}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "datasets" / "rl-agent-opponents").mkdir(parents=True)

    import kaggle_sync as ks
    shutil.copy2(ks.meshes_zip(), out / "datasets" / "rl-collision-meshes.zip")
    files, _ = ks.opponent_files()
    for f in files:
        shutil.copy2(f, out / "datasets" / "rl-agent-opponents" / f.name)
    info = resume_zip(run, out / "datasets" / "rl-agent-resume.zip")
    nb_name = f"rl-agent-{run.name}.ipynb"
    notebook(run.name, args.profiles, out / nb_name)

    tpl = (ROOT / "cloud" / "LISEZMOI_KIT.md").read_text(encoding="utf-8")
    text = (tpl.replace("{RUN}", run.name).replace("{PROFILES}", args.profiles).replace("{NOTEBOOK}", nb_name)
            .replace("{LATEST}", info["latest"]).replace("{STEP}", f"{info['step']:,}")
            .replace("{SLUG}", run.name.replace("_", "-"))
            .replace("{SIZE_MESHES}", mb(out / "datasets" / "rl-collision-meshes.zip"))
            .replace("{SIZE_OPP}", mb(out / "datasets" / "rl-agent-opponents"))
            .replace("{SIZE_RESUME}", mb(out / "datasets" / "rl-agent-resume.zip"))
            .replace("{POOL}", str(info["pool"])).replace("{DATE}", f"{date.today():%d/%m/%Y}"))
    (out / "LISEZMOI.md").write_text(text, encoding="utf-8")
    print(f"[kit] {out} : reprise à {info['latest']} ({info['step']:,} steps), {info['n_files']} fichiers de run")


if __name__ == "__main__":
    main()
