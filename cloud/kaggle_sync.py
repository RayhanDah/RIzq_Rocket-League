"""Passerelle portable ↔ Kaggle : envoie le code, lance la session et rapatrie le run, sans passer par le site.

    python cloud/kaggle_sync.py setup                                   # une fois : vérifie le jeton, dépose les meshes
    python cloud/kaggle_sync.py launch --run-name s7f_spo_kaggle \\
        --init runs/s7e_spo_laptop --anchors-from s7 s7b s7c s7d s7e s7f  # 1re session d'un run (démarrage à chaud)
    python cloud/kaggle_sync.py launch --run-name s7f_spo_kaggle         # sessions suivantes : reprise automatique
    python cloud/kaggle_sync.py status --run-name s7f_spo_kaggle
    python cloud/kaggle_sync.py fetch  --run-name s7f_spo_kaggle --wait  # attend la fin puis rapatrie dans runs/

Un notebook Kaggle privé par run (« rl-agent <run> »), généré depuis cloud/kaggle_train.ipynb :
  - le code (même contenu que dist/rl-agent-code.zip) est EMBARQUÉ dans le notebook : plus de dataset de code à mettre à
    jour, et la session tourne toujours avec le code du portable au moment du `launch` ;
  - les collision meshes (dataset privé `rl-collision-meshes`, déposé par `setup`) ne changent jamais ;
  - le point de départ (`--init`) va dans le dataset privé `rl-agent-init`, seulement quand on le demande ;
  - reprise : le notebook prend en entrée l'Output de sa propre version précédente (comme « Add Input → Notebook Output
    Files » à la main) et y retrouve runs/<run>.
`kernels push` crée une nouvelle version ET la lance (« Save & Run All (Commit) ») : elle tourne navigateur fermé.

Jeton : kaggle.com → Settings → API → « Create New Token » ; soit ~/.kaggle/kaggle.json, soit la variable
KAGGLE_API_TOKEN. Paquet : `pip install kaggle` (dans le .venv).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

TEMPLATE = ROOT / "cloud" / "kaggle_train.ipynb"
WORK = ROOT / "dist" / "kaggle"                    # dist/ n'est pas versionné
STATE = WORK / "state.json"                        # runs lancés avec --init (pour le réattacher si besoin)
MESHES_DATASET = "rl-collision-meshes"
INIT_DATASET = "rl-agent-init"
# machine_shape de Kaggle (kernel-metadata.json) : « GPU T4 x2 » et « TPU v5e-8 »
ACCELERATORS = {"gpu": "NvidiaTeslaT4", "tpu": "TpuV5E8", "none": None}
MAX_EMBEDDED = 900_000                             # octets de base64 : au-delà, le notebook devient trop lourd
POOL_EVERY = 100_000_000                           # fetch : une version du pool tous les 100 M steps…
POOL_RECENT = 20                                   # … plus les dernières (selfplay.pool_recent)


def api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("paquet kaggle absent : .venv\\Scripts\\python -m pip install kaggle")
    k = KaggleApi()
    try:
        k.authenticate()
    except (OSError, ValueError, SystemExit) as err:
        sys.exit(f"authentification Kaggle impossible ({err}).\n"
                 "kaggle.com → Settings → API → Create New Token, puis placer kaggle.json dans ~/.kaggle/ "
                 "(ou définir KAGGLE_API_TOKEN).")
    return k


def username(k) -> str:
    return k.get_config_value(k.CONFIG_NAME_USER) if k else "PSEUDO"     # k = None : --dry-run hors ligne


def kernel_ref(k, run_name: str) -> tuple[str, str]:
    """(titre, owner/slug) du notebook de ce run. Le slug doit être celui que Kaggle déduit du titre."""
    from slugify import slugify

    title = f"rl-agent {run_name}"
    return title, f"{username(k)}/{slugify(title)}"


def kernel_status(k, ref: str):
    """(statut, message d'échec) ; (None, "") si le notebook n'existe pas encore."""
    try:
        resp = k.kernels_status(ref)
    except Exception as err:
        if "404" in str(err) or "denied" in str(err) or "not found" in str(err).lower():
            return None, ""
        raise
    return resp.status.name.lower(), resp.failure_message


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --- datasets -----------------------------------------------------------------------------------------------------

def dataset_exists(k, ref: str) -> bool:
    try:
        k.dataset_status(ref)
        return True
    except Exception:  # noqa: BLE001
        return False


def upload_dataset(k, slug: str, title: str, files: list[Path], note: str) -> str:
    """Crée le dataset privé `slug` ou en publie une nouvelle version, puis attend qu'il soit prêt."""
    ref = f"{username(k)}/{slug}"
    folder = WORK / "datasets" / slug
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, folder / f.name)
    meta = {"title": title, "id": ref, "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    before = dataset_version(k, ref)
    if before is not None:
        print(f"[kaggle] nouvelle version du dataset {ref}")
        resp = k.dataset_create_version(str(folder), note, quiet=True, dir_mode="zip")
    else:
        print(f"[kaggle] création du dataset privé {ref}")
        resp = k.dataset_create_new(str(folder), public=False, quiet=True, dir_mode="zip")
    if getattr(resp, "error", None):
        sys.exit(f"dépôt du dataset {ref} refusé : {resp.error}")
    wait_dataset_ready(k, ref, after=before or 0)
    return ref


def dataset_version(k, ref: str) -> int | None:
    """Numéro de la version courante ; None si le dataset n'existe pas."""
    try:
        return int(json.loads(k.dataset_status(ref, format="json(current_version_number)"))["current_version_number"])
    except Exception:  # noqa: BLE001
        return None


def wait_dataset_ready(k, ref: str, after: int, timeout_s: int = 1800) -> None:
    """Kaggle traite le dépôt quelques minutes : un notebook lancé avant verrait encore l'ancienne version.
    « ready » seul ne suffit pas (c'est encore l'état de l'ancienne version juste après l'envoi) : on attend aussi
    que le numéro de version dépasse `after`."""
    t0 = time.time()
    while True:
        info = json.loads(k.dataset_status(ref, format="json(status,current_version_number)"))
        status, version = info["status"], int(info.get("current_version_number") or 0)
        if status == "ready" and version > after:
            print(f"[kaggle] dataset {ref} prêt (version {version})")
            return
        if status in ("failed", "deleted"):
            sys.exit(f"dataset {ref} : traitement en échec ({status})")
        if time.time() - t0 > timeout_s:
            sys.exit(f"dataset {ref} toujours « {status} » après {timeout_s // 60} min : relancer plus tard")
        time.sleep(15)


def meshes_zip() -> Path:
    """Le dump du PO (collision_meshes/soccar/*.cmf), jamais une autre source."""
    for d in (ROOT / "collision_meshes", ROOT.parent / "collision_meshes"):
        if (d / "soccar").is_dir() and any((d / "soccar").glob("*.cmf")):
            out = WORK / "collision_meshes.zip"
            WORK.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted((d / "soccar").glob("*.cmf")):
                    zf.write(f, f"collision_meshes/soccar/{f.name}")
            return out
    sys.exit("collision_meshes/soccar/*.cmf introuvable (rl-agent/ ou son dossier parent) : "
             "il faut ton dump RLArenaCollisionDumper.")


# --- notebook -----------------------------------------------------------------------------------------------------

def set_param(src: str, name: str, value) -> str:
    """Remplace la valeur de `NAME = ...` dans la cellule des réglages (le commentaire de fin de ligne reste)."""
    pat = re.compile(rf'^({name} = )("[^"]*"|True|False|[0-9.]+)', re.MULTILINE)
    if not pat.search(src):
        sys.exit(f"cloud/kaggle_train.ipynb : réglage {name} introuvable dans la 1re cellule de code")
    return pat.sub(lambda m: m.group(1) + (repr(value) if isinstance(value, bool) else json.dumps(value)), src, count=1)


def build_notebook(params: dict, code_b64: str) -> dict:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    i = next(n for n, c in enumerate(cells) if c["cell_type"] == "code")
    src = "".join(cells[i]["source"])
    for name, value in params.items():
        src = set_param(src, name, value)
    cells[i]["source"] = src
    embed = ("# Code envoyé par cloud/kaggle_sync.py (pas de dataset rl-agent-code)\n"
             "import base64, io, zipfile\n"
             f'CODE_ZIP_B64 = "{code_b64}"\n'
             'zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP_B64))).extractall("/tmp/rl-agent-src")\n'
             'CODE_INPUT = "/tmp/rl-agent-src"\n'
             "del CODE_ZIP_B64\n")
    cells.insert(i + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                         "source": embed})
    cells.insert(0, {"cell_type": "markdown", "metadata": {},
                     "source": "**Généré par `cloud/kaggle_sync.py launch`** : ne pas modifier ici, relancer depuis "
                               "le portable (les modifications faites sur Kaggle seraient écrasées)."})
    return nb


def code_b64() -> str:
    from make_cloud_bundle import build_code_zip

    WORK.mkdir(parents=True, exist_ok=True)
    z = WORK / "rl-agent-code.zip"
    n = build_code_zip(z)
    data = base64.b64encode(z.read_bytes()).decode("ascii")
    print(f"[kaggle] code : {n} fichiers, {z.stat().st_size / 1e3:.0f} ko (embarqué dans le notebook)")
    if len(data) > MAX_EMBEDDED:
        sys.exit(f"code trop gros pour être embarqué ({len(data) / 1e3:.0f} ko en base64)")
    return data


# --- commandes ----------------------------------------------------------------------------------------------------

def cmd_setup(args) -> None:
    k = api()
    print(f"[kaggle] connecté en tant que {username(k)}")
    ref = f"{username(k)}/{MESHES_DATASET}"
    if dataset_exists(k, ref) and not args.force:
        print(f"[kaggle] dataset {ref} déjà là (--force pour le redéposer)")
    else:
        upload_dataset(k, MESHES_DATASET, "rl collision meshes", [meshes_zip()], "dump RLArenaCollisionDumper")
    try:
        q = k.quota_view()
        for name, quota in (("GPU", q.gpu_quota), ("TPU", q.tpu_quota)):
            if quota is not None:
                used, total = quota.time_used.total_seconds() / 3600, quota.total_time_allowed.total_seconds() / 3600
                print(f"[kaggle] quota {name} : {used:.1f} h utilisées sur {total:.0f} h cette semaine")
    except Exception as err:  # noqa: BLE001 — information seulement
        print(f"[kaggle] quota indisponible ({err})")


def cmd_launch(args) -> None:
    k = None if args.dry_run else api()
    title, ref = kernel_ref(k, args.run_name)
    status, _ = kernel_status(k, ref) if k else ("complete", "")
    if status in ("queued", "running") and not args.force:
        sys.exit(f"{ref} : une session est déjà « {status} ». Attendre (fetch --wait) ou --force pour en lancer "
                 "une autre (deux sessions écriraient deux suites différentes du même run).")
    meshes = f"{username(k)}/{MESHES_DATASET}"
    if k and not dataset_exists(k, meshes):
        sys.exit(f"dataset {meshes} absent : lancer d'abord `python cloud/kaggle_sync.py setup`")

    state = load_state()
    if args.fresh and not args.init:
        state.pop(args.run_name, None)
    sources = [meshes]
    if args.init or (args.anchors_from and (status is None or args.fresh or args.dry_run)):
        # --init : démarrage à chaud ; --anchors-from seul (1re session d'un run de zéro) : ancres seules
        from make_cloud_bundle import build_init_zip

        WORK.mkdir(parents=True, exist_ok=True)
        init_zip = build_init_zip(Path(args.init) if args.init else None, args.anchors_from, WORK)
        sources.append(upload_dataset(k, INIT_DATASET, "rl agent init", [init_zip],
                                      f"{args.run_name} depuis {args.init or 'zéro (ancres seules)'}")
                       if k else f"{username(k)}/{INIT_DATASET}")
        state[args.run_name] = {"init": args.init or "anchors-only"}
    elif state.get(args.run_name, {}).get("init"):
        # le notebook ne s'en sert que si le run n'a encore aucun checkpoint (1re session interrompue trop tôt)
        sources.append(f"{username(k)}/{INIT_DATASET}")
    resume = status is not None and not args.fresh

    params = {"PLATFORM": "auto", "MODE": args.mode, "RUN_NAME": args.run_name, "PROFILES": " ".join(args.profiles),
              "HOURS": args.hours, "AUTOTUNE": not args.no_autotune,
              "PREV_OUTPUT": "auto" if resume else ""}
    nb = build_notebook(params, code_b64())
    folder = WORK / "kernels" / args.run_name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "kaggle_train.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    acc = ACCELERATORS[args.accelerator]
    meta = {"id": ref, "title": title, "code_file": "kaggle_train.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": args.accelerator == "gpu",
            "enable_tpu": args.accelerator == "tpu", "enable_internet": True, "dataset_sources": sources,
            "competition_sources": [], "kernel_sources": [ref] if resume else [], "model_sources": []}
    if acc:
        meta["machine_shape"] = acc
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"[kaggle] --dry-run : notebook et métadonnées prêts dans {folder}, rien envoyé")
        return

    resp = k.kernels_push(str(folder), acc=acc)
    if getattr(resp, "error", None):
        sys.exit(f"envoi refusé par Kaggle : {resp.error}")
    save_state(state)
    url = getattr(resp, "url", "") or f"https://www.kaggle.com/code/{ref}"
    print(f"[kaggle] session lancée : {args.run_name} | {args.accelerator} | profils {' '.join(args.profiles)} | "
          f"{'reprise de la version précédente' if resume else 'nouveau départ'}")
    print(f"[kaggle] version {getattr(resp, 'version_number', '?')} : {url}")
    print(f"[kaggle] ensuite : python cloud/kaggle_sync.py fetch --run-name {args.run_name} --wait")


def cmd_status(args) -> None:
    k = api()
    _, ref = kernel_ref(k, args.run_name)
    status, msg = kernel_status(k, ref)
    print(f"{ref} : {status or 'inexistant'}" + (f" ({msg})" if msg else ""))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def download_output(k, ref: str, dest: Path, file_pattern=None, attempts: int = 6, pause: float = 30.0) -> list:
    """Output d'une session, avec reprise après une coupure réseau (25/10 : `IncompleteRead` au milieu des ~2 Go du
    pool de s10 → rien copié). Après une coupure, le fichier le plus récent (celui en cours, tronqué) est supprimé et
    on reprend sans retélécharger les fichiers déjà complets."""
    for attempt in range(1, attempts + 1):
        try:
            files, _ = k.kernels_output(ref, str(dest), file_pattern=file_pattern, force=attempt == 1)
            return files
        except Exception as err:  # noqa: BLE001 — requests / urllib3 : ChunkedEncodingError, ConnectionError…
            if attempt == attempts:
                raise
            # le plus récent est le fichier tronqué ; à égalité de date (résolution du système de fichiers), on retire
            # tous les ex aequo : un fichier complet retiré sera simplement retéléchargé
            found = [(p.stat().st_mtime_ns, p) for p in dest.rglob("*") if p.is_file()]
            newest = max((t for t, _ in found), default=None)
            partial = [p for t, p in found if t == newest]
            for p in partial:
                p.unlink()
            print(f"[kaggle] téléchargement coupé ({type(err).__name__}), essai {attempt + 1}/{attempts} dans "
                  f"{pause:.0f} s (fichier tronqué retiré : {', '.join(p.name for p in partial) or '-'})", flush=True)
            time.sleep(pause)
    return []


def output_files(k, ref: str) -> list:
    """Noms des fichiers de l'Output d'une session, sans rien télécharger."""
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug = ref.split("/")
    names, token = [], None
    with k.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = owner, slug, 500
            if token:
                req.page_token = token
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            names += [f.file_name for f in resp.files]
            token = resp.next_page_token
            if not token:
                return names


def fetch_pattern(names: list, run_name: str) -> str:
    """Filtre de `fetch` : tout runs/<run>, sauf le pool (95 % de l'Output, 1 requête par fichier : ~2 h à 1 600 versions).
    On garde une version tous les POOL_EVERY steps et les POOL_RECENT dernières : l'entraînement ne tire ses adversaires
    que là, sans elles une reprise locale ou un --init jouerait contre de très vieilles versions. Le reste du pool ne
    sert pas sur le portable, et il reste dans l'Output Kaggle (--full-pool pour tout rapatrier). Les versions écartées
    par un retour en arrière (pool/rejected/, train.rollback_to et garde-fou) restent aussi sur Kaggle."""
    run = re.escape(run_name)
    steps = sorted(int(m.group(1)) for n in names if (m := re.fullmatch(rf"runs/{run}/pool/actor_(\d+)\.pt", n)))
    first_of_bucket = {}
    for step in steps:
        first_of_bucket.setdefault(step // POOL_EVERY, step)
    ids = "|".join(f"{s:013d}" for s in sorted({*first_of_bucket.values(), *steps[-POOL_RECENT:]}))
    return rf"^runs/{run}(\.log$|/(?!pool/actor_|pool/rejected/)|/pool/actor_({ids})\.pt$)"


def move_tree(src: Path, target: Path) -> None:
    """Déplace src dans target fichier par fichier : même disque, donc rien n'est recopié et le run n'existe pas en double
    dans dist/. Écrase les fichiers de même nom, ne touche pas aux autres (RAPPORT.md, versions du pool déjà là)."""
    for f in [p for p in src.rglob("*") if p.is_file()]:
        out = target / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        f.replace(out)


def cmd_fetch(args) -> None:
    k = api()
    _, ref = kernel_ref(k, args.run_name)
    last = None
    while True:
        try:
            status, msg = kernel_status(k, ref)
        except Exception as err:  # noqa: BLE001 — coupure réseau passagère pendant une longue attente : on réessaie
            if not args.wait:
                raise
            print(f"[kaggle] {time.strftime('%H:%M')} état indisponible ({type(err).__name__}) : nouvel essai dans "
                  f"{args.poll} s", flush=True)
            time.sleep(args.poll)
            continue
        if status is None:
            sys.exit(f"{ref} : notebook inexistant (lancer `launch` d'abord)")
        if status != last:
            print(f"[kaggle] {time.strftime('%H:%M')} {ref} : {status}" + (f" ({msg})" if msg else ""), flush=True)
            last = status
        if status not in ("queued", "running", "new_script", "cancel_requested"):
            break
        if not args.wait:
            sys.exit("session pas terminée : l'Output n'existe qu'à la fin (relancer avec --wait pour attendre)")
        time.sleep(args.poll)

    dest = WORK / "output" / args.run_name
    shutil.rmtree(dest, ignore_errors=True)
    pattern = (rf"^runs/{re.escape(args.run_name)}(/|\.log$)" if args.full_pool
               else fetch_pattern(output_files(k, ref), args.run_name))
    files = download_output(k, ref, dest, file_pattern=pattern)
    found = sorted(dest.rglob(f"runs/{args.run_name}/latest.txt"))
    if not found:
        print("[kaggle] rien sous runs/<run> avec le filtre : téléchargement de tout l'Output")
        files = download_output(k, ref, dest)
        found = sorted(dest.rglob(f"{args.run_name}/latest.txt"))
    print(f"[kaggle] {len(files)} fichiers téléchargés dans {dest}")
    if not found:
        sys.exit(f"pas de runs/{args.run_name} dans l'Output (session en échec ? voir {dest}/*.log)")
    src = found[0].parent

    target = ROOT / "runs" / args.run_name
    if (target / "metrics.jsonl").exists() and not args.force:
        local = count_lines(target / "metrics.jsonl")
        remote = count_lines(src / "metrics.jsonl")
        if local > remote:
            sys.exit(f"runs/{args.run_name} local est plus avancé ({local} itérations contre {remote} sur Kaggle) : "
                     f"rien copié (--force pour écraser). Téléchargement gardé dans {src}")
    move_tree(src, target)
    for log in dest.rglob(f"{args.run_name}.log"):
        shutil.copy2(log, ROOT / "runs" / log.name)
    print(f"[kaggle] run rapatrié dans {target} ({(target / 'latest.txt').read_text().strip()})")


def main() -> None:
    if not sys.flags.utf8_mode:
        # le paquet kaggle écrit le journal de session avec l'encodage par défaut (cp1252 sous Windows) : il plante sur
        # les « → » et accents des journaux ; on se relance en mode UTF-8
        import subprocess

        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="vérifie le jeton et dépose les collision meshes (une fois)")
    p.add_argument("--force", action="store_true", help="redéposer les meshes même s'ils sont déjà là")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("launch", help="envoie le code et lance une session (reprend le run s'il existe)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--profiles", nargs="+", default=["s7", "s7b", "s7c", "s7d", "s7e", "s7f"],
                   help="réglages d'apprentissage (configs/*.yaml), dans l'ordre")
    p.add_argument("--accelerator", choices=sorted(ACCELERATORS), default="gpu", help="gpu = T4 x2, tpu = v5e-8")
    p.add_argument("--mode", choices=["train", "bench"], default="train")
    p.add_argument("--hours", default="auto", help='"auto" (arrêt 20 min avant la limite : 12 h GPU, 9 h TPU) ou un nombre d\'heures')
    p.add_argument("--no-autotune", action="store_true", help="pas de mesures en début de session (profils seuls)")
    p.add_argument("--init", default=None, help="run local de départ (dernier checkpoint + pool) → dataset rl-agent-init")
    p.add_argument("--anchors-from", nargs="+", default=None, metavar="PROFIL",
                   help="profils dont on fige les selfplay.anchors dans le point de départ")
    p.add_argument("--fresh", action="store_true", help="ne pas reprendre l'Output de la version précédente")
    p.add_argument("--force", action="store_true", help="lancer même si une session de ce run tourne déjà")
    p.add_argument("--dry-run", action="store_true", help="préparer le notebook sans rien envoyer")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("status", help="état de la dernière session du run")
    p.add_argument("--run-name", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("fetch", help="télécharge l'Output et range runs/<run> sur le portable")
    p.add_argument("--run-name", required=True)
    p.add_argument("--wait", action="store_true", help="attendre la fin de la session")
    p.add_argument("--poll", type=int, default=300, help="secondes entre deux vérifications (--wait)")
    p.add_argument("--force", action="store_true", help="écraser même si le run local est plus avancé")
    p.add_argument("--full-pool", action="store_true", help="tout le pool, pas seulement 1 version / 100 M + les 20 dernières")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    if getattr(args, "mode", None) == "bench" and args.no_autotune:
        ap.error("--mode bench fait justement les mesures : incompatible avec --no-autotune")
    args.func(args)


if __name__ == "__main__":
    main()
