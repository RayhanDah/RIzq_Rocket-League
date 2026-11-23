#!/usr/bin/env bash
# Installe l'environnement et lance (ou reprend) un entraînement sur Kaggle / Colab, avec arrêt propre après HOURS.
# Variables (le notebook les définit) :
#   CODE_DIR    dossier du code (contient pyproject.toml)        MESHES_DIR  dossier contenant soccar/*.cmf
#   RUNS_DIR    dossier PERSISTANT des runs                      RUN_NAME    nom du run (défaut s7b_spo_kaggle)
#   PLATFORM    auto (défaut) | kaggle (GPU) | kaggle_tpu | colab : profil d'infra (configs/<PLATFORM>.yaml)
#   PROFILES    réglages d'apprentissage, dans l'ordre (défaut "s7 s7b")
#   HOURS       auto (défaut) : arrêt propre MARGIN_MIN avant la limite Kaggle (12 h GPU, 9 h TPU), comptée depuis
#               KERNEL_T0 (début de session, donné par le notebook) ; ou nb d'heures d'entraînement
#   MARGIN_MIN  (défaut 20) marge avant la limite      MODE        train (défaut) | bench (mesure seule)
#   AUTOTUNE    1 (défaut) : quelques minutes de mesures au début de la session → <run>/autotune.yaml (réutilisé si la
#               session précédente a mesuré la même machine et le même lot) ; 0 : profils seuls
#   INIT_DIR    (option) dossier avec ckpt.pt + pool/ pour le démarrage à chaud (1re session seulement)
#   VENV_DIR    (défaut /tmp/rlenv) : HORS du dossier de sortie Kaggle (sinon ~3 Go sauvegardés)
#   XLA_VERSION (défaut 2.9.0) : torch + torch_xla installés en session TPU (même version pour les deux)
#
# Ce qui tourne où : RocketSim (la simulation) n'existe que sur CPU. L'accélérateur fait l'APPRENTISSAGE (GPU T4 ou
# TPU via PyTorch/XLA) et, en session GPU, éventuellement l'inférence de collecte (2e T4). Les CPU jouent les parties.
set -euo pipefail
: "${CODE_DIR:?}" "${MESHES_DIR:?}" "${RUNS_DIR:?}"
PLATFORM=${PLATFORM:-auto}; RUN_NAME=${RUN_NAME:-s7b_spo_kaggle}; HOURS=${HOURS:-auto}; MODE=${MODE:-train}
VENV_DIR=${VENV_DIR:-/tmp/rlenv}; PROFILES=${PROFILES:-"s7 s7b"}; AUTOTUNE=${AUTOTUNE:-1}; XLA_VERSION=${XLA_VERSION:-2.9.0}
NCPU=$(nproc)
KERNEL_T0=${KERNEL_T0:-$(date +%s)}; MARGIN_MIN=${MARGIN_MIN:-20}
cd "$CODE_DIR"

# 0) Quelle machine ? (GPU NVIDIA, sinon TPU, sinon CPU seul)
has_tpu() {
  [ -n "${TPU_ACCELERATOR_TYPE:-}${TPU_NAME:-}${TPU_WORKER_ID:-}" ] && return 0
  ls /dev/accel0 /dev/vfio/0 >/dev/null 2>&1
}
if [ "$PLATFORM" = "auto" ]; then
  if nvidia-smi -L >/dev/null 2>&1; then PLATFORM=kaggle
  elif has_tpu; then PLATFORM=kaggle_tpu
  else PLATFORM=kaggle; echo "[cloud] ni GPU ni TPU détecté : session CPU (apprentissage sur CPU, lent)"; fi
fi
if [ "$HOURS" = "auto" ]; then           # sessions Kaggle : 12 h en GPU, 9 h en TPU, comptées depuis le début de session
  if [ "$PLATFORM" = "kaggle_tpu" ]; then LIMIT_H=9; else LIMIT_H=12; fi
  STOP_AT=$(( KERNEL_T0 + LIMIT_H * 3600 - MARGIN_MIN * 60 ))
  echo "[cloud] plateforme : $PLATFORM | profils : $PROFILES | run : $RUN_NAME | arrêt propre à $(date -u -d "@$STOP_AT" +%H:%M) UTC ($LIMIT_H h - $MARGIN_MIN min après le début de session)"
else
  STOP_AT=""
  echo "[cloud] plateforme : $PLATFORM | profils : $PROFILES | run : $RUN_NAME | arrêt après $HOURS h d'entraînement"
fi

# 1) Environnement Python isolé (uv : rapide ; Python 3.12 téléchargé si besoin)
if [ ! -x "$VENV_DIR/bin/python" ]; then
  pip install -q uv
  uv venv -q --python 3.12 "$VENV_DIR"
  if [ "$PLATFORM" = "kaggle_tpu" ]; then
    # torch CPU (pas de CUDA sur une machine TPU) + torch_xla de la MÊME version (+ libtpu via l'extra [tpu])
    uv pip install -q --python "$VENV_DIR/bin/python" "torch==$XLA_VERSION" --index-url https://download.pytorch.org/whl/cpu
    uv pip install -q --python "$VENV_DIR/bin/python" "torch_xla[tpu]==$XLA_VERSION" \
      || echo "[cloud] torch_xla $XLA_VERSION non installable : l'apprentissage tournera sur CPU"
  else
    uv pip install -q --python "$VENV_DIR/bin/python" torch              # roue CUDA par défaut sous Linux
  fi
  uv pip install -q --python "$VENV_DIR/bin/python" -r requirements.txt
  uv pip install -q --python "$VENV_DIR/bin/python" -e .
fi
PY="$VENV_DIR/bin/python"
# torch_xla charge libpython dynamiquement : le Python de uv la range dans son propre lib/ (hors du chemin par défaut)
PYLIB=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ "$PLATFORM" = "kaggle_tpu" ] && export PJRT_DEVICE=TPU

# 2) Collision meshes (ton dump perso, déposé en privé) : on cherche le dossier qui contient soccar/
if [ ! -d "$MESHES_DIR/soccar" ]; then
  found=$(find "$MESHES_DIR" -maxdepth 4 -type d -name soccar | head -1 || true)
  [ -n "$found" ] || { echo "ERREUR : aucun dossier soccar/*.cmf sous $MESHES_DIR"; exit 1; }
  MESHES_DIR=$(dirname "$found")
fi
export RS_COLLISION_MESHES="$MESHES_DIR" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# allocateur GPU sans fragmentation (sinon la mémoire réservée grossit jusqu'à l'erreur « out of memory »)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[cloud] meshes : $MESHES_DIR ($(ls "$MESHES_DIR/soccar" | wc -l) fichiers)"
# Session 10 : modèle de Nexto (sparring et imitation), déposé dans le dataset privé rl-agent-opponents
NEXTO_MODEL=$(find /kaggle/input -maxdepth 6 -name nexto-model.pt 2>/dev/null | head -1 || true)
if [ -n "$NEXTO_MODEL" ]; then export RLAGENT_NEXTO_MODEL="$NEXTO_MODEL"; echo "[cloud] Nexto : $NEXTO_MODEL"; fi
# Sparring à plusieurs bots : GalactusV2, Kira, Rain (fichiers .lt, même arborescence que dist/opponents), déposés en
# archive opponents-bots.zip dans le même dataset. Kaggle peut l'avoir déjà décompressée : on cherche les deux formes.
BOTS_ZIP=$(find /kaggle/input -maxdepth 6 -name opponents-bots.zip 2>/dev/null | head -1 || true)
if [ -n "$BOTS_ZIP" ]; then
  mkdir -p /tmp/opponents && "$PY" -m zipfile -e "$BOTS_ZIP" /tmp/opponents && BOTS_ROOT=/tmp/opponents
else
  KIRA_POLICY=$(find /kaggle/input -maxdepth 8 -path '*Kira/rlbot/POLICY.lt' 2>/dev/null | head -1 || true)
  [ -n "$KIRA_POLICY" ] && BOTS_ROOT=$(dirname "$(dirname "$(dirname "$KIRA_POLICY")")")
fi
if [ -n "${BOTS_ROOT:-}" ]; then
  export RLAGENT_GALACTUS_DIR="$BOTS_ROOT/GalactusV2/rlbot" RLAGENT_KIRA_DIR="$BOTS_ROOT" RLAGENT_RAIN_DIR="$BOTS_ROOT"
  echo "[cloud] bots de sparring : $BOTS_ROOT ($(find "$BOTS_ROOT" -name '*.lt' | wc -l) fichiers .lt)"
fi

# 3) Accélérateur utilisable ? (vrai calcul, pas seulement la détection). P100 : plus gérée par les PyTorch récents.
DEVICE=$("$PY" - <<'PYEOF'
import os
import torch
dev = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda") @ torch.zeros(1, device="cuda")
        dev = "cuda"
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"[cloud] GPU : {', '.join(names)}", flush=True)
    except Exception as err:  # noqa: BLE001
        cap = torch.cuda.get_device_capability(0)
        print(f"[cloud] GPU inutilisable ({torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}) : {err}", flush=True)
elif os.environ.get("PJRT_DEVICE") == "TPU":
    try:
        import torch_xla
        import torch_xla.runtime as xr
        d = torch_xla.device()
        assert float((torch.ones(4, device=d) * 2).sum().cpu()) == 8.0
        dev = "xla"
        print(f"[cloud] TPU : {xr.device_type()} ({xr.global_runtime_device_count()} cœurs), torch_xla "
              f"{torch_xla.__version__}", flush=True)
    except Exception as err:  # noqa: BLE001
        print(f"[cloud] TPU inutilisable ({type(err).__name__}: {err}) : apprentissage sur CPU", flush=True)
print(dev)
PYEOF
)
echo "$DEVICE" | sed '$d'
DEVICE=$(echo "$DEVICE" | tail -1)
[ "$DEVICE" = "xla" ] || unset PJRT_DEVICE      # TPU absent ou en panne : personne ne doit plus essayer de l'ouvrir
echo "[cloud] apprentissage sur : $DEVICE | vCPU : $NCPU | RAM : $(free -g | awk '/Mem:/ {print $2}') Go"

PROF=(--profile "$PLATFORM")
for p in $PROFILES; do PROF+=(--profile "$p"); done
INFRA=(--set "ppo.device=$DEVICE")
if [ "$DEVICE" = "cpu" ] && [ "$NCPU" -ge 16 ]; then
  # apprentissage sur CPU d'une grosse machine : un tiers des cœurs pour l'update, le reste pour jouer
  THREADS=$(( NCPU / 3 )); [ "$THREADS" -gt 48 ] && THREADS=48
  WORKERS=$(( NCPU - THREADS - 2 ))
  INFRA+=(--set "train.torch_threads=$THREADS" --set "train.n_workers=$WORKERS")
  echo "[cloud] répartition CPU : $WORKERS workers de simulation, $THREADS threads pour l'update"
fi

# 4) Dossier du run : reprise s'il a déjà un checkpoint, sinon démarrage à chaud depuis INIT_DIR (+ copie du pool)
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR/pool"
INIT=()
if [ -n "${INIT_DIR:-}" ] && [ -f "$INIT_DIR/ckpt.pt" ]; then
  INIT=(--set "train.init_from=$INIT_DIR/ckpt.pt")
  if [ -z "$(ls -A "$RUN_DIR/checkpoints" 2>/dev/null || true)" ]; then
    cp -n "$INIT_DIR"/pool/*.pt "$RUN_DIR/pool/" 2>/dev/null || true
  fi
else
  INIT=(--set "train.init_from=")
fi
# adversaires fixes déjà figés (make_cloud_bundle.py --anchors-from) : les runs sources n'existent pas ici.
# Aussi pour un run de zéro (point de départ « ancres seules », repéré par init.json)
if [ -n "${INIT_DIR:-}" ] && [ -d "$INIT_DIR/pool/anchors" ]; then
  mkdir -p "$RUN_DIR/pool/anchors"
  cp -n "$INIT_DIR"/pool/anchors/*.pt "$RUN_DIR/pool/anchors/" 2>/dev/null || true
fi

# 5) Autoréglage (quelques minutes) : devices + mode de collecte + workers × arènes sur CETTE machine
TUNE="$RUN_DIR/autotune.yaml"
if [ "$MODE" = "bench" ] || [ "$AUTOTUNE" = "1" ]; then
  STEP=0      # step courant du run (taille du lot de ppo.batch_schedule)
  if [ -f "$RUN_DIR/metrics.jsonl" ]; then
    STEP=$(tail -1 "$RUN_DIR/metrics.jsonl" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["global_step"]))' || echo 0)
  elif [ -n "${INIT_DIR:-}" ] && [ -f "$INIT_DIR/ckpt.pt" ]; then
    STEP=$("$PY" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_step"]))' "$INIT_DIR/ckpt.pt" || echo 0)
  fi
  LEARNERS=$DEVICE; [ "$DEVICE" = "xla" ] && LEARNERS="xla,cpu"
  REUSE=(); [ "$MODE" = "bench" ] || REUSE=(--reuse)
  if ! "$PY" scripts/autotune.py "${PROF[@]}" "${INFRA[@]}" --pool "$RUN_DIR/pool" --step "$STEP" \
       --learner "$LEARNERS" --out "$TUNE" "${REUSE[@]}"; then
    echo "[cloud] autoréglage en échec : on garde les profils ($PLATFORM)"
    rm -f "$TUNE"
  fi
  [ "$MODE" = "bench" ] && exit 0
fi

ARGS=("${PROF[@]}")
[ "$AUTOTUNE" = "1" ] && [ -f "$TUNE" ] && ARGS+=(--profile "$TUNE")
ARGS+=(--run-name "$RUN_NAME" --set "paths.runs_dir=$RUNS_DIR")
if [ ! -f "$TUNE" ] || [ "$AUTOTUNE" != "1" ]; then ARGS+=("${INFRA[@]}"); fi   # autotune.yaml fixe déjà tout
ARGS+=("${INIT[@]}")

# 6) Arrêt propre avant la fin de session (checkpoint garanti), puis entraînement au premier plan
# (minuteur SANS accès à la sortie : sinon son `sleep` garderait le flux ouvert et le notebook attendrait des heures)
if [ -n "$STOP_AT" ]; then
  WAIT=$(( STOP_AT - $(date +%s) )); if [ "$WAIT" -lt 60 ]; then WAIT=60; fi
else WAIT=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 }'); fi
echo "[cloud] entraînement : $(awk -v w="$WAIT" 'BEGIN { printf "%.2f", w / 3600 }') h avant l'arrêt propre"
( sleep "$WAIT" && touch "$RUN_DIR/STOP" ) </dev/null >/dev/null 2>&1 &
TIMER=$!
trap 'touch "$RUN_DIR/STOP"' INT TERM
"$PY" -m rlagent.train "${ARGS[@]}" 2>&1 | tee -a "$RUNS_DIR/$RUN_NAME.log"
pkill -P "$TIMER" 2>/dev/null || true
kill "$TIMER" 2>/dev/null || true
echo "[cloud] terminé : $(cat "$RUN_DIR/latest.txt" 2>/dev/null)"
