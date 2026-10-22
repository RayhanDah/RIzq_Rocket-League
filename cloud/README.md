# ☁️ Entraîner sur Kaggle (GPU ou TPU) ou Google Colab (gratuit)

Le portable reste libre pendant que Kaggle (ou Colab) entraîne l'agent. Même code, mêmes réglages d'apprentissage ;
seule l'infra change (profils `configs/kaggle.yaml`, `configs/kaggle_tpu.yaml`, `configs/colab.yaml`, plus le profil
`autotune.yaml` mesuré en début de session).

## Ce qui tourne où

Une itération = **jouer** des parties puis **apprendre** dessus (les deux en parallèle : `async_collect`).

| Étape | Où | Pourquoi |
|---|---|---|
| Simulation (RocketSim), observations, rewards | **CPU uniquement** | RocketSim est un moteur physique C++ CPU : aucun GPU/TPU ne peut le remplacer |
| Choix des actions pendant les parties (inférence) | CPU des workers (mode `local`) **ou** GPU (mode `central`, 2e T4) | c'était 56 % du temps de collecte (mesuré) |
| Apprentissage (GAE + PPO/SPO) | **GPU T4** ou **TPU** (PyTorch/XLA) | caché derrière la collecte |

### Ce qui a été accéléré (mesures du 20/10/2026, vrai RocketSim + pool, 4 cœurs comme Kaggle GPU)

| Collecte (lot de 50 000 steps, 20 % contre le pool) | steps/s |
|---|---|
| Avant : inférence centrale CPU, workers en attente pendant l'inférence | 1 325 |
| Double tampon (`collect_pipeline: 2`) : workers et inférence en parallèle | 2 901 (×2,2) |
| Collecte locale 4 workers (`collect_mode: local`) : chaque worker a son acteur | **3 731 (×2,8)** |

Sur Kaggle GPU, s6 tournait à ~2 200 steps/s : on peut s'attendre à ~5 000-6 000 (les CPU de Kaggle sont plus rapides
que ceux du portable). En session **TPU**, la machine a beaucoup plus de cœurs : la collecte locale passe à l'échelle
(un worker ≈ 900 steps/s) et l'apprentissage sur TPU suit. **Chiffres TPU à mesurer** (`MODE = "bench"`).

Résultats identiques : tests d'équivalence (`tests/test_cloud_accel.py` : mêmes parties, step par step, dans les trois
modes de collecte ; `tests/test_xla.py` : mêmes poids après un update sur XLA que sur CPU, aucune recompilation).

### Autoréglage (`scripts/autotune.py`, ~3-6 min au début de chaque session)

Kaggle ne garantit pas la machine (nombre de cœurs, type de TPU). Le notebook mesure donc, sur la machine obtenue :
1. la durée d'un apprentissage sur chaque device (GPU ; TPU et CPU en session TPU) ;
2. le débit de collecte de quelques configurations (local / central + GPU, nombre de workers × arènes), avec les
   adversaires du pool ;

puis écrit `runs/<run>/autotune.yaml` (gardé dans l'Output) et entraîne avec la meilleure. Garde-fou : jamais tant
d'arènes que chaque agent ne jouerait moins de 32 steps par lot (`train.min_segment_steps`). `AUTOTUNE = False` pour
s'en passer (profils seuls).

| | Kaggle GPU (T4 x2) | Kaggle TPU (v5e-8) | Colab gratuit |
|---|---|---|---|
| Quota | 30 h / semaine | ~20 h / semaine (à vérifier) | variable |
| Session | 12 h, **tourne navigateur fermé** | 9 h (à vérifier) | ~12 h, onglet ouvert |
| CPU pour jouer | 4 vCPU → 4 workers locaux | beaucoup (mesuré au lancement) | 2 vCPU |
| Apprentissage | T4 n°1 | TPU (torch_xla 2.9) ; repli CPU automatique | T4 |
| Arrêt propre (`HOURS = "auto"`) | après 11 h | après 8 h | 11 h |

---

## 0. Kaggle depuis le portable, sans le site : `cloud/kaggle_sync.py` (recommandé)

Plus de dataset de code à mettre à jour, de notebook à importer ni d'Output à télécharger à la main. Le notebook tourne
toujours sur Kaggle (navigateur fermé), mais c'est le portable qui l'envoie, le lance et rapatrie le run.

**Une fois** :
1. kaggle.com → *Settings* → *Phone verification* (GPU/TPU + Internet), puis *Settings → API → Create New Token* :
   placer le `kaggle.json` téléchargé dans `C:\Users\<toi>\.kaggle\` (c'est un identifiant : ne pas le partager).
2. `.venv\Scripts\python -m pip install kaggle`
3. `.venv\Scripts\python cloud/kaggle_sync.py setup` : vérifie le jeton, dépose tes collision meshes dans un dataset
   **privé** `rl-collision-meshes`, affiche le quota GPU/TPU restant.

**Chaque session** :
```bash
# 1re session d'un run (démarrage à chaud depuis un run du portable)
.venv\Scripts\python cloud/kaggle_sync.py launch --run-name s7f_spo_kaggle --init runs/s7e_spo_laptop --anchors-from s7 s7b s7c s7d s7e s7f
# sessions suivantes : reprise automatique là où la précédente s'est arrêtée
.venv\Scripts\python cloud/kaggle_sync.py launch --run-name s7f_spo_kaggle
# attendre la fin puis copier le run dans runs/s7f_spo_kaggle (+ runs/s7f_spo_kaggle.log)
.venv\Scripts\python cloud/kaggle_sync.py fetch --run-name s7f_spo_kaggle --wait
```
Options de `launch` : `--accelerator gpu|tpu|none` (défaut `gpu` = T4 x2), `--profiles s7 s7b ...` (défaut
`s7 s7b s7c s7d s7e s7f`), `--mode bench`, `--hours 6`, `--no-autotune`, `--fresh` (ne pas reprendre l'Output
précédent), `--dry-run` (prépare le notebook dans `dist/kaggle/kernels/<run>/` sans rien envoyer, sans jeton).
`status --run-name <run>` : état de la session.

Ce que ça fait :
- un notebook **privé** par run, « rl-agent <run> », généré depuis `cloud/kaggle_train.ipynb` avec ces réglages ;
  **le code y est embarqué** (le même zip que `rl-agent-code.zip`) : la session tourne avec le code du portable au
  moment du `launch` ;
- `--init` : construit `rl-agent-init.zip` (comme `make_cloud_bundle.py --init`) et le publie dans le dataset privé
  `rl-agent-init`, en attendant que Kaggle l'ait traité ;
- reprise : la nouvelle version prend en entrée l'Output de la version précédente du même notebook ;
- garde-fous : refuse de lancer si une session de ce run tourne déjà ; `fetch` ne copie rien si le run du portable est
  plus avancé que celui de Kaggle (`--force` pour passer outre). Rien n'est jamais supprimé dans `runs/`.

Limites : l'Output n'existe qu'à la **fin** de la session (d'où l'arrêt propre `HOURS`) ; les journaux en direct restent
sur la page du notebook (lien affiché par `launch`). Le quota hebdomadaire ne change pas.

Les sections 1 et 2 ci-dessous décrivent la méthode à la main (toujours valable).

## 1. Préparer les fichiers (une seule fois, sur le portable)

```bash
.venv\Scripts\python scripts/make_cloud_bundle.py --init runs/<run de départ>
```
Ça crée dans `rl-agent/dist/` :
- `rl-agent-code.zip` : le code (≈ 0,1 Mo). **Jamais** le débrief, `runs/`, les venvs.
- `rl-agent-init.zip` (avec `--init`) : le point de départ (dernier checkpoint + adversaires du pool). Sans `--init`
  (ou sans ce dataset), le run part de zéro.
- `--anchors-from <profils>` : ajoute à `rl-agent-init.zip` les adversaires fixes de ces profils (`selfplay.anchors`),
  déjà figés (`pool/anchors/`) : sur Kaggle, les runs sources n'existent pas. Pour s7e :
  `--init runs/s7c_spo_kaggle --anchors-from s7 s7b s7c s7d s7e` ; pour s7f :
  `--init runs/s7e_spo_laptop --anchors-from s7 s7b s7c s7d s7e s7f`.

Et à part : ton dossier `rl-agent/collision_meshes/` (avec `soccar/*.cmf`, ton dump). Tout reste **privé**.

## 2. Kaggle, étape par étape

### 2.1 Compte (une fois)
1. Créer un compte sur kaggle.com.
2. *Settings → Phone verification* : obligatoire pour avoir le GPU/TPU et Internet dans les notebooks.

### 2.2 Les datasets privés (une fois)
Pour chacun : *Create → New Dataset*, glisser le fichier, **Visibility : Private**, puis *Create*.
| Nom du dataset | Contenu |
|---|---|
| `rl-agent-code` | `dist/rl-agent-code.zip` (Kaggle le décompresse tout seul, ce n'est pas grave) |
| `rl-collision-meshes` | le dossier `collision_meshes` (ou un zip de ce dossier) |
| `rl-agent-init` | `dist/rl-agent-init.zip` (optionnel : démarrage à chaud) |

Quand le code change (nouvelle version de ma part) : *rl-agent-code → New Version* avec le nouveau zip.
**À faire pour cette version** (collecte accélérée, TPU, autoréglage) : sinon Kaggle garde l'ancien code.

### 2.3 Le notebook (une fois)
1. *Create → New Notebook*, puis *File → Import Notebook* → `cloud/kaggle_train.ipynb`.
2. Panneau de droite, *Add Input* → chercher et ajouter tes datasets (onglet *Your Work*).
3. Rien à régler pour les chemins : le notebook trouve tout seul le code, les meshes et le point de départ
   n'importe où sous `/kaggle/input`. Un chemin explicite peut être mis dans `CODE_INPUT`, `MESHES_INPUT`, `INIT_INPUT`.
4. *Settings* (panneau de droite) : **Internet : On**.

### 2.4 Choisir GPU ou TPU
*Settings → Accelerator* :
- **GPU T4 x2** (pas P100 : les PyTorch récents ne la gèrent plus) ;
- **TPU** (v5e-8) : apprentissage sur le TPU, parties sur ses nombreux CPU.

`PLATFORM = "auto"` détecte la machine. Pour comparer, une session courte `MODE = "bench"` (5-10 min, *Run All* en
interactif) affiche le tableau des mesures et la configuration retenue, puis s'arrête.

### 2.5 Entraîner (la vraie session)
1. 1re cellule : `MODE = "train"`, `RUN_NAME` (ex. `s7f_spo_kaggle`), `PROFILES` (réglages d'apprentissage, défaut
   `"s7 s7b s7c s7d s7e s7f"`), `HOURS = "auto"`.
2. **Save Version** (en haut à droite) → **Save & Run All (Commit)** → *Save*.
   Le notebook tourne tout seul, **tu peux fermer le navigateur**. Il s'arrête proprement avant la coupure de Kaggle
   (checkpoint garanti).
3. Suivre : *View Active Events* (en haut) ou la page du notebook → la version en cours → *Logs*. Au début : les lignes
   `[cloud]` (machine, device) et `[autotune]` (mesures, configuration retenue) ; ensuite une ligne toutes les 10
   itérations : `steps`, `sps` (débit), `elo`, `buts/min`…

### 2.6 Session suivante (reprendre le run)
1. Ouvrir le notebook (*Edit*). *Add Input* → *Notebook Output Files* → choisir **ce notebook** (la dernière
   version) : ses fichiers apparaissent sous `/kaggle/input/<nom-du-notebook>/`.
2. Laisser `PREV_OUTPUT = "auto"` et le même `RUN_NAME` : le notebook retrouve le run précédent dans les Inputs.
3. *Save Version → Save & Run All (Commit)*. Le run reprend exactement où il s'était arrêté — que la session précédente
   ait été GPU ou TPU (les checkpoints sont toujours écrits sur CPU).

### 2.7 Récupérer le run sur le portable
Page du notebook → onglet **Output** de la version → télécharger le dossier `runs/<RUN_NAME>` → le copier dans
`rl-agent/runs/`. Ensuite tout marche comme en local : RLViser, `evaluate.py`, export RLBot, ou continuer
l'entraînement sur le portable avec le même `--run-name`.

⚠️ Ne pas faire tourner le **même** `RUN_NAME` à la fois sur Kaggle et sur le portable (deux suites différentes).

## 3. Google Colab

1. Google Drive : créer `MyDrive/rl-agent/` et y mettre `rl-agent-code.zip`, `rl-agent-init.zip` et le dossier
   `collision_meshes/`.
2. Colab : *Fichier → Importer le notebook* → `cloud/colab_train.ipynb`. *Exécution → Modifier le type d'exécution →
   GPU T4*. Puis *Exécution → Tout exécuter* (autoriser l'accès à Drive).
3. Checkpoints dans `MyDrive/rl-agent/runs/` : si la session coupe, relancer le notebook reprend le run.
   Bouton *Stop* = arrêt propre avec checkpoint. Colab gratuit : garder l'onglet ouvert, 2 vCPU seulement.

## 4. Ce que fait `cloud/run_cloud.sh`

1. Détecte la machine (`PLATFORM=auto` : GPU NVIDIA, sinon TPU, sinon CPU) et la durée (`HOURS=auto`).
2. Crée un venv isolé (`uv`, Python 3.12) **hors** du dossier de sortie : PyTorch CUDA en GPU ; en TPU, `torch` et
   `torch_xla[tpu]` 2.9.0 (même version, `XLA_VERSION`). Puis RLGym, RocketSim (roue Linux) et le package.
3. Vérifie que l'accélérateur calcule vraiment (sinon apprentissage sur CPU, avec un message).
4. Reprend le run s'il a un checkpoint, sinon démarrage à chaud depuis `ckpt.pt` (+ copie du pool).
5. `AUTOTUNE=1` : `scripts/autotune.py` → `runs/<run>/autotune.yaml`. `MODE=bench` : s'arrête là.
6. Entraîne, arrêt propre après `HOURS` heures.

## 5. Réglages à la main (sans autoréglage)

Tout passe par la config (`--set` ou un profil) ; rien ne change ce que l'agent apprend :
| Clé | Valeurs | Effet |
|---|---|---|
| `train.collect_mode` | `central` (défaut) / `local` | qui calcule les actions pendant les parties |
| `train.collect_pipeline` | `1` (défaut) / `2` | (central) double tampon |
| `train.collect_device` | `cpu` (défaut) / `auto` / `cuda:1` | (central) device de l'inférence, versions du pool comprises |
| `train.stack_hist` | `true` (défaut) / `false` | versions du pool de même architecture jouées en un seul passage (`models/stacked.py`) |
| `train.n_workers` | nombre / `auto` (cœurs − 1) / `all` (tous) | processus de simulation |
| `ppo.device` | `auto` / `cpu` / `cuda` / `xpu` / `xla` | apprentissage |
| `ppo.batch_on_device` | `auto` (oui sur GPU NVIDIA) / `true` / `false` | lot copié une fois sur le GPU |

Mesurer à la main : `python scripts/autotune.py --profile kaggle --profile s7 --profile s7b --candidates
"central:3x16:p2:cuda:1,local:4x16"` ; durée d'un update : `python scripts/benchmark_update.py --device xla`.

**État des tests** : collecte (3 modes) et apprentissage testés sous Windows et sous Linux (WSL) ; le chemin TPU est
testé avec le backend CPU de PyTorch/XLA (mêmes résultats, pas de recompilation, entraînement + reprise) ; le script
complet `run_cloud.sh` a tourné sous Linux (WSL) en modes bench et train, y compris le repli « TPU absent → CPU ».
**Pas encore exécuté sur Kaggle même** (ni sur un vrai TPU / GPU T4) : si la 1re session échoue, copie-moi la sortie
de la cellule.
