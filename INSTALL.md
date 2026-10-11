# 🛠️ Installation

Deux machines :
- **Portable** : développement, tests et premiers runs.
- **Gros PC** : longs entraînements, et démo dans le vrai jeu (Windows + Rocket League).


---

## 1. Prérequis

| Outil | Pourquoi | Vérification |
|---|---|---|
| **Python 3.10 – 3.12** (64 bits) | RLGym v2 / PyTorch | `python --version` |
| **Git** | dépôt | `git --version` |
| **GPU NVIDIA** (optionnel) | accélère l'apprentissage ; la simulation reste sur le CPU | `nvidia-smi` |
| **Rocket League** (Windows) | uniquement pour dumper les collision meshes (§4) et pour la démo RLBot | — |

Pas besoin de Rust ni de compilateur C++ : RocketSim est installé en binaire (`rocketsim` sur PyPI), et on
n'utilise pas rlgym-learn pour l'instant (voir README, « Écarts »).

---

## 2. Environnement virtuel et dépendances

```bash
cd rl-agent
python -m venv .venv
# Windows (PowerShell) : .venv\Scripts\Activate.ps1      Linux/macOS : source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# PyTorch D'ABORD (choisir la commande sur https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU NVIDIA
pip install torch                                                       # CPU seulement

pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## 3. Tests (aucun besoin du jeu)

```bash
pytest
```
Inclut un mini-entraînement complet sur un **moteur factice** : toute la chaîne est validée avant RocketSim.

---

## 4. Collision meshes ⚠️ obligatoire pour RocketSim

Sans eux, RocketSim **ferme le processus** à la création de l'arène. Ils viennent de TON jeu :

0. ⚠️ Depuis le 28/04/2026, Rocket League PC tourne avec **Easy Anti-Cheat**, qui empêche le dumper de lire
   le processus (erreur « Failed to get main module of process … GetLastError(): 6 »). Lancer le jeu **avec EAC
   désactivé** (option officielle pour entraînement / hors ligne / LAN / mods). Ne jamais tenter de contourner EAC.
   RLBot (§8) a besoin du même mode.
1. Lancer Rocket League → **Entraînement → Free play** sur un **stade standard** (pas de terrain custom).
2. Télécharger `RLArenaCollisionDumper.exe` (dernière release) : https://github.com/ZealanL/RLArenaCollisionDumper/releases
3. Le lancer (en administrateur si besoin) pendant que tu es en free play.
4. Il produit `collision_meshes/soccar/mesh_0.cmf …` (≈ 16 fichiers). Fermer le jeu.
5. Copier **le dossier `collision_meshes`** à la racine du dépôt (à côté de `README.md`). Il est ignoré par git :
   le copier à la main sur chaque machine.

> Le dumper lit la mémoire du jeu (méthode standard de la communauté RLGym). À faire **hors ligne**, en free play.
> Autre emplacement possible : variable d'environnement `RS_COLLISION_MESHES=<chemin>`.

---

## 5. Vérification complète

```bash
python scripts/check_install.py
```
Vérifie Python, PyTorch (CPU/GPU), RLGym v2, RocketSim, le package, les meshes, puis crée un **vrai** env
RocketSim en 1v1/2v2/3v3 et joue quelques centaines de steps.

---

## 6. Premiers lancements

```bash
python -m rlagent.train --profile debug                       # fumée : 2-5 min
python scripts/benchmark_sps.py --workers 2 4 6 --envs 1 4 8  # régler train.n_workers / train.envs_per_worker
python -m rlagent.train --profile laptop --run-name s1_spo_laptop
python -m rlagent.train --profile laptop --set ppo.policy_loss=ppo --run-name s1_ppo_laptop   # ablation
```
- Arrêt propre : créer `runs/<run>/STOP` (ou Ctrl+C) → checkpoint puis sortie.
- Reprise : même commande et même `--run-name`, ou `--resume runs/<run>`.
- Métriques : `runs/<run>/metrics.jsonl` (+ W&B si `wandb login` a été fait).

---

## 7. Outils de visualisation et d'évaluation

```bash
pip install "rlgym[rl-rlviser]==2.0.1"
python scripts/watch.py --policy runs/<run> --format 1                       # voir le bot (RLViser)
python scripts/evaluate.py --a runs/<runA> --b runs/<runB> --format 1 --games 50   # matchs simulés
python scripts/evaluate.py --a runs/<run> --scenario empty_goal --trials 200      # critère S1-20 (but vide)
```

---

## 8. RLBot v5 (vrai jeu) — **gros PC Windows uniquement**

1. Installer **RLBot v5** en suivant le wiki officiel (https://wiki.rlbot.org/v5/).
2. Créer un **venv dédié au bot** (séparé du venv d'entraînement : `rlgym-rlbot` impose ses versions de `rlbot`) :
   ```bash
   py -3.12 -m venv .venv-bot
   .venv-bot\Scripts\python -m pip install rlgym-rlbot
   .venv-bot\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv-bot\Scripts\python -m pip install -e .
   ```
   Testé le 12/10/2026 : rlgym-rlbot 0.3.0, rlbot 2.0.0b52 ; `bot/bot.py` se charge et joue sur une observation.
3. `python scripts/export_bot.py --run runs/<run>` → crée `bot/model.pt` (non versionné).
4. Ajouter le dossier `bot/` dans RLBot (`bot/bot.toml`, format v5 vérifié avec le wiki ; `run_command` utilise `.venv-bot`).
5. Ajouter au botpack les bots de référence : Psyonix (Rookie → All-Star), Necto, Nexto.

> ⚠️ Jamais en ligne. RLBot lance le jeu dans un mode hors ligne.

---

## 9. Arborescence

```
rl-agent/
├── README.md  INSTALL.md  ARCHITECTURE.md  SPRINT_1.md
├── pyproject.toml  requirements.txt
├── configs/            default.yaml + profils laptop / bigpc / debug
├── src/rlagent/
│   ├── env/            obs, actions, mutators (formats + états de départ), rewards, factory, moteur factice
│   ├── models/         acteur à attention + critique
│   ├── algo/           worker, collector, learner (PPO/SPO), losses, gae
│   ├── selfplay/       pool d'adversaires + Elo
│   ├── metrics/        stats d'épisode (jeu d'équipe, kickoffs)
│   ├── utils/          journalisation
│   └── train.py        boucle principale
├── scripts/            check_install, benchmark_sps, evaluate, watch, export_bot
├── bot/                bot RLBot v5
├── tests/
├── collision_meshes/   (non versionné) soccar/mesh_*.cmf
└── runs/               (non versionné) un dossier par run : config, checkpoints, pool, metrics.jsonl
```

---

## 10. Checklist

- [ ] Portable : §2 + `pytest` vert
- [ ] Collision meshes dumpés et copiés (§4)
- [ ] `check_install.py` → « Tout est prêt »
- [ ] Run `debug` OK, débit mesuré, `n_workers` réglé
- [ ] Gros PC : idem + RLBot v5 + un match contre Psyonix Rookie
- [ ] `pip freeze > requirements.lock.txt` commité
