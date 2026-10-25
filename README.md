# Agent Rocket League — PPO / SPO

Agent de reinforcement learning pour Rocket League, entraîné en self-play dans **RocketSim** via **RLGym v2**.
Le projet sera jugé en **match 1v1 à opposition directe** contre l'agent PPO d'une autre équipe ; à terme, le même
code doit aussi tenir en 2v2/3v3.

Tout est écrit en Python : environnement, collecte multi-processus, PPO/SPO, self-play, Elo et export vers le vrai jeu
(RLBot). RocketSim est utilisé tel quel.

---

## Où on en est (25/10/2026)

L'agent actuel, **s10**, est reparti de zéro après 15 runs qui plafonnaient (voir [Historique des runs](#historique-des-runs)) :

- PPO clippé, lr fixe, réseau MLP (acteur 512×3, critique 1024-1024-512), 1v1 uniquement ;
- un **curriculum par paliers** : le bot passe au palier suivant quand il maîtrise le précédent (`curriculum.py`) ;
- une **échelle Elo absolue** sur les versions figées du bot (`selfplay/ladder.py`), avec l'ancien meilleur agent (s6)
  comme repère fixe.

Première session sur Kaggle (TPU v5e-8, 224 vCPU) : **2,9 milliards de steps en 8 h** (~100 000 steps/s).
Contre s6, notre meilleur agent jusque-là, en 40 matchs 1v1 de 5 min : **38 victoires, 1 nul, 1 défaite (buts 142-25)**.

En cours : un 4e palier « précision » (viser le but plutôt que la touche, kickoffs, vraies aériennes) et la
sauvegarde automatique de la meilleure version. Détails : [`docs/REFONTE_S10.md`](docs/REFONTE_S10.md).

---

## Stack

| Brique | Choix |
|---|---|
| Simulateur | **RocketSim** (physique de Rocket League, bien plus rapide que le temps réel) |
| Environnement | **RLGym v2** (`rlgym[rl-sim]`) |
| Collecte | collecteur multi-processus maison, plusieurs arènes par worker, mode asynchrone (`algo/collector.py`) |
| Algorithme | **PPO et SPO** dans le même code, choisis par `ppo.policy_loss` |
| Réseaux | MLP (s10) ou acteur à attention + critique centralisé type MAPPO (s1-s9, pour le 2v2/3v3) |
| Self-play | pool d'adversaires historiques, ancres fixes, Elo |
| Calcul de l'update | CPU, iGPU Intel (`xpu`), GPU NVIDIA (`cuda`) ou TPU via XLA (`ppo.device`) |
| Cloud | Kaggle (GPU / TPU) et Google Colab, pilotés depuis le PC (`cloud/`) |
| Suivi | `metrics.jsonl` par run, Weights & Biases en option, **RLViser** pour voir les parties |
| Vrai jeu | **RLBot v5** + `rlgym-rlbot` |

---

## Démarrage rapide

Installation complète (PyTorch, collision meshes, RLBot) : [`INSTALL.md`](INSTALL.md).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS : source .venv/bin/activate
pip install torch                      # ou la version CUDA / xpu, cf. INSTALL.md
pip install -r requirements.txt
pip install -e ".[dev]"

pytest                                 # tests, sans le jeu ni RocketSim (moteur factice)
python scripts/check_install.py        # vérifie RocketSim + collision meshes
python -m rlagent.train --profile debug                                           # run de fumée (2-5 min)
python -m rlagent.train --profile laptop --profile s10 --run-name s10_ppo_laptop  # vrai run
```

- Arrêt propre : créer `runs/<run>/STOP` (ou Ctrl+C) ; reprise : même commande, même `--run-name`.
- Voir le bot jouer : `python scripts/watch.py --policy runs/<run> --format 1`
- Matchs entre deux runs : `python scripts/evaluate.py --a runs/<runA> --b runs/<runB> --format 1 --games 50`
- Entraîner sur Kaggle / Colab : [`cloud/README.md`](cloud/README.md)

---

## Arborescence

```
├── src/rlagent/
│   ├── env/          observations, actions, états de départ, rewards, moteur factice pour les tests
│   ├── models/       MLP, acteur à attention, critique
│   ├── algo/         workers, collecteur, learner PPO/SPO, GAE
│   ├── selfplay/     pool d'adversaires, Elo, échelle Elo absolue
│   ├── metrics/      statistiques par épisode
│   ├── curriculum.py paliers de compétence
│   └── train.py      boucle d'entraînement
├── configs/          default.yaml + profils machine (laptop, kaggle…) et profils de run (s2 → s10)
├── scripts/          benchmarks, évaluation, visualisation, export du bot, préparation cloud
├── cloud/            notebooks Kaggle / Colab et synchronisation
├── bot/              bot RLBot v5
├── tests/
└── docs/             architecture, sprint 1, notes de conception des runs
```

Non versionnés : `collision_meshes/` (à dumper depuis son propre jeu), `runs/` (checkpoints, pool, métriques).

---

## Historique des runs

| Runs | Idée | Résultat |
|---|---|---|
| s1-s2 | pipeline de bout en bout, SPO, acteur à attention, rewards de base | le bot apprend à toucher la balle (0,6 → 11 touches/min) |
| s3-s5 | lr piloté par le KL, plus de 1v1 dans les données | chaque run repart du précédent (démarrage à chaud) |
| s6 | même recette, longue session Kaggle (163 M steps) | meilleur agent jusqu'à s10, mais plafonne dès ~20 M et ne saute presque plus |
| s7-s7f | refonte des rewards, exercices, curriculum ([`docs/REWARDS_S7.md`](docs/REWARDS_S7.md)) | ne bat jamais s6 |
| s8-s9 | réglage de l'optimiseur (contrôle des normes, taille des pas) | aucun gain sur s6 ; les grands pas font régresser (gradient trop bruité) |
| **s10** | de zéro : PPO + MLP + curriculum par paliers + Elo absolu | **bat s6 38-1-1** |

## Débit mesuré

| Machine | Config | De bout en bout |
|---|---|---|
| Portable (Core Ultra 5 125H), update sur CPU | 4 workers × 8 arènes | ~1 300 steps/s |
| Portable, update sur l'iGPU Intel Arc | 10 workers × 12 arènes, asynchrone | ~5 000 steps/s (s10 : 6 600-8 000) |
| Kaggle TPU v5e-8 (224 vCPU) | 168 workers × 8 arènes, update sur TPU | **~100 000 steps/s** |

Optimisations sans effet sur l'apprentissage (vérifiées par des tests d'équivalence contre des copies figées dans
`tests/reference/`) : plusieurs arènes par worker, Transformer limité aux jetons réels, update sur accélérateur pendant
la collecte du lot suivant, observations et rewards en arithmétique Python directe (×1,9 par cœur).

---

## Choix écartés

- **PQN / DQfD (Q-learning)** : abandonné au profit de PPO/SPO.
- **GigaLearnCPP** : C++, et sa version publique provient d'une fuite.
- **Un « cerveau central » qui pilote 3 voitures** : l'espace d'actions explose et le modèle ne passe pas d'un mode à
  l'autre ; remplacé par un acteur partagé (MAPPO).
- **Behavior cloning sur des replays** : les replays ne contiennent pas les touches pressées ; gardé en bonus éventuel.
- **rlgym-learn** : y ajouter SPO, l'acteur à attention et le pool d'adversaires demandait de réécrire son learner ;
  on a notre propre collecteur et PPO/SPO, testés.

## Documents

| Fichier | Contenu |
|---|---|
| [`INSTALL.md`](INSTALL.md) | installation sur le portable et sur le gros PC, RLBot |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | conception : observations, actions, réseaux, PPO/SPO, self-play, métriques |
| [`docs/SPRINT_1.md`](docs/SPRINT_1.md) | backlog du sprint 1 et avancement |
| [`docs/REWARDS_S7.md`](docs/REWARDS_S7.md) | refonte des rewards et de l'entraînement (s7) |
| [`docs/REFONTE_S10.md`](docs/REFONTE_S10.md) | diagnostic des runs s1-s9 et conception de s10 |
| [`cloud/README.md`](cloud/README.md) | entraîner sur Kaggle / Colab |

## Règle d'éthique

Le bot ne joue **jamais en ligne / en ranked**. Il tourne uniquement hors ligne via RLBot. Des bots RL (Nexto) ont déjà
été détournés pour tricher, et Psyonix bannit ces comptes.
