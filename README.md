# Rizq — agent Rocket League par apprentissage par renforcement

Rizq est un agent qui joue à Rocket League. Il a appris **de zéro**, sans démonstration humaine, en jouant contre
lui-même dans **RocketSim** (la physique de Rocket League, sans graphismes et bien plus rapide que le temps réel) via
**RLGym v2**. Il joue ensuite dans le vrai jeu grâce à **RLBot v5**.
---

## Résultats

État au 26/11/2026, version **v33875** (le bot fourni dans `bot/model.pt`).

**Dans le vrai jeu** (RLBot, matchs de 5 minutes, 120 FPS) :

| Adversaire | Résultat |
|---|---|
| v30160 (version précédente du bot) | BO3 gagné 2-1 : 6-2, 7-4, 5-6 (prolongation) |
| Necto (bot RL de référence de la communauté RLBot) | victoire 6-4 (v30160 : 4 victoires sur 4) |
| Psyonix All-Star (meilleure IA du jeu de base) | battu par toutes nos versions testées depuis novembre (jusqu'à 15-3) |
| Nexto (successeur de Necto) | défaite 4-11 (v30160 : 3-13, 3-15, 3-11). C'est le prochain objectif |

**En simulation** (24 matchs 1v1) : 23 victoires sur 24 contre Necto (74 % des buts) ; contre Nexto, 9 victoires,
1 nul, 14 défaites (47 % des buts, contre 40 % pour v30160) ; 55 % des buts contre v30160. Le bot dispute
maintenant les kickoffs (1re touche dans 31 à 46 % des kickoffs contre Necto et Nexto, contre 0 % avant).

Entraînement : près de **34 milliards de steps** de jeu simulé, jusqu'à **~165 000 steps/s** sur Kaggle (TPU v5e-8,
224 cœurs CPU pour la simulation).

---

## Comment ça marche

```
   RocketSim (centaines de parties en parallèle, CPU)
          │  observations          ▲ actions
          ▼                        │
   Acteur (réseau de neurones) ────┘
          │  trajectoires
          ▼
   Apprentissage PPO (GPU / TPU) ──► nouvelle version ──► pool d'adversaires + Elo
                                                            │
                                                            ▼
                                                   vrai jeu via RLBot
```

- **Algorithme** : PPO clippé, en self-play contre la version courante et contre d'anciennes versions figées du bot.
- **Curriculum par paliers** : l'entraînement avance par étapes, des bases (aller à la balle, la frapper) jusqu'aux
  kickoffs, aériennes et duels ; un palier s'ouvre quand le bot maîtrise le précédent (`src/rlagent/curriculum.py`).
- **Échelle Elo absolue** sur les versions figées (`src/rlagent/selfplay/ladder.py`), pour savoir si une nouvelle
  version est vraiment meilleure, et un garde-fou qui coupe une récompense dont le bot abuse (`src/rlagent/guard.py`).
- **Évaluation** : chaque version retenue est comparée en simulation aux précédentes et à des bots de référence
  (Necto, Nexto), puis jouée dans le vrai jeu.

### Le modèle

| | |
|---|---|
| Entrée | état du jeu vu depuis la voiture (positions, vitesses, rotations, boost, pads), symétrisé pour jouer des deux côtés du terrain, plus des prédictions physiques (trajectoire de la balle, qui l'atteindra en premier) |
| Réseau | acteur MLP 3 × 512 (~0,7 M de paramètres), critique MLP 1024-1024-512 |
| Sortie | une action parmi 90 combinaisons de commandes, répétée sur 8 ticks de physique (15 décisions par seconde) |

### Du simulateur au vrai jeu

RLBot reconstruit mal certains indicateurs de la voiture (temps de saut, boost en cours…). Le bot fait donc tourner
en parallèle une copie RocketSim de sa voiture pour les recalculer exactement comme à l'entraînement
(`src/rlagent/env/shadow_sim.py`). Les observations du bot et celles du simulateur ont été comparées tick par tick sur
des matchs réels.

---

## Stack

| Brique | Choix |
|---|---|
| Simulateur | **RocketSim** |
| Environnement | **RLGym v2** (`rlgym[rl-sim]`) |
| Collecte | collecteur multi-processus maison, plusieurs arènes par worker (`algo/collector.py`) |
| Algorithme | PPO (SPO disponible via `ppo.policy_loss`, utilisé jusqu'à s9) |
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
- Voir le bot jouer en simulation : `python scripts/watch.py --policy runs/<run> --format 1`
- Matchs entre deux runs : `python scripts/evaluate.py --a runs/<runA> --b runs/<runB> --format 1 --games 50`
- Entraîner sur Kaggle / Colab : [`cloud/README.md`](cloud/README.md)
- Match dans le vrai jeu : `.venv-bot\Scripts\python scripts\run_match.py --config bot/match_allstar.toml`
  (venv du bot et RLBot : [`INSTALL.md`](INSTALL.md) §8)

Les collision meshes ne sont pas fournis : chacun les extrait de son propre jeu (`INSTALL.md` §4).

---

## Arborescence

```
├── src/rlagent/
│   ├── env/          observations, actions, états de départ, récompenses, prédictions physiques
│   ├── models/       MLP, acteur à attention, critique
│   ├── algo/         workers, collecteur, learner PPO/SPO, GAE
│   ├── selfplay/     pool d'adversaires, Elo, échelle Elo absolue
│   ├── opponents/    adaptateurs de bots de référence pour l'évaluation (Necto, Nexto…)
│   ├── curriculum.py paliers de compétence
│   ├── guard.py      garde-fou contre l'abus d'une récompense
│   └── train.py      boucle d'entraînement
├── configs/          default.yaml, profils machine (laptop, kaggle…) et profils de run (s2 → s10)
├── scripts/          benchmarks, évaluation, audits de style, visualisation, export du bot, préparation cloud
├── cloud/            notebooks Kaggle / Colab et synchronisation
├── bot/              bot RLBot v5, poids de v33875 (model.pt) et configurations de matchs
├── tests/
└── docs/             architecture, sprint 1, notes de conception des runs
```

Le bot fourni (`bot/model.pt`) est la version v33875. Non versionnés : `collision_meshes/` et `runs/` (checkpoints,
pool, métriques).

---

## Historique

| Runs | Idée | Résultat |
|---|---|---|
| s1-s2 | pipeline de bout en bout, SPO, acteur à attention | le bot apprend à toucher la balle |
| s3-s6 | lr piloté par le KL, plus de 1v1, longue session Kaggle | s6, meilleur agent avant s10, plafonne dès ~20 M steps |
| s7-s9 | refonte des récompenses, réglage de l'optimiseur | ne battent jamais s6 |
| **s10** | de zéro : PPO + MLP + curriculum par paliers + Elo absolu | bat s6 38-1-1 dès la 1re session |
| s10, sessions 2-9 | nouveaux paliers (précision, kickoffs, démos, aériennes, duels, arrêts…) | premières victoires contre Necto dans le vrai jeu (17/11), puis 3-0 avec v25785 |
| s10, session 10 | sparring contre Nexto et imitation de Nexto | l'imitation fait chuter le bot : abandonnée |
| s10, session 11 | sparring contre quatre bots (Nexto, GalactusV2, Kira, Rain) | v30160 bat v25785 3-1 et Necto 2-0 dans le vrai jeu |
| s10, session 12 | paliers kickoff, contacts, boost, tir cadré, mobilité | v33875 dispute les kickoffs, bat v30160 2-1 et Necto 6-4 |

Le détail de chaque session est dans [`docs/REFONTE_S10.md`](docs/REFONTE_S10.md).

## Documents

| Fichier | Contenu |
|---|---|
| [`INSTALL.md`](INSTALL.md) | installation (portable, gros PC, RLBot) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | conception : observations, actions, réseaux, PPO/SPO, self-play, métriques |
| [`docs/SPRINT_1.md`](docs/SPRINT_1.md) | backlog du sprint 1 |
| [`docs/REWARDS_S7.md`](docs/REWARDS_S7.md) | refonte des récompenses (s7) |
| [`docs/REFONTE_S10.md`](docs/REFONTE_S10.md) | diagnostic des runs s1-s9, conception de s10 et journal des sessions |
| [`cloud/README.md`](cloud/README.md) | entraîner sur Kaggle / Colab |

## Choix écartés

- **Q-learning (PQN / DQfD)** : abandonné au profit de PPO.
- **Un « cerveau central » qui pilote 3 voitures** : l'espace d'actions explose ; remplacé par un acteur partagé.
- **Behavior cloning sur des replays** : les replays ne contiennent pas les touches pressées.
- **rlgym-learn** : y ajouter nos briques demandait de réécrire son learner ; on a notre propre collecteur et PPO/SPO.

## Remerciements

- [RLGym](https://rlgym.org) et [RocketSim](https://github.com/ZealanL/RocketSim) pour l'environnement et la simulation ;
- [RLBot](https://rlbot.org) pour l'accès au vrai jeu ;
