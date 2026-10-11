# 🚀 Agent Rocket League — Projet RL

Agent de reinforcement learning pour Rocket League capable de jouer en **1v1, 2v2 et 3v3**, avec un seul modèle.
**Priorité : le 1v1.** Le projet se jugera en **match à opposition directe** contre l'agent PPO d'une autre équipe.

> Product Owner : *(ton nom)*

---

## 🎯 Objectif

1. **Battre l'agent adverse en 1v1** en match direct.
2. Garder un agent **compétent en 2v2/3v3**, avec de vraies rotations : pas de double commit, pas de collisions entre coéquipiers.
3. Se rapprocher au maximum du niveau des bots ML de référence (Necto → Nexto → Ripple/Seer) et le mesurer par un Elo.

## 🧱 Stack (décisions validées)

| Brique | Choix |
|---|---|
| Langage | **Python** uniquement (RocketSim est utilisé tel quel, sans modification C++) |
| Simulateur | **RocketSim** (physique de Rocket League, bien plus rapide que le temps réel) |
| Environnement | **RLGym v2** (`rlgym[rl-sim]`) |
| Parallélisation | **Collecteur multi-processus maison** (`src/rlagent/algo/collector.py`) ; rlgym-learn reporté (voir Écarts) |
| Algorithme | **PPO + SPO** dans le même code, choisis par un flag (`--policy-loss ppo|spo`) |
| Multi-agent | **MAPPO** : acteur partagé à attention + critique centralisé |
| Suivi | **Weights & Biases** + **RLViser** pour voir les parties |
| Démo / évaluation | **RLBot v5** + wrapper **`rlgym-rlbot`** (vrai jeu) |

## ❌ Décisions écartées

- **PQN / DQfD (Q-learning)** : abandonné.
- **GigaLearnCPP** : C++, et sa version publique provient d'une fuite → écarté.
- **Un « cerveau central » qui pilote 3 voitures** : l'espace d'actions explose et le modèle ne passe pas d'un mode à l'autre. On le remplace par MAPPO.
- **Behavior cloning sur des replays** : mis de côté, éventuellement en bonus de fin de projet. Les replays ne contiennent pas les touches pressées, et les résultats au championnat RLBot sont peu convaincants.

## ↔️ Écarts par rapport au plan initial (à valider par le PO)

- **rlgym-learn n'est pas utilisé pour l'instant.** Son PPO est enfoui dans sa bibliothèque : y ajouter SPO,
  l'acteur à attention et le pool d'adversaires historiques demandait de réécrire une partie de son learner.
  On a donc un collecteur multi-processus et un PPO/SPO à nous (~600 lignes, testés), ce qui colle aussi au
  « PPO maison » du sprint 1. Si le débit devient le goulot d'étranglement sur le gros PC, on pourra brancher
  rlgym-learn plus tard sans toucher à l'env, au modèle ni aux rewards.
- **Pool d'adversaires + Elo dès le sprint 1** (prévu au sprint 2) : sans lui, un run long sans surveillance
  risque de tourner en rond contre lui-même.
- **Critique « centralisé »** : Rocket League étant entièrement observable, l'acteur voit déjà tout le terrain ;
  le critique est donc un réseau séparé recevant l'état global vu depuis l'agent (variante MAPPO standard).

## ⏱️ Débit mesuré (S1-21)

| Machine | Config | Collecte seule | De bout en bout |
|---|---|---|---|
| Portable (Core Ultra 5 125H, 14 cœurs, sans GPU) | 4 workers × 8 arènes, synchrone, 8 threads (`configs/laptop.yaml`) | ~6 700 steps/s | **~1 400 steps/s** (≈ 5 M steps/heure) |
| Gros PC | à mesurer : `python scripts/benchmark_sps.py --workers 4 8 12 --envs 4 8` | — | — |

Sur CPU, le goulot est l'**update PPO/SPO** (~25 s pour 50k steps), pas la simulation : un GPU (gros PC ou Kaggle) est
le principal levier restant. Optimisations déjà faites, sans effet sur l'apprentissage (vérifiées par des tests
d'équivalence) : plusieurs arènes par worker, Transformer limité aux jetons réels (×1,65 sur l'update),
observations optimisées en 2v2/3v3.

## 🎯 Objectif « battre un Grand Champion 1 »

GC1 ≈ le niveau de **Nexto** (≈ GC1 en 1v1/2v2/3v3), qui a demandé l'équivalent de ~250 000 heures de jeu
simulées. C'est atteignable en principe avec cette stack, mais c'est une question de **compute** : le portable
sert à valider et à démarrer ; le gros PC tournant jour et nuit pendant des semaines est indispensable.
Le suivi Elo + les matchs contre Necto/Nexto dans RLBot diront honnêtement où on en est.

## 📁 Documents

| Fichier | Contenu |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Installation complète, sur le portable puis sur le gros PC |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Conception technique : observations, actions, réseau, rewards, SPO, self-play |
| [`SPRINT_1.md`](SPRINT_1.md) | Backlog du sprint 1, critères d'acceptation et avancement |

## ⚠️ Règle d'éthique

Le bot ne joue **jamais en ligne / en ranked**. Il tourne uniquement hors ligne via RLBot. Des bots RL (Nexto) ont déjà été détournés pour tricher, et Psyonix bannit ces comptes.

## 🗺️ Roadmap (environ 2,5 mois)

| Phase | Contenu |
|---|---|
| **Sprint 1** | Installation, pipeline de bout en bout, PPO/SPO, premier bot qui marque, export vers RLBot |
| **Sprint 2** | Observations enrichies (prédiction de balle), états de départ variés, kickoffs, self-play avec pool et Elo |
| **Sprint 3** | Longs runs sur le gros PC, ablation PPO vs SPO, montée en 2v2/3v3 (esprit d'équipe) |
| **Sprint 4** | Affinage contre les bots de référence (Necto/Nexto), préparation du match et de la démo |

## ❓ Questions ouvertes (PO)

- [ ] Format du match : nombre de matchs, durée, mode(s) ? Leur bot est-il visible avant le match ?
- [ ] Date exacte de la démo
- [ ] Configuration du gros PC (CPU : nombre de cœurs / GPU / RAM)
- [ ] Répartition des rôles dans l'équipe
