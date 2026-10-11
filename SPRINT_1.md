# 🏁 Sprint 1 — Pipeline de bout en bout

**Dates proposées** : lundi 12 → dimanche 25 octobre 2026 (2 semaines)
**Objectif du sprint** : *un premier bot entraîné par notre PPO/SPO, qui marque dans un but vide en simulation et joue un match dans le vrai jeu via RLBot.*

> Rôles proposés (à confirmer par le PO) :
> **Dev A : Env & Rewards** (observations, actions, états de départ, rewards, RocketSim)
> **Dev B : Algo & Infra** (PPO/SPO, réseaux, rlgym-learn, W&B, export RLBot)

---

## 📋 Backlog

### 🔧 Installation & outillage

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-01 | Installer l'environnement sur le **portable** (INSTALL.md §1-6) | A+B | `smoke_test.py` passe | 0,5 j |
| S1-02 | Installer le **gros PC** (INSTALL.md §1-8, dont RLBot v5) | B | smoke test OK + match Psyonix vs Psyonix lancé via RLBot | 1 j |
| S1-03 | Créer le dépôt Git + arborescence + `requirements.txt` figé | B | structure conforme à INSTALL.md §9, `pip install -r` fonctionne | 0,5 j |
| S1-04 | Créer l'équipe W&B + logging de base | B | une run test apparaît dans W&B avec les métriques | 0,5 j |
| S1-05 | Extraire et partager les **collision meshes** (si RocketSim les demande) | A | env créé sans erreur sur les 2 machines | 0,5 j |

### 🔍 RocketSim (verdict : opérationnel tel quel pour le soccar, on ne le modifie pas)

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-06 | **Dumper les collision meshes** avec RLArenaCollisionDumper (jeu installé requis), les placer là où RocketSim les attend, puis les partager sur les 2 machines | A | env RocketSim créé sans erreur sur les 2 machines | 0,5 j |
| S1-07 | **Test de cohérence simulation / vrai jeu** : même état initial (kickoff + quelques tirs) dans RocketSim et dans le vrai jeu via RLBot, comparaison des trajectoires sur 1 à 2 s | A | écart documenté dans `docs/sim_vs_game.md` | 1 j |
| S1-08 | Si un **vrai** écart gênant apparaît : le documenter, le PO décide (correctif ou contournement) | PO | décision notée | — |

### 🌍 Environnement

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-09 | **Obs builder** v1 (cf. ARCHITECTURE §3, sans prédiction de balle) normalisé côté équipe | A | test unitaire : obs identique pour bleu et orange en situation miroir | 1,5 j |
| S1-10 | **Action parser** : lookup table + répétition 8 ticks | A | test unitaire : nombre d'actions et ticks corrects | 0,5 j |
| S1-11 | **Tirage du format** 1v1 / 2v2 / 3v3 (70/20/10, configurable) | A | sur 1000 resets, la répartition respecte les poids à ±3 % | 0,5 j |
| S1-12 | **États de départ** : kickoff + position aléatoire | A | visualisés dans RLViser | 0,5 j |
| S1-13 | **Rewards v1** : but, touche, vitesse vers la balle, balle vers le but, boost + **esprit d'équipe (τ)** | A | chaque reward a un test unitaire, poids dans un YAML | 1,5 j |

### 🧠 Algorithme

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-14 | **Acteur à attention** partagé + **critique centralisé** | B | forward OK en 1v1, 2v2 et 3v3 avec le même modèle | 1,5 j |
| S1-15 | **PPO** maison (GAE, normalisations, entropie, minibatches) branché sur rlgym-learn | B | le bot apprend sur un scénario simple (toucher la balle) | 2 j |
| S1-16 | **Flag SPO** (`--policy-loss spo`) | B | test unitaire : loss conforme à la formule, les deux modes tournent | 0,5 j |
| S1-17 | **Checkpoints** + reprise d'entraînement | B | un run arrêté reprend sans perte | 0,5 j |
| S1-18 | Métriques de jeu + **métriques d'équipe** (double commit, collisions coéquipiers, distance) | A+B | visibles dans W&B | 1 j |

### 🎮 Vrai jeu

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-19 | **Export vers RLBot v5** via `rlgym-rlbot` (même code d'obs/actions qu'à l'entraînement) | B | notre bot joue un match complet contre Psyonix Rookie dans le vrai jeu | 1 j |

### 🧪 Validation

| # | Tâche | Resp. | Critère d'acceptation | Est. |
|---|---|---|---|---|
| S1-20 | **Premier run** de quelques heures (gros PC si dispo, sinon portable) | A+B | le bot **marque dans un but vide** de manière fiable en simulation | — |
| S1-21 | Mesurer le **débit** (steps/s) sur les 2 machines | B | chiffres notés dans le README | 0,5 j |

---

## ✅ Definition of Done (sprint)

- [ ] Installation reproductible sur les 2 machines (`requirements.txt` figé)
- [ ] Un seul modèle tourne en 1v1, 2v2 et 3v3
- [ ] PPO **et** SPO fonctionnent (flag)
- [ ] Le bot marque dans un but vide en simulation
- [ ] Le bot joue un match complet dans le vrai jeu via RLBot
- [ ] Métriques d'apprentissage, de jeu et d'équipe visibles dans W&B
- [ ] Collision meshes en place et test de cohérence simulation / vrai jeu documenté
- [ ] Tests unitaires verts (`pytest`)

## 🔜 Aperçu du sprint 2

Prédiction de trajectoire de balle dans les observations, états de départ avancés (air, mur, défense), entraînement aux kickoffs, pool de self-play avec Elo, premiers runs longs sur le gros PC.

## ❓ À trancher par le PO avant/pendant le sprint

- [ ] Confirmer la répartition des rôles (A / B)
- [ ] Format exact du match final et date de la démo
- [ ] Config du gros PC (cœurs CPU / GPU / RAM)

---

## 📍 Avancement au 11/10/2026 (code livré, à valider sur machine réelle)

| # | État | Remarque |
|---|---|---|
| S1-01/02 | ⏳ | Procédure complète dans `INSTALL.md` |
| S1-03 | ✅ | Dépôt, arborescence, `pyproject.toml`, `requirements.txt` (figer avec `pip freeze` après install) |
| S1-04 | ✅ code | W&B optionnel (auto : en ligne si `wandb login`, sinon hors ligne) + `metrics.jsonl` toujours |
| S1-06 | ⛔ bloqué | Dumper bloqué par Easy Anti-Cheat → relancer le jeu sans EAC (INSTALL §4) |
| S1-07 | ⏳ | Après S1-06 et RLBot |
| S1-09 | ✅ testé | Obs à jetons masqués, normalisées côté équipe ; test de symétrie miroir vert |
| S1-10 | ✅ testé | 90 actions, répétition 8, action précédente dans l'obs |
| S1-11 | ✅ testé | Formats 70/20/10, écart < 3 % sur 3000 tirages |
| S1-12 | ✅ testé | Kickoff + aléatoire (visualisation RLViser : `scripts/watch.py`) |
| S1-13 | ✅ testé | 7 termes + esprit d'équipe τ + zéro-somme, poids en YAML |
| S1-14 | ✅ code | Transformer à jetons, acteur + critique ; tests d'invariance écrits (PyTorch requis) |
| S1-15/16 | ✅ code | PPO/SPO maison, flag `ppo.policy_loss` ; test « bandit » qui vérifie l'apprentissage |
| S1-17 | ✅ code | Checkpoints, reprise, arrêt propre (fichier STOP), protection anti-NaN |
| S1-18 | ✅ code | Métriques de jeu et d'équipe (double commit, contacts, distance, kickoffs vs historique) |
| S1-19 | ✅ code | `bot/` + `scripts/export_bot.py` ; `bot.toml` à valider avec le wiki RLBot v5 |
| S1-20/21 | ⏳ | `scripts/benchmark_sps.py`, puis run long |
| bonus | ✅ code | Pool d'adversaires + Elo (prévu sprint 2), moteur factice pour tester toute la chaîne sans le jeu |
