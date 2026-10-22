# 🧠 Architecture technique

Ce document fixe la conception. Toute modification passe par le PO.

---

## 1. Vue d'ensemble

```
            ┌──────────── Collecte (CPU, multi-processus) ────────┐
            │  N parties RocketSim en parallèle                    │
            │  format tiré au hasard : 1v1 / 2v2 / 3v3             │
            │  bleu = politique courante                           │
            │  orange = politique courante OU ancienne version     │
            └──────────────┬───────────────────────────────────────┘
                           │ trajectoires
                           ▼
            ┌──── Apprentissage (GPU NVIDIA / iGPU Intel / TPU) ──┐
            │  GAE → PPO ou SPO (flag) → acteur + critique         │
            └──────────────┬───────────────────────────────────────┘
                           │ checkpoints
                           ▼
            Pool d'adversaires + Elo  ──►  Évaluation RLBot (vrai jeu)
```

Le principe : **un seul modèle** pour les trois formats. Chaque voiture est contrôlée indépendamment par le même acteur partagé.

**Où tourne quoi (infra, sans effet sur ce qui est appris).** RocketSim n'existe que sur CPU. Deux modes de collecte
(`train.collect_mode`, `algo/collector.py`), qui produisent exactement les mêmes parties pour une même politique
(tests d'équivalence) :
- `central` : le processus principal calcule les actions de toutes les arènes, sur CPU ou GPU (`train.collect_device`,
  ex. 2e T4 de Kaggle) ; `train.collect_pipeline: 2` = double tampon (les workers simulent une moitié des arènes
  pendant le calcul des actions de l'autre).
- `local` : chaque worker joue ses arènes avec sa propre copie de l'acteur (poids par mémoire partagée) ; aucun
  aller-retour par step, passe à l'échelle sur les machines à beaucoup de cœurs.

L'apprentissage tourne sur CPU, GPU (cuda / xpu) ou TPU (PyTorch/XLA, formes de tenseurs fixes : `src/rlagent/accel.py`).
Sur le cloud, `scripts/autotune.py` choisit la combinaison la plus rapide en début de session (cf. `cloud/README.md`).

---

## 2. Formats de jeu (anti « bot 1v1 qui ne sait pas jouer en équipe »)

Dès le **sprint 1**, chaque partie tire son format au hasard :

| Format | Poids initial |
|---|---|
| 1v1 | 70 % |
| 2v2 | 20 % |
| 3v3 | 10 % |

Le 1v1 reste prioritaire, mais le bot ne « désapprend » jamais la présence d'un coéquipier. Les poids sont configurables dans `configs/`.

⚠️ Ces poids portent sur les **épisodes**. Un épisode 3v3 produit 3× plus de transitions d'apprentissage qu'un 1v1 :
avec 70/20/10 en épisodes, le 1v1 ne fait que **50 %** des données. Depuis le run s3 (16/10/2026), on vise 70/20/10
en **données**, soit des poids d'épisodes 0,84 / 0,12 / 0,04 (`configs/laptop_s3.yaml`).

---

## 3. Observations

On normalise **du point de vue de l'équipe** : on inverse le terrain pour l'équipe orange, pour que le bot « attaque » toujours dans la même direction.

**Par entité** (mêmes features pour chaque voiture) :
- position, vitesse linéaire, vitesse angulaire, orientation (forward + up)
- boost, au sol / en l'air, flip disponible, démoli
- **distance à la balle** et **rang de proximité dans son équipe** (qui est le 1er homme ?)
- rôle : `soi`, `coéquipier` ou `adversaire` (one-hot)

**Global** :
- balle : position, vitesse, vitesse angulaire
- **prédiction de trajectoire de la balle** (option `env.ball_prediction`, depuis s6) : calcul **analytique** (gravité,
  un rebond au sol, murs en miroir), 14 valeurs : temps et position de retombée, hauteur max, positions à 0,5 / 1 /
  1,5 s. Quasi gratuit (pas de simulation). Les anciens agents ignorent ces colonnes (`fit_obs`), et un agent existant
  est élargi sans rien perdre (nouvelles entrées à poids nuls).
- état des pads de boost, score, temps restant, indicateur de kickoff
- action précédente

Les autres voitures forment un **ensemble de taille variable** (0 à 5) traité par l'attention : les emplacements vides sont **masqués** (le réseau ne les voit pas), pas simplement mis à zéro.

Implémentation : `src/rlagent/env/obs.py` — 105 features « soi + global », 34 par autre voiture (×5), + masque (5) = 280. Test de symétrie : une situation miroir donne exactement la même observation au bleu et à l'orange.

---

## 4. Actions

- **Table discrète (lookup table)** d'environ 90 actions, standard dans la communauté RLGym, avec **répétition de 8 ticks** (≈ 15 décisions/s).
- À tester plus tard : répétition 4 vs 8, et masquage d'actions (pas de boost à 0 %, etc.).

---

## 5. Réseaux

### Acteur partagé (décentralisé)
```
obs_soi ──► MLP ──┐
                  ├──► attention (soi = query ; autres voitures + balle = keys/values) ──► MLP ──► logits (≈90)
obs_autres ─► MLP ┘
```
- Invariant au nombre de joueurs : ça marche en 1v1, 2v2 et 3v3 sans modification.
- LayerNorm, activation ReLU/GELU (configurable).
- Implémentation : petit Transformer (2 couches, d = 128, 4 têtes) sur [jeton « soi » + 5 jetons voitures], masque de padding ; la sortie du jeton « soi » alimente la tête (`src/rlagent/models/attention.py`).
- Taille de départ modeste. On grossit au sprint 3 : c'est là que SPO est censé briller.

### Critique centralisé (MAPPO)
- Voit **l'état complet** (toutes les voitures + la balle), ce qui aide à attribuer correctement le mérite d'une action collective.
- Le jeu étant entièrement observable, c'est un réseau séparé (même architecture, sans poids partagés) qui reçoit l'état global vu depuis l'agent : la variante « agent-specific global state » de MAPPO.
- Utilisé uniquement à l'entraînement : en match, seul l'acteur tourne.

---

## 6. Algorithme : PPO / SPO

Même code, seule la loss de politique change (`--policy-loss ppo|spo`), avec `r = π_θ(a|s) / π_old(a|s)` :

```python
def policy_loss(log_prob, old_log_prob, adv, eps, kind="ppo"):
    ratio = (log_prob - old_log_prob).exp()
    if kind == "ppo":
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - eps, 1 + eps) * adv
        return -torch.min(surr1, surr2).mean()
    elif kind == "spo":
        # Simple Policy Optimization (Xie et al., ICML 2025)
        return -(ratio * adv - adv.abs() / (2 * eps) * (ratio - 1) ** 2).mean()
```

Commun aux deux : GAE, normalisation des avantages, normalisation des retours, bonus d'entropie, gradient clipping, minibatches.

**Hyperparamètres de départ** (à ajuster) : batch ≈ 100k steps, minibatch ≈ 50k, 2 epochs, lr 2e-4, γ = 0,99, λ = 0,95, ε = 0,2, entropie 0,01.

**Learning rate piloté par le KL** (`ppo.lr_schedule: adaptive_kl`, depuis s3) : à chaque itération, on mesure le
déplacement réel de la politique (KL entre la politique de collecte et la politique mise à jour) et on ajuste le lr
de l'acteur pour viser `target_kl` = 0,01 (lr ÷ 1,5 si KL > 2 × cible, × 1,5 si KL < cible / 2, borné à
[5e-5, 1e-3]). Garde-fou : on arrête l'itération si le KL dépasse 4 × cible. Motif : avec un lr fixe de 2e-4, le KL
par itération était de 1e-4 à 1e-3 et la politique restait quasi uniforme après 5 M steps.

**Collecte asynchrone → objectif « découplé »** (Hilton et al. 2021). Quand l'update tourne pendant la collecte du lot
suivant, ce lot a été joué par la politique d'il y a une itération. Le ratio est alors calculé par rapport à la politique
au début de l'update (π_prox, centre de la zone de confiance) et chaque terme est pondéré par w = π_prox / π_collecte
(sans gradient, borné à 2). Sans retard, w = 1 et on retrouve exactement la formule ci-dessus. Sans ce correctif, la
pénalité de SPO ramenait la politique vers la version en retard et les updates oscillaient (run s3).

**Ablation prévue (sprint 3)** : PPO vs SPO, **même seed, même compute, même config**, comparés en Elo.

---

## 7. Rewards

On combine des rewards denses en début d'entraînement, puis on les réduit progressivement au profit des rewards « de match ».

| Reward | Rôle | Phase |
|---|---|---|
| But marqué / encaissé | objectif final | toujours |
| Toucher la balle | apprendre à jouer | début, puis ↓ |
| Vitesse joueur → balle | aller à la balle | début, puis ↓ |
| Vitesse balle → but adverse | pousser vers le but | toujours |
| Collecte / économie de boost | gestion des ressources | toujours, faible |
| Gagner le kickoff | crucial en 1v1 | sprint 2 |
| Pénalité de double commit | jeu d'équipe | 2v2/3v3, **faible** |
| Tir cadré (`shot`) : gain de vitesse de la balle dans l'axe du but adverse | viser le but | depuis s5 |
| Placement derrière la balle (`align_ball_goal`), en l'air (`in_air`), réserve de boost (`boost_keep`) | faibles | depuis s5 |
| Arrêt (`save`), boost gaspillé (`boost_waste`), placement défensif (`goalside`) | défense, boost | depuis s6 |

### Esprit d'équipe (team spirit)
```
r_final(i) = (1 - τ) * r_i + τ * moyenne(r de l'équipe) - moyenne(r de l'équipe adverse)
```
`τ` augmente progressivement (0 → ~0,3, puis plus). Le bot apprend ainsi que laisser son coéquipier prendre la balle peut rapporter autant que la prendre lui-même.

---

## 8. États de départ

| État | Poids | Sprint |
|---|---|---|
| Kickoff standard | élevé | 1 |
| Position aléatoire (voitures + balle) | moyen | 1 |
| Balle en l'air / au mur / en défense / en attaque | progressif | 2 |
| Scénarios de kickoff spécifiques | — | 2 |

**Exercices dédiés** (`drill_*`, depuis s6) : défense (tir adverse vers notre but), tir (balle devant la surface
adverse), aérienne (balle qui retombe de haut, plein de boost), mur, boost bas, récupération (voiture en l'air mal
orientée). Ce sont des **états de départ** : l'épisode continue en match normal, avec les mêmes rewards. Chaque
exercice a un critère de réussite mesuré (`drill/<nom>/success`) et un **dosage automatique** : ceux qui sont ratés
reviennent plus souvent, à part totale constante (50 % de matchs normaux dans s6).

---

## 9. Self-play et Elo

- **Pool d'adversaires** : on sauvegarde une version de la politique tous les N steps.
- Répartition des adversaires : environ **80 % politique courante, 20 % anciennes versions** (tirées selon leur Elo).
- **Elo** recalculé en continu sur les matchs du pool. C'est notre métrique principale de progression.
- Ancres externes (évaluation dans RLBot) : Psyonix Rookie → All-Star, puis **Necto**, puis **Nexto**.

---

## 10. Métriques suivies (W&B)

**Apprentissage** : reward moyenne, entropie, KL approx., fraction de ratios hors de [1-ε, 1+ε], value loss, steps/s (débit de simulation).

**Jeu** : buts pour/contre, taux de touches, vitesse moyenne, boost moyen, temps en l'air, **taux de kickoffs gagnés**.

**Mécaniques** (`mech/*`) : boost ramassé par minute, boost moyen, sauts par minute, temps en l'air, temps
supersonique, boost gaspillé. **Exercices** (`drill/*`) : taux de réussite par exercice.

**Jeu d'équipe (2v2/3v3)** :
- **taux de double commit** (deux coéquipiers à moins de X m de la balle en même temps)
- **collisions entre coéquipiers**
- **distance moyenne entre coéquipiers**
- **répartition des rôles** (part du temps en 1er / 2e / 3e homme)

Ces métriques d'équipe permettent de voir une régression même quand le bot gagne.

---

## 11. RocketSim : utilisé tel quel

**Verdict : opérationnel directement pour notre usage** (soccar 1v1/2v2/3v3). D'après son README, il est assez précis pour entraîner des bots jusqu'au niveau SSL. Il gère aussi les pinches, le contrôle aérien et les tirs sous différents angles.

- **Seul prérequis** : dumper les collision meshes de l'arène avec RLArenaCollisionDumper (le jeu doit être installé), puis les placer là où RocketSim les attend.
- **Limite connue** : de petites erreurs s'accumulent sur des simulations longues sans retour du jeu. Ce n'est pas un problème pour l'entraînement, car le bot réagit à chaque step. En revanche, une prédiction de balle à long horizon sera de moins en moins fiable : rester sur ≤ 3 s.
- **Garde-fou** : test de cohérence simulation / vrai jeu au sprint 1 (`docs/sim_vs_game.md`).
- **Pas de fork**, sauf si ce test révèle un vrai écart gênant (décision PO).

---

## 12. Export vers le vrai jeu

- Wrapper `rlgym-rlbot` : le bot RLBot reconstruit **exactement** les mêmes observations et le même action parser qu'à l'entraînement (même code importé, jamais dupliqué).
- Test : les observations calculées dans le vrai jeu doivent avoir la même forme et le même ordre de grandeur qu'en simulation.
