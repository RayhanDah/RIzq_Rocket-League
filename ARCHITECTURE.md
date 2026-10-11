# 🧠 Architecture technique

Ce document fixe la conception. Toute modification passe par le PO.

---

## 1. Vue d'ensemble

```
            ┌──────────── Collecte (CPU, rlgym-learn) ────────────┐
            │  N parties RocketSim en parallèle                    │
            │  format tiré au hasard : 1v1 / 2v2 / 3v3             │
            │  bleu = politique courante                           │
            │  orange = politique courante OU ancienne version     │
            └──────────────┬───────────────────────────────────────┘
                           │ trajectoires
                           ▼
            ┌──────────── Apprentissage (GPU si dispo) ───────────┐
            │  GAE → PPO ou SPO (flag) → acteur + critique         │
            └──────────────┬───────────────────────────────────────┘
                           │ checkpoints
                           ▼
            Pool d'adversaires + Elo  ──►  Évaluation RLBot (vrai jeu)
```

Le principe : **un seul modèle** pour les trois formats. Chaque voiture est contrôlée indépendamment par le même acteur partagé.

---

## 2. Formats de jeu (anti « bot 1v1 qui ne sait pas jouer en équipe »)

Dès le **sprint 1**, chaque partie tire son format au hasard :

| Format | Poids initial |
|---|---|
| 1v1 | 70 % |
| 2v2 | 20 % |
| 3v3 | 10 % |

Le 1v1 reste prioritaire, mais le bot ne « désapprend » jamais la présence d'un coéquipier. Les poids sont configurables dans `configs/`.

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
- **prédiction de trajectoire de la balle** via RocketSim (ex. à 0,5 s, 1 s, 2 s, 3 s) *(sprint 2)*
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
