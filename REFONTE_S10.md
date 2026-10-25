# Refonte s10 : ce qui ne va pas, et le plan pour devenir compétitif (23/10/2026)

## 1. Le constat, chiffré

**~550 M steps dépensés en 15 runs, et aucun run de zéro n'a dépassé 35 M steps.** Chaque run repartait du précédent
(s4 → s5 → s6 → s8 → s9 ; s7c → s7e → s7f) en changeant 5 à 8 choses à la fois. Le meilleur agent, s6, a 163 M steps.

**Défaut n°1 : le bot ne saute plus.** C'est le plus grave, et il n'était suivi par aucun critère :

| | sauts / min | flips / min | temps en l'air | vitesse moy. | supersonique |
|---|---|---|---|---|---|
| politique aléatoire (s7c, 0 M) | 32 | 29 | 83 % | 920 uu/s | 4 % |
| s7c à 15 M / 35 M | 12 / 7 | 5,6 / 2,2 | 33 % / 21 % | 980 / 1 090 | 1 % |
| s6 à 66 M → 163 M | 0,4 → **0,04** | — | 6 % | ~1 220 | 2 % |
| s9 à 185 M | **0,07** | **0,006** | 6 % | 1 230 | 2,5 % |

Le bot de la lignée s6 est un « pousseur de balle au sol » : il ne saute qu'une fois toutes les 15-25 minutes de jeu.
Il ne peut donc ni flipper pour aller vite, ni frapper fort, ni défendre une balle haute, ni faire d'aérienne. Les
exercices aériens (0 % de réussite) et muraux (5 %) étaient impossibles pour lui.

À l'inverse, la lignée s7 (rewards potentiels, course à la balle, `flip_speed`) sautait encore (4-7 sauts/min, 2-3
flips/min jusqu'à 172 M ; `flip_speed` avait fait passer les flips de 0,4 à 3,3/min en 30 M dans s7e) et perdait
pourtant 47 à 0 contre s6. Sauter ne suffit pas : c'est la recette entière qui était en cause.

Pourquoi : au départ, sauter au hasard fait perdre le contrôle, la politique apprend vite à ne plus sauter. Les bots de
référence réapprennent ensuite des sauts utiles parce que **la vitesse vers la balle reste payée** (et qu'un flip la
fait gagner) et que **les frappes fortes sont payées** (flip dans la balle). Chez nous :
- la vitesse vers la balle (`vel_player_to_ball`) décroissait jusqu'à **0 à 40 M steps** (`laptop_s2`), `face_ball` à 0
  à 20 M ;
- la touche payait **n'importe quel contact** (pousser la balle), pas la force de la frappe ;
- aucun reward « en l'air » avant s5, alors que la politique ne sautait déjà plus ;
- les démarrages à chaud ont ensuite figé ce comportement dans toutes les versions suivantes.

**Défaut n°2 : trop de choses à la fois.** ~20 termes de rewards (dont 4 potentiels s7), 6 exercices = 50 % des
épisodes dès s6, 3 formats, prédiction de balle, course à la balle, pads de boost… et des runs qui changent 5 à 8
réglages d'un coup. Aucun changement n'était attribuable, et les défauts de base restaient masqués.

**Défaut n°3 : on a soigné l'optimiseur au lieu du comportement.** lr de 1e-3 à 6e-3 piloté par le KL, SPO, NaP,
remise des normes : des semaines sur des symptômes (KL, normes), alors que la politique avait un défaut de jeu.

**Défaut n°4 : pas de repère absolu.** L'`elo` est mis à jour contre des adversaires dont l'Elo date de leur
sauvegarde : il oscille de ±150 d'une itération à l'autre (s9 : 1 417 → 1 673 en 10 itérations). Impossible de voir
« ça monte » ou « ça baisse » avant des dizaines de millions de steps.

## 2. Réponses aux cinq questions

| Question | Réponse | Preuve / raison |
|---|---|---|
| **Le SPO pose-t-il problème ?** | **Non, ce n'est pas la cause.** On passe quand même au PPO pour la base. | Au KL de nos runs (0,002-0,005), le ratio s'écarte de ~4 % (`ratio_dev` 0,04, `clip_frac` 0,03) : le clip PPO n'agit presque jamais et la pénalité SPO réduit le gradient de ~20 % au plus. Les deux font quasiment la même chose. PPO = la référence de tous les bots RLGym, avec des hyperparamètres connus : une inconnue de moins. SPO reste dans le code (`ppo.policy_loss`) pour l'ablation du projet, **à faire sur une base saine** (A/B à config identique). |
| **Les outils sont-ils le problème ?** | **Pas pour le défaut actuel, mais c'est notre plus gros levier de débit.** | Mesuré le 23/10 sur le portable : **RocketSim seul = 334 000 ticks/s sur 1 cœur ≈ 83 000 steps d'agent/s en 1v1**, contre ~900 steps/s par worker pour notre collecte complète. La simulation pèse ~1 % du temps ; le reste est du Python (profil d'une arène : observations 31 %, conversion des états RLGym 22 %, rewards 19 %, simulation 11 %, hors réseau et échanges entre processus). Optimiser ce Python : ×1,5-2 au mieux. GigaLearn (C++, libtorch) annonce ~10× rlgym-ppo. Autre piste, sans rien réécrire : la **session TPU de Kaggle** (VM v5e-8 : 224 vCPU chez Google Cloud, non vérifié sur Kaggle) ferait tourner notre code sur ~50× plus de cœurs. Mais le défaut « ne saute pas » serait le même avec 10× plus de steps : il fallait d'abord la bonne recette. |
| **L'architecture est-elle compétitive ?** | **Le principe oui, notre version non.** On passe au MLP pour le 1v1. | L'attention est celle de Necto/Nexto (bons bots), mais la nôtre est petite (d_model 128, 2 couches), et en 1v1 le Transformer ne voit que 2 jetons (soi + adversaire) : aucun gain, plus de calcul, et LayerNorm + Adam sans weight decay fait dériver le lr effectif (bilan du 22/10). Les bots 1v1 de référence (rlgym-ppo, GigaLearn) sont des MLP. Le match se joue en 1v1 : on y met 100 % des données. L'attention reste disponible pour le 2v2/3v3 plus tard. |
| **8 ticks, est-ce compétitif ?** | **Oui, on garde.** | Standard de Necto, Nexto et de l'exemple GigaLearn (`tickSkip = 8`). 15 décisions/s, horizon plus court, apprentissage plus rapide. |
| **Stratégies trop agressives ?** | **Oui.** On repart sur une base simple, et on avance par paliers mesurés. | Exercices à 50 % des épisodes avant que le bot sache sauter ; exercice aérien impossible (0 %) ; rewards changés à chaque run. |

## 3. Le nouvel agent : s10 (`configs/s10.yaml`)

De zéro, sans démarrage à chaud. L'infra (RocketSim, workers, collecte, Kaggle) est gardée : elle n'est pas en cause
et elle est testée.

| | s6 / s9 | **s10** |
|---|---|---|
| Algo | SPO, lr piloté par le KL (jusqu'à 6e-3) | **PPO clippé, lr fixe 2e-4** (guide rlgym-ppo), 1e-4 au dernier palier |
| Lot | 50 k → 300 k, 4 epochs | **100 k / 50 k, 2 epochs**, puis 200 k à 50 M et 400 k à 150 M |
| Réseau | attention d_model 128 + LayerNorm | **MLP** acteur 512×3, critique 1024-1024-512, ReLU, sans LayerNorm |
| Formats | 84 / 12 / 4 % | **1v1 uniquement** |
| Départs | kickoff, aléatoire + 6 exercices (50 %) | **kickoff 50 % / aléatoire 50 %**, aucun exercice |
| Rewards | ~15 termes, aides décroissantes en steps | **recette GigaLearn** (exemple officiel, ÷ 15) : vitesse vers la balle (signée, jamais retirée), frappe forte (0 sous 20 km/h, plein à 100 km/h), en l'air, face à la balle, boost ramassé / gardé, balle → but, but, démo et bump en zéro-somme. **Plus un garde-fou** : `flip_speed` (flip qui fait gagner de la vitesse vers la balle), seuil 300 uu/s (cf. §5) |
| Progression | décroissances fixées en steps | **paliers de compétence** (`curriculum.py`) |
| Mesure | Elo glissant | **échelle Elo absolue** (`selfplay/ladder.py`) + s6 comme repère fixe |

### Paliers (le bot avance à son rythme)

| Palier | Ce qui change | Pour en sortir (moyennes glissantes, 20 itérations de suite) |
|---|---|---|
| 1. **bases** | recette complète + petite prime de touche | ≥ 15 touches/min, ≥ 3 flips/min, vitesse moy. ≥ 1 250 uu/s, après 30 M steps au moins |
| 2. **jeu** | retire les aides d'amorce (touche simple, face à la balle), moins de poursuite pure, plus de balle → but | dernière version **≥ s6 + 50 Elo** sur l'échelle, après 40 M steps au moins |
| 3. **avancé** | aériennes (`aerial_touch`), kickoffs gagnés (`kickoff_win`), lr 1e-4 | — |

Les poids passent d'un palier à l'autre en douceur (10 M steps). Si un critère n'est jamais atteint, le bot reste au
palier : c'est voulu (pas d'exercice avancé tant que la base n'est pas là). Les exercices (fast kickoff, aériennes)
reviendront au palier 3 ou après, une fois validés sur un bot qui saute.

### L'échelle Elo absolue (`ladder/elo`)

- La version de départ (réseau aléatoire) est entrée dans le pool à 0 step : c'est **l'Elo 0**.
- Chaque version du pool (toutes les 5 M steps) est un joueur **figé** ; la politique en cours est « current » et ses
  parties lui sont réattribuées quand elle entre dans le pool.
- Classement Bradley-Terry par maximum de vraisemblance sur toutes les parties depuis le début (nul = ½), avec un nul
  fictif par paire (pas de note infinie). Aucune partie en plus : ce sont les parties contre le pool (20 %).
- `ladder/elo` monte **par marches** (une par version figée) ; `ladder/anchor/s6_spo_kaggle` situe s6 sur la même
  échelle ; `ladder/gap/s6_spo_kaggle` > 0 = s10 au-dessus de s6.

## 4. Le plan

1. **Validation sur le portable** (~20 M steps, ~1 h 30) : débit, KL au lr de 2e-4, et surtout les courbes de base
   comparées à s7c au même step (touches, sauts/flips qui ne s'effondrent pas, vitesse, Elo qui monte).
2. **Kaggle 12 h** : `cloud/kaggle_sync.py launch --run-name s10_ppo_kaggle --profiles s10 --anchors-from s10`
   (de zéro, s6 figé comme repère). Reprise automatique aux sessions suivantes (même commande).
3. **Mesurer le débit avant de changer d'outil** : session TPU en mode `bench` (notre code, beaucoup de cœurs, ~15 min
   de quota TPU) et essai de GigaLearn sur le portable et sur Kaggle. On ne réécrit en C++ que si le gain mesuré est
   ≥ 5×. Point favorable : les rewards de s10 sont déjà ceux de l'exemple GigaLearn, le portage de la recette serait
   direct. Points à trancher (PO) : le projet doit-il garder son propre PPO/SPO en Python ? Licence de GigaLearn non
   précisée (dépôt d'origine « GigaLearnCPP-Leak »), pas de guide d'installation, bot RLBot à refaire en C++.
4. **Quand s10 bat s6** (palier 3) : aériennes, kickoffs, puis retour des exercices un par un, chacun avec un critère.
5. **Ablation SPO vs PPO** pour le projet, sur la base s10 (même config, seul `ppo.policy_loss` change).
6. **2v2 / 3v3** si le format du match l'exige : attention ou MLP, à décider sur données.

### Ce qu'on peut attendre de 12 h sur Kaggle (~250 M steps)

Honnêtement : un bot qui fonce, flippe, frappe fort et marque, au palier 2, avec une échelle Elo qui monte ; battre
s6 est plausible mais pas garanti dans la 1re session. Pas d'aériennes construites ni de fast kickoff à ce stade :
il faut plusieurs sessions (ou un débit bien plus élevé) pour ça.

## 5. Validation sur le portable (23/10, nuit)

Quatre runs de zéro, 1v1 (sauf le 1er), même config sauf ce qui est indiqué. Détails : `runs/s10_ppo_laptop/RAPPORT.md`.

| Run | Différence | Leçon |
|---|---|---|
| `s10_formatbug_laptop` (0-9,9 M) | `format_weights: {1: 1.0}` | **les profils fusionnent les dictionnaires** : 2v2 et 3v3 de `default.yaml` restaient (77 / 15 / 8 %). Corrigé + test de la config réelle après fusion avec chaque profil d'infra |
| `s10_noflipspeed_laptop` (0-15,9 M) | recette GigaLearn seule | apprend 2× plus vite que s7c, mais flips 25 → 2,0/min en 15 M |
| `s10_flipspeed1000_laptop` (0-15,6 M) | + `flip_speed` 0,5 au seuil de s7e (1 000 uu/s) | **le terme ne paie rien** (0,0001 / épisode) : flips 1,4/min à 15 M. L'écart avec le run précédent à 5 M (19 contre 11 flips/min) est de la **variance entre runs**, pas l'effet du terme |
| `s10_ppo_laptop` | seuil de `flip_speed` à 300 uu/s | ci-dessous |

Pourquoi les flips s'effondrent : un flip demande 3 décisions de suite (saut, relâcher, saut + direction). Au début, la
politique les enchaîne par hasard (25 flips/min) ; dès qu'elle apprend à ne plus sauter au hasard (sauter fait rater
la balle), la séquence n'est plus jamais tirée, et le gain de vitesse d'un flip, payé seulement en différé par
`vel_player_to_ball`, ne suffit pas à la faire revenir. Il faut payer le flip utile **tout de suite** et **tant que
la politique flippe encore**.

### Mesures (moyennes de 7 itérations autour du step indiqué)

| Step | Run | touches/min | flips/min | sauts/min | vitesse uu/s | entropie | KL | steps/s |
|---|---|---|---|---|---|---|---|---|
| 5 M | **s10 (flip_speed, seuil 300)** | 3,6 | 24,6 | 28,4 | 920 | 3,85 | 0,0037 | 6 610 |
| 5 M | flip_speed seuil 1 000 | 3,7 | 18,8 | 25,1 | 940 | 3,85 | 0,0030 | 8 120 |
| 5 M | GigaLearn seul | 4,5 | 11,4 | 21,0 | 935 | 3,77 | 0,0049 | 8 144 |
| 5 M | s7c (ancienne recette) | 2,6 | 23,9 | 28,7 | 938 | 4,25 | 0,0040 | 4 039 |
| 10 M | **s10 (flip_speed, seuil 300)** | 14,0 | 9,4 | 13,4 | 1 019 | 3,53 | 0,0056 | 6 878 |
| 10 M | flip_speed seuil 1 000 | 12,5 | 3,9 | 9,7 | 1 061 | 3,48 | 0,0048 | 8 087 |
| 10 M | GigaLearn seul | 14,1 | 3,4 | 6,2 | 1 077 | 3,55 | 0,0047 | 7 503 |
| 10 M | s7c (ancienne recette) | 7,2 | 12,8 | 21,3 | 926 | 4,13 | 0,0033 | 3 152 |
| 15 M | **s10 (flip_speed, seuil 300)** | 15,4 | 7,2 | 8,7 | 1 121 | 3,45 | 0,0023 | 6 638 |
| 15 M | flip_speed seuil 1 000 | 17,6 | 1,3 | 2,6 | 1 168 | 3,48 | 0,0054 | 6 387 |
| 15 M | GigaLearn seul | 18,4 | 2,0 | 3,2 | 1 108 | 3,48 | 0,0034 | 6 890 |
| 15 M | s7c (ancienne recette) | 14,4 | 5,6 | 12,4 | 985 | 4,01 | 0,0028 | 3 685 |

Ce qu'on en tire :
- **la nouvelle recette apprend ~2× plus vite** que l'ancienne : à 10 M, 12-14 touches/min contre 7 pour s7c ;
  vitesse plus haute ; **débit ×2** (MLP : 6 600-8 100 steps/s contre 3 200-4 000) ;
- **lr fixe de 2e-4 suffisant** : KL 0,002-0,006 par itération (la zone des anciens runs à lr 5× plus fort), sans
  arrêt anticipé ni garde-fou ;
- **flips** : le seuil 300 en garde 4 à 6× plus à 15 M, et ils restent **en plateau à ~7/min de 10 à 18 M** (7,1 / 6,8 / 7,5 / 6,6 par tranche de 2,5 M ; run arrêté à 18,3 M). Prix :
  ~15 % de touches en moins à 15 M. Les trois variantes sont dans le même ordre à 5, 10 et 15 M. Mais un seul run par
  variante : à 5 M, deux configs presque identiques différaient déjà de 19 contre 11 flips/min ;
- l'échelle Elo bouge peu avant ~15 M : 75 % des parties contre le pool sont nulles (personne ne marque avant le délai
  de 10 s sans touche). Elle s'écarte quand le bot marque régulièrement ; pour la session Kaggle, c'est elle qu'il
  faut regarder à partir de ~30 M.

## 6. Lancer la session de 12 h sur Kaggle

Depuis `rl-agent` (PowerShell), avec le quota GPU disponible :

```
.venv\Scripts\python cloud/kaggle_sync.py launch --run-name s10_ppo_kaggle --profiles s10 --anchors-from s10
```

- de zéro, s6 figé comme repère (dataset `rl-agent-init` = ancres seules, repéré par `init.json`) ; arrêt propre après
  11 h (`--hours auto`) ; autoréglage du débit au début (~5 min) ;
- rapatrier : `.venv\Scripts\python cloud/kaggle_sync.py fetch --run-name s10_ppo_kaggle --wait` ;
- session suivante : **la même commande** (reprise automatique de la version précédente) ;
- sans quota GPU : ajouter `--accelerator none` (CPU seul, plus lent) ; la session TPU (`--accelerator tpu`) n'a
  jamais été mesurée : d'abord `--mode bench`.

**À surveiller** (ordre de priorité), dans `metrics.jsonl` ou le journal en direct :

| Métrique | Attendu | Si ce n'est pas le cas |
|---|---|---|
| `mech/flips_per_min` | ≥ 4 et stable après 20 M, puis en hausse | < 1 à 50 M : syndrome s6, arrêter et revoir `flip_speed` / `in_air` |
| `touches_per_min` | ≥ 15 vers 15-20 M, ~20 ensuite | |
| `ladder/elo` | monte par marches à partir de ~30 M | plat ou en baisse sur 50 M : regarder `learn/kl_final`, l'entropie |
| `ladder/gap/s6_spo_kaggle` | négatif au début, vers 0 puis positif | c'est le critère du passage au palier 3 (+50) |
| `curriculum/stage` | 0 → 1 quand les bases sont là (≥ 30 M) | rester à 0 n'est pas une panne : un critère n'est pas atteint (`curriculum/ema/*`) |
| `learn/kl_final` | 0,002-0,01 | |

## 7. Résultat de la 1re session Kaggle (TPU v5e-8, 23-24/10/2026)

Rapatrié le 25/10 (le rapatriement automatique s'était arrêté avec le PC). Copie complète : `runs/s10_ppo_kaggle`
(sinon `dist/kaggle/output/s10_ppo_kaggle`).

**Débit** : la VM TPU a **224 vCPU** ; l'autoréglage a choisi 168 workers × 8 arènes en collecte locale et
l'apprentissage sur TPU (0,4 s par lot de 100 k, contre 4 s sur CPU). **~100 000 steps/s pendant 8 h = 2,91 milliards
de steps**, 15× le portable, 17× Kaggle GPU. Sans réécrire une ligne : la file d'attente est le seul prix.

| Step | touches/min | flips/min | sauts/min | touches aériennes/min | vitesse | supersonique | `ladder/elo` | écart avec s6 | victoires / défaites contre s6 | palier |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 M | 18,4 | 6,1 | 7,4 | 0,1 | 1 180 | 3 % | 33 | −439 | 8 % / 84 % | bases |
| 100 M | 27,1 | **1,9** | 2,3 | 0,2 | 1 130 | 3 % | 128 | −269 | 4 % / 80 % | bases |
| 300 M | 45,5 | 4,4 | 4,6 | 0,3 | 1 210 | 4 % | 214 | −169 | 26 % / 68 % | bases |
| 500 M | 47,7 | 5,2 | 5,1 | 0,2 | 1 250 | 5 % | 287 | −92 | 37 % / 55 % | jeu (453 M) |
| 1 000 M | 29,8 | 15,2 | 15,9 | 2,4 | 1 400 | 10 % | 499 | +121 | 55 % / 42 % | avancé (775 M) |
| 2 000 M | 30,2 | 16,9 | 18,4 | 4,8 | 1 500 | 14 % | 612 | +234 | 73 % / 23 % | avancé |
| 2 910 M | 30,6 | **17,4** | 19,9 | **6,8** | 1 550 | 16 % | 608 | +230 | 77 % / 20 % | avancé |

**Match de contrôle** (`scripts/evaluate.py`, 40 matchs 1v1 de 5 min, version finale contre s6) : **38 victoires,
1 défaite, 1 nul, buts 142 à 25**. s10 est de loin notre meilleur agent.

Ce qu'on en tire :
- **la recette marche et le curriculum a fait son travail seul** : palier 2 à 453 M (bases acquises), palier 3 à
  775 M (s6 dépassé de 50 Elo), puis aériennes et kickoffs ;
- **les flips ont encore plongé (1,9/min à 100 M) puis sont remontés** quand le jeu l'a demandé (17/min à la fin) ;
  le garde-fou `flip_speed` a tenu jusque-là. Touches aériennes 0,1 → 6,8/min, vitesse 1 180 → 1 550 uu/s ;
- l'échelle Elo est montée de façon continue jusqu'à ~1,9 G steps (version la meilleure : 638 à 1 915 M), puis
  **plateau depuis ~2 G** (591 → 614 → 608). Kickoffs gagnés ~50 % (pas de progrès), démos quasi absentes.

**Suite proposée** (à décider) : session suivante = reprise du même run (même commande, `--accelerator tpu`), avec
au choix un lot plus gros (800 k-1 M, le gradient reste bruité), un lr qui baisse encore, un exercice de kickoff, et
des adversaires plus variés (PFSP) pour sortir du plateau.

## 8. Session 2 sur TPU : palier 4 « précision » (25/10/2026)

Ce que le PO voit en match (RLViser) : il vise mal le but, kickoffs moyens, touches en l'air seulement en sautant, il
cherche la touche plus que le but. Les métriques de fin de palier 3 (> 2,5 G steps) le confirment :

| Part du signal d'apprentissage | |
|---|---|
| `vel_player_to_ball` (aller vite à la balle) | **44,6 %** |
| `strong_touch` (frapper fort, n'importe où) | **17,4 %** |
| `vel_ball_to_goal` (balle vers le but adverse) | 16,7 % |
| `goal` | 9,0 % |
| `aerial_touch` / `kickoff_win` | 1,4 % / **0,1 %** |

30 touches/min, dont **6 % de tirs** (45 % des tirs marquent). `aerial_touch` payait dès 220 uu de hauteur : de simples
touches en sautant. Correction **sans toucher à la structure** : un palier de plus dans le curriculum, mêmes termes
(tous déjà testés), autre dosage, transition progressive sur 100 M steps :

| Poids | palier 3 | **palier 4** |
|---|---|---|
| `vel_player_to_ball` | 0,15 | 0,06 |
| `strong_touch` | 4,0 | 1,5 |
| `shot` (gain de vitesse de la balle **dans l'axe du but adverse**, négatif vers notre but) | 0 | 5,0 |
| `vel_ball_to_goal` | 0,2 | 0,35 |
| `kickoff_win` | 1,0 | 3,0 |
| `aerial_touch` (au-dessus de **500 uu** au lieu de 220) | 1,0 | 2,0 |
| `air_approach` (en l'air, balle haute : vitesse vers la balle) | 0 | 0,05 |

Passage au palier 4 quand l'écart avec s6 dépasse +150 Elo (+230 à la fin de la session 1 : passage dès la reprise).

**Garde-fou contre le surentraînement** : à chaque itération, la version du pool la mieux notée sur l'échelle Elo est
copiée dans `runs/s10_ppo_kaggle/best_actor.pt` (jouable par `evaluate.py`, `watch.py` et le bot). Nouvelles
métriques `ladder/best_elo`, `ladder/best_step` et `ladder/drop` (meilleure − dernière : > 50 de façon durable =
régression). Quoi qu'il arrive, on garde la meilleure version, pas la dernière.

À regarder au retour : `shots_per_min` / `touches_per_min` (6 % au départ), `goals_per_min`,
`vs_hist/kickoff_win_rate` (~50 %), `aerial_touches_per_min` au-dessus de 500 uu, `ladder/drop`.
