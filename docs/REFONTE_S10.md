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

## 9. Session 3 sur TPU : palier 5 « finition » (04/11/2026)

Premiers matchs dans le vrai jeu (RLBot, 5,315 G en argmax, 120 FPS) : 3-1 et 5-3 contre Psyonix All-Star, 2-7 contre le
PO. Ce que le PO voit : il cherche la touche au point de dévier sa propre balle qui rentre, il tire souvent à côté, il
touche des balles en sautant du sol ou des murs mais sans boost. **Contrainte : garder le jeu agressif et rush de
l'agent, qui fait partie de son identité.**

`scripts/style_audit.py` (40 matchs de 5 min en RocketSim, 5,315 G en argmax contre 5,315 G échantillonné) confirme tout :

| Défaut | Mesure |
|---|---|
| Dévie sa propre balle | 50 % de ses touches sur une balle cadrée la font sortir du cadre (0,68 / min, pour 1,13 but / min) |
| Tire à côté | 35 % de frappes cadrées (1,85 cadrées, 3,50 hors cadre / min) |
| Pas d'aériennes | 0,10 vraie aérienne / min, 1,22 touche depuis un mur / min, 0 but aérien ; boost sur 10 % du temps en l'air |
| Cause cachée | moins de 10 de boost 71 % du temps (moyenne 16), 48 % du temps en l'air à flipper pour garder de la vitesse |

Pourquoi le signal produisait ça :

- `strong_touch` paie toute frappe forte, quelle que soit sa direction, et rien ne reprenait une balle cadrée perdue ;
- `shot` paie la direction générale du but : une frappe à côté rapporte presque autant qu'une frappe cadrée ;
- `aerial_touch` ne regarde pas le boost : 86 % des touches payées étaient des sauts depuis un mur ;
- presque aucun épisode ne propose une balle haute avec du boost en réserve, et l'entropie (3,18 nats ≈ 24 actions) rend
  impossible une aérienne, qui demande 1 à 2 s d'actions cohérentes.

### Ce qui change

| | Palier 4 | **Palier 5** |
|---|---|---|
| `shot_threat` (nouveau, potentiel : balle cadrée payée, déviation et arrêt repris) | 0 | **3,0** |
| `strong_touch` | 1,5 | 1,0 |
| `shot` hors cadre dans le camp adverse (`shot_off_target_factor`) | payé en entier | payé au quart |
| `aerial_touch` (boost exigé : 0,25 s en l'air, `aerial_min_air_boost_s`) | 2,0, sans boost | **4,0**, avec boost |
| `air_approach` (seulement en boostant, `air_approach_boost_only`) | 0,05 | 0,15 |
| `boost_gain` | 0,67 | 1,0 |
| `ent_coef` | 0,01 | 0,005 |
| Départs | kickoff 50 %, aléatoire 50 % | kickoff 40 %, aléatoire 30 %, aérien 20 %, tir 10 % |

**Non modifiés, pour protéger le rush** : `vel_player_to_ball`, `flip_speed`, `demo`, `bump`, `vel_ball_to_goal`, `goal`,
`boost_keep`. Dépenser du boost reste gratuit. Mesuré avec le palier 5 sur le jeu actuel de l'agent (rejeu, puis log de la
validation) : `vel_player_to_ball` 15 à 18 % du signal (17,9 % avant : poids inchangé, mais un terme de plus dans le
total), `shot_threat` 10 à 14 %.

`shot_threat` est un potentiel (`r = γΦ(s') − Φ(s)`) : il ne se farme pas, sa somme sur un épisode se télescope. Il
ne punit que les touches qui font perdre une balle cadrée, pas les touches en général : la pression reste rentable.

Exercices : `drill_aerial` et `drill_shot` existaient déjà (s7), leur niveau suit la réussite (`env.drill_curriculum`).
Plancher à 0,4 : l'aérien n'est réussi qu'à 3 % au niveau 0,3, et sans plancher le niveau redescendait vers des balles à
200 uu, sous le seuil des rewards aériens. Reprise du run : le palier 4 a maintenant un critère de sortie (écart s6 ≥ 150, +465 aujourd'hui), le passage
se fait dès les premières itérations, avec transition sur 100 M steps.

Débit : `selfplay.ladder_fit_every: 10`. L'échelle était recalculée à chaque itération, pour un surcoût de 0,8 s sur
4,9 s à 1 000 versions, en n². Métriques ajoutées au log : `spoils_per_min`, `off_target_per_min`,
`true_aerials_per_min`, `wall_aerials_per_min`, `mech/air_boost_frac`, `mech/ball_dist`, `mech/opp_half_frac`,
`mech/bumps_per_min`.

### Critères fixés avant la session

Mêmes conditions que le tableau ci-dessus, contre le même adversaire (5,315 G échantillonné) :

```bash
.venv\Scripts\python scripts/style_audit.py --a <nouvel agent> --b <5,315 G> --det-a --games 40
```

| Objectif | Avant | Visé |
|---|---|---|
| Déviations (% des touches sur balle cadrée) | 50 % | < 25 % |
| Précision des frappes | 35 % | > 45 % |
| Vraies aériennes / min | 0,10 | > 1 |
| Buts aériens | 0 | > 0 |
| Boost moyen | 16 | > 25 |

| Plancher rush (à ne pas franchir) | Avant | Minimum |
|---|---|---|
| Vitesse moyenne | 1 488 | 1 400 |
| Supersonique | 14,5 % | 12 % |
| Plus proche de la balle | 57 % | 52 % |
| Frappes / min | 16,6 | 13 |
| Contacts / min | 2,4 | 2,0 |

Un agent qui atteint les objectifs mais franchit un plancher n'est pas retenu : on cherche alors, dans le pool (une version
tous les 5 M steps), la meilleure version qui respecte les planchers. `best_actor.pt` choisit à l'Elo seul, il ne suffit
donc pas à trancher. Les aériennes sont l'objectif le moins sûr : c'est une compétence neuve pour l'agent, une seule
session peut ne pas suffire.

## 10. Session 4 sur TPU : palier 6 « maîtrise » (06/11/2026)

Bilan de la session 3 : `runs/s10_ppo_kaggle/RAPPORT.md`. Le PO a joué les versions dans le vrai jeu (4-3, 6-5 et 5-4 pour
lui, contre 7-2 face au 5,315 G) et garde **`v8100`** (8,10 G) dans le bot : la plus forte à son goût, même sous le
plancher de proximité. **La force passe désormais avant ce plancher.** Ce qu'il voit : mieux en aérienne, dans les duels
et depuis les murs, gère mieux son boost, dévie moins ses balles. À améliorer, sans urgence : les kickoffs, et les sauts
à répétition au lieu de ramasser du boost sur son chemin.

### Ce qui se fait de mieux

| Source | Ce qu'on en retient |
|---|---|
| Necto (champion RLBot 2022, base de Nexto, GC1), code public | γ 0,995 ; dépense de boost facturée seulement près du sol ; `flip_reset_w` = 10, autant qu'un but ; 70 % de départs depuis des replays, 8 % de kickoffs |
| Opti (Kaiyotech), code public | un modèle par compétence (kickoff, pinch, pinch au plafond, flip reset, double tap, flick, mur, recovery…) et un sélecteur, chacun avec ses états de départ |
| Guide RLGym-PPO | les mécaniques ne se découvrent pas seules ; boost en racine ; entropie vers 2 |
| Fork de GigaLearn | une récompense dédiée par mécanique (speedflip, aérienne rapide, air dribble, atterrissage sur boost) |
| RLBot Championship | ML : Necto 2022, Seer 2023, LeBot+ 2024, Slater 2025 (finale gagnée sur un tir au plafond) |

### Diagnostic

- **On paie encore les sauts** : `in_air` et `flip_speed` sont des aides du palier 1. L'agent passe 45 % du temps en l'air
  et flippe 18 fois par minute.
- **Rouler à sec rapporte** : `boost_gain` ne compte pas la dépense et paie plus à sec. Sous 10 de boost 63 % du temps ;
  l'exercice boost à sec (atteindre 50 en 6 s) ne réussit qu'à 10 %.
- **Horizon court** : γ 0,99 = 4,6 s de demi-vie, trop peu pour voir l'intérêt d'un boost ramassé.
- **L'exercice aérien ne sert pas** : niveau bloqué à son plancher, 3,7 % de réussite, pour 20 % des épisodes.
- **Adversaires étroits** : les 20 dernières versions couvrent 100 M steps, 17 minutes de run, et les duels ne sont pas
  transitifs (`v6800` bat `v7000` 24-6, 8,14 G bat `v6800` 20-8).
- **`train.total_steps` valait 10 G** (default.yaml) : la session se serait arrêtée après 5 h.

### Ce qui change

| | Palier 5 | **Palier 6** |
|---|---|---|
| `ppo.gamma` | 0,99 | **0,995** |
| `ent_coef` | 0,005 | 0,003 |
| `boost_spend` (nouveau : Δ√boost, atténué avec la hauteur, nul à partir de la barre) | 0 | **0,5** |
| `in_air` | 0,008 | 0,002 |
| `flip_speed` | 0,3 | 0,1 |
| `kickoff_rush` (vitesse vers la balle jusqu'à la 1re touche, déjà codé) | 0 | 0,05 |
| `flip_reset` (nouveau, +1 par reset, 1 s de délai ; le contact des roues ne paie pas `aerial_touch`) | 0 | **5,0** |
| Kickoffs / aléatoire | 40 % / 30 % | 30 % / 15 % |
| Situations variées (`aerial`, `attack_defense`) | 0 | 10 % |
| Exercices | aérien 20 %, tir 10 % | **vol 15 %, reset 5 %**, aérien, tir, mur, boost à sec et défense aérienne 5 % chacun |
| Adversaires | 20 dernières versions + s6 | + 5,315 G, `v6800`, `v7000`, 7,565 G, `v8100` en ancres, PFSP, 12,5 % des épisodes |

Inchangés : `goal`, `shot`, `shot_threat`, `vel_ball_to_goal`, `vel_player_to_ball`, `aerial_touch`, `strong_touch`.

**Exercice de vol** (`AirDrill`) : curriculum inversé, on part de la fin du geste. Niveau 0 : voiture déjà lancée vers la
balle ; niveau 1 : juste décollée, loin, mal orientée. **Exercice reset** (`ResetDrill`) : roues face à la balle, flip déjà
utilisé. Mesuré avec `v8100` tel quel dans RocketSim :

| Réussite | Niveau 0 | Niveau 0,5 | Niveau 1 |
|---|---|---|---|
| Vol (touche au-dessus de 300 uu en 3 s) | 94 % | 14 % | 4 % |
| Reset (un flip reset en 3 s) | 24 % | 10 % | 8 % |

RocketSim reproduit le flip reset : un contact des roues avec la balle rend le saut et le flip, et compte comme un retour
au sol (`on_ground` vrai à 800 uu). Contact du toit : rien. Une balle sans vitesse reste suspendue : ces exercices lui en donnent.

**Validation sur le portable** (reprise du checkpoint à 8,14 G, 5,6 M steps) : variance expliquée stable (0,87-0,88) malgré
le changement de γ, KL 0,001-0,0016, entropie 2,67 → 2,60. Parts du signal : `vel_ball_to_goal` 27 %, `goal` 17 %,
`shot_threat` 15 %, `vel_player_to_ball` 14 %, `shot` 8 %, `boost_spend` 1,5 %, `aerial_touch` 1,2 %, `kickoff_rush` 0,7 %,
`in_air` et `flip_speed` 0,4 %. Exercice de vol à 85 % : le niveau monte seul.

**Essayé et écarté** : moyenner les poids des 20 dernières versions. Le modèle moyenné bat `v8100` 35-8-17 en argmax contre
argmax, mais fait moins bien que lui contre `v8100` échantillonné (55,2 % des buts contre 57,2 %). Un duel argmax contre
argmax mesure un duel, pas une force.

### Critères fixés avant la session

Contre le même adversaire, `v8100` échantillonné, l'agent en argmax, 200 matchs :

```bash
.venv\Scripts\python scripts/style_audit.py --a <nouvel agent> --b runs/s10_ppo_kaggle_s3/pool/actor_0008100114172.pt --det-a --games 200 --config runs/s10_ppo_kaggle_s3/config.yaml
```

Avant de rapatrier, renommer `runs/s10_ppo_kaggle` en `runs/s10_ppo_kaggle_s3` (et son `.log`).

| Objectif | `v8100` | Visé |
|---|---|---|
| Part des buts | 57,2 % | > 65 % |
| Vraies aériennes / min | 0,69 | > 1 |
| Boost moyen | 18,3 | > 22 |
| Moins de 10 de boost | 62,5 % | < 50 % |
| Flips / min | 18,5 | < 15 |
| Déviations | 35,7 % | < 30 % |
| Précision | 44,2 % | > 47 % |
| Flip resets / min | 0,003 | > 0,05 |

À surveiller, sans seuil bloquant : proximité 52,8 %, vitesse 1 477, contacts 2,4 / min, kickoffs (balle chez l'adversaire
à 3 s, contre `v8100`). L'entropie ne doit pas descendre vers 1,5. Le dernier mot reste aux matchs du PO dans le vrai jeu.

Si les flips chutent sans que le boost monte, l'agent perd sa seule source de vitesse : c'est le risque de ce palier, à
regarder en premier. Et les resets resteront rares : un reset utile suppose un contrôle aérien que l'agent n'a pas encore.
La suite logique, d'après Opti : entraîner à part un modèle de kickoff, puis d'autres spécialistes, avec un sélecteur.

## 11. Session 5 : retour à `v8400`, failles fermées, garde-fou (08/11/2026)

### Ce qui s'est passé en session 4

Le palier 6 a marché jusqu'à 8,6 G : `v8400` bat `v8100` 59-1-0 en argmax et 41-6-13 en échantillonné, avec plus de
boost (25 contre 18) et moins de déviations (29 % contre 36 %). À 8,7 G, l'agent a trouvé une faille de `aerial_touch` :
les touches enchaînées sans retomber étaient payées ×1,5 puis ×2, et jongler sur place rapportait ~30 / s contre 10 pour
un but. `aerial_touch` est passé de 1 à 66 % du signal, l'agent a cessé de marquer (0,06 but / min), son Elo est tombé de
916 à 270, et rien n'a arrêté le run : 2,2 G steps perdus. Détails : `runs/s10_ppo_kaggle/RAPPORT.md`.

Le palier a rendu la faille accessible (vol boosté gratuit avec `boost_spend`, boost 21 → 52, exercice de vol, entropie
plus basse), mais elle existait depuis le palier 3. **Aucune boucle n'avait été chiffrée avant le lancement.**

### Revue de tous les termes

Chaque terme qui a un poids dans un palier a été revu pour la même question : une boucle qui ne fait pas avancer le jeu
peut-elle rapporter autant qu'un but ? Trois oui :

| Terme | Boucle | Correction | Une minute de boucle |
|---|---|---|---|
| `aerial_touch` | jongler sur place sans retomber | payé selon le gain de vitesse vers le but adverse (plein à 1 000 uu/s) ; 1re touche du vol au moins ¼ ; touches suivantes × 0,5 par rang ; le vol finit au vrai retour au sol | 883 → **0,99** |
| `flip_reset` | rebondir sur la balle sans sauter (5 / s) | payé seulement si le saut avait été utilisé ; 3 par vol au plus (1 + 0,5 + 0,25) | 250 → **5** |
| `air_approach` | aller-retour sous la balle en boostant ; planer en la retouchant | plus rien après la 1re touche du vol ; l'éloignement compte en négatif | 41 → **0,8** |

Les autres sont zéro-somme, potentiels, signés, ou bornés par le jeu (raison écrite pour chacun dans
`tests/test_exploits.py`). **Règle désormais testée** : un terme qui reçoit un poids sans figurer dans cette revue fait
échouer les tests.

### Garde-fou (`train.guard`, `guard.py`)

Trois signaux, chacun maintenu 60 itérations (~24 M steps) : `ladder/drop` > 120, un terme surveillé > 25 % du signal,
entropie < 1,5. Déclenché, il recharge le meilleur jalon (checkpoint complet gardé tous les 250 M, les 8 derniers),
écarte du pool les versions postérieures (`pool/rejected/`, rien n'est supprimé) et coupe le terme dont la part a triplé
depuis ce jalon. Deux retours au plus ; au troisième déclenchement, il recharge le jalon et arrête le run proprement.
Tout est noté dans `<run>/guard.json`. **Rejoué sur la session 4** : déclenché à 8,854 G (drop 144), retour au jalon de
8,5 G (Elo 905), `aerial_touch` désigné ; 2 G steps gardés au lieu d'être perdus.

### Départ de la session 5

- `train.rollback_to: pool/actor_0008400059217.pt` : à la reprise sur Kaggle (10,88 G), l'acteur de `v8400` remplace
  l'acteur effondré, une seule fois par run. Le pool ne garde pas le critique : celui de 10,88 G est recalé pendant 50
  itérations, acteur figé (lr 0). Les ~500 versions postérieures à 8,4 G quittent le pool et l'échelle Elo. Le compteur de
  steps continue (les nouvelles versions s'appellent `actor_00108…` et plus).
- `ent_coef` 0,003 → 0,005 (entropie 2,1 avant la faille, 1,13 après).
- `drill_aerial` retiré (0 % de réussite sur la session 4) ; ses 5 % vont au jeu libre.
- `v8400` ajoutée aux adversaires fixes : `vs_anchor/actor_0008400059217` mesure le progrès depuis le départ.

### Validation sur le portable (08/11)

Copie du vrai run de 10,88 G (pool partiel, comme après `fetch`), reprise avec le profil s10 :

- retour à `v8400` appliqué une fois, versions postérieures écartées, jalon de départ posé dès la fin du recalage ;
- recalage du critique : KL nul (acteur figé), variance expliquée 0,59 → 0,80 en 4 itérations, puis 0,85 ;
- jeu de `v8400` retrouvé : 2,2 à 3,0 buts / min (0,06 avant), 32 touches / min (100 avant), entropie 2,2, flips 16 / min,
  boost 28 ;
- parts du signal : `vel_ball_to_goal` 28 %, `goal` 18 %, `shot_threat` 14 %, `vel_player_to_ball` 14 %, `shot` 8 %,
  `aerial_touch` 1,4 %, `air_approach` 1,5 %, `flip_reset` 0,1 % ;
- **bug trouvé et corrigé** : l'échelle Elo gardait les versions effondrées dont le fichier n'était pas dans le pool
  local ; `ladder/drop` restait à 664 et le garde-fou se serait déclenché à tort. Après correction : 1 681 versions,
  dernière notée `v8400` (923), drop 17. Test ajouté (pool partiel).

### Critères de la session 5

Contre `v8400` échantillonné, l'agent en argmax, 200 matchs (`v8400` contre elle-même : à mesurer au retour) :
part des buts > 60 %, duels gagnés dans les deux modes, et pas de recul sur ses acquis (boost > 22, déviations < 30 %,
flips < 16 / min). À surveiller : vraies aériennes (0,73 / min), kickoffs (première touche perdue 3 fois sur 4 contre
`v8100`). Puis les matchs du PO dans le vrai jeu.

## 12. Session 6 : palier 7 « lucidité » (11/11/2026)

### Bilan de la session 5

10,88 → 13,54 G en 7,8 h. Le garde-fou s'est déclenché deux fois sur `ladder/drop` (12,66 G → jalon 11,75 ; 13,49 G →
jalon 12,75, `aerial_touch` coupé). Les deux dérives ont la même forme, plus lente que celle de la session 4 : buts
2,4 → 0,5 / min, épisodes 25 → 103 s, vraies aériennes 1,4 → 4,7 / min. `aerial_touch` (poids 4, hors zéro-somme)
y rapporte ~10 par minute, autant que les buts perdus, pour une part de 6 % seulement. **1,65 G steps et 4,8 h
écartés sur 2,66 G et 7,8 h (62 %).** `v13535` ne descend de `v8400` que par 1,0 G steps, dont 48 M sans `aerial_touch`.

`v13535` (A argmax contre `v8400` échantillonné, 200 matchs) : 184-10-6, 71,7 % des buts (`v8400` contre elle-même :
55,7 %) ; 63 % en échantillonné. Vrai jeu, un match chacun : All-Star 15-3, `v8100` 10-2, **Necto 1-10**. Détails :
`runs/s10_ppo_kaggle/RAPPORT.md`. C'est le bot actuel (`bot/model.pt`).

### Le défaut vu par le PO contre Necto

Il colle la balle et « commit » sur tout ; Necto en profite par des tirs forts de loin. En simulation contre `v8400` :
distance à la balle 1 091 contre 1 191 uu, 22 touches / min contre 19, vitesse inchangée (~1 500 uu/s). Le résultat du
match ne faisait que ~17 % du signal (but ±10) ; `vel_player_to_ball` (14 %) paie l'approche même quand l'adversaire
arrive avant, et encaisser ne coûtait pas plus que marquer ne rapporte.

### Ce qui change (inspiré de « goal / concede only »)

- Palier 7 `lucidite` : `goal` 10 → **20** (zéro-somme : encaisser −20) et `concede` **5** (hors zéro-somme) →
  marquer +20, encaisser **−25**. But et encaissement ≈ 33 % du signal au lieu de 17 %. Un seul changement de jeu.
- `maitrise` reçoit un critère toujours rempli : passage au palier 7 ~20 itérations après la reprise, poids mélangés
  sur 100 M steps.
- `aerial_touch` à 0 dans le palier (en plus de `guard.json`).
- Ancre `v13535` : `vs_anchor/actor_0013535043833` mesure l'effet du palier.
- Garde-fou : quota de retours compté par session (la session 5 en a fait 2 ; compté sur le run, le premier
  déclenchement aurait arrêté la session 6).
- Boucles : `concede` ne paie qu'à un but encaissé et n'est jamais positif ; `goal` est zéro-somme. Aucune boucle
  nouvelle (`tests/test_exploits.py`, qui garde aussi les tests d'`aerial_touch` au poids maximal des paliers).

### Critères de la session 6

Contre `v13535` échantillonné, l'agent en argmax, 200 matchs : part des buts > 60 %, sans perdre les buts / min
(> 1,2). Style : distance à la balle en hausse, arrêts en hausse, buts encaissés en baisse ; à surveiller, un bot trop
passif (buts / min, touches / min). Puis Necto (audit RocketSim si l'adaptateur est prêt, sinon vrai jeu).

Validation sur portable (copie du run dans `dist/val_s7`) : passage à « lucidité » au bout de 20 itérations, blend
+0,4 % par itération, `concede` payé, entropie stable, pas de retour du garde-fou. Elle a trouvé un plantage : le
message de passage de palier (« → ») plantait le run sous Windows quand la sortie est redirigée vers un fichier
(cp1252). `train.main` passe désormais la console en UTF-8 (`tests/test_train_console.py`).

## 13. Session 7 : retour à `v13600`, paliers 8-10 kickoff, démos, aériennes (14/11/2026)

### Ce qu'a donné la session 6 (Kaggle GPU, 13,537 → 13,762 G)
Contre sa lignée, la session 6 progresse : la meilleure version du classement, `v13720`, bat `v13535` 156-15-29. Dans le
vrai jeu contre Necto, elle perd 3-12, puis 4-9 une fois le bot corrigé (§ suivant). Le PO relève 3 buts vides manqués,
2 buts encaissés sur kickoff et 1 contre son camp.

Deux causes trouvées le 14/11 :
1. **Le bot voyait mal le vrai jeu.** `rlgym_compat` reconstruit faux trois valeurs que le réseau reçoit :
   `jump_time` (faux 44-49 % du temps : jamais remis à 0 au sol), `air_time_since_jump` (41-44 % : figé après un
   flip), `is_boosting` (15-33 % : vrai réservoir vide). Corrigé par une copie RocketSim recalée sur le jeu à chaque
   paquet (`env/shadow_sim.py`, branchée dans `bot/bot.py`) : < 0,3 % d'écart mesuré dans le vrai jeu
   (`scripts/real_flags_check.py`).
2. **« Lucidité » à plein poids a appris à ne plus disputer les kickoffs.** Audits contre Necto en simulation
   (100 matchs, nouveau bloc « kickoff » de `style_audit.py`) :

| Version | V-N-D | buts | 1er contact au kickoff | kickoffs convertis par Necto |
|---|---|---|---|---|
| v13535 | 24-15-61 | 46 % | 11 % | 6,8 % |
| v13600 (lucidité à 62 %) | 58-9-33 | 53 % | 10 % | 19 % |
| v13665 (lucidité à 100 %) | 6-5-89 | 33 % | 0 % | 27 % |
| v13720 | 1-1-98 | 26 % | 7 % | 26 % |

Marquer +20 et encaisser −25 rend tout duel à 50/50 perdant d'avance : l'agent laisse le ballon au kickoff. Contre
lui-même, cela ne coûte presque rien (v13535 ne convertit que 0,7 % de ces kickoffs), contre Necto, c'est fatal.
v13600 bat v13535 150-17-33, presque autant que v13720.

### Décisions du PO
- Retour à `v13600` (`train.rollback_to`, critique recalé 50 itérations) : 13,60 → 13,76 G écartés.
- Kickoff, démos et aériennes introduits lentement, un palier à la fois, sans boucle de reward.
- `concede` 5 → 2 au palier kickoff (marquer +20, encaisser −22).

### Paliers
| Palier | Changement | Transition | Passage au suivant |
|---|---|---|---|
| 7 lucidité | (inchangé) | — | 20 itérations après la reprise |
| 8 kickoff | `kickoff_win` 3 → 6, `kickoff_rush` 0,05 → 0,1, `concede` 5 → 2 | 100 M | 500 M steps, buts ≥ 2 / min, pas sous v13535, `vs_hist/kickoff_win_rate` ≥ 0,5 |
| 9 démo | `demo` 5,3 → 8 (zéro-somme) | 200 M | 500 M steps, buts ≥ 2 / min, pas sous v13535 |
| 10 aérien | `aerial_touch` 0 → 1, zéro-somme désormais | 300 M | dernier palier |

Pourquoi pas de boucle :
- `kickoff_win` : ±1 une fois par kickoff, antisymétrique ;
- `kickoff_rush` : 4 s au plus, soit 6 par kickoff au plus ;
- `demo` : zéro-somme, 8 = 40 % d'un but ;
- `aerial_touch` : zéro-somme (les deux dérives de la session 5 payaient les deux camps à la fois), au quart de son
  ancien poids, avec les bornes du 08/11.

### Garde-fou renforcé (`guard.py`)
- Plafond de part propre aux termes renforcés : 5 % (`kickoff_win`, `kickoff_rush`, `aerial_touch`), 8 % pour `demo`.
  La dérive de la session 5 n'a jamais dépassé ~6 %, invisible au seuil général de 25 %.
- Retour au jalon si les buts / min tombent sous 60 % du meilleur des 4 derniers jalons (session 5 : 2,4 → 0,5).
- Un terme fautif revient à son ancien poids (`fallback_weights` : démo 5,3, kickoff 3 et 0,05), pas à 0.
- `aerial_touch` remis en jeu une seule fois par run (`reenable_terms`, noté dans guard.json).
- Le coupable se cherche d'abord parmi les termes renforcés (part ×2 depuis le jalon).
- 3 retours par session au lieu de 2.

### Critères pour la session 7
- Contre Necto en simulation (100 matchs) : premier contact au kickoff > 30 %, kickoffs convertis par Necto < 10 %,
  part des buts > 53 % (v13600).
- Contre v13600 et v13535 : pas de recul (> 55 % des buts).
- Démos > 0,3 / min, vraies aériennes > 0,4 / min, sans hausse de la durée des épisodes ni chute des buts.
