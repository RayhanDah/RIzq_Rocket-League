# Run s7 : refonte des rewards et de la stratégie d'entraînement (19/10/2026)

Profil : `configs/s7.yaml` (autonome, **départ de zéro**). Code : `env/predict.py` (nouveau), `env/rewards.py`
(section « rewards s7 »), `env/obs.py` (`race_features`), `env/mutators.py` (exercices à niveaux),
`metrics/episode.py`, `train.py` (curriculum, lot progressif). Tests : `tests/test_rewards_s7.py` (21 tests).

```
Portable : .venv-xpu\Scripts\python -m rlagent.train --profile laptop --profile laptop_xpu --profile s7 --run-name s7_spo_laptop
Kaggle   : --profile kaggle --profile s7 --run-name s7_spo_kaggle
```

---

## 1. Diagnostic de s6 (163 M steps)

| Constat (métriques) | Cause |
|---|---|
| **Plateau depuis ~20 M steps** : touches ~18/min, entropie ~3,5 de 20 M à 163 M | Après 40 M, les rewards de base (`vel_player_to_ball`, `face_ball`) sont à 0 ; le signal restant est dominé par le but (rare, bruité) |
| **lr de l'acteur bloqué au plafond (1e-3) et KL 6× sous la cible** | Gradient trop bruité sur des lots de 50k : la politique ne bouge presque plus |
| **Saut éteint** : 0,05 saut/min (0,3 au début de s6), 6 % du temps en l'air | s4 n'avait aucun reward lié au saut : sauter faisait perdre la balle → le bot a appris à ne jamais sauter. P(saut) ≈ 5e-5/step : 2 à 3 sauts par lot, plus aucun gradient ne peut le réapprendre |
| **Exercice aérien : 0 % de réussite** | Balle à 600-1800 uu pour un bot qui ne saute pas : aucun succès, donc rien à apprendre |
| **Boost moyen 12, supersonique 2 % du temps, exercice « peu de boost » 6 %** | Le boost n'avait de valeur qu'au ramassage (0,3 × √gain) |
| **Exercice défense : 35-45 % de réussite** (plus d'un tir sur deux encaissé) | Rien ne punissait un tir cadré avant le but lui-même |

## 2. Principe des nouveaux rewards

1. **Potentiels (Ng, Harada & Russell 1999)** pour tout ce qui est placement ou valeur : `r = γ·Φ(s') − Φ(s)`.
   Somme sur un épisode = Φ(fin) − Φ(début) : **impossible à farmer**, la stratégie optimale reste celle qui gagne
   les matchs, mais le signal est immédiat. Bonus : le critique apprend Φ, donc ces termes n'ajoutent presque pas de
   bruit aux avantages. (Testé : `test_potential_rewards_telescope`.)
2. **Événements de compétence** (aérienne, kickoff, but encaissé) : ce qu'on veut voir, payé quand ça arrive.
3. **Aides de démarrage qui décroissent** (toucher, en l'air, face à la balle, ramassage).
4. Une **prédiction physique partagée** (`env/predict.py`, 13 µs/step en 1v1) : trajectoire de balle avec rebonds,
   **tir cadré** (la balle entre-t-elle dans le cadre, dans combien de temps), **ETA** de chaque voiture (qui
   touchera la balle en premier, où).

## 3. Les rewards s7

| Terme | Poids | Type | Ce qu'il apprend | Faiblesse visée |
|---|---|---|---|---|
| `goal` | 10 | match, zéro-somme | marquer | attaque |
| `concede` | 3 | match, **non** compensé | encaisser coûte 13, marquer rapporte 10 | défense |
| `threat` | 3 | potentiel | Φ = urgence d'un tir cadré vers le but adverse − vers le nôtre. Tir cadré : récompense **immédiate** ; arrêt : récompense immédiate pour le défenseur, reprise au tireur ; balle qui file vers notre but : pénalité qui grandit chaque step | défense, attaque |
| `ball_goal` | 2 | potentiel | balle près du but adverse (remplace `vel_ball_to_goal`) | attaque |
| `positioning` | 1 | potentiel | course gagnée (mon ETA < ETA adverse) → aller à la balle ; course perdue → se placer entre la balle et notre but. En 2v2/3v3, seul le 1er homme attaque | **anticiper l'adversaire**, défense, vitesse |
| `boost_potential` | 0,6 | potentiel | Φ = √boost : ramasser rapporte, gaspiller coûte, dépenser n'est rentable que si ça sert (remplace `boost_keep`) | boost |
| `aerial_touch` | 1,5 | événement | touche en l'air : 0,75 au ras du sol → 3 au plafond ; ×1,5 puis ×2 en double touche / air dribble ; 1 touche payée / 0,5 s (anti-farming) ; pas au mur | aérien, saut |
| `kickoff_win` | 1,5 | événement | balle dans le camp adverse 3 s après l'engagement | kickoff |
| `touch` | 0,5 → 0,1 (80 M) | aide | toucher la balle | bases |
| `air_approach` | 0,03 → 0,009 (150 M) | aide | en l'air, voler vers une balle haute (1er homme) | aérien |
| `kickoff_rush` | 0,03 → 0,009 (100 M) | aide | foncer au kickoff (1er homme) : boost + flip | kickoff, vitesse |
| `in_air` | 0,002 → 0 (30 M) | aide | ne pas « oublier » de sauter | saut |
| `face_ball` | 0,01 → 0 (20 M) | aide | s'orienter | bases |
| `boost_gain` | 0,15 → 0 (60 M) | aide | ramasser | boost |
| `boost_waste`, `double_commit` | 0,003 / 0,05 | hygiène | inchangés | |

Désactivés (remplacés) : `vel_player_to_ball`, `vel_ball_to_goal`, `shot`, `save`, `align_ball_goal`, `goalside`,
`boost_keep`, `touch_strength`. Le code reste là (anciens profils inchangés).

**Bilan pour un but** : le tireur reçoit ~10 + 3 (menace) + ~2 (balle au fond), l'équipe qui encaisse
~−10 − 3 (concede) − 3 (menace) − 2. **Encaisser coûte ~1,4× ce que marquer rapporte** : c'est le levier
« impossible de marquer contre lui », sans rendre le bot passif (marquer reste très rentable).

## 4. Observations : « qui touche en premier ? » (+10 entrées)

Ajoutées à la fin du bloc « soi » (`env.race_features`) : mon ETA, meilleure ETA d'un coéquipier, meilleure ETA
adverse, mon avance sur l'adversaire et sur le coéquipier, point d'interception (x, y, z), urgence d'un tir cadré vers
notre but et vers le but adverse. Symétrie bleu/orange testée. Les anciens agents (pool, RLViser, évaluations) jouent
toujours dans cet env : `fit_obs` retire les colonnes qu'ils ne connaissent pas.

## 5. Curriculum : du saut simple à l'aérienne au plafond

Les exercices `aerial`, `aerial_defense` (nouveau : lob qui retombe dans notre but), `defense` et `shot` ont un
**niveau 0 → 1** piloté par le processus principal sur la réussite de toutes les arènes : +0,01 par itération si
réussite > 65 %, −0,01 si < 35 % (toujours dans la zone de progrès). Sauvegardé dans les checkpoints.

| Exercice | Niveau 0 | Niveau 1 | Réussite |
|---|---|---|---|
| `aerial` | balle à 250-450 uu, tout près, qui monte puis retombe | 1150-1800 uu, jusqu'à 1400 uu de distance | touche au-dessus de 200 → 700 uu en 4 s |
| `aerial_defense` | lob à 350-700 uu, voiture près du but | lob à 1200-1800 uu, voiture loin et décalée | pas de but en 6 s |
| `defense` | tirs à 900-1700 uu/s, surtout au sol | 1800-3200 uu/s, souvent en hauteur | pas de but en 6 s |
| `shot` | balle près du but, gardien 15 % | plus loin, gardien 75 % | but en 6 s |

Répartition des départs : kickoff 30 %, aléatoire 15 %, exercices 55 % (aérien 14, défense aérienne 8, défense 10,
tir 8, mur / peu de boost / récupération 5 chacun). Chaque exercice se poursuit en match normal.

## 6. Stratégie d'entraînement

| Réglage | s6 | s7 | Pourquoi |
|---|---|---|---|
| Départ | chaud depuis s5 | **de zéro** | un réseau neuf saute 34 fois/min (mesuré) ; s6 0,05. Critique à réapprendre de toute façon (rewards, γ, entrées). s4 avait atteint le niveau actuel en ~20 M steps (~2 h) |
| γ | 0,99 (demi-vie 4,6 s) | **0,995** (9,2 s) | défense, gestion du boost, constructions d'actions |
| Lot | 50k fixe | **50k → 100k (25 M) → 200k (80 M)** | gradient moins bruité quand le bot progresse (lr adaptatif inchangé) |
| Formats, lr adaptatif, 4 epochs, SPO | — | inchangés | |

Coût : collecte −6 % (10 049 contre 10 707 steps/s, 10 workers × 12 arènes).

## 7. Comment juger s7 (nouvelles métriques)

| Métrique | Attendu |
|---|---|
| `mech/jumps_per_min`, `mech/flips_per_min` | ne s'effondrent pas (s6 : 0,05 saut/min) |
| `aerial_touches_per_min` (voiture en l'air, nouveau) | monte (repère s6 : `air_touches_per_min` = 0,25/min, balle > 300 uu) |
| `curriculum/aerial/level` | monte régulièrement vers 1 |
| `shots_per_min`, `saves_per_min`, `drill/defense/success` | montent ; défense > 70 % |
| `mech/avg_boost`, `mech/low_boost_frac`, `mech/avg_speed`, `mech/supersonic_frac` | boost moyen > 25, moins de temps à sec, plus vite |
| `vs_hist/kickoff_win_rate` | > 0,55 |
| `reward_abs/<terme>` (nouveau) | poids réel de chaque terme ; `reward_term/` vaut ~0 pour les termes antisymétriques |
| **Matchs contre s6** : `evaluate.py --a runs/s7_spo_laptop --b runs/s6_spo_kaggle --format 1 --games 12 --game-seconds 120` à 20, 40 et 60 M steps | s7 doit battre s6 au plus tard à 60 M ; sinon, on revoit |

## 8. Ce qui viendra ensuite (non fait)

- **Self-play priorisé + ancres : codé, désactivé** (`selfplay.pfsp`, `selfplay.anchors`, cf. `configs/s7.yaml`) :
  versions tirées selon (1 − p_victoire)² + 30 % d'uniforme, ancres s4/s6 fixes (via `fit_obs`) ; à juger sur
  `vs_hist/pfsp_entropy` et `vs_anchor/<run>/win_rate`. Reste : bots Psyonix comme ancres ; activation = décision PO.
- Modèle plus large (d_model 192-256) dès qu'on a un vrai GPU.

## 9. Journal

- **20/10/2026, s7 → s7b.** s7 (13 M steps) apprenait ~6× moins vite que s4. Cause : ETA mal estimée en l'air →
  `positioning` très bruité à chaque saut. Correctif dans `env/predict.py` + amorce `vel_player_to_ball` 0,05 → 0 sur
  30 M + γ 0,99 au début (`configs/s7b.yaml`). Run `s7b_spo_laptop`.
- **20/10/2026 03:49, s7b à 29 M steps.** Match contre s6 : 0-12 (buts 6-68). Critique des rewards → reprise avec
  `positioning` 0,4 (transition 0,6 s), ramassage de boost 0,4 sous 50 de boost, exercices 35 %, niveau 0 aérien
  vraiment facile + passage de niveau à 50 % (`configs/s7b.yaml` points 4-7, `env/mutators.py`, `env/rewards.py`).
- **21/10/2026, s7b → s7c.** s7b arrêté à 32 M steps (~10 touches/min depuis 25 M, s4 ~18). Nouveau run de zéro
  `s7c_spo_laptop` : config s7b actuelle (correctifs 1-7 dès le départ) + aides vers la balle sur le calendrier de s4
  (`configs/s7c.yaml`). Comparaison avec s4 à 5, 10 et 20 M steps (≥ 4, ≥ 9, ≥ 15 touches/min).
- **21/10/2026, s7c → s7d (même run).** s7c rattrape s4 en touches dès 17 M steps (20,9/min à 23,7 M), mais au niveau 0,
  l'exercice aérien ne réussit qu'à 3,5 %, et le bot est à 0 de boost 81 % du temps
  (`runs/s7c_spo_laptop/DIAGNOSTIC_AERIEN_BOOST.md`). Correctifs dans `configs/s7d.yaml` :
  - A1 : exercice aérien en rendez-vous. Une politique scriptée y réussit à 81 %.
  - A2 : touche aérienne payée dès 200 uu.
  - B1 : potentiel de boost à γ = 1.
  - B2 : aide d'approche plafonnée à 1410 uu/s.

  Code dans `env/mutators.py` et `env/rewards.py`, 2 tests ajoutés ou mis à jour. Pas encore relancé.
- **22/10/2026, s7c_spo_kaggle (s7d, 34,6 → 72 M steps sur Kaggle).** Toujours battu par s6 : 0 V / 12 D (buts 11-46),
  1-10 en temps réel ; à peine mieux que s7c laptop (16-14). Cause principale : lr de l'acteur au plafond (1e-3) dans
  99-100 % des itérations de tous les runs, KL final 6× sous la cible. B1/B2 sans effet (boost moyen 12, à sec 75 %),
  aérien toujours au niveau 0, défense aérienne à 0 % (exercice irréalisable). `runs/s7c_spo_kaggle/RAPPORT.md`.
- **22/10/2026, s7e (décision PO : « meilleur partout »).** Nouveau run `s7e_spo_kaggle`, démarrage à chaud depuis
  s7c_spo_kaggle (72 M). `configs/s7e.yaml`, points E1-E8 :
  - E1 : plafond du lr 1e-3 → 3e-3, lot gardé à 100 000 steps après 80 M.
  - E2 : γ 0,99 → 0,995.
  - E3 : ancres s4, s5, s6, s7c laptop + PFSP ; 30 % des épisodes contre l'historique, dont la moitié contre une ancre.
  - E4 : reward `flip_speed` (gain de vitesse vers la balle dû au flip, record de la dernière seconde et boost déduits).
  - E5 : 3 pads de boost « à prendre » dans les observations (+12 entrées, couche d'entrée élargie à poids nuls) ;
    ramassage maintenu.
  - E6 : touche aérienne 3, aide d'approche en l'air 0,06.
  - E7 : défense aérienne en rendez-vous. La politique scriptée (grossière) réussit 45-56 % au niveau 0 contre 0 %
    avant ; ne rien faire : 0 %.
  - E8 : frappe vers le but adverse (`shot` 0,5).

  Code : `env/obs.py`, `env/rewards.py`, `env/mutators.py`, `models/attention.py` (SELF_DIMS), options branchées
  partout (factory, config, train, bot, evaluate, watch). Zip d'init avec ancres figées :
  `scripts/make_cloud_bundle.py --anchors-from`. Tests : `tests/test_s7e.py`.

  **Critères** (à 82 M puis 92 M steps) :
  - sain : `learn/kl_final` entre 0,004 et 0,02, `learn/lr_actor` sous le plafond une partie du temps, entropie qui
    baisse lentement (pas sous ~2,5 avant 100 M), `learn/early_stop` rare, `learn/explained_variance` > 0,6 après le
    changement de γ ;
  - progrès : `vs_anchor/s6_spo_kaggle/p_win` qui monte (≥ 0,3 à 92 M), `vs_anchor/s4…/s5…/s7c…` > 0,6, curriculum
    aérien et défense aérienne qui quittent 0, `mech/avg_boost` > 20, `mech/flips_per_min` qui monte.
  - Si le KL dépasse 0,05 de façon répétée ou que l'entropie s'effondre : STOP, reprise avec `--set ppo.lr_max=2e-3`.
- **22/10/2026, s7e arrêté à 105 M (échec), mesure du bruit du gradient → profil s7f (prêt, pas lancé).** Taille de lot
  critique ≈ 1,1-1,3 M steps par pas d'optimiseur : à 50 000, chaque pas est à ~96 % du bruit ; le bonus d'entropie vaut
  70 % du signal et le contre (`runs/s7e_spo_laptop/RAPPORT.md`). `configs/s7f.yaml` : lot 1 M / minibatch 500 000,
  `ent_coef` 0,005, tout le reste de s7e inchangé ; démarrage à chaud depuis s7e_spo_laptop (105 M). Juger à +30 M et
  +60 M : `learn/kl_final` qui répond au lr, entropie qui baisse, `vs_anchor/s6_spo_kaggle/p_win` qui monte.
- **22/10/2026, s7f_spo_kaggle (105 → 172 M steps, Kaggle GPU 5 h 30, 3 365 steps/s).** Pas de progrès mesurable.
  `p_win` contre s6 plat à 0,15-0,16. Gain sur s7c (0,48 → 0,63) et s5 (0,32 → 0,40), acquis dès les 10 premiers M
  puis plat. Aérien toujours au niveau 0, boost 13. KL final médian 0,0005, lr au plafond 65 itérations sur 67 : le KL
  ne répond toujours pas. Hypothèse : état d'Adam (2e moment) hérité des minibatchs bruités de s7e. Pistes à décider :
  optimiseur neuf, `lr_max` plus haut, plus d'epochs. Matchs 1v1 12 × 120 s : contre s6 0 V / 12 D (buts 3-49 ;
  déterministe 3-42), contre s7e 6 V / 2 D / 4 N (20-12). Détails : `runs/s7f_spo_kaggle/RAPPORT.md`.
