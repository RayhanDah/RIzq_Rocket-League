# Lancer la session d'entraînement sur Kaggle (run `{RUN}`)

Kit préparé le {DATE}. Le run reprend au checkpoint `{LATEST}` ({STEP} steps).
Compte environ 30 minutes de mise en place la première fois, puis une session tourne seule jusqu'à 9 h, navigateur
fermé.

## Contenu du dossier

| Fichier | Rôle | Taille |
|---|---|---|
| `{NOTEBOOK}` | le notebook à importer (le code du projet est déjà dedans) | < 1 Mo |
| `datasets/rl-collision-meshes.zip` | la géométrie du terrain pour le simulateur | {SIZE_MESHES} |
| `datasets/rl-agent-opponents/` | les bots adverses du sparring : 2 fichiers, `nexto-model.pt` et `opponents-bots.zip` | {SIZE_OPP} |
| `datasets/rl-agent-resume.zip` | le run à reprendre : checkpoint, jalons, {POOL} fichiers du pool, classement | {SIZE_RESUME} |

Tout reste **privé** : ne rends aucun dataset ni le notebook public. Il y a les poids d'autres bots et des
fichiers tirés du jeu.

## 1. Préparer le compte Kaggle (une seule fois)

1. Sur kaggle.com, connecté : photo de profil → **Settings**.
2. **Phone verification** : vérifie ton numéro. Sans ça, ni TPU ni Internet dans les notebooks.
3. Le quota TPU est d'environ 20 h par semaine, par compte. Il est indiqué en haut à droite de l'éditeur de notebook.

## 2. Créer les 3 datasets (une seule fois)

Pour chacun : **Create** (bouton +) → **New Dataset** → glisser les fichiers → donner le nom indiqué → laisser
**Private** → **Create**. L'envoi du plus gros prend du temps : attends que Kaggle affiche le dataset comme prêt.

| Nom du dataset | Fichiers à envoyer |
|---|---|
| `rl-collision-meshes` | `datasets/rl-collision-meshes.zip` |
| `rl-agent-opponents` | les 2 fichiers de `datasets/rl-agent-opponents/` (pas le dossier lui-même) |
| `rl-agent-resume` | `datasets/rl-agent-resume.zip` |

Si Kaggle décompresse les zips à l'envoi, ce n'est pas grave : le notebook gère les deux cas.

## 3. Importer le notebook (une seule fois)

1. **Create** → **New Notebook**.
2. Dans l'éditeur : menu **File** → **Import Notebook** → choisir `{NOTEBOOK}`.
3. Renomme le notebook `{SLUG}` (clic sur le titre en haut à gauche). Le nom sert à le retrouver ensuite.
4. Ne modifie aucune cellule. Les réglages (run `{RUN}`, profils `{PROFILES}`, arrêt automatique) sont déjà remplis.

## 4. Réglages de la session

Dans le panneau de droite de l'éditeur (**Session options** / **Settings**) :

- **Accelerator** : **TPU v5e-8** (ou « TPU VM »). Pas de GPU P100. À défaut de TPU, **GPU T4 x2** marche aussi, environ 3 fois moins vite.
- **Internet** : **On**. Le notebook installe ses dépendances au démarrage.

Puis **Add Input** (même panneau) → onglet **Datasets** / **Your Work** → ajoute les 3 datasets de l'étape 2.

## 5. Lancer

1. En haut à droite : **Save Version**.
2. Choisis **Save & Run All (Commit)** → **Save**.
3. Tu peux fermer le navigateur : la version tourne seule sur Kaggle, jusqu'à 9 h en TPU. Elle s'arrête proprement 20 min avant la limite et sauvegarde tout.

N'utilise pas le bouton **Run All** de l'éditeur (mode interactif) : la session s'arrêterait dès que tu fermes la page.

La version peut rester un moment **Queued** (en attente d'un TPU libre, parfois plusieurs heures, voire une
journée). C'est normal, il n'y a rien à faire.

## 6. Vérifier que tout est bon (15-20 min après le démarrage)

Ouvre la version en cours → onglet **Logs** (ou **Output** → journal). Tu dois voir, dans l'ordre :

```
reprise du run précédent : .../runs/{RUN}
[cloud] Nexto : /kaggle/input/.../nexto-model.pt
[cloud] bots de sparring : ... (10 fichiers .lt)
[train] reprise depuis {LATEST} (step {STEP})
{SPARRING}
[train] curriculum : palier ...
it ... | steps ... | sps ...
```

Les lignes `[train] sparring : ... reprend à beta ...` qui peuvent suivre sont normales : le run garde le niveau
déjà atteint contre chaque bot.

Ensuite, une ligne `it ... | steps ... | sps ...` toutes les 10 itérations. En TPU, `sps` doit être autour de
100 000 à 170 000.

| Message ou symptôme | Cause | Que faire |
|---|---|---|
| `reprise demandée, mais aucun run {RUN}` | dataset `rl-agent-resume` pas ajouté en Input | l'ajouter (étape 4), relancer |
| `collision meshes introuvables` | dataset `rl-collision-meshes` pas ajouté | l'ajouter, relancer |
| `Modèle Nexto absent` ou pas de ligne `bots de sparring` | dataset `rl-agent-opponents` manquant ou incomplet | vérifier ses 2 fichiers, relancer |
| `step 0` dans la ligne `reprise depuis` | le run n'a pas été retrouvé | **arrêter** la version (Cancel) et prévenir |
| `sps` sous 20 000 | pas de TPU (session CPU) | Accelerator : TPU, relancer |
| `garde-fou : ... retour au jalon ...` | sécurité automatique du run | rien à faire, la session continue |

Pour arrêter une version en cours : ouvre-la → **Cancel run**. Ce qui est déjà sauvegardé reste dans son Output.

## 7. Sessions suivantes

Quand une version est **Complete** :

1. Ouvre le notebook → **Edit**.
2. **Add Input** → onglet **Notebooks** / **Your Work** → ce notebook (`{SLUG}`), dernière version. Son Output devient le point de reprise.
3. Garde les 3 datasets en Input : le notebook prend tout seul le point de départ le plus avancé.
4. **Save Version** → **Save & Run All (Commit)**, puis vérifie comme à l'étape 6. Le `step` de la ligne `reprise depuis` doit avoir avancé.

## 8. Rendre les résultats

Le plus simple : **partager le notebook** avec le compte Kaggle du responsable du projet (bouton **Share** →
ajouter son pseudo, droit *Can view*), puis lui envoyer le lien du notebook. Il rapatrie le run depuis son PC.
Pas besoin de télécharger l'Output à la main : il pèse plusieurs Go à cause du pool de versions.

Si le partage n'est pas possible : onglet **Output** de la dernière version → télécharger au moins
`runs/{RUN}.log`, `runs/{RUN}/best_actor.pt`, `runs/{RUN}/best_actor.txt`, `runs/{RUN}/ladder.json` et le
dossier `runs/{RUN}/checkpoints/`.

## Règles du projet

- Le bot ne joue **jamais en ligne ni en ranked** : uniquement hors ligne, via RLBot.
- Ne modifie pas le notebook ni les réglages d'entraînement. Si quelque chose ne colle pas avec ce document, arrête la version et préviens.

