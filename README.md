# PA-ML — Bibliothèque d’apprentissage automatique en C avec orchestration Python

## Présentation

Ce projet construit une bibliothèque d’apprentissage automatique en C, puis l’exploite depuis Python pour préparer les données, lancer les expérimentations, analyser les résultats et produire des éléments de rapport.

Depuis l’implémentation du modèle linéaire, la chaîne expérimentale a été complétée autour du passage vers le perceptron multicouches:

1. implémentation et validation du modèle linéaire;
2. constitution du jeu d’images dog / cat / others;
3. création d’une baseline linéaire sur pixels bruts 64x64;
4. implémentation du perceptron multicouches;
5. validation du MLP sur cas contrôlés;
6. diagnostic du passage aux images réelles;
7. ajout d’une sortie continue de diagnostic;
8. correction de la stratégie majoritaire par équilibrage du train;
9. première validation croisée MLP complète sur cinq plis;
10. comparaison MLP / baseline linéaire.

Le résultat principal est le suivant: le MLP à une couche cachée améliore la baseline linéaire, mais ne produit pas encore une classification robuste. La valeur du projet vient donc autant de la démarche expérimentale que du score final: baseline, diagnostic, instrumentation, correction, comparaison et limites.

## État actuel

À ce stade:

- la bibliothèque C compile sous forme de DLL;
- Python charge la DLL avec ctypes;
- le modèle linéaire est implémenté, testé et utilisé comme baseline;
- le MLP est implémenté pour la classification;
- l’interface C / Python du MLP expose create_mlp_model, train_mlp_model, predict_mlp_model, predict_mlp_model_raw et destroy_mlp_model;
- les cas contrôlés principaux du MLP sont validés;
- le jeu d’images dataset_v1_64x64 est préparé;
- la baseline linéaire 64x64 est exécutée et analysée;
- l’expérience MLP 64x64 complète est exécutée sur cinq plis;
- les artefacts CSV et figures sont disponibles;
- la synthèse comparative est rédigée.

Le projet MLP peut donc être présenté comme une étape expérimentale cohérente, avec une amélioration mesurable mais des limites assumées.

## Périmètre validé

### Modèle linéaire

Le modèle linéaire couvre:

- régression simple;
- régression à plusieurs entrées;
- classification binaire;
- classification multi-sorties;
- apprentissage multi-classes;
- prédiction avec sorties bipolaires;
- libération mémoire.

Il sert maintenant de baseline basse sur les images 64x64.

### Perceptron multicouches

Le MLP couvre actuellement:

- classification;
- couches cachées configurables;
- activation tanh;
- initialisation déterministe corrigée;
- propagation avant;
- rétropropagation;
- mise à jour des poids et biais;
- prédiction bipolaire finale;
- sortie continue de diagnostic avant décision;
- appel depuis Python avec ctypes.

Le périmètre MLP actuel est volontairement centré sur la classification image. La régression MLP n’est pas traitée dans cette étape et doit être mentionnée comme hors périmètre si le rapport général évoque classification et régression.

## Pile technique

### Langages

- C: cœur algorithmique, modèles, API publique et bibliothèque dynamique.
- Python: orchestration, tests, bindings ctypes, chargement des images, validation croisée, métriques, figures et analyse.

### Outils principaux

- Git / GitHub;
- CLion;
- PyCharm;
- CMake;
- MinGW GCC;
- Ninja;
- uv;
- ctypes;
- NumPy;
- Pillow;
- pandas;
- Matplotlib;
- tqdm.

### Configuration locale validée

- système: Windows;
- toolset MinGW: C:\mingw64;
- dépendances MinGW: C:\mingw64\bin;
- build CMake recommandé pour les expériences longues: cmake-build-release;
- bibliothèque attendue: cmake-build-release/libpa_ml.dll.

Le profil Release doit être privilégié pour les expériences MLP, car les boucles C sur images 64x64 sont coûteuses.

## Arborescence utile

~~~plain text
pa-ml/
├── src/
│   ├── api/
│   │   ├── ml_library.h
│   │   └── ml_library.c
│   ├── linear/
│   │   ├── linear_model.h
│   │   └── linear_model.c
│   └── mlp/
│       ├── mlp_model.h
│       └── mlp_model.c
├── python/
│   ├── bindings/
│   │   └── c_api.py
│   └── experiments/
│       ├── run_linear_baseline_64x64.py
│       ├── run_mlp_64x64.py
│       └── debug_mlp_64x64_signal.py
├── tests/
│   └── python/
│       ├── test_linear_interface.py
│       ├── test_linear_data_conventions.py
│       ├── test_linear_multiclass_predict.py
│       ├── test_linear_multiclass_training.py
│       ├── test_linear_model_cases.py
│       ├── test_mlp_interface.py
│       ├── test_mlp_model_cases.py
│       └── test_mlp_raw_output.py
├── data/
│   ├── processed/
│   │   └── 64x64/
│   └── splits/
│       ├── folds.csv
│       └── test.csv
└── reports/
    ├── tables/
    └── figures/
        ├── confusion_matrices/
        └── learning_curves/
~~~

## Jeu de données

Le jeu de données image est constitué pour le projet à partir d’images collectées puis filtrées.

### Classes

| Classe | Nombre d’images |
|---|---:|
| dog | 2114 |
| cat | 1668 |
| others | 2457 |
| Total | 6239 |

### Organisation

~~~plain text
data/
├── raw/
│   ├── dog/
│   ├── cat/
│   └── others/
├── processed/
│   ├── 32x32/
│   ├── 64x64/
│   └── 128x128/
└── splits/
    ├── test.csv
    └── folds.csv
~~~

La résolution utilisée pour les expériences documentées ici est 64x64. Chaque image est convertie en RGB, aplatie, puis normalisée entre 0.0 et 1.0.

### Séparation expérimentale

- test final: 15%;
- développement: 85%;
- validation croisée: 5 plis stratifiés;
- graine: 42.

Règle importante:

~~~plain text
data/splits/test.csv ne doit pas être utilisé pour choisir une architecture, un taux d’apprentissage ou un nombre d’époques.
~~~

Les expériences actuelles utilisent data/splits/folds.csv. Le fichier test.csv reste gelé tant que la configuration finale n’est pas explicitement figée.

## Conventions de classification

Les sorties de classification utilisent une convention bipolaire.

| Classe | Cible |
|---|---|
| dog | [+1.0, -1.0, -1.0] |
| cat | [-1.0, +1.0, -1.0] |
| others | [-1.0, -1.0, +1.0] |

La prédiction finale renvoyée par predict_mlp_model est également bipolaire.

Pour l’analyse, predict_mlp_model_raw expose les activations continues de sortie avant décision. Cette sortie est indispensable pour éviter de diagnostiquer seulement le vecteur final discrétisé.

## Compilation

### Release recommandé

~~~bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
~~~

Le fichier attendu est:

~~~plain text
cmake-build-release/libpa_ml.dll
~~~

### Debug

~~~bash
cmake -S . -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
~~~

Le profil Debug reste utile pour inspecter le code, mais il n’est pas recommandé pour les expériences longues.

## Tests techniques

### Tests du modèle linéaire

~~~bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
uv run python tests/python/test_linear_model_cases.py
~~~

### Tests du MLP

~~~bash
uv run python tests/python/test_mlp_interface.py
uv run python tests/python/test_mlp_model_cases.py
uv run python tests/python/test_mlp_raw_output.py
~~~

Résultats MLP observés sur cas contrôlés:

| Cas | Résultat observé |
|---|---:|
| Linear Simple | accuracy=1.0000 |
| Linear Multiple | accuracy=1.0000 |
| XOR | accuracy=1.0000 |
| Cross | accuracy=0.9960 |
| Multi Linear 3 classes | accuracy=1.0000 |
| Multi Cross | accuracy=0.5240, limite connue |

Multi Cross ne doit pas être présenté comme validé. Il reste une limite connue non bloquante pour les expériences images actuelles.

## Baseline linéaire 64x64

### Commande

~~~bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
~~~

### Artefacts

~~~plain text
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
~~~

### Résultats

| Métrique | Valeur |
|---|---:|
| train_accuracy moyenne | 0.3940 |
| validation_accuracy moyenne | 0.3936 |

Matrice de confusion cumulée:

| Classe réelle | Prédit dog | Prédit cat | Prédit others |
|---|---:|---:|---:|
| dog | 2 | 0 | 1795 |
| cat | 1 | 1 | 1415 |
| others | 3 | 1 | 2084 |

Lecture:

- le pipeline fonctionne;
- le modèle linéaire sur pixels bruts sous-apprend;
- la stratégie est presque entièrement concentrée sur others;
- cette baseline sert de référence basse avant le MLP.

## MLP 64x64

### Script principal

~~~plain text
python/experiments/run_mlp_64x64.py
~~~

Le script:

- lit uniquement data/splits/folds.csv;
- ne lit pas data/splits/test.csv;
- charge les images 64x64;
- applique le même prétraitement que la baseline linéaire;
- encode les cibles en bipolaire;
- crée le MLP C via ctypes;
- entraîne pli par pli;
- évalue avec les sorties continues predict_mlp_model_raw;
- produit les CSV et figures;
- libère le modèle dans un bloc finally.

### Configuration retenue

~~~plain text
hidden_sizes=64
learning_rate=0.001
epochs=16
eval_every=2
balanced_train=true
activation=tanh
initialization=centered*1.0
fold=all
~~~

Commande:

~~~bash
uv run python python/experiments/run_mlp_64x64.py --fold all --epochs 16 --eval-every 2 --learning-rate 0.001 --hidden-sizes 64 --balanced-train
~~~

### Artefacts

~~~plain text
reports/tables/mlp_64x64_folds.csv
reports/tables/mlp_64x64_history.csv
reports/figures/confusion_matrices/mlp_64x64.png
reports/figures/learning_curves/mlp_64x64.png
~~~

### Résultats par pli

| Pli | train_accuracy | validation_accuracy |
|---:|---:|---:|
| 0 | 0.5528 | 0.4972 |
| 1 | 0.4790 | 0.4473 |
| 2 | 0.5212 | 0.4698 |
| 3 | 0.5135 | 0.4731 |
| 4 | 0.5062 | 0.4400 |

Synthèse:

| Métrique | Valeur |
|---|---:|
| train_accuracy moyenne finale | 0.5145 |
| validation_accuracy moyenne finale | 0.4655 |
| baseline linéaire | 0.3936 |
| gain absolu MLP vs linéaire | +0.0719 |

### Matrice de confusion cumulée

| Classe réelle | Prédit dog | Prédit cat | Prédit others | Rappel |
|---|---:|---:|---:|---:|
| dog | 724 | 418 | 655 | 0.4029 |
| cat | 400 | 434 | 583 | 0.3063 |
| others | 402 | 376 | 1310 | 0.6274 |

Lecture:

- le MLP dépasse la baseline linéaire;
- le modèle ne prédit plus presque tout en others;
- dog et cat sont prédits en quantité significative;
- others reste la classe la mieux reconnue;
- cat reste la classe la plus faible;
- les courbes train / validation restent proches et basses, ce qui suggère surtout du sous-apprentissage.

## Diagnostics importants

### Stratégie majoritaire others

Les premiers essais MLP sur le pli 0 donnaient une validation accuracy de 0.3936, exactement au niveau de la proportion others.

Matrice observée:

~~~plain text
dog    -> dog=0, cat=0, others=360
cat    -> dog=0, cat=0, others=284
others -> dog=0, cat=0, others=418
~~~

Cette situation a bloqué le lancement des cinq plis tant que dog et cat n’étaient pas prédits.

### Mini-jeu équilibré 30/30/30

Un diagnostic sur mini-jeu équilibré a montré que le modèle bougeait bien, mais que learning_rate=0.01 était instable.

Commandes utiles:

~~~bash
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.01 --hidden-sizes 64
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.003 --hidden-sizes 64
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.001 --hidden-sizes 64
~~~

Résultats retenus:

| learning_rate | Mini-train final | Sonde finale | Lecture |
|---:|---:|---:|---|
| 0.01 | 0.4889 | 0.4000 | instable |
| 0.003 | 0.9556 | 0.4000 | mémorisation forte, généralisation faible |
| 0.001 | 0.9111 | 0.5333 | configuration prudente retenue |

### Sortie continue predict_mlp_model_raw

La sortie continue a été ajoutée pour lire les activations avant décision.

Fichiers concernés:

- src/mlp/mlp_model.h;
- src/mlp/mlp_model.c;
- src/api/ml_library.h;
- src/api/ml_library.c;
- python/bindings/c_api.py;
- tests/python/test_mlp_raw_output.py;
- python/experiments/debug_mlp_64x64_signal.py.

Cette correction a permis de distinguer:

- le signal continu réellement produit par le modèle;
- la prédiction finale bipolaire;
- les erreurs de diagnostic liées à np.argmax sur des sorties déjà discrétisées.

### Entraînement équilibré

L’option --balanced-train sous-échantillonne le train de chaque pli pour obtenir autant d’exemples dog, cat et others.

Sur le pli 0, l’équilibrage observé était:

~~~plain text
dog=1133, cat=1133, others=1133
~~~

Cette correction a permis de sortir de la stratégie strictement majoritaire.

## Comparaison linéaire / MLP

| Modèle | Configuration | Validation accuracy moyenne | Lecture |
|---|---|---:|---|
| Linéaire | 64x64, pixels bruts, epochs=5, lr=0.001 | 0.3936 | baseline basse, stratégie presque majoritaire others |
| MLP | hidden_sizes=64, epochs=16, lr=0.001, --balanced-train | 0.4655 | amélioration réelle, mais classification encore fragile |

Conclusion:

- le MLP apporte un gain mesurable;
- l’amélioration ne suffit pas à parler de modèle robuste;
- la faiblesse principale reste la classe cat;
- la représentation par pixels bruts 64x64 reste probablement difficile pour un MLP simple;
- la suite doit être cadrée: soit clôture pour rapport, soit variation ciblée.

## Décisions expérimentales

Décisions déjà prises:

- ne pas optimiser davantage la baseline linéaire;
- ne pas lancer les cinq plis MLP tant que le pli 0 prédit tout en others;
- ajouter une sortie continue de diagnostic;
- utiliser learning_rate=0.001 comme réglage prudent;
- utiliser hidden_sizes=64;
- utiliser --balanced-train;
- retenir epochs=16 pour la première expérience complète, car le pli 0 avait montré une instabilité après ce point;
- conserver data/splits/test.csv hors des réglages.

Décisions encore à expliciter dans le rapport ou la soutenance:

- le test final data/splits/test.csv n’a pas encore été utilisé;
- le MLP régression est hors périmètre de cette étape;
- le résultat MLP améliore la baseline mais reste insuffisant pour une application robuste;
- une variation future peut être planifiée, mais ne doit pas rouvrir une campagne d’optimisation non maîtrisée.

## Variations possibles à reporter

Si une variation minimale est nécessaire pour renforcer le rapport, privilégier un seul axe à la fois.

### Capacité

~~~bash
uv run python python/experiments/run_mlp_64x64.py --fold all --epochs 16 --eval-every 2 --learning-rate 0.001 --hidden-sizes 128 --balanced-train
~~~

But: vérifier si le sous-apprentissage vient d’une capacité insuffisante.

### Résolution plus faible

Créer une expérience 32x32 si les données préparées et le script adapté sont disponibles.

But: réduire la dimension d’entrée, accélérer les essais et tester si le bruit des pixels 64x64 pénalise le MLP.

### Perte continue

Utiliser predict_mlp_model_raw pour calculer une erreur quadratique moyenne entre sorties continues et cibles bipolaires.

But: remplacer la perte provisoire 1 - accuracy, trop pauvre pour diagnostiquer l’apprentissage.

## Points de vigilance

- Ne pas utiliser data/splits/test.csv pour régler le modèle.
- Ne pas conclure uniquement sur l’accuracy globale.
- Toujours lire la matrice de confusion.
- Toujours regarder les rappels dog, cat et others.
- Ne pas présenter le MLP comme robuste: il améliore la baseline mais reste limité.
- Ne pas présenter Multi Cross comme validé.
- Ne pas mélanger prédiction bipolaire finale et sortie continue de diagnostic.
- Ne pas oublier que --balanced-train réduit le nombre d’exemples utilisés au train.
- Documenter la graine et les effectifs retenus si de nouvelles expériences sont lancées.
- Préférer une variation courte et interprétable à une grande campagne d’hyperparamètres.

## Commandes utiles

### Synchroniser l’environnement Python

~~~bash
uv sync
~~~

### Compiler en Release

~~~bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
~~~

### Lancer les tests principaux

~~~bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
uv run python tests/python/test_linear_model_cases.py
uv run python tests/python/test_mlp_interface.py
uv run python tests/python/test_mlp_model_cases.py
uv run python tests/python/test_mlp_raw_output.py
~~~

### Relancer la baseline linéaire

~~~bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
~~~

### Relancer l’expérience MLP principale

~~~bash
uv run python python/experiments/run_mlp_64x64.py --fold all --epochs 16 --eval-every 2 --learning-rate 0.001 --hidden-sizes 64 --balanced-train
~~~

### Relancer le diagnostic du signal MLP

~~~bash
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.001 --hidden-sizes 64
~~~

## Artefacts à conserver

### Baseline linéaire

~~~plain text
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
~~~

### MLP

~~~plain text
reports/tables/mlp_64x64_folds.csv
reports/tables/mlp_64x64_history.csv
reports/figures/confusion_matrices/mlp_64x64.png
reports/figures/learning_curves/mlp_64x64.png
reports/tables/mlp_64x64_signal_probe.csv
reports/tables/mlp_64x64_mini_balanced_history.csv
~~~

## Formulation recommandée pour le rapport

Le modèle linéaire constitue une baseline volontairement simple sur pixels bruts. Il obtient une validation accuracy moyenne de 0.3936 et prédit presque toujours la classe others, ce qui révèle un sous-apprentissage important.

Le MLP à une couche cachée, entraîné avec un sous-échantillonnage équilibré du train, atteint une validation accuracy moyenne de 0.4655. Il améliore donc la baseline linéaire et sort de la stratégie strictement majoritaire. Toutefois, les rappels par classe montrent que la classification reste fragile: others reste la classe la mieux reconnue, tandis que cat demeure la plus faible.

L’expérience montre donc que le MLP apporte un signal non linéaire utile, mais que la représentation par pixels bruts 64x64, la capacité limitée du modèle et la stratégie d’entraînement actuelle ne suffisent pas à obtenir une classification robuste.

## Prochaines actions recommandées

1. Vérifier que les artefacts MLP et linéaires sont inclus dans le rendu.
2. Ajouter au rapport un tableau de comparaison des rappels par classe.
3. Indiquer explicitement que data/splits/test.csv n’a pas encore été utilisé.
4. Mentionner que le MLP régression est hors périmètre de cette étape.
5. Décider si le projet MLP est clôturé pour rapport ou si une seule variation ciblée est ajoutée.
6. Si une variation est lancée, ne modifier qu’un axe: capacité, résolution ou perte continue.

## Résumé final

Le projet a dépassé le stade de l’implémentation: il dispose maintenant d’une chaîne expérimentale complète, d’une baseline, d’un MLP instrumenté, d’un diagnostic documenté, d’une expérience complète et d’une comparaison exploitable.

Le MLP améliore la baseline linéaire, mais le résultat doit être présenté avec nuance: amélioration mesurable, pas performance applicative robuste.