# PA-ML — Bibliothèque d'apprentissage automatique en C avec orchestration Python

## Présentation

Ce projet a pour objectif de construire une bibliothèque d'apprentissage automatique en **C**, puis de l'exploiter depuis **Python** pour préparer les données, lancer les expérimentations, analyser les résultats et produire une démonstration.

Le projet est organisé autour de trois objectifs principaux:

1. implémenter progressivement plusieurs modèles de machine learning en C;
2. utiliser Python comme couche d'orchestration, de validation, de notebooks, de visualisation et de démonstration;
3. conserver une trace reproductible des jeux de données, des expériences, des résultats et des décisions.

Le périmètre déjà validé couvre:

- la structure du dépôt;
- l'environnement Python avec `uv`;
- la compilation d'une bibliothèque C dynamique avec CMake;
- la génération de `libpa_ml.dll` sous Windows;
- l'appel de fonctions C depuis Python avec `ctypes`;
- la préparation du jeu de données image `dog` / `cat` / `others`;
- la séparation stable du jeu de données avec `test.csv` final et validation croisée 5 plis;
- l'interface technique du modèle linéaire;
- les conventions de données C / Python;
- la prédiction multi-sorties;
- l'apprentissage multi-classes;
- les cas professeur du modèle linéaire;
- une première baseline linéaire sur `dataset_v1_64x64`.

Le modèle linéaire est actuellement utilisé comme **baseline explicable** avant les modèles non linéaires, notamment le MLP.

## Pile technique

### Langages

- **C**: cœur algorithmique, modèles et bibliothèque dynamique.
- **Python**: scripts, bindings `ctypes`, notebooks, prétraitement des données, expérimentation, visualisation et démonstration.

### Outils principaux

- **Git / GitHub**: versionnement et collaboration.
- **CLion**: développement C, configuration CMake, compilation Debug / Release.
- **PyCharm**: développement Python, scripts, notebooks, orchestration et visualisation de CSV.
- **CMake**: configuration du build C.
- **MinGW GCC**: compilation C sous Windows.
- **Ninja**: moteur de build utilisé par CLion / CMake.
- **GDB**: débogage C.
- **uv**: gestion de l'environnement Python et des dépendances.
- **ctypes**: interface entre Python et la bibliothèque C.
- **NumPy**: tableaux numériques et conventions de données.
- **Pillow / OpenCV**: chargement et prétraitement des images.
- **pandas**: analyse des CSV expérimentaux.
- **Matplotlib / Plotly**: visualisation des métriques et figures.
- **tqdm**: barres de progression pour les traitements longs.
- **Gradio**: démonstration applicative prévue.

### Configuration locale validée

La configuration locale utilisée pour le projet est la suivante:

- système: Windows;
- toolset MinGW: `C:\mingw64`;
- dépendances MinGW disponibles dans: `C:\mingw64\bin`;
- CMake: CMake intégré à CLion;
- générateur CMake: Ninja;
- compilateur C: `cc.exe`;
- compilateur C++ détecté par CLion: `c++.exe`;
- débogueur: GDB intégré / configuré dans CLion;
- type de projet CLion: `C Library`;
- cible principale CMake: bibliothèque partagée.

Le terminal externe peut ne pas reconnaître `cmake` si le CMake intégré à CLion n'est pas présent dans le `PATH`. Ce n'est pas bloquant: la compilation peut être faite depuis CLion.

## Arborescence du projet

~~~plain text
pa-ml/
├── README.md
├── Makefile
├── CMakeLists.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── dog/
│   │   ├── cat/
│   │   └── others/
│   ├── processed/
│   │   ├── 32x32/
│   │   ├── 64x64/
│   │   └── 128x128/
│   └── splits/
│       ├── test.csv
│       └── folds.csv
├── models/
│   ├── linear/
│   ├── mlp/
│   ├── rbf/
│   └── svm/
├── src/
│   ├── core/
│   ├── linear/
│   │   ├── linear_model.c
│   │   └── linear_model.h
│   ├── mlp/
│   ├── rbf/
│   ├── svm/
│   └── api/
│       ├── ml_library.h
│       └── ml_library.c
├── tests/
│   ├── c/
│   └── python/
│       ├── test_linear_interface.py
│       ├── test_linear_data_conventions.py
│       ├── test_linear_multiclass_predict.py
│       ├── test_linear_multiclass_training.py
│       └── test_linear_model_cases.py
├── notebooks/
│   ├── 00_dataset_exploration.ipynb
│   ├── 01_linear_model.ipynb
│   ├── 02_mlp.ipynb
│   ├── 03_rbf.ipynb
│   ├── 04_svm.ipynb
│   └── 05_comparative_analysis.ipynb
├── python/
│   ├── bindings/
│   │   ├── __init__.py
│   │   └── c_api.py
│   ├── data/
│   │   ├── resize.py
│   │   └── split.py
│   ├── experiments/
│   │   └── run_linear_baseline_64x64.py
│   └── app/
├── reports/
│   ├── figures/
│   │   ├── confusion_matrices/
│   │   ├── learning_curves/
│   │   └── comparisons/
│   ├── tables/
│   │   ├── linear_model_classic_tests.csv
│   │   ├── linear_baseline_64x64_folds.csv
│   │   └── linear_baseline_64x64_history.csv
│   └── experiment_log.md
└── scripts/
~~~

### Rôle des principaux dossiers

| Dossier | Rôle |
|---|---|
| `src/api/` | Interface C exposée à Python via `ctypes`. |
| `src/core/` | Fonctions communes: matrices, données, métriques, utilitaires. |
| `src/linear/` | Implémentation du modèle linéaire. |
| `src/mlp/` | Futur perceptron multicouches. |
| `src/rbf/` | Futur modèle RBF. |
| `src/svm/` | Futur SVM ou modèle alternatif. |
| `python/bindings/` | Chargement de la bibliothèque C et déclaration des signatures `ctypes`. |
| `python/data/` | Chargement, préparation, redimensionnement et séparation des données. |
| `python/experiments/` | Scripts d'expérimentation et de suivi. |
| `python/app/` | Application de démonstration, par exemple avec Gradio. |
| `notebooks/` | Exploration, expérimentation et analyse comparative. |
| `data/raw/` | Images brutes conservées dans leur forme d'origine. |
| `data/processed/` | Images redimensionnées et préparées. |
| `data/splits/` | Fichiers de séparation test final et validation croisée. |
| `models/` | Modèles sauvegardés. |
| `reports/` | Figures, tableaux, résultats et traces d'expériences. |
| `tests/` | Tests C et Python. |

## Installation

### 1. Cloner le dépôt

~~~bash
git clone <url-du-depot>
cd pa-ml
~~~

### 2. Installer uv

Si `uv` n'est pas encore installé, suivre la documentation officielle de `uv` ou l'installer avec la méthode adaptée au poste.

Vérifier l'installation:

~~~bash
uv --version
~~~

### 3. Synchroniser l'environnement Python

Depuis la racine du projet:

~~~bash
uv sync
~~~

Si les dépendances doivent être ajoutées:

~~~bash
uv add numpy pandas pillow opencv-python plotly matplotlib seaborn tensorboard gradio tqdm
~~~

### 4. Vérifier l'environnement Python

~~~bash
uv run python --version
~~~

## Compilation de la bibliothèque C

La bibliothèque C est compilée sous forme de bibliothèque dynamique Windows:

~~~plain text
libpa_ml.dll
~~~

Le projet utilise CMake et une cible de type bibliothèque partagée.

### Compilation avec CLion

La méthode recommandée est de compiler depuis CLion.

#### Profil Debug

Le profil `Debug` sert au développement, au débogage et à la validation rapide de l'interopérabilité C / Python.

Procédure:

1. ouvrir le projet dans CLion;
2. sélectionner le profil `Debug`;
3. lancer `Build`;
4. vérifier que le fichier suivant existe:

~~~plain text
cmake-build-debug/libpa_ml.dll
~~~

#### Profil Release

Le profil `Release` sert aux versions optimisées.

Procédure:

1. ouvrir le projet dans CLion;
2. sélectionner le profil `Release`;
3. lancer `Build`;
4. vérifier que le fichier suivant existe:

~~~plain text
cmake-build-release/libpa_ml.dll
~~~

La configuration Release par défaut de CLion / CMake est suffisante à ce stade. Les flags plus agressifs doivent être envisagés seulement après mesure des performances, comparaison Debug / Release et vérification de la stabilité numérique.

### Compilation en terminal

#### Debug

~~~bash
cmake -S . -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
~~~

#### Release

~~~bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
~~~

### Erreur possible: `cmake: command not found`

Si le terminal affiche:

~~~plain text
bash: cmake: command not found
~~~

cela signifie que le terminal ne trouve pas l'exécutable `cmake`.

Solutions possibles:

- compiler depuis CLion;
- utiliser le terminal intégré de CLion;
- ajouter CMake et Ninja au `PATH`;
- installer CMake / Ninja séparément si nécessaire.

Pour diagnostiquer depuis PowerShell:

~~~powershell
where.exe cmake
where.exe ninja
~~~

## Interface Python / C

Le fichier principal de bindings est:

~~~plain text
python/bindings/c_api.py
~~~

Il charge `libpa_ml.dll` avec `ctypes`, déclare les signatures des fonctions C exposées et fournit les utilitaires Python nécessaires pour transmettre des tableaux NumPy au C.

### Gestion du chemin de la DLL

Le chargement de la DLL doit être calculé depuis l'emplacement réel de `c_api.py`, afin d'éviter les chemins relatifs fragiles.

Le chargement doit notamment:

- ajouter `C:\mingw64\bin` aux dossiers DLL Windows;
- chercher la DLL dans `cmake-build-debug/` ou `cmake-build-release/`;
- échouer explicitement si la DLL attendue est absente.

## Modèle linéaire

Le modèle linéaire est le premier modèle implémenté et validé.

Il couvre actuellement:

- création d'un modèle linéaire;
- régression simple;
- régression à plusieurs entrées;
- classification binaire;
- classification multi-sorties;
- apprentissage multi-classes;
- prédiction avec sorties bipolaires;
- libération mémoire.

### Convention des sorties de classification

Les sorties de classification utilisent une convention bipolaire:

| Valeur | Signification |
|---:|---|
| `+1.0` | classe positive / classe prédite |
| `-1.0` | classe négative |

Pour les trois classes du projet:

| Classe | Cible bipolaire |
|---|---|
| `dog` | `[+1.0, -1.0, -1.0]` |
| `cat` | `[-1.0, +1.0, -1.0]` |
| `others` | `[-1.0, -1.0, +1.0]` |

### Point de vigilance sur les labels one-hot

Une erreur potentielle a été identifiée dans la normalisation des cibles de classification côté C: une cible `0.0` pouvait être convertie en `+1.0` si la règle était `target >= 0.0`.

Cette convention est dangereuse avec les labels one-hot classiques:

~~~plain text
[1.0, 0.0, 0.0]
~~~

car ils peuvent devenir:

~~~plain text
[1.0, 1.0, 1.0]
~~~

au lieu de:

~~~plain text
[1.0, -1.0, -1.0]
~~~

La correction recommandée est:

| Label reçu | Label interne |
|---:|---:|
| `1.0` | `+1.0` |
| `0.0` | `-1.0` |
| `-1.0` | `-1.0` |
| autre valeur | erreur |

Après correction, il faut relancer les tests du modèle linéaire puis confirmer la baseline.

## Tests

Les tests Python actuellement utilisés pour le modèle linéaire sont:

~~~plain text
tests/python/test_linear_interface.py
tests/python/test_linear_data_conventions.py
tests/python/test_linear_multiclass_predict.py
tests/python/test_linear_multiclass_training.py
tests/python/test_linear_model_cases.py
~~~

### Lancer les tests techniques

~~~bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
~~~

Résultats attendus:

~~~plain text
Test d'interface linéaire réussi.
Test de conventions de données réussi.
Test de prédiction multi-sorties réussi.
Test d'apprentissage multi-classes réussi.
~~~

### Lancer les cas professeur

~~~bash
uv run python tests/python/test_linear_model_cases.py
~~~

Le script produit:

~~~plain text
reports/tables/linear_model_classic_tests.csv
~~~

Les cas professeur servent à vérifier:

- les cas linéaires de classification attendus `OK`;
- les cas linéaires de régression attendus `OK`;
- la classification multi-classes à trois sorties;
- les cas non linéaires attendus `KO`;
- l'absence de bug probable sur les cas simples.

## Jeu de données

Le jeu de données image est constitué spécifiquement pour le projet.

### Méthode de collecte

Méthode retenue:

~~~plain text
Google Images → téléchargement massif par mots-clefs → filtrage humain
~~~

Le projet ne reprend pas directement un jeu de données public existant.

### Classes

Les classes finales sont:

| Classe | Description |
|---|---|
| `dog` | Images de chiens. |
| `cat` | Images de chats. |
| `others` | Animaux identifiables qui ne sont ni des chiens ni des chats. |

### Volumes après filtrage

| Classe | Nombre d'images |
|---|---:|
| `dog` | 2114 |
| `cat` | 1668 |
| `others` | 2457 |
| **Total** | **6239** |

### Organisation des données

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

### Convention de nommage

Les images finales sont au format `.jpg`.

Convention:

~~~plain text
<classe>_<identifiant-numérique>.jpg
~~~

Exemple:

~~~plain text
dog_00001.jpg
~~~

### Résolutions préparées

Les images préparées sont prévues en plusieurs résolutions:

- `32x32`;
- `64x64`;
- `128x128`.

La résolution de départ pour les premières expériences est:

~~~plain text
64x64
~~~

### Redimensionnement

Le redimensionnement conserve le ratio de l'image et ajoute un padding noir si nécessaire. Cette règle évite de déformer les animaux.

## Protocole expérimental

La séparation des données doit rester stable pour permettre des comparaisons fiables entre modèles.

### Séparation retenue

- test final: 15% du jeu de données;
- développement: 85% restants;
- validation croisée: 5 plis stratifiés sur les 85%;
- graine aléatoire: `42`.

### Fichiers de split

~~~plain text
data/splits/test.csv
data/splits/folds.csv
~~~

### Règles importantes

- Le jeu de test final ne doit pas être utilisé pour choisir un modèle ou régler des hyperparamètres.
- Les comparaisons pendant le développement doivent utiliser les plis de validation.
- Toute modification des données, des splits ou du prétraitement doit être documentée.
- Une expérience échouée doit être conservée et analysée.
- Les résultats doivent être tracés dans le carnet d'expérimentations.

## Baseline linéaire 64x64

La première baseline image utilise le modèle linéaire sur les pixels bruts `64x64`.

### Script

~~~plain text
python/experiments/run_linear_baseline_64x64.py
~~~

Le script:

- lit `data/splits/folds.csv`;
- ne charge pas `data/splits/test.csv`;
- charge les images depuis `data/processed/64x64/`;
- convertit les images en RGB;
- vectorise les pixels;
- normalise les valeurs entre `0.0` et `1.0`;
- encode les classes en sorties bipolaires;
- entraîne le modèle sur un ou plusieurs plis;
- enregistre les métriques et figures;
- utilise `tqdm` pour les traitements longs.

### Commande de contrôle

~~~bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold 0 --epochs 1 --learning-rate 0.001
~~~

### Commande 5 plis

~~~bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
~~~

Une relance de diagnostic a aussi été effectuée sur 10 époques.

### Artefacts produits

~~~plain text
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
~~~

### Résultat observé

La baseline linéaire `64x64` donne une validation accuracy moyenne autour de:

~~~plain text
0.39
~~~

La matrice de confusion montre que le modèle prédit presque toujours:

~~~plain text
others
~~~

Interprétation:

- la chaîne expérimentale fonctionne;
- le modèle linéaire ne surapprend pas;
- les courbes train / validation restent basses et proches;
- le comportement correspond à un sous-apprentissage ou à une baseline majoritaire;
- le modèle linéaire sur pixels bruts `64x64` ne sépare pas utilement `dog`, `cat` et `others`;
- ce résultat reste utile comme baseline basse avant le MLP.

### Analyse des CSV

`linear_baseline_64x64_folds.csv` sert à comparer les résultats finaux par pli.

Colonnes principales:

~~~plain text
fold
train_accuracy
validation_accuracy
train_loss
validation_loss
~~~

Vue recommandée dans PyCharm:

- Categories: `fold`
- Groups: aucun, ou `dataset` si un champ est requis
- Values: `train_accuracy`, `validation_accuracy`
- Stacked: désactivé
- Horizontal: désactivé

`linear_baseline_64x64_history.csv` sert à suivre l'évolution par époque.

Colonnes principales:

~~~plain text
fold
epoch
train_accuracy
validation_accuracy
train_loss
validation_loss
~~~

Vue recommandée dans PyCharm:

- Categories: `epoch`
- Groups: `fold`
- Values: `train_accuracy`, `validation_accuracy`
- Stacked: désactivé
- Horizontal: désactivé

## Carnet d'expérimentations

Chaque expérience doit documenter au minimum:

~~~plain text
Expérience:
Statut:
Type:
Modèle:
Résolution:
Hypothèse:
Hyperparamètres:
Résultat:
Analyse:
Décision suivante:
Fichiers associés:
~~~

L'entrée actuelle importante est:

~~~plain text
Linéaire — baseline — 64x64
~~~

Elle documente:

- l'hypothèse de départ;
- les hyperparamètres;
- les métriques moyennes;
- la matrice de confusion;
- le diagnostic de sous-apprentissage;
- la décision suivante.

## Notebooks

Les notebooks prévus sont:

| Notebook | Rôle |
|---|---|
| `00_dataset_exploration.ipynb` | Exploration du jeu de données. |
| `01_linear_model.ipynb` | Expériences liées au modèle linéaire. |
| `02_mlp.ipynb` | Expériences liées au perceptron multicouches. |
| `03_rbf.ipynb` | Expériences liées au modèle RBF. |
| `04_svm.ipynb` | Expériences liées au SVM ou modèle alternatif. |
| `05_comparative_analysis.ipynb` | Analyse comparative finale. |

## Résultats attendus

Les sorties expérimentales doivent être rangées dans:

~~~plain text
reports/
├── figures/
│   ├── learning_curves/
│   ├── confusion_matrices/
│   └── comparisons/
├── tables/
└── experiment_log.md
~~~

Les modèles sauvegardés doivent être rangés dans:

~~~plain text
models/
├── linear/
├── mlp/
├── rbf/
└── svm/
~~~

## Nettoyage des builds

Depuis un terminal compatible Unix:

~~~bash
rm -rf cmake-build-debug cmake-build-release
~~~

Depuis PowerShell:

~~~powershell
Remove-Item -Recurse -Force cmake-build-debug, cmake-build-release -ErrorAction SilentlyContinue
~~~

## Problèmes fréquents

### `cmake: command not found`

Le terminal ne trouve pas CMake.

Correction recommandée:

- compiler depuis CLion;
- ou ajouter CMake au `PATH`;
- ou utiliser le terminal intégré de CLion.

### `libpa_ml.dll` introuvable

Vérifier que:

- le projet a été compilé;
- le bon profil CLion a été utilisé;
- le fichier existe dans `cmake-build-debug/` ou `cmake-build-release/`;
- le chemin dans `c_api.py` correspond au build testé.

### Erreur de dépendance DLL MinGW

Si Python trouve `libpa_ml.dll` mais échoue à la charger, il peut manquer une dépendance runtime de MinGW.

Vérifier que le chemin suivant est bien ajouté avant `ctypes.CDLL`:

~~~python
os.add_dll_directory(r"C:\mingw64\bin")
~~~

### Labels one-hot mal interprétés

Si les labels Python utilisent `0.0` / `1.0`, vérifier que la normalisation C transforme:

~~~plain text
1.0  -> +1.0
0.0  -> -1.0
-1.0 -> -1.0
~~~

et refuse les valeurs ambiguës.

## État actuel

À ce stade:

- l'environnement de développement est en place;
- la bibliothèque C compile sous forme de DLL;
- Python charge la DLL avec `ctypes`;
- le modèle linéaire est implémenté et testé;
- les conventions de données sont documentées;
- les cas professeur sont transformés en validation reproductible;
- la baseline linéaire `64x64` est exécutée et analysée;
- le résultat linéaire sert de baseline basse avant les modèles non linéaires.

## Actions restantes

- Corriger et sécuriser la normalisation des cibles de classification côté C pour accepter proprement `0.0`, `1.0` et `-1.0`.
- Ajouter ou adapter un test Python dédié aux labels one-hot.
- Relancer les tests du modèle linéaire après correction.
- Confirmer la baseline linéaire après correction.
- Passer au MLP comme premier modèle non linéaire.
- Compléter les tests C.
- Sauvegarder les modèles entraînés.
- Préparer la démonstration finale avec Gradio.
- Alimenter le carnet d'expérimentations à chaque essai.

## Règle de contribution

Recommandations:

- faire des commits courts et lisibles;
- séparer les changements C, Python, données et documentation;
- éviter de commiter les fichiers générés lourds;
- documenter toute décision structurante;
- vérifier que le projet compile avant de pousser une modification importante;
- relancer les tests Python après toute modification de l'interface C / Python ou du modèle linéaire.

## Commandes utiles

### Synchroniser Python

~~~bash
uv sync
~~~

### Compiler depuis CLion

1. ouvrir CLion;
2. sélectionner `Debug` ou `Release`;
3. lancer `Build`.

### Tester le modèle linéaire

~~~bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
uv run python tests/python/test_linear_model_cases.py
~~~

### Lancer la baseline linéaire

~~~bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
~~~

### Lire les résultats

~~~plain text
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
~~~