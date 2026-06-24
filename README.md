# PA-ML — Bibliothèque d"apprentissage automatique en C avec orchestration Python

## Présentation

Ce projet a pour objectif de construire une bibliothèque d'apprentissage automatique en **C**, puis de l'exploiter depuis **Python** pour préparer les données, lancer les expérimentations, analyser les résultats et produire une démonstration.

Le projet est organisé autour de trois objectifs principaux:

1. implémenter progressivement plusieurs modèles de machine learning en C;
2. utiliser Python comme couche d'orchestration, de validation, de notebooks et de visualisation;
3. conserver une trace reproductible des jeux de données, des expériences, des résultats et des décisions.

Le premier périmètre technique validé concerne la mise en place de l'environnement:
- structure du dépôt;
- environnement Python avec `uv`;
- compilation d'une bibliothèque C dynamique avec CMake;
- génération de `libpa_ml.dll` sous Windows;
- appel de fonctions C depuis Python avec `ctypes`;
- préparation du jeu de données image;
- création d'un protocole expérimental de départ.

## Pile technique

### Langages

- C: cœur algorithmique et future bibliothèque de machine learning.
- Python: scripts, bindings `ctypes`, notebooks, prétraitement des données, expérimentation et démonstration.

### Outils principaux

- Git / GitHub: versionnement et collaboration.
- CLion: développement C, configuration CMake, compilation Debug / Release.
- PyCharm: développement Python, scripts, notebooks et orchestration.
- CMake: configuration du build C.
- MinGW GCC: compilation C sous Windows.
- Ninja: moteur de build utilisé par CLion / CMake.
- GDB: débogage C.
- uv: gestion de l'environnement Python et des dépendances.

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
│   ├── processed/
│   └── splits/
├── models/
│   ├── linear/
│   ├── mlp/
│   ├── rbf/
│   └── svm/
├── src/
│   ├── core/
│   ├── linear/
│   ├── mlp/
│   ├── rbf/
│   ├── svm/
│   └── api/
│       ├── ml_library.h
│       └── ml_library.c
├── tests/
│   ├── c/
│   └── python/
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
│   ├── experiments/
│   └── app/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── experiment_log.md
└── scripts/
~~~

### Rôle des principaux dossiers

| Dossier | Rôle |
|---|---|
| `src/api/` | Interface C exposée à Python via `ctypes`. |
| `src/core/` | Fonctions communes: matrices, données, métriques, utilitaires. |
| `src/linear/` | Futur modèle linéaire. |
| `src/mlp/` | Futur perceptron multicouches. |
| `src/rbf/` | Futur modèle RBF. |
| `src/svm/` | Futur SVM ou modèle alternatif. |
| `python/bindings/` | Chargement de la bibliothèque C et déclaration des signatures `ctypes`. |
| `python/data/` | Chargement, préparation et séparation des données. |
| `python/experiments/` | Scripts d'expérimentation et de suivi. |
| `python/app/` | Application de démonstration, par exemple avec Gradio. |
| `notebooks/` | Exploration, expérimentation et analyse comparative. |
| `data/raw/` | Images brutes conservées dans leur forme d'origine. |
| `data/processed/` | Images redimensionnées et préparées. |
| `data/splits/` | Fichiers de séparation train / validation / test. |
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

Si le projet vient d'être initialisé et que les dépendances doivent être ajoutées:

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

Le projet utilise CMake et une cible de type bibliothèque partagée:

~~~cmake
add_library(
        pa_ml SHARED
        src/api/ml_library.c
)
~~~

## Compilation avec CLion

La méthode recommandée est de compiler depuis CLion.

### Profil Debug

Le profil `Debug` sert au développement, au débogage et à la validation rapide de l'interopérabilité C / Python.

Procédure:

1. ouvrir le projet dans CLion;
2. sélectionner le profil `Debug`;
3. lancer `Build`;
4. vérifier que le fichier suivant existe:

~~~plain text
cmake-build-debug/libpa_ml.dll
~~~

### Profil Release

Le profil `Release` sert aux versions optimisées.

Procédure:

1. ouvrir le projet dans CLion;
2. sélectionner le profil `Release`;
3. lancer `Build`;
4. vérifier que le fichier suivant existe:

~~~plain text
cmake-build-release/libpa_ml.dll
~~~

La configuration Release par défaut de CLion / CMake est suffisante à ce stade. Aucun flag personnalisé comme `-O3`, `-march=native` ou `-ffast-math` n'est ajouté pour l'instant.

Les flags plus agressifs devront être envisagés seulement après:
- mesure des performances;
- comparaison Debug / Release;
- vérification de la stabilité numérique;
- validation du fait que le binaire reste exploitable par le groupe.

## Compilation en terminal

Les commandes CMake équivalentes sont les suivantes.

### Debug

~~~bash
cmake -S . -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
~~~

### Release

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

Ce problème ne remet pas en cause la configuration CLion. CLion peut utiliser son CMake intégré sans que celui-ci soit disponible dans un terminal externe.

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

## Validation Python / C

Le fichier de validation minimal est:

~~~plain text
python/bindings/c_api.py
~~~

Il charge `libpa_ml.dll` avec `ctypes`, déclare les signatures des fonctions C exposées et vérifie les appels suivants:
- appel d'une fonction simple `my_add`;
- création d'un pointeur opaque `LinearModel*`;
- prédiction minimale via ce pointeur;
- libération du pointeur;
- transmission d'un tableau `numpy.float32` vers une fonction C.

### Lancer la validation

~~~bash
uv run python/bindings/c_api.py
~~~

### Résultat attendu

~~~plain text
55
93.0
110.0
~~~

Si ce résultat s'affiche, cela valide que:
- la DLL est trouvée;
- les dépendances MinGW sont accessibles;
- les symboles C sont exportés;
- Python peut appeler la bibliothèque C;
- les types simples, pointeurs opaques et tableaux traversent correctement la frontière C / Python.

## Gestion du chemin de la DLL

Le chargement de la DLL doit être calculé depuis l'emplacement réel du fichier `c_api.py`, afin d'éviter les chemins relatifs fragiles.

Version minimale actuelle:

~~~python
import ctypes
import os
from pathlib import Path

import numpy as np


def load_library():
    os.add_dll_directory(r"C:\mingw64\bin")

    project_root = Path(__file__).resolve().parents[2]
    dll_path = project_root / "cmake-build-debug" / "libpa_ml.dll"

    if not dll_path.exists():
        raise FileNotFoundError(f"Bibliothèque introuvable: {dll_path}")

    return ctypes.CDLL(str(dll_path))
~~~

Si la DLL Release doit être testée, adapter le chemin vers:

~~~plain text
cmake-build-release/libpa_ml.dll
~~~

Une amélioration prévue consiste à ajouter un choix de build `Debug` / `Release` via une variable d'environnement, pour ne pas modifier le code à la main.

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
| Total | 6239 |

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
- validation croisée: 5 folds stratifiés sur les 85%;
- graine aléatoire: `42`.

### Fichiers de split

~~~plain text
data/splits/test.csv
data/splits/folds.csv
~~~

### Règles importantes

- Le jeu de test final ne doit pas être utilisé pour choisir un modèle ou régler des hyperparamètres.
- Les comparaisons pendant le développement doivent utiliser les folds de validation.
- Toute modification des données, des splits ou du prétraitement doit être documentée.
- Une expérience échouée doit être conservée et analysée.
- Les résultats doivent être tracés dans le carnet d'expérimentations.

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

Les expériences doivent permettre de comprendre:
- ce qui est testé;
- pourquoi c'est testé;
- avec quels réglages;
- quels résultats sont observés;
- quelle décision est prise ensuite.

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
│   ├── metrics_by_fold.csv
│   └── model_comparisons.csv
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

## Limites actuelles

À ce stade:
- la bibliothèque C minimale ne contient pas encore de vrai modèle de machine learning;
- les fonctions exposées servent surtout à valider l'interopérabilité C / Python;
- les modèles linéaire, MLP, RBF et SVM seront implémentés dans des tâches séparées;
- les scripts de build terminal dépendent de la disponibilité de `cmake` et `ninja` dans le `PATH`;
- la validation Python charge actuellement une DLL selon le chemin configuré dans `c_api.py`;
- les flags d'optimisation avancés ne sont pas encore activés.

## Actions restantes

- Finaliser les scripts `scripts/` si le groupe souhaite une compilation hors CLion.
- Ajouter un choix Debug / Release sans modification manuelle du chemin de DLL.
- Compléter les tests C.
- Compléter les tests Python.
- Implémenter les modèles dans des tâches dédiées.
- Alimenter le carnet d'expérimentations à chaque essai.
- Préparer la démonstration finale avec l'application Python.

## Règle de contribution

Recommandations:
- faire des commits courts et lisibles;
- séparer les changements C, Python, données et documentation;
- éviter de commiter les fichiers générés lourds;
- documenter toute décision structurante;
- vérifier que le projet compile avant de pousser une modification importante.

## Commandes utiles

### Synchroniser Python

~~~bash
uv sync
~~~

### Compiler depuis CLion

1. ouvrir CLion;
2. sélectionner `Debug` ou `Release`;
3. lancer `Build`.

### Tester l'appel C depuis Python

~~~bash
uv run python/bindings/c_api.py
~~~

### Résultat attendu

~~~plain text
55
93.0
110.0
~~~