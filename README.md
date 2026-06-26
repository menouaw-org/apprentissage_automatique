# PA-ML — Bibliothèque d'apprentissage automatique en C avec orchestration Python

## Présentation

Ce projet implémente des algorithmes d'apprentissage automatique basiques en C, puis les expose à Python au moyen d'une bibliothèque dynamique et de bindings `ctypes`.

L'objectif n'est pas seulement d'obtenir un modèle qui fonctionne, mais de construire une chaîne complète et explicable:

1. implémenter les modèles en C;
2. compiler une bibliothèque dynamique;
3. appeler cette bibliothèque depuis Python;
4. préparer un jeu d'images original;
5. entraîner et comparer plusieurs modèles;
6. sauvegarder des modèles pré-entraînés;
7. recharger ces modèles pour l'inférence;
8. exposer une démonstration locale minimale avec Gradio.

Le projet couvre actuellement:

- un modèle linéaire;
- un perceptron multicouches, ou MLP;
- un socle Python commun pour les données 64x64;
- une couche d'inférence réutilisable;
- une base Gradio locale;
- la sauvegarde et la recharge de modèles linéaires et MLP.

Le RBF, le SVM ou une alternative validée ne sont pas encore intégrés dans la bibliothèque publique.

## État actuel

À ce stade:

- la bibliothèque C compile sous forme de DLL;
- Python charge la DLL avec `ctypes`;
- le modèle linéaire est implémenté, testé et utilisé comme baseline;
- le MLP est implémenté pour la classification;
- l'API C expose la création, l'entraînement, la prédiction, la sauvegarde, la recharge et la destruction des modèles linéaire et MLP;
- le MLP expose aussi une sortie continue de diagnostic avec `predict_mlp_model_raw`;
- les conventions d'images 64x64 sont centralisées dans `python/common/dataset_64x64.py`;
- la couche d'inférence est isolée dans `python/inference/predict_64x64.py`;
- l'application Gradio minimale est disponible dans `app.py`;
- deux artefacts de démonstration peuvent être générés dans `models/linear/` et `models/mlp/`;
- les scripts d'expérience linéaire et MLP restent séparés de l'application;
- `data/splits/test.csv` reste gelé pour l'évaluation finale.

Le projet peut donc être présenté comme une chaîne expérimentale et applicative complète pour les modèles linéaire et MLP, avec des performances encore limitées mais une architecture désormais stabilisée.

## Pile technique

### Langages

- C: cœur algorithmique, modèles, API publique et bibliothèque dynamique.
- Python: orchestration, tests, bindings `ctypes`, chargement des images, validation croisée, métriques, figures, inférence et interface Gradio.

### Outils principaux

- Git / GitHub;
- CMake;
- MinGW GCC;
- Ninja;
- uv;
- `ctypes`;
- NumPy;
- Pillow;
- pandas;
- Matplotlib;
- Plotly;
- TensorBoard;
- tqdm;
- Gradio.

### Configuration locale validée

- système principal: Windows;
- toolset MinGW: `C:\mingw64`;
- dépendances MinGW: `C:\mingw64\bin`;
- profil CMake recommandé: `cmake-build-release`;
- bibliothèque attendue: `cmake-build-release/libpa_ml.dll`.

Le profil Release doit être privilégié pour les expériences longues et pour l'inférence applicative.

## Arborescence utile

```
pa-ml/
├── app.py
├── CMakeLists.txt
├── pyproject.toml
├── README.md
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
│   ├── common/
│   │   ├── dataset_64x64.py
│   │   └── reports.py
│   ├── experiments/
│   │   ├── run_linear_baseline_64x64.py
│   │   ├── run_mlp_64x64.py
│   │   ├── debug_mlp_64x64_signal.py
│   │   └── export_demo_artifacts_64x64.py
│   └── inference/
│       ├── __init__.py
│       └── predict_64x64.py
├── tests/
│   └── python/
│       ├── test_linear_interface.py
│       ├── test_linear_data_conventions.py
│       ├── test_linear_multiclass_predict.py
│       ├── test_linear_multiclass_training.py
│       ├── test_linear_model_cases.py
│       ├── test_linear_model_persistence.py
│       ├── test_mlp_interface.py
│       ├── test_mlp_model_cases.py
│       ├── test_mlp_raw_output.py
│       └── test_mlp_model_persistence.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── 64x64/
│   └── splits/
│       ├── folds.csv
│       └── test.csv
├── models/
│   ├── linear/
│   │   └── linear_64x64_demo.pa_model
│   └── mlp/
│       └── mlp_64x64_demo.pa_model
├── reports/
│   ├── tables/
│   └── figures/
│       ├── confusion_matrices/
│       └── learning_curves/
└── tmp/
    └── gradio_uploads/
```

## Jeu de données

Le jeu de données image est organisé autour de trois classes:

| Classe | Nombre d'images |
| --- | --- |
| --- | ---: |
| dog | 2114 |
| cat | 1668 |
| others | 2457 |
| Total | 6239 |

La résolution utilisée dans les expériences documentées est `64x64`.

Chaque image est:

1. convertie en RGB;
2. contrôlée au format `64x64x3`;
3. aplatie en vecteur;
4. convertie en `np.float64`;
5. normalisée entre `0.0` et `1.0`.

Les conventions sont centralisées dans:

```
python/common/dataset_64x64.py
```

## Séparation expérimentale

```
data/splits/
├── folds.csv
└── test.csv
```

Règles:

- `folds.csv` sert aux entraînements, diagnostics et validations croisées;
- `test.csv` reste gelé pour l'évaluation finale;
- `test.csv` ne doit pas être utilisé pour choisir l'architecture, le taux d'apprentissage, le nombre d'époques ou l'image de démonstration;
- les expériences actuelles reposent sur 5 plis stratifiés;
- la graine de référence est `42`.

## Conventions de classification

Les sorties de classification utilisent une convention bipolaire:

| Classe | Cible |
| --- | --- |
| dog | `[+1.0, -1.0, -1.0]` |
| cat | `[-1.0, +1.0, -1.0]` |
| others | `[-1.0, -1.0, +1.0]` |

La prédiction finale du modèle linéaire et du MLP est bipolaire.

Pour le MLP, `predict_mlp_model_raw` expose les activations continues avant décision. Cette sortie est utilisée pour le diagnostic et pour l'affichage des scores bruts dans la démonstration.

Ces scores ne doivent pas être interprétés comme des probabilités.

## API C publique

L'API publique se trouve dans:

```
src/api/ml_library.h
src/api/ml_library.c
```

### Modèle linéaire

Fonctions exposées:

```c
void* create_linear_model(
    int32_t input_size,
    int32_t output_size,
    int32_t task_type
);

int32_t train_linear_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

int32_t predict_linear_model(
    void* model,
    const double* x,
    double* y_pred
);

int32_t save_linear_model(void* model, const char* path);

void* load_linear_model(const char* path);

void destroy_linear_model(void* model);
```

### MLP

Fonctions exposées:

```c
void* create_mlp_model(
    int32_t input_size,
    int32_t output_size,
    int32_t hidden_layer_count,
    const int32_t* hidden_layer_sizes,
    int32_t task_type
);

int32_t train_mlp_model(
    void* model,
    const double* x,
    const double* y,
    int32_t n_samples,
    double learning_rate,
    int32_t epochs
);

int32_t predict_mlp_model(
    void* model,
    const double* x,
    double* y_pred
);

int32_t predict_mlp_model_raw(
    void* model,
    const double* x,
    double* y_raw
);

int32_t save_mlp_model(void* model, const char* path);

void* load_mlp_model(const char* path);

void destroy_mlp_model(void* model);
```

## Bindings Python

Les bindings sont définis dans:

```
python/bindings/c_api.py
```

Ce module est volontairement limité à:

- charger la DLL;
- configurer les signatures `ctypes`;
- exposer les constantes de type de tâche;
- fournir des utilitaires de conversion NumPy vers pointeurs C;
- encoder les chemins de fichiers pour la sauvegarde et la recharge.

Il ne contient pas de logique d'expérience, de chargement d'image, de métrique ou d'interface applicative.

La DLL chargée par défaut est:

```
cmake-build-release/libpa_ml.dll
```

## Compilation

### Compiler en Release

```bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Fichier attendu:

```
cmake-build-release/libpa_ml.dll
```

### Compiler en Debug

```bash
cmake -S . -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
```

Le profil Debug reste utile pour inspecter le code, mais il n'est pas recommandé pour les expériences longues.

## Environnement Python

Synchroniser l'environnement:

```bash
uv sync
```

Dépendances principales:

- `numpy`;
- `pillow`;
- `pandas`;
- `matplotlib`;
- `plotly`;
- `seaborn`;
- `tensorboard`;
- `tqdm`;
- `gradio`.

## Tests techniques

Après compilation Release, lancer les tests principaux.

### Tests du modèle linéaire

```bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
uv run python tests/python/test_linear_model_cases.py
uv run python tests/python/test_linear_model_persistence.py
```

### Tests du MLP

```bash
uv run python tests/python/test_mlp_interface.py
uv run python tests/python/test_mlp_model_cases.py
uv run python tests/python/test_mlp_raw_output.py
uv run python tests/python/test_mlp_model_persistence.py
```

Les tests de persistance vérifient que les modèles sauvegardés puis rechargés produisent les mêmes prédictions qu'avant sauvegarde.

## Expériences

## Baseline linéaire 64x64

Commande:

```bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
```

Artefacts produits:

```
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
```

Résultats observés:

| Métrique | Valeur |
| --- | --- |
| --- | ---: |
| train_accuracy moyenne | 0.3940 |
| validation_accuracy moyenne | 0.3936 |

Lecture:

- le pipeline fonctionne;
- le modèle linéaire sur pixels bruts sous-apprend;
- la stratégie est presque entièrement concentrée sur `others`;
- cette baseline sert de référence basse avant le MLP.

## MLP 64x64

Commande principale:

```bash
uv run python python/experiments/run_mlp_64x64.py --fold all --epochs 16 --eval-every 2 --learning-rate 0.001 --hidden-sizes 64 --balanced-train
```

Configuration retenue:

```
hidden_sizes=64
learning_rate=0.001
epochs=16
eval_every=2
balanced_train=true
activation=tanh
fold=all
```

Artefacts produits:

```
reports/tables/mlp_64x64_folds.csv
reports/tables/mlp_64x64_history.csv
reports/figures/confusion_matrices/mlp_64x64.png
reports/figures/learning_curves/mlp_64x64.png
```

Résultats par pli:

| Pli | train_accuracy | validation_accuracy |
| --- | --- | --- |
| ---: | ---: | ---: |
| 0 | 0.5528 | 0.4972 |
| 1 | 0.4790 | 0.4473 |
| 2 | 0.5212 | 0.4698 |
| 3 | 0.5135 | 0.4731 |
| 4 | 0.5062 | 0.4400 |

Synthèse:

| Métrique | Valeur |
| --- | --- |
| --- | ---: |
| train_accuracy moyenne finale | 0.5145 |
| validation_accuracy moyenne finale | 0.4655 |
| baseline linéaire | 0.3936 |
| gain absolu MLP vs linéaire | +0.0719 |

Lecture:

- le MLP dépasse la baseline linéaire;
- le modèle ne prédit plus presque tout en `others`;
- `dog` et `cat` sont prédits en quantité significative;
- `others` reste la classe la mieux reconnue;
- `cat` reste la classe la plus faible;
- les courbes train / validation restent proches et basses, ce qui suggère surtout du sous-apprentissage.

## Diagnostic MLP

Commande de diagnostic:

```bash
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.001 --hidden-sizes 64
```

Artefacts produits:

```
reports/tables/mlp_64x64_signal_probe.csv
reports/tables/mlp_64x64_mini_balanced_history.csv
```

Ce diagnostic permet de distinguer:

- la sortie continue du modèle;
- la prédiction finale bipolaire;
- les cas où le modèle apprend sur le mini-jeu mais généralise mal;
- les effets d'un taux d'apprentissage trop élevé;
- la stratégie majoritaire `others`.

## Sauvegarde et recharge des modèles

La persistance est disponible pour:

- le modèle linéaire;
- le MLP.

Les artefacts de démonstration sont générés dans:

```
models/linear/linear_64x64_demo.pa_model
models/mlp/mlp_64x64_demo.pa_model
```

Commande de génération:

```bash
uv run python python/experiments/export_demo_artifacts_64x64.py --fold 0 --linear-learning-rate 0.001 --linear-epochs 5 --mlp-learning-rate 0.001 --mlp-epochs 2 --mlp-hidden-sizes 64 --mlp-balanced-train
```

Ces artefacts servent à valider la chaîne:

```
modèle entraîné -> sauvegarde -> recharge -> inférence -> Gradio
```

Ils ne doivent pas être présentés comme les meilleurs modèles scientifiques du projet.

## Couche d'inférence

La couche d'inférence est définie dans:

```
python/inference/predict_64x64.py
```

Elle expose:

```python
predict_image(image_path, model_path, model_type)
```

Elle retourne un objet contenant:

```
label
scores
image_path
model_path
model_type
```

Modèles acceptés:

```
linear
mlp
```

Exemple d'inférence linéaire:

```bash
uv run python -m python.inference.predict_64x64 \
  --model-path models/linear/linear_64x64_demo.pa_model \
  --image-path data/processed/64x64/cat/cat_00007.jpg \
  --model-type linear
```

Exemple d'inférence MLP:

```bash
uv run python -m python.inference.predict_64x64 \
  --model-path models/mlp/mlp_64x64_demo.pa_model \
  --image-path data/processed/64x64/cat/cat_00007.jpg \
  --model-type mlp
```

La couche d'inférence:

- réutilise le prétraitement commun;
- recharge un modèle sauvegardé;
- appelle les bindings C / Python;
- libère le modèle dans un bloc `finally`;
- ne relance pas d'entraînement;
- ne lit pas `data/splits/test.csv`;
- ne dépend pas de Gradio.

## Démonstration Gradio

L'application Gradio minimale est disponible dans:

```
app.py
```

Lancement:

```bash
uv run python app.py
```

L'interface permet:

- de déposer une image;
- de choisir `mlp` ou `linear`;
- de charger l'artefact correspondant;
- d'exécuter une prédiction;
- d'afficher un JSON minimal.

Format de sortie:

```json
{
  "label": "others",
  "scores": {
    "dog": -0.48,
    "cat": -0.52,
    "others": -0.07
  },
  "model_type": "mlp",
  "model_path": "models/mlp/mlp_64x64_demo.pa_model",
  "image_path": "tmp/gradio_uploads/gradio_input_64x64.jpg"
}
```

Points importants:

- les scores affichés sont des sorties brutes, pas des probabilités;
- Gradio ne relance pas d'entraînement;
- Gradio ne manipule pas directement `ctypes`;
- Gradio ne duplique pas l'aplatissement ni la normalisation;
- l'image déposée est convertie en RGB puis redimensionnée en `64x64`;
- le dossier `tmp/gradio_uploads/` est temporaire et ne constitue pas un artefact scientifique.

## Chaîne applicative

```
Image déposée dans Gradio
-> conversion RGB et copie 64x64
-> fichier temporaire tmp/gradio_uploads/gradio_input_64x64.jpg
-> predict_image(image_path, model_path, model_type)
-> prétraitement commun load_image_as_vector
-> chargement du modèle sauvegardé
-> prédiction via bindings C / Python
-> libération mémoire
-> résultat JSON
```

## Commandes utiles

### Préparer l'environnement

```bash
uv sync
```

### Compiler la bibliothèque

```bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

### Lancer tous les tests principaux

```bash
uv run python tests/python/test_linear_interface.py
uv run python tests/python/test_linear_data_conventions.py
uv run python tests/python/test_linear_multiclass_predict.py
uv run python tests/python/test_linear_multiclass_training.py
uv run python tests/python/test_linear_model_cases.py
uv run python tests/python/test_linear_model_persistence.py

uv run python tests/python/test_mlp_interface.py
uv run python tests/python/test_mlp_model_cases.py
uv run python tests/python/test_mlp_raw_output.py
uv run python tests/python/test_mlp_model_persistence.py
```

### Relancer la baseline linéaire

```bash
uv run python python/experiments/run_linear_baseline_64x64.py --fold all --epochs 5 --learning-rate 0.001
```

### Relancer l'expérience MLP principale

```bash
uv run python python/experiments/run_mlp_64x64.py --fold all --epochs 16 --eval-every 2 --learning-rate 0.001 --hidden-sizes 64 --balanced-train
```

### Relancer le diagnostic MLP

```bash
uv run python python/experiments/debug_mlp_64x64_signal.py --fold 0 --per-class 30 --probe-per-class 10 --epochs 50 --learning-rate 0.001 --hidden-sizes 64
```

### Générer les artefacts de démonstration

```bash
uv run python python/experiments/export_demo_artifacts_64x64.py --fold 0 --linear-learning-rate 0.001 --linear-epochs 5 --mlp-learning-rate 0.001 --mlp-epochs 2 --mlp-hidden-sizes 64 --mlp-balanced-train
```

### Tester l'inférence

```bash
uv run python -m python.inference.predict_64x64 \
  --model-path models/linear/linear_64x64_demo.pa_model \
  --image-path data/processed/64x64/cat/cat_00007.jpg \
  --model-type linear

uv run python -m python.inference.predict_64x64 \
  --model-path models/mlp/mlp_64x64_demo.pa_model \
  --image-path data/processed/64x64/cat/cat_00007.jpg \
  --model-type mlp
```

### Lancer Gradio

```bash
uv run python app.py
```

## Artefacts à conserver

### Baseline linéaire

```
reports/tables/linear_baseline_64x64_folds.csv
reports/tables/linear_baseline_64x64_history.csv
reports/figures/confusion_matrices/linear_baseline_64x64.png
reports/figures/learning_curves/linear_baseline_64x64.png
```

### MLP

```
reports/tables/mlp_64x64_folds.csv
reports/tables/mlp_64x64_history.csv
reports/figures/confusion_matrices/mlp_64x64.png
reports/figures/learning_curves/mlp_64x64.png
reports/tables/mlp_64x64_signal_probe.csv
reports/tables/mlp_64x64_mini_balanced_history.csv
```

### Modèles de démonstration

```
models/linear/linear_64x64_demo.pa_model
models/mlp/mlp_64x64_demo.pa_model
```

## Points de vigilance

- Ne pas utiliser `data/splits/test.csv` pour régler le modèle.
- Ne pas conclure uniquement sur l'accuracy globale.
- Toujours lire la matrice de confusion.
- Toujours regarder les rappels `dog`, `cat` et `others`.
- Ne pas présenter le MLP comme robuste: il améliore la baseline mais reste limité.
- Ne pas présenter les scores Gradio comme des probabilités.
- Ne pas confondre démonstration technique et validation scientifique.
- Ne pas relancer d'entraînement depuis Gradio.
- Ne pas importer les scripts d'expérience dans l'application.
- Ne pas dupliquer le prétraitement image dans l'inférence ou dans Gradio.
- Ne pas ajouter de signatures RBF tant que le modèle RBF n'est pas implémenté, compilable et testé.
- Ne pas confondre les artefacts de démonstration avec les meilleurs modèles du projet.

## Limites actuelles

- Le modèle linéaire sous-apprend fortement sur pixels bruts.
- Le MLP améliore la baseline mais reste insuffisant pour une classification robuste.
- La classe `cat` reste la plus fragile.
- Les modèles de démonstration servent d'abord à valider la chaîne technique.
- Le RBF n'est pas encore implémenté.
- Le SVM ou l'alternative validée n'est pas encore implémenté.
- Le MLP régression n'est pas couvert dans l'étape actuelle.
- L'application Gradio est volontairement minimale et locale.

## Formulation recommandée pour le rapport

Le modèle linéaire constitue une baseline volontairement simple sur pixels bruts. Il obtient une validation accuracy moyenne de `0.3936` et prédit presque toujours la classe `others`, ce qui révèle un sous-apprentissage important.

Le MLP à une couche cachée, entraîné avec un sous-échantillonnage équilibré du train, atteint une validation accuracy moyenne de `0.4655`. Il améliore donc la baseline linéaire et sort de la stratégie strictement majoritaire. Toutefois, les rappels par classe montrent que la classification reste fragile: `others` reste la classe la mieux reconnue, tandis que `cat` demeure la plus faible.

La réduction de dette technique a stabilisé la chaîne logicielle: conventions de données centralisées, bindings nettoyés, persistance des modèles, inférence isolée et démonstration Gradio locale. Le projet dispose donc maintenant d'une base plus fiable pour poursuivre vers RBF, SVM ou une application plus complète.

## Prochaines actions recommandées

1. Vérifier que les artefacts de démonstration sont inclus ou régénérables dans le rendu.
2. Ajouter au rapport un tableau de comparaison des rappels par classe.
3. Indiquer explicitement que `data/splits/test.csv` n'a pas encore été utilisé pour le réglage.
4. Présenter Gradio comme une démonstration technique locale, pas comme une preuve de robustesse scientifique.
5. Implémenter le RBF comme modèle séparé, avec API C, bindings Python, tests et persistance.
6. Décider ensuite si le SVM est implémenté ou si une alternative validée est retenue.

## Résumé final

Le projet dispose désormais d'une chaîne complète pour le modèle linéaire et le MLP: compilation C, bindings Python, préparation du jeu de données, expériences, diagnostics, sauvegarde, recharge, inférence et démonstration Gradio.

Le MLP améliore la baseline linéaire, mais le résultat doit être présenté avec nuance: amélioration mesurable, pas performance applicative robuste.

La principale avancée récente est la réduction de dette technique: le code est mieux séparé, les conventions sont centralisées, les modèles peuvent être sauvegardés et rechargés, et l'application Gradio consomme une couche d'inférence propre plutôt que des scripts d'expérience.