import argparse
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    lib,
)
from python.common.dataset_64x64 import (  # noqa: E402
    CLASSES,
    PROJECT_ROOT,
    balance_training_data,
    compute_accuracy,
    compute_confusion_matrix,
    compute_recalls,
    labels_from_predictions,
    labels_from_targets,
    load_split,
    read_folds_csv,
)
from python.common.reports import (  # noqa: E402
    ensure_directories,
    plot_confusion_matrix,
    plot_learning_curves,
    write_csv,
)

EXPERIMENT_NAME = "MLP — 1 couche cachée — 64x64"
DATASET_NAME = "dataset_v1_64x64"

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CONFUSION_DIR = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices"
CURVES_DIR = PROJECT_ROOT / "reports" / "figures" / "learning_curves"

FOLDS_REPORT = TABLES_DIR / "mlp_64x64_folds.csv"
HISTORY_REPORT = TABLES_DIR / "mlp_64x64_history.csv"
CONFUSION_FIGURE = CONFUSION_DIR / "mlp_64x64.png"
CURVES_FIGURE = CURVES_DIR / "mlp_64x64.png"

FOLDS_FIELDNAMES = [
    "experiment",
    "dataset",
    "fold",
    "hidden_sizes",
    "learning_rate",
    "epochs",
    "balanced_train",
    "train_samples",
    "validation_samples",
    "train_accuracy",
    "validation_accuracy",
    "train_loss",
    "validation_loss",
    "dog_recall",
    "cat_recall",
    "others_recall",
    "notes",
]

HISTORY_FIELDNAMES = [
    "experiment",
    "dataset",
    "fold",
    "epoch",
    "hidden_sizes",
    "learning_rate",
    "balanced_train",
    "train_samples",
    "validation_samples",
    "train_accuracy",
    "validation_accuracy",
    "train_loss",
    "validation_loss",
]


@dataclass
class RunConfig:
    learning_rate: float
    epochs: int
    eval_every: int
    hidden_sizes: list[int]
    balanced_train: bool
    seed: int


def predict_one(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_scores = np.zeros(len(CLASSES), dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_scores_array, y_scores_ptr = as_double_pointer(y_scores)

    predict_status = lib.predict_mlp_model_raw(model, x_ptr, y_scores_ptr)
    if predict_status != 0:
        raise RuntimeError(f"predict_mlp_model_raw a renvoyé {predict_status}")

    return y_scores_array.copy()


def predict_many(model: int, x: np.ndarray) -> np.ndarray:
    predictions = [
        predict_one(model, x_sample)
        for x_sample in tqdm(
            x,
            desc="Prédiction",
            unit="image",
            leave=False,
        )
    ]

    return np.asarray(predictions, dtype=np.float64)


def evaluate_model(
        model: int,
        x: np.ndarray,
        expected_targets: np.ndarray,
) -> tuple[float, list[str], list[str]]:
    predictions = predict_many(model, x)
    expected_labels = labels_from_targets(expected_targets)
    predicted_labels = labels_from_predictions(predictions)
    accuracy = compute_accuracy(expected_labels, predicted_labels)

    return accuracy, expected_labels, predicted_labels


def hidden_sizes_label(hidden_sizes: list[int]) -> str:
    return ",".join(str(size) for size in hidden_sizes)


def create_model(input_size: int, hidden_sizes: list[int]) -> int:
    hidden_sizes_array = np.ascontiguousarray(hidden_sizes, dtype=np.int32)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes_array)

    model = lib.create_mlp_model(
        int(input_size),
        len(CLASSES),
        int(hidden_sizes_array.shape[0]),
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )

    if not model:
        raise RuntimeError("Le modèle MLP n'a pas été créé.")

    return model


def train_one_epoch(
        model: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
) -> None:
    x_array, x_ptr = as_double_pointer(x_train)
    _, y_ptr = as_double_pointer(y_train)

    train_status = lib.train_mlp_model(
        model,
        x_ptr,
        y_ptr,
        x_array.shape[0],
        learning_rate,
        1,
    )

    if train_status != 0:
        raise RuntimeError(f"train_mlp_model a renvoyé {train_status}")


def run_fold(
        rows: list[dict[str, str]],
        fold: int,
        config: RunConfig,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    print(f"\n=== Fold {fold} ===")
    split_data = load_split(rows, validation_fold=fold)
    if config.balanced_train:
        split_data = balance_training_data(
            split_data=split_data,
            seed=config.seed + fold,
        )

    input_size = int(split_data.x_train.shape[1])
    model = create_model(input_size, config.hidden_sizes)
    hidden_label = hidden_sizes_label(config.hidden_sizes)

    history_rows: list[dict[str, object]] = []

    try:
        for epoch in tqdm(
                range(1, config.epochs + 1),
                desc=f"Entraînement MLP — fold {fold}",
                unit="epoch",
        ):
            train_one_epoch(
                model=model,
                x_train=split_data.x_train,
                y_train=split_data.y_train,
                learning_rate=config.learning_rate,
            )

            if epoch % config.eval_every == 0 or epoch == config.epochs:
                train_accuracy, _, _ = evaluate_model(
                    model,
                    split_data.x_train,
                    split_data.y_train,
                )
                validation_accuracy, _, _ = evaluate_model(
                    model,
                    split_data.x_validation,
                    split_data.y_validation,
                )

                history_rows.append(
                    {
                        "experiment": EXPERIMENT_NAME,
                        "dataset": DATASET_NAME,
                        "fold": fold,
                        "epoch": epoch,
                        "hidden_sizes": hidden_label,
                        "learning_rate": config.learning_rate,
                        "balanced_train": config.balanced_train,
                        "train_samples": int(split_data.x_train.shape[0]),
                        "validation_samples": int(split_data.x_validation.shape[0]),
                        "train_accuracy": train_accuracy,
                        "validation_accuracy": validation_accuracy,
                        "train_loss": 1.0 - train_accuracy,
                        "validation_loss": 1.0 - validation_accuracy,
                    }
                )

                print(
                    f"epoch={epoch} "
                    f"train_accuracy={train_accuracy:.4f} "
                    f"validation_accuracy={validation_accuracy:.4f}"
                )

        final_train_accuracy, _, _ = evaluate_model(
            model,
            split_data.x_train,
            split_data.y_train,
        )
        final_validation_accuracy, expected_labels, predicted_labels = evaluate_model(
            model,
            split_data.x_validation,
            split_data.y_validation,
        )

        confusion_matrix = compute_confusion_matrix(
            expected_labels=expected_labels,
            predicted_labels=predicted_labels,
        )

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))

    recalls = compute_recalls(confusion_matrix)

    fold_row = {
        "experiment": EXPERIMENT_NAME,
        "dataset": DATASET_NAME,
        "fold": fold,
        "hidden_sizes": hidden_label,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "balanced_train": config.balanced_train,
        "train_samples": int(split_data.x_train.shape[0]),
        "validation_samples": int(split_data.x_validation.shape[0]),
        "train_accuracy": final_train_accuracy,
        "validation_accuracy": final_validation_accuracy,
        "train_loss": 1.0 - final_train_accuracy,
        "validation_loss": 1.0 - final_validation_accuracy,
        "dog_recall": recalls["dog_recall"],
        "cat_recall": recalls["cat_recall"],
        "others_recall": recalls["others_recall"],
        "notes": (
                     "train équilibré par sous-échantillonnage; "
                     if config.balanced_train
                     else ""
                 )
                 + (
                     "loss=1-accuracy car l’API MLP actuelle expose une prédiction "
                     "bipolaire, pas une fonction de perte continue."
                 ),
    }

    return fold_row, history_rows, confusion_matrix


def parse_hidden_sizes(raw_value: str) -> list[int]:
    if raw_value is None:
        raise ValueError("hidden_sizes est obligatoire.")

    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(
            "hidden_sizes doit contenir au moins une taille, par exemple 128 ou 128,64."
        )

    hidden_sizes = [int(part) for part in parts]

    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Chaque taille cachée doit être strictement positive.")

    return hidden_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute le MLP sur dataset_v1_64x64."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Taux d'apprentissage utilisé par train_mlp_model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Nombre d'époques par pli.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Fréquence d'évaluation des métriques.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Pli à exécuter: all, ou un entier entre 0 et 4.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        required=True,
        help="Tailles des couches cachées, par exemple 128 ou 128,64.",
    )
    parser.add_argument(
        "--balanced-train",
        action="store_true",
        help=(
            "Sous-échantillonne le train de chaque pli pour avoir autant "
            "d’exemples dog, cat et others."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine utilisée pour le sous-échantillonnage équilibré.",
    )

    return parser.parse_args()


def select_folds(fold_argument: str) -> list[int]:
    if fold_argument == "all":
        return [0, 1, 2, 3, 4]

    fold = int(fold_argument)
    if fold < 0 or fold > 4:
        raise ValueError("Le pli doit être compris entre 0 et 4.")

    return [fold]


def validate_run_config(config: RunConfig) -> None:
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate doit être strictement positif.")

    if config.epochs <= 0:
        raise ValueError("epochs doit être strictement positif.")

    if config.eval_every <= 0:
        raise ValueError("eval_every doit être strictement positif.")

    if not config.hidden_sizes:
        raise ValueError("hidden_sizes ne doit pas être vide.")

    if any(size <= 0 for size in config.hidden_sizes):
        raise ValueError("Chaque taille cachée doit être strictement positive.")


def main() -> None:
    args = parse_args()

    config = RunConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        eval_every=args.eval_every,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        balanced_train=args.balanced_train,
        seed=args.seed,
    )
    validate_run_config(config)

    ensure_directories(TABLES_DIR, CONFUSION_DIR, CURVES_DIR)

    rows = read_folds_csv()
    folds = select_folds(args.fold)

    fold_rows = []
    history_rows = []
    total_confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

    for fold in tqdm(folds, desc="Validation croisée MLP", unit="fold"):
        fold_row, fold_history_rows, fold_confusion_matrix = run_fold(
            rows=rows,
            fold=fold,
            config=config,
        )

        fold_rows.append(fold_row)
        history_rows.extend(fold_history_rows)
        total_confusion_matrix += fold_confusion_matrix

    write_csv(FOLDS_REPORT, fold_rows, fieldnames=FOLDS_FIELDNAMES)
    write_csv(HISTORY_REPORT, history_rows, fieldnames=HISTORY_FIELDNAMES)
    plot_learning_curves(history_rows, CURVES_FIGURE, "MLP — 64x64")
    plot_confusion_matrix(
        total_confusion_matrix,
        CONFUSION_FIGURE,
        "MLP — matrice de confusion — validation croisée",
    )

    print("\nArtefacts écrits:")
    print(f"- {FOLDS_REPORT}")
    print(f"- {HISTORY_REPORT}")
    print(f"- {CONFUSION_FIGURE}")
    print(f"- {CURVES_FIGURE}")


if __name__ == "__main__":
    main()