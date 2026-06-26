import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    LINEAR_TASK_CLASSIFICATION,
    as_double_pointer,
    lib,
)
from python.common.dataset_64x64 import (  # noqa: E402
    CLASSES,
    PROJECT_ROOT,
    compute_accuracy,
    compute_confusion_matrix,
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

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CONFUSION_DIR = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices"
CURVES_DIR = PROJECT_ROOT / "reports" / "figures" / "learning_curves"

FOLDS_REPORT = TABLES_DIR / "linear_baseline_64x64_folds.csv"
HISTORY_REPORT = TABLES_DIR / "linear_baseline_64x64_history.csv"
CONFUSION_FIGURE = CONFUSION_DIR / "linear_baseline_64x64.png"
CURVES_FIGURE = CURVES_DIR / "linear_baseline_64x64.png"

FOLDS_FIELDNAMES = [
    "experiment",
    "dataset",
    "fold",
    "train_accuracy",
    "validation_accuracy",
    "train_loss",
    "validation_loss",
    "notes",
]

HISTORY_FIELDNAMES = [
    "experiment",
    "dataset",
    "fold",
    "epoch",
    "train_accuracy",
    "validation_accuracy",
    "train_loss",
    "validation_loss",
]


def predict_one(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(len(CLASSES), dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_linear_model(model, x_ptr, y_pred_ptr)
    if predict_status != 0:
        raise RuntimeError(f"predict_linear_model a renvoyé {predict_status}")

    return y_pred_array.copy()


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


def create_model(input_size: int) -> int:
    model = lib.create_linear_model(
        input_size,
        len(CLASSES),
        LINEAR_TASK_CLASSIFICATION,
    )

    if not model:
        raise RuntimeError("Le modèle linéaire n'a pas été créé.")

    return model


def train_one_epoch(
        model: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
) -> None:
    x_array, x_ptr = as_double_pointer(x_train)
    _, y_ptr = as_double_pointer(y_train)

    train_status = lib.train_linear_model(
        model,
        x_ptr,
        y_ptr,
        x_array.shape[0],
        learning_rate,
        1,
    )

    if train_status != 0:
        raise RuntimeError(f"train_linear_model a renvoyé {train_status}")


def run_fold(
        rows: list[dict[str, str]],
        fold: int,
        learning_rate: float,
        epochs: int,
        eval_every: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    print(f"\n=== Fold {fold} ===")
    split_data = load_split(rows, validation_fold=fold)

    input_size = int(split_data.x_train.shape[1])
    model = create_model(input_size)

    history_rows: list[dict[str, object]] = []

    try:
        for epoch in tqdm(
                range(1, epochs + 1),
                desc=f"Entraînement — fold {fold}",
                unit="epoch",
        ):
            train_one_epoch(
                model=model,
                x_train=split_data.x_train,
                y_train=split_data.y_train,
                learning_rate=learning_rate,
            )

            if epoch % eval_every == 0 or epoch == epochs:
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
                        "experiment": "Linéaire — baseline — 64x64",
                        "dataset": "dataset_v1_64x64",
                        "fold": fold,
                        "epoch": epoch,
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
        lib.destroy_linear_model(ctypes.c_void_p(model))

    fold_row = {
        "experiment": "Linéaire — baseline — 64x64",
        "dataset": "dataset_v1_64x64",
        "fold": fold,
        "train_accuracy": final_train_accuracy,
        "validation_accuracy": final_validation_accuracy,
        "train_loss": 1.0 - final_train_accuracy,
        "validation_loss": 1.0 - final_validation_accuracy,
        "notes": (
            "loss=1-accuracy car l'API actuelle expose la prédiction bipolaire, "
            "pas les scores bruts ni une fonction de perte."
        ),
    }

    return fold_row, history_rows, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute la baseline du modèle linéaire sur dataset_v1_64x64."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage utilisé par train_linear_model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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

    return parser.parse_args()


def select_folds(fold_argument: str) -> list[int]:
    if fold_argument == "all":
        return [0, 1, 2, 3, 4]

    fold = int(fold_argument)
    if fold < 0 or fold > 4:
        raise ValueError("Le pli doit être compris entre 0 et 4.")

    return [fold]


def main() -> None:
    args = parse_args()

    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate doit être strictement positif.")

    if args.epochs <= 0:
        raise ValueError("epochs doit être strictement positif.")

    if args.eval_every <= 0:
        raise ValueError("eval_every doit être strictement positif.")

    ensure_directories(TABLES_DIR, CONFUSION_DIR, CURVES_DIR)

    rows = read_folds_csv()
    folds = select_folds(args.fold)

    fold_rows = []
    history_rows = []
    total_confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

    for fold in tqdm(folds, desc="Validation croisée", unit="fold"):
        fold_row, fold_history_rows, fold_confusion_matrix = run_fold(
            rows=rows,
            fold=fold,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            eval_every=args.eval_every,
        )

        fold_rows.append(fold_row)
        history_rows.extend(fold_history_rows)
        total_confusion_matrix += fold_confusion_matrix

    write_csv(FOLDS_REPORT, fold_rows, fieldnames=FOLDS_FIELDNAMES)
    write_csv(HISTORY_REPORT, history_rows, fieldnames=HISTORY_FIELDNAMES)
    plot_learning_curves(
        history_rows,
        CURVES_FIGURE,
        "Modèle linéaire — baseline 64x64",
    )
    plot_confusion_matrix(
        total_confusion_matrix,
        CONFUSION_FIGURE,
        "Matrice de confusion — validation croisée",
    )

    print("\nArtefacts écrits:")
    print(f"- {FOLDS_REPORT}")
    print(f"- {HISTORY_REPORT}")
    print(f"- {CONFUSION_FIGURE}")
    print(f"- {CURVES_FIGURE}")


if __name__ == "__main__":
    main()