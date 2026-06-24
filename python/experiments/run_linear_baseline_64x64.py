import argparse
import csv
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    LINEAR_TASK_CLASSIFICATION,
    as_double_pointer,
    lib,
)

CLASSES = ["dog", "cat", "others"]

LABEL_TO_TARGET = {
    "dog": np.array([1.0, -1.0, -1.0], dtype=np.float64),
    "cat": np.array([-1.0, 1.0, -1.0], dtype=np.float64),
    "others": np.array([-1.0, -1.0, 1.0], dtype=np.float64),
}

FOLDS_CSV = PROJECT_ROOT / "data" / "splits" / "folds.csv"

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CONFUSION_DIR = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices"
CURVES_DIR = PROJECT_ROOT / "reports" / "figures" / "learning_curves"

FOLDS_REPORT = TABLES_DIR / "linear_baseline_64x64_folds.csv"
HISTORY_REPORT = TABLES_DIR / "linear_baseline_64x64_history.csv"
CONFUSION_FIGURE = CONFUSION_DIR / "linear_baseline_64x64.png"
CURVES_FIGURE = CURVES_DIR / "linear_baseline_64x64.png"


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


def ensure_output_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)


def read_folds_csv() -> list[dict[str, str]]:
    if not FOLDS_CSV.exists():
        raise FileNotFoundError(f"Fichier introuvable: {FOLDS_CSV}")

    with FOLDS_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    expected_columns = {"path", "label", "fold"}
    missing_columns = expected_columns - set(reader.fieldnames or [])
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {FOLDS_CSV}: {sorted(missing_columns)}"
        )

    return rows


def load_image_as_vector(relative_path: str) -> np.ndarray:
    image_path = PROJECT_ROOT / relative_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float64)

    return (array.reshape(-1) / 255.0).astype(np.float64)


def encode_label(label: str) -> np.ndarray:
    if label not in LABEL_TO_TARGET:
        raise ValueError(f"Label inconnu: {label}")

    return LABEL_TO_TARGET[label].copy()


def load_split(rows: list[dict[str, str]], validation_fold: int) -> SplitData:
    train_vectors = []
    train_targets = []
    validation_vectors = []
    validation_targets = []

    for row in tqdm(
            rows,
            desc=f"Chargement des images — fold {validation_fold}",
            unit="image",
    ):
        row_fold = int(row["fold"])
        label = row["label"]

        image_vector = load_image_as_vector(row["path"])
        target = encode_label(label)

        if row_fold == validation_fold:
            validation_vectors.append(image_vector)
            validation_targets.append(target)
        else:
            train_vectors.append(image_vector)
            train_targets.append(target)

    return SplitData(
        x_train=np.ascontiguousarray(train_vectors, dtype=np.float64),
        y_train=np.ascontiguousarray(train_targets, dtype=np.float64),
        x_validation=np.ascontiguousarray(validation_vectors, dtype=np.float64),
        y_validation=np.ascontiguousarray(validation_targets, dtype=np.float64),
    )


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


def labels_from_targets(targets: np.ndarray) -> list[str]:
    indices = np.argmax(targets, axis=1)
    return [CLASSES[index] for index in indices]


def labels_from_predictions(predictions: np.ndarray) -> list[str]:
    indices = np.argmax(predictions, axis=1)
    return [CLASSES[index] for index in indices]


def compute_accuracy(expected_labels: list[str], predicted_labels: list[str]) -> float:
    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Les listes de labels n’ont pas la même taille.")

    correct_count = sum(
        expected == predicted
        for expected, predicted in zip(expected_labels, predicted_labels)
    )

    return correct_count / len(expected_labels)


def compute_confusion_matrix(
        expected_labels: list[str],
        predicted_labels: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    class_to_index = {label: index for index, label in enumerate(CLASSES)}

    for expected, predicted in zip(expected_labels, predicted_labels):
        expected_index = class_to_index[expected]
        predicted_index = class_to_index[predicted]
        matrix[expected_index, predicted_index] += 1

    return matrix


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
        raise RuntimeError("Le modèle linéaire n’a pas été créé.")

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
            "loss=1-accuracy car l’API actuelle expose la prédiction bipolaire, "
            "pas les scores bruts ni une fonction de perte."
        ),
    }

    return fold_row, history_rows, confusion_matrix


def write_folds_report(rows: list[dict[str, object]]) -> None:
    with FOLDS_REPORT.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "experiment",
            "dataset",
            "fold",
            "train_accuracy",
            "validation_accuracy",
            "train_loss",
            "validation_loss",
            "notes",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_history_report(rows: list[dict[str, object]]) -> None:
    with HISTORY_REPORT.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "experiment",
            "dataset",
            "fold",
            "epoch",
            "train_accuracy",
            "validation_accuracy",
            "train_loss",
            "validation_loss",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def plot_learning_curves(history_rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10, 6))

    folds = sorted({int(row["fold"]) for row in history_rows})

    for fold in folds:
        fold_rows = [row for row in history_rows if int(row["fold"]) == fold]
        epochs = [int(row["epoch"]) for row in fold_rows]
        train_values = [float(row["train_accuracy"]) for row in fold_rows]
        validation_values = [float(row["validation_accuracy"]) for row in fold_rows]

        plt.plot(
            epochs,
            train_values,
            linestyle="--",
            marker="o",
            alpha=0.6,
            label=f"Fold {fold} train",
        )
        plt.plot(
            epochs,
            validation_values,
            linestyle="-",
            marker="o",
            alpha=0.9,
            label=f"Fold {fold} validation",
        )

    plt.title("Modèle linéaire — baseline 64x64")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(CURVES_FIGURE, dpi=150)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Matrice de confusion — validation croisée")
    plt.colorbar()

    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES)
    plt.yticks(tick_marks, CLASSES)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > threshold else "black"

            plt.text(
                column_index,
                row_index,
                str(value),
                horizontalalignment="center",
                color=color,
            )

    plt.ylabel("Classe réelle")
    plt.xlabel("Classe prédite")
    plt.tight_layout()
    plt.savefig(CONFUSION_FIGURE, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute la baseline du modèle linéaire sur dataset_v1_64x64."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d’apprentissage utilisé par train_linear_model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d’époques par pli.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Fréquence d’évaluation des métriques.",
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

    ensure_output_directories()

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

    write_folds_report(fold_rows)
    write_history_report(history_rows)
    plot_learning_curves(history_rows)
    plot_confusion_matrix(total_confusion_matrix)

    print("\nArtefacts écrits:")
    print(f"- {FOLDS_REPORT}")
    print(f"- {HISTORY_REPORT}")
    print(f"- {CONFUSION_FIGURE}")
    print(f"- {CURVES_FIGURE}")


if __name__ == "__main__":
    main()