import argparse
import csv
import ctypes
import random
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
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    lib,
)

CLASSES = ["dog", "cat", "others"]

LABEL_TO_TARGET = {
    "dog": np.array([1.0, -1.0, -1.0], dtype=np.float64),
    "cat": np.array([-1.0, 1.0, -1.0], dtype=np.float64),
    "others": np.array([-1.0, -1.0, 1.0], dtype=np.float64),
}

EXPERIMENT_NAME = "MLP — 1 couche cachée — 64x64"
DATASET_NAME = "dataset_v1_64x64"

FOLDS_CSV = PROJECT_ROOT / "data" / "splits" / "folds.csv"

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CONFUSION_DIR = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices"
CURVES_DIR = PROJECT_ROOT / "reports" / "figures" / "learning_curves"

FOLDS_REPORT = TABLES_DIR / "mlp_64x64_folds.csv"
HISTORY_REPORT = TABLES_DIR / "mlp_64x64_history.csv"
CONFUSION_FIGURE = CONFUSION_DIR / "mlp_64x64.png"
CURVES_FIGURE = CURVES_DIR / "mlp_64x64.png"


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


@dataclass
class RunConfig:
    learning_rate: float
    epochs: int
    eval_every: int
    hidden_sizes: list[int]
    balanced_train: bool
    seed: int


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
        fieldnames = set(reader.fieldnames or [])

    expected_columns = {"path", "label", "fold"}
    missing_columns = expected_columns - fieldnames
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

    split_data = SplitData(
        x_train=np.ascontiguousarray(train_vectors, dtype=np.float64),
        y_train=np.ascontiguousarray(train_targets, dtype=np.float64),
        x_validation=np.ascontiguousarray(validation_vectors, dtype=np.float64),
        y_validation=np.ascontiguousarray(validation_targets, dtype=np.float64),
    )

    validate_split_shapes(split_data)

    return split_data


def validate_split_shapes(split_data: SplitData) -> None:
    expected_input_size = 64 * 64 * 3

    if split_data.x_train.ndim != 2:
        raise ValueError("x_train doit être une matrice 2D.")

    if split_data.x_validation.ndim != 2:
        raise ValueError("x_validation doit être une matrice 2D.")

    if split_data.y_train.ndim != 2 or split_data.y_train.shape[1] != len(CLASSES):
        raise ValueError("y_train doit avoir trois sorties.")

    if (
            split_data.y_validation.ndim != 2
            or split_data.y_validation.shape[1] != len(CLASSES)
    ):
        raise ValueError("y_validation doit avoir trois sorties.")

    if split_data.x_train.shape[1] != expected_input_size:
        raise ValueError(
            "Dimension d’entrée inattendue pour x_train: "
            f"{split_data.x_train.shape[1]} au lieu de {expected_input_size}."
        )

    if split_data.x_validation.shape[1] != expected_input_size:
        raise ValueError(
            "Dimension d’entrée inattendue pour x_validation: "
            f"{split_data.x_validation.shape[1]} au lieu de {expected_input_size}."
        )


def balance_training_data(split_data: SplitData, seed: int) -> SplitData:
    train_class_indices = np.argmax(split_data.y_train, axis=1)
    per_class_counts = [
        int(np.sum(train_class_indices == class_index))
        for class_index in range(len(CLASSES))
    ]

    if any(count == 0 for count in per_class_counts):
        raise ValueError(
            "Impossible d’équilibrer l’entraînement: au moins une classe est absente."
        )

    target_count = min(per_class_counts)
    rng = random.Random(seed)
    selected_indices: list[int] = []

    for class_index in range(len(CLASSES)):
        class_indices = np.where(train_class_indices == class_index)[0].tolist()
        selected_indices.extend(rng.sample(class_indices, target_count))

    rng.shuffle(selected_indices)
    selected_array = np.asarray(selected_indices, dtype=np.int64)

    print(
        "Entraînement équilibré activé: "
        + ", ".join(f"{class_name}={target_count}" for class_name in CLASSES)
    )

    return SplitData(
        x_train=np.ascontiguousarray(
            split_data.x_train[selected_array],
            dtype=np.float64,
        ),
        y_train=np.ascontiguousarray(
            split_data.y_train[selected_array],
            dtype=np.float64,
        ),
        x_validation=split_data.x_validation,
        y_validation=split_data.y_validation,
    )


def labels_from_targets(targets: np.ndarray) -> list[str]:
    indices = np.argmax(targets, axis=1)
    return [CLASSES[index] for index in indices]


def labels_from_predictions(predictions: np.ndarray) -> list[str]:
    indices = np.argmax(predictions, axis=1)
    return [CLASSES[index] for index in indices]


def compute_accuracy(expected_labels: list[str], predicted_labels: list[str]) -> float:
    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Les listes de labels n'ont pas la même taille.")

    if not expected_labels:
        raise ValueError("Impossible de calculer l’accuracy sans exemples.")

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


def compute_recalls(confusion_matrix: np.ndarray) -> dict[str, float]:
    recalls: dict[str, float] = {}

    for class_index, class_name in enumerate(CLASSES):
        row_total = int(confusion_matrix[class_index].sum())
        if row_total == 0:
            recalls[f"{class_name}_recall"] = 0.0
        else:
            recalls[f"{class_name}_recall"] = (
                    float(confusion_matrix[class_index, class_index]) / row_total
            )

    return recalls


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
                 ) + (
                     "loss=1-accuracy car l’API MLP actuelle expose une prédiction "
                     "bipolaire, pas une fonction de perte continue."
                 ),
    }

    return fold_row, history_rows, confusion_matrix


def write_folds_report(rows: list[dict[str, object]]) -> None:
    with FOLDS_REPORT.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
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

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def plot_learning_curves(history_rows: list[dict[str, object]]) -> None:
    if not history_rows:
        raise ValueError("Aucun historique disponible pour tracer les courbes.")

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

    plt.title("MLP — 64x64")
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
    plt.title("MLP — matrice de confusion — validation croisée")
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

    ensure_output_directories()

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