import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSES = ["dog", "cat", "others"]

LABEL_TO_TARGET = {
    "dog": np.array([1.0, -1.0, -1.0], dtype=np.float64),
    "cat": np.array([-1.0, 1.0, -1.0], dtype=np.float64),
    "others": np.array([-1.0, -1.0, 1.0], dtype=np.float64),
}

INPUT_WIDTH = 64
INPUT_HEIGHT = 64
INPUT_CHANNELS = 3
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS

FOLDS_CSV = PROJECT_ROOT / "data" / "splits" / "folds.csv"


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


def read_folds_csv(path: Path = FOLDS_CSV) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    expected_columns = {"path", "label", "fold"}
    missing_columns = expected_columns - fieldnames
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing_columns)}")

    return rows


def load_image_as_vector(relative_path: str) -> np.ndarray:
    image_path = PROJECT_ROOT / relative_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float64)

    if array.shape != (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS):
        raise ValueError(
            "Dimension d'image inattendue: "
            f"{array.shape}, attendu {(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)}"
        )

    return np.ascontiguousarray(array.reshape(-1) / 255.0, dtype=np.float64)


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

    if split_data.x_train.shape[1] != INPUT_SIZE:
        raise ValueError(
            "Dimension d'entrée inattendue pour x_train: "
            f"{split_data.x_train.shape[1]} au lieu de {INPUT_SIZE}."
        )

    if split_data.x_validation.shape[1] != INPUT_SIZE:
        raise ValueError(
            "Dimension d'entrée inattendue pour x_validation: "
            f"{split_data.x_validation.shape[1]} au lieu de {INPUT_SIZE}."
        )


def balance_training_data(split_data: SplitData, seed: int) -> SplitData:
    train_class_indices = np.argmax(split_data.y_train, axis=1)
    per_class_counts = [
        int(np.sum(train_class_indices == class_index))
        for class_index in range(len(CLASSES))
    ]

    if any(count == 0 for count in per_class_counts):
        raise ValueError(
            "Impossible d'équilibrer l'entraînement: au moins une classe est absente."
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
        x_train=np.ascontiguousarray(split_data.x_train[selected_array], dtype=np.float64),
        y_train=np.ascontiguousarray(split_data.y_train[selected_array], dtype=np.float64),
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
        raise ValueError("Impossible de calculer l'accuracy sans exemples.")

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