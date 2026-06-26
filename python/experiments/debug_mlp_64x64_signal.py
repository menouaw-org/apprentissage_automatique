import argparse
import ctypes
import random
import sys
from collections import Counter
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
    compute_accuracy,
    encode_label,
    labels_from_predictions,
    load_image_as_vector,
    read_folds_csv,
)
from python.common.reports import write_csv  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
SIGNAL_PROBE_CSV = TABLES_DIR / "mlp_64x64_signal_probe.csv"
MINI_HISTORY_CSV = TABLES_DIR / "mlp_64x64_mini_balanced_history.csv"


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    labels: list[str]
    paths: list[str]


@dataclass
class RunConfig:
    fold: int
    per_class: int
    probe_per_class: int
    epochs: int
    learning_rate: float
    hidden_sizes: list[int]
    seed: int


def sample_balanced_rows(
        rows: list[dict[str, str]],
        per_class: int,
        seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected_rows: list[dict[str, str]] = []

    for class_name in CLASSES:
        class_rows = [row for row in rows if row["label"] == class_name]
        if len(class_rows) < per_class:
            raise ValueError(
                f"Pas assez d’exemples pour {class_name}: "
                f"{len(class_rows)} disponibles, {per_class} demandés."
            )

        selected_rows.extend(rng.sample(class_rows, per_class))

    rng.shuffle(selected_rows)
    return selected_rows


def load_dataset(rows: list[dict[str, str]], description: str) -> Dataset:
    vectors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    for row in tqdm(rows, desc=f"Chargement — {description}", unit="image"):
        vectors.append(load_image_as_vector(row["path"]))
        targets.append(encode_label(row["label"]))
        labels.append(row["label"])
        paths.append(row["path"])

    x = np.ascontiguousarray(vectors, dtype=np.float64)
    y = np.ascontiguousarray(targets, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Dimension d’entrée inattendue: {x.shape}")

    if y.ndim != 2 or y.shape[1] != len(CLASSES):
        raise ValueError(f"Dimension de sortie inattendue: {y.shape}")

    return Dataset(x=x, y=y, labels=labels, paths=paths)


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
        raise RuntimeError("Le modèle MLP n’a pas été créé.")

    return model


def train_one_epoch(
        model: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
) -> None:
    x_array, x_ptr = as_double_pointer(x_train)
    _, y_ptr = as_double_pointer(y_train)

    status = lib.train_mlp_model(
        model,
        x_ptr,
        y_ptr,
        x_array.shape[0],
        learning_rate,
        1,
    )

    if status != 0:
        raise RuntimeError(f"train_mlp_model a renvoyé {status}")


def predict_one(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(len(CLASSES), dtype=np.float64)
    _, x_ptr = as_double_pointer(np.ascontiguousarray(x_sample, dtype=np.float64))
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
    if status != 0:
        raise RuntimeError(f"predict_mlp_model a renvoyé {status}")

    return y_pred_array.copy()


def predict_many(model: int, x: np.ndarray) -> np.ndarray:
    predictions = [predict_one(model, sample) for sample in x]
    return np.ascontiguousarray(predictions, dtype=np.float64)


def predict_one_raw(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_raw = np.zeros(len(CLASSES), dtype=np.float64)
    _, x_ptr = as_double_pointer(np.ascontiguousarray(x_sample, dtype=np.float64))
    y_raw_array, y_raw_ptr = as_double_pointer(y_raw)

    status = lib.predict_mlp_model_raw(model, x_ptr, y_raw_ptr)
    if status != 0:
        raise RuntimeError(f"predict_mlp_model_raw a renvoyé {status}")

    return y_raw_array.copy()


def predict_many_raw(model: int, x: np.ndarray) -> np.ndarray:
    predictions = [predict_one_raw(model, sample) for sample in x]
    return np.ascontiguousarray(predictions, dtype=np.float64)


def compute_recalls(
        expected_labels: list[str],
        predicted_labels: list[str],
) -> dict[str, float]:
    recalls: dict[str, float] = {}

    for class_name in CLASSES:
        class_total = sum(label == class_name for label in expected_labels)
        if class_total == 0:
            recalls[f"{class_name}_recall"] = 0.0
            continue

        true_positive = sum(
            expected == class_name and predicted == class_name
            for expected, predicted in zip(expected_labels, predicted_labels)
        )
        recalls[f"{class_name}_recall"] = true_positive / class_total

    return recalls


def summarize_predictions(
        epoch: int,
        split_name: str,
        dataset: Dataset,
        raw_outputs: np.ndarray,
) -> dict[str, object]:
    predicted_labels = labels_from_predictions(raw_outputs)
    accuracy = compute_accuracy(dataset.labels, predicted_labels)
    recalls = compute_recalls(dataset.labels, predicted_labels)
    prediction_counts = Counter(predicted_labels)

    summary: dict[str, object] = {
        "epoch": epoch,
        "split": split_name,
        "accuracy": accuracy,
        "dog_pred_count": prediction_counts.get("dog", 0),
        "cat_pred_count": prediction_counts.get("cat", 0),
        "others_pred_count": prediction_counts.get("others", 0),
        "dog_recall": recalls["dog_recall"],
        "cat_recall": recalls["cat_recall"],
        "others_recall": recalls["others_recall"],
    }

    for real_class in CLASSES:
        class_indices = [
            index for index, label in enumerate(dataset.labels) if label == real_class
        ]
        if not class_indices:
            for output_class in CLASSES:
                summary[f"mean_{output_class}_score_on_{real_class}"] = ""
            continue

        class_raw_outputs = raw_outputs[class_indices]
        mean_scores = class_raw_outputs.mean(axis=0)
        for output_index, output_class in enumerate(CLASSES):
            summary[f"mean_{output_class}_score_on_{real_class}"] = float(
                mean_scores[output_index]
            )

    return summary


def build_probe_rows(
        epoch: int,
        split_name: str,
        dataset: Dataset,
        raw_outputs: np.ndarray,
        binary_predictions: np.ndarray,
) -> list[dict[str, object]]:
    predicted_labels = labels_from_predictions(raw_outputs)
    rows: list[dict[str, object]] = []

    for index, (
            path,
            expected_label,
            predicted_label,
            raw_scores,
            binary_scores,
    ) in enumerate(
        zip(
            dataset.paths,
            dataset.labels,
            predicted_labels,
            raw_outputs,
            binary_predictions,
        )
    ):
        rows.append(
            {
                "epoch": epoch,
                "split": split_name,
                "sample_index": index,
                "path": path,
                "label": expected_label,
                "predicted": predicted_label,
                "dog_raw": float(raw_scores[0]),
                "cat_raw": float(raw_scores[1]),
                "others_raw": float(raw_scores[2]),
                "dog_pred_score": float(binary_scores[0]),
                "cat_pred_score": float(binary_scores[1]),
                "others_pred_score": float(binary_scores[2]),
            }
        )

    return rows


def parse_hidden_sizes(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("hidden_sizes doit contenir au moins une taille.")

    hidden_sizes = [int(part) for part in parts]
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Chaque taille cachée doit être strictement positive.")

    return hidden_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostique le signal MLP sur un mini-jeu équilibré 64x64."
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--probe-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--hidden-sizes", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_config(config: RunConfig) -> None:
    if config.fold < 0 or config.fold > 4:
        raise ValueError("fold doit être compris entre 0 et 4.")
    if config.per_class <= 0:
        raise ValueError("per_class doit être strictement positif.")
    if config.probe_per_class <= 0:
        raise ValueError("probe_per_class doit être strictement positif.")
    if config.epochs <= 0:
        raise ValueError("epochs doit être strictement positif.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate doit être strictement positif.")
    if not config.hidden_sizes:
        raise ValueError("hidden_sizes ne doit pas être vide.")


def main() -> None:
    args = parse_args()
    config = RunConfig(
        fold=args.fold,
        per_class=args.per_class,
        probe_per_class=args.probe_per_class,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        seed=args.seed,
    )
    validate_config(config)

    rows = read_folds_csv()
    train_pool = [row for row in rows if int(row["fold"]) != config.fold]
    validation_pool = [row for row in rows if int(row["fold"]) == config.fold]

    mini_train_rows = sample_balanced_rows(
        train_pool,
        per_class=config.per_class,
        seed=config.seed,
    )
    probe_rows = sample_balanced_rows(
        validation_pool,
        per_class=config.probe_per_class,
        seed=config.seed,
    )

    mini_train = load_dataset(mini_train_rows, description="mini train équilibré")
    probe = load_dataset(probe_rows, description="sonde validation équilibrée")

    model = create_model(
        input_size=mini_train.x.shape[1],
        hidden_sizes=config.hidden_sizes,
    )

    summary_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    try:
        for epoch in range(0, config.epochs + 1):
            if epoch > 0:
                train_one_epoch(
                    model=model,
                    x_train=mini_train.x,
                    y_train=mini_train.y,
                    learning_rate=config.learning_rate,
                )

            for split_name, dataset in [("mini_train", mini_train), ("probe", probe)]:
                raw_outputs = predict_many_raw(model, dataset.x)
                binary_predictions = predict_many(model, dataset.x)

                summary_rows.append(
                    summarize_predictions(
                        epoch=epoch,
                        split_name=split_name,
                        dataset=dataset,
                        raw_outputs=raw_outputs,
                    )
                )
                signal_rows.extend(
                    build_probe_rows(
                        epoch=epoch,
                        split_name=split_name,
                        dataset=dataset,
                        raw_outputs=raw_outputs,
                        binary_predictions=binary_predictions,
                    )
                )

            last_train = summary_rows[-2]
            last_probe = summary_rows[-1]
            print(
                f"epoch={epoch} "
                f"mini_train_accuracy={last_train['accuracy']:.4f} "
                f"probe_accuracy={last_probe['accuracy']:.4f} "
                f"probe_pred_counts="
                f"dog:{last_probe['dog_pred_count']} "
                f"cat:{last_probe['cat_pred_count']} "
                f"others:{last_probe['others_pred_count']}"
            )

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))

    write_csv(MINI_HISTORY_CSV, summary_rows)
    write_csv(SIGNAL_PROBE_CSV, signal_rows)

    print("\nArtefacts écrits:")
    print(f"- {MINI_HISTORY_CSV}")
    print(f"- {SIGNAL_PROBE_CSV}")


if __name__ == "__main__":
    main()