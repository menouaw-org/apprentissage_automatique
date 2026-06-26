import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    LINEAR_TASK_CLASSIFICATION,
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    encode_path,
    lib,
)
from python.common.dataset_64x64 import (  # noqa: E402
    CLASSES,
    INPUT_SIZE,
    balance_training_data,
    labels_from_predictions,
    load_image_as_vector,
    load_split,
    read_folds_csv,
)

LINEAR_MODEL_PATH = PROJECT_ROOT / "models" / "linear" / "linear_64x64_demo.pa_model"
MLP_MODEL_PATH = PROJECT_ROOT / "models" / "mlp" / "mlp_64x64_demo.pa_model"
DEFAULT_CONTROL_IMAGE = "data/processed/64x64/cat/cat_00007.jpg"


def parse_hidden_sizes(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("hidden_sizes doit contenir au moins une taille.")

    hidden_sizes = [int(part) for part in parts]
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Chaque taille cachée doit être strictement positive.")

    return hidden_sizes


def train_linear_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
        epochs: int,
) -> int:
    model = lib.create_linear_model(
        INPUT_SIZE,
        len(CLASSES),
        LINEAR_TASK_CLASSIFICATION,
    )
    if not model:
        raise RuntimeError("Le modèle linéaire n’a pas été créé.")

    x_array, x_ptr = as_double_pointer(x_train)
    _, y_ptr = as_double_pointer(y_train)

    train_status = lib.train_linear_model(
        model,
        x_ptr,
        y_ptr,
        int(x_array.shape[0]),
        learning_rate,
        epochs,
    )
    if train_status != 0:
        lib.destroy_linear_model(ctypes.c_void_p(model))
        raise RuntimeError(f"train_linear_model a renvoyé {train_status}.")

    return model


def train_mlp_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
        epochs: int,
        hidden_sizes: list[int],
) -> int:
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(
        np.asarray(hidden_sizes, dtype=np.int32)
    )

    model = lib.create_mlp_model(
        INPUT_SIZE,
        len(CLASSES),
        int(hidden_sizes_array.shape[0]),
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    if not model:
        raise RuntimeError("Le modèle MLP n’a pas été créé.")

    x_array, x_ptr = as_double_pointer(x_train)
    _, y_ptr = as_double_pointer(y_train)

    train_status = lib.train_mlp_model(
        model,
        x_ptr,
        y_ptr,
        int(x_array.shape[0]),
        learning_rate,
        epochs,
    )
    if train_status != 0:
        lib.destroy_mlp_model(ctypes.c_void_p(model))
        raise RuntimeError(f"train_mlp_model a renvoyé {train_status}.")

    return model


def save_linear_model(model: int, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_status = lib.save_linear_model(model, encode_path(model_path))
    if save_status != 0:
        raise RuntimeError(f"save_linear_model a renvoyé {save_status}.")


def save_mlp_model(model: int, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_status = lib.save_mlp_model(model, encode_path(model_path))
    if save_status != 0:
        raise RuntimeError(f"save_mlp_model a renvoyé {save_status}.")


def predict_linear_label(model: int, image_vector: np.ndarray) -> tuple[str, list[float]]:
    output = np.zeros(len(CLASSES), dtype=np.float64)
    _, x_ptr = as_double_pointer(image_vector)
    output_array, output_ptr = as_double_pointer(output)

    predict_status = lib.predict_linear_model(model, x_ptr, output_ptr)
    if predict_status != 0:
        raise RuntimeError(f"predict_linear_model a renvoyé {predict_status}.")

    predictions = np.asarray([output_array.copy()], dtype=np.float64)
    return labels_from_predictions(predictions)[0], [float(value) for value in output_array]


def predict_mlp_label(model: int, image_vector: np.ndarray) -> tuple[str, list[float]]:
    output = np.zeros(len(CLASSES), dtype=np.float64)
    _, x_ptr = as_double_pointer(image_vector)
    output_array, output_ptr = as_double_pointer(output)

    predict_status = lib.predict_mlp_model_raw(model, x_ptr, output_ptr)
    if predict_status != 0:
        raise RuntimeError(f"predict_mlp_model_raw a renvoyé {predict_status}.")

    predictions = np.asarray([output_array.copy()], dtype=np.float64)
    return labels_from_predictions(predictions)[0], [float(value) for value in output_array]


def verify_saved_models(
        linear_model_path: Path,
        mlp_model_path: Path,
        control_image: str,
) -> None:
    image_vector = load_image_as_vector(control_image)

    linear_model = lib.load_linear_model(encode_path(linear_model_path))
    if not linear_model:
        raise RuntimeError(f"Impossible de recharger {linear_model_path}.")

    try:
        label, scores = predict_linear_label(linear_model, image_vector)
        print(f"Contrôle linéaire: label={label} scores={scores}")
    finally:
        lib.destroy_linear_model(ctypes.c_void_p(linear_model))

    mlp_model = lib.load_mlp_model(encode_path(mlp_model_path))
    if not mlp_model:
        raise RuntimeError(f"Impossible de recharger {mlp_model_path}.")

    try:
        label, scores = predict_mlp_label(mlp_model, image_vector)
        print(f"Contrôle MLP: label={label} scores={scores}")
    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(mlp_model))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère les artefacts durables linéaire et MLP pour l’inférence 64x64."
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--linear-learning-rate", type=float, default=0.001)
    parser.add_argument("--linear-epochs", type=int, default=5)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-epochs", type=int, default=2)
    parser.add_argument("--mlp-hidden-sizes", type=str, default="64")
    parser.add_argument("--mlp-balanced-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-image", default=DEFAULT_CONTROL_IMAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold < 0 or args.fold > 4:
        raise ValueError("fold doit être compris entre 0 et 4.")
    if args.linear_learning_rate <= 0.0 or args.mlp_learning_rate <= 0.0:
        raise ValueError("Les taux d’apprentissage doivent être strictement positifs.")
    if args.linear_epochs <= 0 or args.mlp_epochs <= 0:
        raise ValueError("Les nombres d’époques doivent être strictement positifs.")

    rows = read_folds_csv()
    split_data = load_split(rows, validation_fold=args.fold)

    linear_model = train_linear_model(
        x_train=split_data.x_train,
        y_train=split_data.y_train,
        learning_rate=args.linear_learning_rate,
        epochs=args.linear_epochs,
    )
    try:
        save_linear_model(linear_model, LINEAR_MODEL_PATH)
    finally:
        lib.destroy_linear_model(ctypes.c_void_p(linear_model))

    mlp_split_data = split_data
    if args.mlp_balanced_train:
        mlp_split_data = balance_training_data(
            split_data=split_data,
            seed=args.seed + args.fold,
        )

    mlp_model = train_mlp_model(
        x_train=mlp_split_data.x_train,
        y_train=mlp_split_data.y_train,
        learning_rate=args.mlp_learning_rate,
        epochs=args.mlp_epochs,
        hidden_sizes=parse_hidden_sizes(args.mlp_hidden_sizes),
    )
    try:
        save_mlp_model(mlp_model, MLP_MODEL_PATH)
    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(mlp_model))

    verify_saved_models(
        linear_model_path=LINEAR_MODEL_PATH,
        mlp_model_path=MLP_MODEL_PATH,
        control_image=args.control_image,
    )

    print("\nArtefacts durables écrits:")
    print(f"- {LINEAR_MODEL_PATH}")
    print(f"- {MLP_MODEL_PATH}")


if __name__ == "__main__":
    main()