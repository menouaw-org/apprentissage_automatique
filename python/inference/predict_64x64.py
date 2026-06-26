import argparse
import ctypes
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from python.bindings.c_api import as_double_pointer, lib
from python.common.dataset_64x64 import CLASSES, INPUT_SIZE, load_image_as_vector


@dataclass(frozen=True)
class PredictionResult:
    label: str
    scores: list[float]
    image_path: str
    model_path: str
    model_type: str


def encode_path(path: str | Path) -> bytes:
    return str(Path(path)).encode("utf-8")


def label_from_scores(scores: np.ndarray) -> str:
    if scores.shape != (len(CLASSES),):
        raise ValueError(
            "Dimension de sortie inattendue: "
            f"{scores.shape}, attendu {(len(CLASSES),)}."
        )

    class_index = int(np.argmax(scores))
    return CLASSES[class_index]


def load_model(model_path: str | Path, model_type: str) -> int:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")

    if model_type == "linear":
        model = lib.load_linear_model(encode_path(path))
    elif model_type == "mlp":
        model = lib.load_mlp_model(encode_path(path))
    else:
        raise ValueError("model_type doit être 'linear' ou 'mlp'.")

    if not model:
        raise RuntimeError(f"Impossible de charger le modèle {model_type}: {path}")

    return model


def destroy_model(model: int, model_type: str) -> None:
    if model_type == "linear":
        lib.destroy_linear_model(ctypes.c_void_p(model))
    elif model_type == "mlp":
        lib.destroy_mlp_model(ctypes.c_void_p(model))
    else:
        raise ValueError("model_type doit être 'linear' ou 'mlp'.")


def predict_with_loaded_model(
        model: int,
        image_vector: np.ndarray,
        model_type: str,
) -> np.ndarray:
    if image_vector.shape != (INPUT_SIZE,):
        raise ValueError(
            "Dimension d'entrée inattendue: "
            f"{image_vector.shape}, attendu {(INPUT_SIZE,)}."
        )

    _, x_ptr = as_double_pointer(image_vector)
    output = np.zeros(len(CLASSES), dtype=np.float64)
    output_array, output_ptr = as_double_pointer(output)

    if model_type == "linear":
        predict_status = lib.predict_linear_model(model, x_ptr, output_ptr)
    elif model_type == "mlp":
        predict_status = lib.predict_mlp_model_raw(model, x_ptr, output_ptr)
    else:
        raise ValueError("model_type doit être 'linear' ou 'mlp'.")

    if predict_status != 0:
        raise RuntimeError(
            f"Erreur pendant la prédiction {model_type}: {predict_status}"
        )

    return np.asarray(output_array, dtype=np.float64)


def predict_image(
        image_path: str | Path,
        model_path: str | Path,
        model_type: str,
) -> PredictionResult:
    if model_type not in {"linear", "mlp"}:
        raise ValueError("model_type doit être 'linear' ou 'mlp'.")

    image_vector = load_image_as_vector(str(image_path))
    model = load_model(model_path=model_path, model_type=model_type)

    try:
        scores = predict_with_loaded_model(
            model=model,
            image_vector=image_vector,
            model_type=model_type,
        )
    finally:
        destroy_model(model=model, model_type=model_type)

    return PredictionResult(
        label=label_from_scores(scores),
        scores=[float(value) for value in scores],
        image_path=str(image_path),
        model_path=str(model_path),
        model_type=model_type,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute une inférence 64x64 avec un modèle sauvegardé."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--model-type", choices=["linear", "mlp"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
        model_type=args.model_type,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()