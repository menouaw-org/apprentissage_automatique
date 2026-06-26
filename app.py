from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from python.common.dataset_64x64 import (
    CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PROJECT_ROOT,
)
from python.inference import predict_image

UPLOAD_DIR = PROJECT_ROOT / "tmp" / "gradio_uploads"

MODEL_TYPES = ("mlp", "linear")

MODEL_PATHS = {
    "linear": PROJECT_ROOT / "models" / "linear" / "linear_64x64_demo.pa_model",
    "mlp": PROJECT_ROOT / "models" / "mlp" / "mlp_64x64_demo.pa_model",
}


def prepare_uploaded_image(image: Image.Image | None) -> str:
    if image is None:
        raise ValueError("Aucune image n’a été fournie.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    prepared_image = image.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT))
    prepared_path = UPLOAD_DIR / "gradio_input_64x64.jpg"
    prepared_image.save(prepared_path, format="JPEG", quality=95)

    return prepared_path.relative_to(PROJECT_ROOT).as_posix()


def get_model_path(model_type: str) -> Path:
    if model_type not in MODEL_PATHS:
        raise ValueError("Le modèle doit être 'linear' ou 'mlp'.")

    model_path = MODEL_PATHS[model_type]
    if not model_path.exists():
        raise FileNotFoundError(f"Artefact modèle introuvable: {model_path}")

    return model_path


def scores_by_class(scores: list[float]) -> dict[str, float]:
    if len(scores) != len(CLASSES):
        raise ValueError(
            "Nombre de scores incohérent: "
            f"{len(scores)} score(s) reçu(s) pour {len(CLASSES)} classe(s)."
        )

    return {
        class_name: float(score)
        for class_name, score in zip(CLASSES, scores)
    }


def run_prediction(image: Image.Image | None, model_type: str) -> dict[str, Any]:
    try:
        prepared_image_path = prepare_uploaded_image(image)
        model_path = get_model_path(model_type)

        result = predict_image(
            image_path=prepared_image_path,
            model_path=model_path,
            model_type=model_type,
        )

        return {
            "label": result.label,
            "scores": scores_by_class(result.scores),
            "model_type": result.model_type,
            "model_path": result.model_path,
            "image_path": result.image_path,
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "model_type": model_type,
        }


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=run_prediction,
        inputs=[
            gr.Image(type="pil"),
            gr.Radio(choices=list(MODEL_TYPES), value="mlp"),
        ],
        outputs=gr.JSON(),
    )


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()