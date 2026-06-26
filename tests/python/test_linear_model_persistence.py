import ctypes
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    LINEAR_TASK_CLASSIFICATION,
    as_double_pointer,
    encode_path,
    lib,
)


def predict_one(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(3, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_linear_model(model, x_ptr, y_pred_ptr)
    assert predict_status == 0, (
        f"predict_linear_model a renvoyé {predict_status}."
    )

    return y_pred_array.copy()


def main() -> None:
    x = np.array(
        [
            [2.0, 2.0],
            [2.5, 2.0],
            [-2.0, 2.0],
            [-2.5, 2.0],
            [0.0, -2.0],
            [0.5, -2.5],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )

    x_test = np.array([2.2, 2.1], dtype=np.float64)

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)

    model = lib.create_linear_model(
        2,
        3,
        LINEAR_TASK_CLASSIFICATION,
    )
    assert model, "Le modèle linéaire n’a pas été créé."

    loaded_model = None

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            0.1,
            200,
        )
        assert train_status == 0, (
            f"train_linear_model a renvoyé {train_status}."
        )

        prediction_before = predict_one(model, x_test)

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "linear_model.pa_model"

            save_status = lib.save_linear_model(
                model,
                encode_path(model_path),
            )
            assert save_status == 0, (
                f"save_linear_model a renvoyé {save_status}."
            )
            assert model_path.exists(), "Le fichier de modèle n’a pas été créé."

            loaded_model = lib.load_linear_model(encode_path(model_path))
            assert loaded_model, "Le modèle linéaire sauvegardé n’a pas été rechargé."

            prediction_after = predict_one(loaded_model, x_test)

        np.testing.assert_array_equal(prediction_after, prediction_before)

    finally:
        if loaded_model:
            lib.destroy_linear_model(ctypes.c_void_p(loaded_model))
        lib.destroy_linear_model(ctypes.c_void_p(model))

    print("Test de sauvegarde / recharge linéaire réussi.")


if __name__ == "__main__":
    main()