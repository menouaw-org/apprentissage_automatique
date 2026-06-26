import ctypes
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    encode_path,
    lib,
)


def predict_final(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(3, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
    assert predict_status == 0, (
        f"predict_mlp_model a renvoyé {predict_status}."
    )

    return y_pred_array.copy()


def predict_raw(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_raw = np.zeros(3, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_raw_array, y_raw_ptr = as_double_pointer(y_raw)

    predict_status = lib.predict_mlp_model_raw(model, x_ptr, y_raw_ptr)
    assert predict_status == 0, (
        f"predict_mlp_model_raw a renvoyé {predict_status}."
    )

    return y_raw_array.copy()


def main() -> None:
    x = np.array(
        [
            [2.0, 2.0],
            [2.5, 2.0],
            [2.0, 2.5],
            [-2.0, 2.0],
            [-2.5, 2.0],
            [-2.0, 2.5],
            [0.0, -2.0],
            [0.5, -2.5],
            [-0.5, -2.5],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )

    hidden_sizes = np.array([4], dtype=np.int32)
    x_test = np.array([2.2, 2.1], dtype=np.float64)

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes)

    model = lib.create_mlp_model(
        2,
        3,
        int(hidden_sizes_array.shape[0]),
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    assert model, "Le modèle MLP n’a pas été créé."

    loaded_model = None

    try:
        train_status = lib.train_mlp_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            0.05,
            1000,
        )
        assert train_status == 0, (
            f"train_mlp_model a renvoyé {train_status}."
        )

        prediction_before = predict_final(model, x_test)
        raw_before = predict_raw(model, x_test)

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "mlp_model.pa_model"

            save_status = lib.save_mlp_model(model, encode_path(model_path))
            assert save_status == 0, (
                f"save_mlp_model a renvoyé {save_status}."
            )
            assert model_path.exists(), "Le fichier de modèle n’a pas été créé."

            loaded_model = lib.load_mlp_model(encode_path(model_path))
            assert loaded_model, "Le modèle MLP sauvegardé n’a pas été rechargé."

            prediction_after = predict_final(loaded_model, x_test)
            raw_after = predict_raw(loaded_model, x_test)

        np.testing.assert_array_equal(prediction_after, prediction_before)
        np.testing.assert_allclose(raw_after, raw_before, rtol=0.0, atol=0.0)

    finally:
        if loaded_model:
            lib.destroy_mlp_model(ctypes.c_void_p(loaded_model))
        lib.destroy_mlp_model(ctypes.c_void_p(model))

    print("Test de sauvegarde / recharge MLP réussi.")


if __name__ == "__main__":
    main()