import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    lib,
)


def predict_mlp(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(3, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
    assert predict_status == 0, (
        f"predict_mlp_model a renvoyé {predict_status}."
    )

    return y_pred_array.copy()


def test_mlp_interface_minimal() -> None:
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

    test_points = [
        (
            np.array([2.2, 2.1], dtype=np.float64),
            np.array([1.0, -1.0, -1.0], dtype=np.float64),
        ),
        (
            np.array([-2.2, 2.1], dtype=np.float64),
            np.array([-1.0, 1.0, -1.0], dtype=np.float64),
        ),
        (
            np.array([0.0, -2.2], dtype=np.float64),
            np.array([-1.0, -1.0, 1.0], dtype=np.float64),
        ),
    ]

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes)

    model = lib.create_mlp_model(
        2,
        3,
        hidden_sizes_array.shape[0],
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    assert model, "Le modèle MLP n’a pas été créé."

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

        for x_test, expected in test_points:
            prediction = predict_mlp(model, x_test)
            np.testing.assert_array_equal(prediction, expected)

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))


if __name__ == "__main__":
    test_mlp_interface_minimal()
    print("Test d’interface MLP réussi.")