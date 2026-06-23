import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    LINEAR_TASK_CLASSIFICATION,
    as_double_pointer,
    lib,
)


def test_linear_multiclass_training() -> None:
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

    test_points = [
        (np.array([2.2, 2.1], dtype=np.float64), np.array([1.0, -1.0, -1.0])),
        (np.array([-2.2, 2.1], dtype=np.float64), np.array([-1.0, 1.0, -1.0])),
        (np.array([0.0, -2.2], dtype=np.float64), np.array([-1.0, -1.0, 1.0])),
    ]

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)

    model = lib.create_linear_model(2, 3, LINEAR_TASK_CLASSIFICATION)
    assert model, "Le modèle linéaire multi-classes n’a pas été créé."

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            0.1,
            50,
        )
        assert train_status == 0

        for x_test, expected in test_points:
            y_pred = np.zeros(3, dtype=np.float64)

            _, x_test_ptr = as_double_pointer(x_test)
            y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

            predict_status = lib.predict_linear_model(
                model,
                x_test_ptr,
                y_pred_ptr,
            )
            assert predict_status == 0

            np.testing.assert_array_equal(y_pred_array, expected)

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))


if __name__ == "__main__":
    test_linear_multiclass_training()
    print("Test d’apprentissage multi-classes réussi.")